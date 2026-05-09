from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from xsignal.strategies.volume_price_efficiency_v1.live.bar_buffer import RollingBarBuffer
from xsignal.strategies.volume_price_efficiency_v1.live.binance_adapter import (
    build_usd_futures_broker,
)
from xsignal.strategies.volume_price_efficiency_v1.live.binance_rest import BinanceRestClient
from xsignal.strategies.volume_price_efficiency_v1.live.market_data import fetch_closed_klines
from xsignal.strategies.volume_price_efficiency_v1.live.realtime import RealtimeStrategyService
from xsignal.strategies.volume_price_efficiency_v1.live.reconcile import run_reconciliation_pass
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore
from xsignal.strategies.volume_price_efficiency_v1.live.ws_market import (
    BINANCE_MAX_COMBINED_STREAMS,
    BINANCE_USD_FUTURES_LIVE_WS_BASE_URL,
    BINANCE_USD_FUTURES_TESTNET_WS_BASE_URL,
    build_combined_stream_urls,
    parse_kline_stream_event,
    validate_interval,
)


DEFAULT_REALTIME_INTERVALS = ("1h", "4h", "1d")


@dataclass(frozen=True)
class StreamDaemonConfig:
    mode: str
    db_path: Path | str
    intervals: tuple[str, ...] = DEFAULT_REALTIME_INTERVALS
    lookback_bars: int = 120
    max_symbols: int | None = None
    max_streams: int = BINANCE_MAX_COMBINED_STREAMS
    reconnect_backoff_seconds: float = 5.0
    reconcile_interval_seconds: float = 300.0
    seed_sleep_ms: int = 20
    stop_after_events: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"testnet", "live"}:
            raise ValueError("mode must be testnet or live")
        if self.lookback_bars <= 0:
            raise ValueError("lookback_bars must be positive")
        if self.max_symbols is not None and self.max_symbols <= 0:
            raise ValueError("max_symbols must be positive")
        for interval in self.intervals:
            validate_interval(interval)


def ws_base_url_for_mode(mode: str) -> str:
    if mode == "testnet":
        return BINANCE_USD_FUTURES_TESTNET_WS_BASE_URL
    if mode == "live":
        return BINANCE_USD_FUTURES_LIVE_WS_BASE_URL
    raise ValueError(f"unsupported mode: {mode}")


def build_daemon_stream_urls(
    *,
    mode: str,
    symbols: list[str],
    intervals: list[str] | tuple[str, ...],
    max_streams: int,
) -> list[str]:
    return build_combined_stream_urls(
        symbols=symbols,
        intervals=list(intervals),
        base_url=ws_base_url_for_mode(mode),
        max_streams=max_streams,
    )


def seed_rolling_buffers(
    rest_client: BinanceRestClient,
    *,
    symbols: list[str],
    intervals: list[str] | tuple[str, ...],
    lookback_bars: int,
    server_time_ms: int,
    max_bars: int,
    seed_sleep_ms: int = 0,
) -> dict[str, RollingBarBuffer]:
    buffers = {
        interval: RollingBarBuffer(interval=interval, max_bars=max_bars)
        for interval in intervals
    }
    for interval, buffer in buffers.items():
        for symbol in symbols:
            rows = fetch_closed_klines(
                rest_client,
                symbol=symbol,
                interval=interval,
                limit=lookback_bars,
                server_time_ms=server_time_ms,
            )
            buffer.seed_rows(rows)
            if seed_sleep_ms > 0:
                time.sleep(seed_sleep_ms / 1000)
    return buffers


def run_stream_daemon(*, config: StreamDaemonConfig, credentials) -> int:
    return asyncio.run(run_stream_daemon_async(config=config, credentials=credentials))


async def run_stream_daemon_async(*, config: StreamDaemonConfig, credentials) -> int:
    broker = build_usd_futures_broker(mode=config.mode, credentials=credentials)
    store = LiveStore.open(Path(config.db_path))
    store.initialize()
    metadata_by_symbol = broker.list_trading_usdt_perpetual_metadata()
    symbols = list(metadata_by_symbol)
    if config.max_symbols is not None:
        symbols = symbols[: config.max_symbols]
        metadata_by_symbol = {
            symbol: metadata_by_symbol[symbol]
            for symbol in symbols
        }
    if not symbols:
        raise RuntimeError("no Binance USD-M symbols selected")

    server_time_ms = int(broker.rest_client.request("GET", "/fapi/v1/time")["serverTime"])
    _print_event(
        "seed_started",
        mode=config.mode,
        symbols=len(symbols),
        intervals=list(config.intervals),
    )
    buffers = seed_rolling_buffers(
        broker.rest_client,
        symbols=symbols,
        intervals=config.intervals,
        lookback_bars=config.lookback_bars,
        server_time_ms=server_time_ms,
        max_bars=config.lookback_bars,
        seed_sleep_ms=config.seed_sleep_ms,
    )
    _print_event("seed_finished", mode=config.mode, symbols=len(symbols))
    _print_event(
        "stream_daemon_started",
        mode=config.mode,
        symbols=len(symbols),
        intervals=list(config.intervals),
    )
    run_reconciliation_pass(
        store=store,
        broker=broker,
        symbols=symbols,
        environment=config.mode,
        allow_repair=False,
        now=datetime.now(timezone.utc),
    )

    service = RealtimeStrategyService(
        store=store,
        broker=broker,
        config=_live_config(config.mode),
        environment=config.mode,
        buffers=buffers,
        metadata_by_symbol=metadata_by_symbol,
        account_provider=lambda: broker.get_account_snapshot(
            mode=config.mode,
            daily_realized_pnl=0.0,
        ),
        now_provider=lambda: datetime.now(timezone.utc),
    )
    urls = build_daemon_stream_urls(
        mode=config.mode,
        symbols=symbols,
        intervals=config.intervals,
        max_streams=config.max_streams,
    )
    stop_event = asyncio.Event()
    counter = _EventCounter(limit=config.stop_after_events)
    tasks = [
        asyncio.create_task(_consume_stream_url(url, service, stop_event, counter, config))
        for url in urls
    ]
    if config.reconcile_interval_seconds > 0:
        tasks.append(
            asyncio.create_task(
                _periodic_reconcile(
                    store=store,
                    broker=broker,
                    symbols=symbols,
                    environment=config.mode,
                    interval_seconds=config.reconcile_interval_seconds,
                    stop_event=stop_event,
                )
            )
        )
    await asyncio.gather(*tasks)
    return 0


async def _consume_stream_url(
    url: str,
    service: RealtimeStrategyService,
    stop_event: asyncio.Event,
    counter: "_EventCounter",
    config: StreamDaemonConfig,
) -> None:
    import websockets

    while not stop_event.is_set():
        try:
            async with websockets.connect(url, ping_interval=180, ping_timeout=600) as websocket:
                _print_event("stream_connected", url=url.split("?streams=", 1)[0])
                async for message in websocket:
                    if stop_event.is_set():
                        return
                    payload = json.loads(message)
                    event = parse_kline_stream_event(payload)
                    result = service.process_event(event)
                    if result.entries or result.adds or result.stop_updates:
                        _print_event(
                            "strategy_action",
                            symbol=event.symbol,
                            interval=event.interval,
                            closed=event.is_closed,
                            entries=result.entries,
                            adds=result.adds,
                            stop_updates=result.stop_updates,
                        )
                    if counter.incremented_past_limit():
                        stop_event.set()
                        return
        except Exception as exc:  # noqa: BLE001
            _print_event("stream_error", error=str(exc))
            await asyncio.sleep(config.reconnect_backoff_seconds)


async def _periodic_reconcile(
    *,
    store: LiveStore,
    broker,
    symbols: list[str],
    environment: str,
    interval_seconds: float,
    stop_event: asyncio.Event,
) -> None:
    while not stop_event.is_set():
        await asyncio.sleep(interval_seconds)
        if stop_event.is_set():
            return
        summary = run_reconciliation_pass(
            store=store,
            broker=broker,
            symbols=symbols,
            environment=environment,
            allow_repair=False,
            now=datetime.now(timezone.utc),
        )
        _print_event(
            "reconcile_pass",
            status="error" if summary.error_count else "clean",
            errors=summary.error_count,
        )


class _EventCounter:
    def __init__(self, *, limit: int | None) -> None:
        self.limit = limit
        self.count = 0

    def incremented_past_limit(self) -> bool:
        self.count += 1
        return self.limit is not None and self.count >= self.limit


def _live_config(mode: str):
    from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig

    return LiveTradingConfig(mode=mode, live_acknowledgement=(mode == "live"))


def _print_event(event: str, **fields) -> None:
    print(json.dumps({"event": event, **fields}, sort_keys=True), flush=True)
