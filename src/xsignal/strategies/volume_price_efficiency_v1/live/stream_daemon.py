from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import time

from xsignal.strategies.volume_price_efficiency_v1.live.bar_aggregator import (
    MultiIntervalAggregator,
)
from xsignal.strategies.volume_price_efficiency_v1.live.bar_buffer import RollingBarBuffer
from xsignal.strategies.volume_price_efficiency_v1.live.binance_adapter import (
    build_usd_futures_broker,
)
from xsignal.strategies.volume_price_efficiency_v1.live.binance_rest import (
    BinanceApiError,
    BinanceRestClient,
)
from xsignal.strategies.volume_price_efficiency_v1.live.health_gate import EntryHealthGate
from xsignal.strategies.volume_price_efficiency_v1.live.market_data import (
    fetch_closed_klines,
    fetch_closed_klines_range,
)
from xsignal.strategies.volume_price_efficiency_v1.live.realtime import RealtimeStrategyService
from xsignal.strategies.volume_price_efficiency_v1.live.reconcile import run_reconciliation_pass
from xsignal.strategies.volume_price_efficiency_v1.live.recovery import (
    event_from_market_bar,
    latest_closed_1m_open_time,
    market_bar_from_event,
    recovery_start_time,
    replay_closed_1m_events,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore
from xsignal.strategies.volume_price_efficiency_v1.live.ws_market import (
    BINANCE_USD_FUTURES_LIVE_WS_BASE_URL,
    BINANCE_USD_FUTURES_TESTNET_WS_BASE_URL,
    build_combined_stream_urls,
    parse_kline_stream_event,
    validate_interval,
)


DEFAULT_REALTIME_INTERVALS = ("1h", "4h", "1d")
DEFAULT_REALTIME_MAX_STREAMS_PER_CONNECTION = 100


@dataclass(frozen=True)
class StreamUrlSpec:
    url: str
    symbols: tuple[str, ...]


@dataclass(frozen=True)
class StreamDaemonConfig:
    mode: str
    db_path: Path | str
    intervals: tuple[str, ...] = DEFAULT_REALTIME_INTERVALS
    lookback_bars: int = 120
    max_symbols: int | None = None
    max_streams: int = DEFAULT_REALTIME_MAX_STREAMS_PER_CONNECTION
    reconnect_backoff_seconds: float = 5.0
    rate_limit_backoff_seconds: float = 60.0
    reconcile_interval_seconds: float = 300.0
    seed_sleep_ms: int = 20
    recovery_sleep_ms: int = 500
    stop_after_events: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"testnet", "live"}:
            raise ValueError("mode must be testnet or live")
        if self.lookback_bars <= 0:
            raise ValueError("lookback_bars must be positive")
        if self.max_symbols is not None and self.max_symbols <= 0:
            raise ValueError("max_symbols must be positive")
        if self.rate_limit_backoff_seconds < 0:
            raise ValueError("rate_limit_backoff_seconds must be non-negative")
        if self.recovery_sleep_ms < 0:
            raise ValueError("recovery_sleep_ms must be non-negative")
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
    for interval in intervals:
        validate_interval(interval)
    return [spec.url for spec in build_daemon_stream_specs(mode=mode, symbols=symbols, max_streams=max_streams)]


def build_daemon_stream_specs(
    *,
    mode: str,
    symbols: list[str],
    max_streams: int,
) -> list[StreamUrlSpec]:
    specs = []
    for chunk in _chunk_symbols(symbols, max_streams=max_streams):
        url = build_combined_stream_urls(
            symbols=list(chunk),
            intervals=["1m"],
            base_url=ws_base_url_for_mode(mode),
            max_streams=max_streams,
        )[0]
        specs.append(StreamUrlSpec(url=url, symbols=chunk))
    return specs


def _chunk_symbols(symbols: list[str], *, max_streams: int) -> list[tuple[str, ...]]:
    if max_streams <= 0:
        raise ValueError("max_streams must be positive")
    return [
        tuple(symbols[index : index + max_streams])
        for index in range(0, len(symbols), max_streams)
    ]


def seed_rolling_buffers(
    rest_client: BinanceRestClient,
    *,
    symbols: list[str],
    intervals: list[str] | tuple[str, ...],
    lookback_bars: int,
    server_time_ms: int,
    max_bars: int,
    seed_sleep_ms: int = 0,
    rate_limit_backoff_seconds: float = 60.0,
) -> dict[str, RollingBarBuffer]:
    buffers = {
        interval: RollingBarBuffer(interval=interval, max_bars=max_bars)
        for interval in intervals
    }
    for interval, buffer in buffers.items():
        for symbol in symbols:
            while True:
                try:
                    rows = fetch_closed_klines(
                        rest_client,
                        symbol=symbol,
                        interval=interval,
                        limit=lookback_bars,
                        server_time_ms=server_time_ms,
                    )
                    break
                except BinanceApiError as exc:
                    if not _is_rate_limit_error(exc):
                        raise
                    _print_event(
                        "seed_rate_limited",
                        symbol=symbol,
                        interval=interval,
                        backoff_seconds=rate_limit_backoff_seconds,
                    )
                    time.sleep(rate_limit_backoff_seconds)
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
        rate_limit_backoff_seconds=config.rate_limit_backoff_seconds,
    )
    _print_event("seed_finished", mode=config.mode, symbols=len(symbols))
    _print_event(
        "stream_daemon_started",
        mode=config.mode,
        symbols=len(symbols),
        intervals=list(config.intervals),
    )
    entry_gate = EntryHealthGate()
    summary = run_reconciliation_pass(
        store=store,
        broker=broker,
        symbols=symbols,
        environment=config.mode,
        allow_repair=False,
        now=datetime.now(timezone.utc),
    )
    entry_gate.mark_reconcile(summary)
    _print_reconcile_event(summary=summary, entry_gate=entry_gate)

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
    aggregator = MultiIntervalAggregator(intervals=config.intervals)
    specs = build_daemon_stream_specs(
        mode=config.mode,
        symbols=symbols,
        max_streams=config.max_streams,
    )
    stop_event = asyncio.Event()
    recovery_lock = asyncio.Lock()
    counter = _EventCounter(limit=config.stop_after_events)
    _print_event("startup_recovery_started", mode=config.mode, symbols=len(symbols))
    await _recover_symbols_1m_gap_async(
        recovery_lock=recovery_lock,
        store=store,
        rest_client=broker.rest_client,
        aggregator=aggregator,
        service=service,
        symbols=symbols,
        intervals=config.intervals,
        recovery_sleep_ms=config.recovery_sleep_ms,
    )
    _print_event("startup_recovery_finished", mode=config.mode, symbols=len(symbols))
    tasks = [
        asyncio.create_task(
            _consume_stream_url(
                spec,
                store,
                broker.rest_client,
                aggregator,
                service,
                entry_gate,
                stop_event,
                counter,
                config,
                recovery_lock,
                recover_before_connect=False,
            )
        )
        for spec in specs
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
                    entry_gate=entry_gate,
                    stop_event=stop_event,
                )
            )
        )
    await asyncio.gather(*tasks)
    return 0


async def _consume_stream_url(
    spec: StreamUrlSpec,
    store: LiveStore,
    rest_client: BinanceRestClient,
    aggregator: MultiIntervalAggregator,
    service: RealtimeStrategyService,
    entry_gate: EntryHealthGate,
    stop_event: asyncio.Event,
    counter: "_EventCounter",
    config: StreamDaemonConfig,
    recovery_lock: asyncio.Lock,
    recover_before_connect: bool = True,
) -> None:
    import websockets

    needs_recovery = recover_before_connect
    while not stop_event.is_set():
        try:
            if needs_recovery:
                await _recover_symbols_1m_gap_async(
                    recovery_lock=recovery_lock,
                    store=store,
                    rest_client=rest_client,
                    aggregator=aggregator,
                    service=service,
                    symbols=list(spec.symbols),
                    intervals=config.intervals,
                    recovery_sleep_ms=config.recovery_sleep_ms,
                )
            needs_recovery = True
            async with websockets.connect(spec.url, ping_interval=180, ping_timeout=600) as websocket:
                _print_event("stream_connected", url=spec.url.split("?streams=", 1)[0])
                async for message in websocket:
                    if stop_event.is_set():
                        return
                    if not _should_parse_stream_message(message, service):
                        if counter.incremented_past_limit():
                            stop_event.set()
                            return
                        continue
                    payload = json.loads(message)
                    if not _should_parse_stream_payload(payload, service):
                        if counter.incremented_past_limit():
                            stop_event.set()
                            return
                        continue
                    event = parse_kline_stream_event(payload)
                    if event.is_closed:
                        _process_closed_1m_event(
                            store=store,
                            aggregator=aggregator,
                            service=service,
                            event=event,
                            entry_gate=entry_gate,
                        )
                    else:
                        result = service.process_price_event(event, allow_pyramid_add=True)
                        _print_strategy_action(event=event, result=result)
                    if counter.incremented_past_limit():
                        stop_event.set()
                        return
        except Exception as exc:  # noqa: BLE001
            entry_gate.mark_stream_error(str(exc))
            _print_event("stream_error", error=str(exc), entry_gate=entry_gate.snapshot())
            await asyncio.sleep(_stream_error_backoff_seconds(exc, config))


def _should_parse_stream_message(message: str | bytes, service: RealtimeStrategyService) -> bool:
    if isinstance(message, bytes):
        try:
            message = message.decode("utf-8")
        except UnicodeDecodeError:
            return True
    if '"x":true' in message:
        return True
    if '"x":false' not in message:
        return True
    symbol = _extract_json_string_field(message, '"s":"')
    if not symbol:
        return True
    return service.has_active_symbol_position(symbol)


def _extract_json_string_field(message: str, marker: str) -> str | None:
    start = message.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = message.find('"', start)
    if end < 0:
        return None
    return message[start:end]


def _should_parse_stream_payload(payload: dict, service: RealtimeStrategyService) -> bool:
    data = payload.get("data", payload)
    if data.get("e") != "kline":
        return True
    kline = data.get("k")
    if not isinstance(kline, dict):
        return True
    if bool(kline.get("x")):
        return True
    symbol = str(kline.get("s") or data.get("s") or "")
    if not symbol:
        return True
    return service.has_active_symbol_position(symbol)


def _stream_error_backoff_seconds(exc: Exception, config: StreamDaemonConfig) -> float:
    if isinstance(exc, BinanceApiError) and _is_rate_limit_error(exc):
        return max(config.reconnect_backoff_seconds, config.rate_limit_backoff_seconds)
    text = str(exc)
    if "-1003" in text or " 429 " in text:
        return max(config.reconnect_backoff_seconds, config.rate_limit_backoff_seconds)
    return config.reconnect_backoff_seconds


def _is_rate_limit_error(exc: BinanceApiError) -> bool:
    return exc.status == 429 or exc.code == -1003


def _recover_symbols_1m_gap(
    *,
    store: LiveStore,
    rest_client: BinanceRestClient,
    aggregator: MultiIntervalAggregator,
    service: RealtimeStrategyService,
    symbols: list[str],
    intervals: tuple[str, ...],
    recovery_sleep_ms: int,
) -> None:
    server_time_ms = int(rest_client.request("GET", "/fapi/v1/time")["serverTime"])
    end_time = latest_closed_1m_open_time(server_time_ms)
    recovered_source = 0
    recovered_aggregates = 0
    for symbol in symbols:
        start_time = recovery_start_time(
            store=store,
            symbol=symbol,
            target_intervals=intervals,
            server_time_ms=server_time_ms,
        )
        rows = fetch_closed_klines_range(
            rest_client,
            symbol=symbol,
            interval="1m",
            start_time=start_time,
            end_time=end_time,
            server_time_ms=server_time_ms,
        )
        result = replay_closed_1m_events(
            store=store,
            aggregator=aggregator,
            sink=service,
            events=[event_from_market_bar(row) for row in rows],
            allow_entries=False,
            allow_pyramid_add=False,
        )
        recovered_source += result.source_bars
        recovered_aggregates += result.aggregated_bars
        if recovery_sleep_ms > 0:
            time.sleep(recovery_sleep_ms / 1000)
    if recovered_source or recovered_aggregates:
        _print_event(
            "market_gap_recovered",
            symbols=len(symbols),
            source_bars=recovered_source,
            aggregated_bars=recovered_aggregates,
        )


async def _recover_symbols_1m_gap_async(
    *,
    recovery_lock: asyncio.Lock,
    store: LiveStore,
    rest_client: BinanceRestClient,
    aggregator: MultiIntervalAggregator,
    service: RealtimeStrategyService,
    symbols: list[str],
    intervals: tuple[str, ...],
    recovery_sleep_ms: int,
    fetch_range=fetch_closed_klines_range,
) -> None:
    async with recovery_lock:
        server_time_ms = await asyncio.to_thread(
            lambda: int(rest_client.request("GET", "/fapi/v1/time")["serverTime"])
        )
        end_time = latest_closed_1m_open_time(server_time_ms)
        recovered_source = 0
        recovered_aggregates = 0
        for symbol in symbols:
            start_time = recovery_start_time(
                store=store,
                symbol=symbol,
                target_intervals=intervals,
                server_time_ms=server_time_ms,
            )
            rows = await asyncio.to_thread(
                fetch_range,
                rest_client,
                symbol=symbol,
                interval="1m",
                start_time=start_time,
                end_time=end_time,
                server_time_ms=server_time_ms,
            )
            result = replay_closed_1m_events(
                store=store,
                aggregator=aggregator,
                sink=service,
                events=[event_from_market_bar(row) for row in rows],
                allow_entries=False,
                allow_pyramid_add=False,
            )
            recovered_source += result.source_bars
            recovered_aggregates += result.aggregated_bars
            if recovery_sleep_ms > 0:
                await asyncio.sleep(recovery_sleep_ms / 1000)
        if recovered_source or recovered_aggregates:
            _print_event(
                "market_gap_recovered",
                symbols=len(symbols),
                source_bars=recovered_source,
                aggregated_bars=recovered_aggregates,
            )


def _process_closed_1m_event(
    *,
    store: LiveStore,
    aggregator: MultiIntervalAggregator,
    service: RealtimeStrategyService,
    event,
    entry_gate: EntryHealthGate | None = None,
) -> None:
    store.upsert_market_bar(market_bar_from_event(event), commit=False)
    store.advance_market_cursor(
        symbol=event.symbol,
        interval="1m",
        open_time=event.open_time,
        commit=False,
    )
    price_result = service.process_price_event(event, allow_pyramid_add=True)
    _print_strategy_action(event=event, result=price_result)
    for aggregate in aggregator.apply_1m_event(event):
        store.upsert_market_bar(market_bar_from_event(aggregate), commit=False)
        result = service.process_closed_bar(
            aggregate,
            allow_entry=True if entry_gate is None else entry_gate.allow_entries,
            allow_pyramid_add=True,
        )
        _print_strategy_action(event=aggregate, result=result)
    store.connection.commit()


def _print_strategy_action(*, event, result) -> None:
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


async def _periodic_reconcile(
    *,
    store: LiveStore,
    broker,
    symbols: list[str],
    environment: str,
    interval_seconds: float,
    entry_gate: EntryHealthGate,
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
        entry_gate.mark_reconcile(summary)
        _print_reconcile_event(summary=summary, entry_gate=entry_gate)


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


def _print_reconcile_event(*, summary, entry_gate: EntryHealthGate) -> None:
    _print_event(
        "reconcile_pass",
        status="error" if summary.error_count else "clean",
        errors=summary.error_count,
        entry_gate=entry_gate.snapshot(),
    )
