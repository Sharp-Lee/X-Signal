from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from collections.abc import Awaitable, Callable
from typing import TypeVar

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
from xsignal.strategies.volume_price_efficiency_v1.live.market_pipeline import (
    ClosedBarBatchWorker,
    MarketEventRouter,
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
DEFAULT_REALTIME_MAX_STREAMS_PER_CONNECTION = 200
DEFAULT_STREAM_MAX_LIFETIME_SECONDS = 23 * 60 * 60
DEFAULT_STREAM_ROTATION_JITTER_SECONDS = 30 * 60
_T = TypeVar("_T")


class _StreamRotationDue(Exception):
    pass


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
    stream_max_lifetime_seconds: float = DEFAULT_STREAM_MAX_LIFETIME_SECONDS
    stream_rotation_jitter_seconds: float = DEFAULT_STREAM_ROTATION_JITTER_SECONDS
    reconnect_backoff_seconds: float = 5.0
    rate_limit_backoff_seconds: float = 60.0
    reconcile_interval_seconds: float = 300.0
    seed_sleep_ms: int = 20
    recovery_sleep_ms: int = 500
    closed_poll_sleep_ms: int = 25
    closed_poll_grace_seconds: float = 2.0
    closed_poll_fetch_limit: int = 99
    stop_after_events: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in {"testnet", "live"}:
            raise ValueError("mode must be testnet or live")
        if self.lookback_bars <= 0:
            raise ValueError("lookback_bars must be positive")
        if self.max_symbols is not None and self.max_symbols <= 0:
            raise ValueError("max_symbols must be positive")
        if self.max_streams <= 0:
            raise ValueError("max_streams must be positive")
        if self.stream_max_lifetime_seconds < 0:
            raise ValueError("stream_max_lifetime_seconds must be non-negative")
        if self.stream_rotation_jitter_seconds < 0:
            raise ValueError("stream_rotation_jitter_seconds must be non-negative")
        if self.rate_limit_backoff_seconds < 0:
            raise ValueError("rate_limit_backoff_seconds must be non-negative")
        if self.recovery_sleep_ms < 0:
            raise ValueError("recovery_sleep_ms must be non-negative")
        if self.closed_poll_sleep_ms < 0:
            raise ValueError("closed_poll_sleep_ms must be non-negative")
        if self.closed_poll_grace_seconds < 0:
            raise ValueError("closed_poll_grace_seconds must be non-negative")
        if self.closed_poll_fetch_limit <= 0:
            raise ValueError("closed_poll_fetch_limit must be positive")
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


def _active_recovery_symbols(service: RealtimeStrategyService, symbols: list[str]) -> list[str]:
    service.refresh_active_symbols()
    active = set(service.active_symbols())
    if not active:
        return []
    return [symbol for symbol in symbols if symbol in active]


def _stream_rotation_deadline(
    spec: StreamUrlSpec,
    *,
    config: StreamDaemonConfig,
    now_monotonic: float | None = None,
) -> float | None:
    if config.stream_max_lifetime_seconds <= 0:
        return None
    now = time.monotonic() if now_monotonic is None else now_monotonic
    jitter = _stream_rotation_jitter_seconds(spec, config=config)
    return now + max(config.stream_max_lifetime_seconds - jitter, 0.0)


def _stream_rotation_jitter_seconds(spec: StreamUrlSpec, *, config: StreamDaemonConfig) -> float:
    max_jitter = min(config.stream_rotation_jitter_seconds, config.stream_max_lifetime_seconds)
    if max_jitter <= 0:
        return 0.0
    seed = "|".join(spec.symbols) or spec.url
    digest = hashlib.blake2b(seed.encode("utf-8"), digest_size=8).digest()
    fraction = int.from_bytes(digest, "big") / ((1 << 64) - 1)
    return fraction * max_jitter


async def _next_stream_message_or_rotate(
    stream_iterator,
    *,
    rotation_deadline: float | None,
    monotonic=time.monotonic,
) -> str | bytes:
    if rotation_deadline is None:
        return await stream_iterator.__anext__()
    remaining = rotation_deadline - monotonic()
    if remaining <= 0:
        raise _StreamRotationDue
    try:
        return await asyncio.wait_for(stream_iterator.__anext__(), timeout=remaining)
    except asyncio.TimeoutError as exc:
        raise _StreamRotationDue from exc


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


def seed_rolling_buffers_from_store(
    store: LiveStore,
    *,
    symbols: list[str],
    intervals: list[str] | tuple[str, ...],
    lookback_bars: int,
    max_bars: int,
) -> dict[str, RollingBarBuffer]:
    buffers = {
        interval: RollingBarBuffer(interval=interval, max_bars=max_bars)
        for interval in intervals
    }
    for interval, buffer in buffers.items():
        for symbol in symbols:
            buffer.seed_rows(
                store.list_recent_market_bars(
                    symbol=symbol,
                    interval=interval,
                    limit=lookback_bars,
                )
            )
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

    _print_event(
        "seed_started",
        mode=config.mode,
        symbols=len(symbols),
        intervals=list(config.intervals),
        source="store",
    )
    buffers = seed_rolling_buffers_from_store(
        store,
        symbols=symbols,
        intervals=config.intervals,
        lookback_bars=config.lookback_bars,
        max_bars=config.lookback_bars,
    )
    _print_event("seed_finished", mode=config.mode, symbols=len(symbols), source="store")
    _print_event(
        "stream_daemon_started",
        mode=config.mode,
        symbols=len(symbols),
        intervals=list(config.intervals),
    )
    entry_gate = EntryHealthGate()
    async def initial_reconcile():
        return run_reconciliation_pass(
            store=store,
            broker=broker,
            symbols=symbols,
            environment=config.mode,
            allow_repair=False,
            now=datetime.now(timezone.utc),
        )

    summary = await _retry_startup_step_after_rate_limit(
        step="reconcile",
        config=config,
        entry_gate=entry_gate,
        action=initial_reconcile,
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
    stop_event = asyncio.Event()
    recovery_lock = asyncio.Lock()
    counter = _EventCounter(limit=config.stop_after_events)
    startup_recovery_symbols = _active_recovery_symbols(service, symbols)
    _print_event(
        "startup_recovery_started",
        mode=config.mode,
        symbols=len(startup_recovery_symbols),
        skipped_symbols=len(symbols) - len(startup_recovery_symbols),
        universe_symbols=len(symbols),
    )

    async def initial_recovery() -> None:
        if not startup_recovery_symbols:
            return
        await _recover_symbols_1m_gap_async(
            recovery_lock=recovery_lock,
            store=store,
            rest_client=broker.rest_client,
            aggregator=aggregator,
            service=service,
            symbols=startup_recovery_symbols,
            intervals=config.intervals,
            recovery_sleep_ms=config.recovery_sleep_ms,
            fetch_limit=config.closed_poll_fetch_limit,
        )

    await _retry_startup_step_after_rate_limit(
        step="startup_recovery",
        config=config,
        entry_gate=entry_gate,
        action=initial_recovery,
    )
    _print_event(
        "startup_recovery_finished",
        mode=config.mode,
        symbols=len(startup_recovery_symbols),
        skipped_symbols=len(symbols) - len(startup_recovery_symbols),
        universe_symbols=len(symbols),
    )
    tasks = _build_market_data_tasks(
        store=store,
        rest_client=broker.rest_client,
        aggregator=aggregator,
        service=service,
        entry_gate=entry_gate,
        symbols=symbols,
        stop_event=stop_event,
        counter=counter,
        config=config,
        recovery_lock=recovery_lock,
    )
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
                    service=service,
                    stop_event=stop_event,
                )
            )
        )
    await asyncio.gather(*tasks)
    return 0


def _build_market_data_tasks(
    *,
    store: LiveStore,
    rest_client: BinanceRestClient,
    aggregator: MultiIntervalAggregator,
    service: RealtimeStrategyService,
    entry_gate: EntryHealthGate,
    symbols: list[str],
    stop_event: asyncio.Event,
    counter: "_EventCounter",
    config: StreamDaemonConfig,
    recovery_lock: asyncio.Lock,
) -> list[asyncio.Task]:
    return [
        asyncio.create_task(
            _full_universe_stream_manager(
                store=store,
                rest_client=rest_client,
                aggregator=aggregator,
                service=service,
                entry_gate=entry_gate,
                symbols=symbols,
                stop_event=stop_event,
                counter=counter,
                config=config,
                recovery_lock=recovery_lock,
            )
        )
    ]


async def _full_universe_stream_manager(
    *,
    store: LiveStore,
    rest_client: BinanceRestClient,
    aggregator: MultiIntervalAggregator,
    service: RealtimeStrategyService,
    entry_gate: EntryHealthGate,
    symbols: list[str],
    stop_event: asyncio.Event,
    counter: "_EventCounter",
    config: StreamDaemonConfig,
    recovery_lock: asyncio.Lock,
) -> None:
    specs = build_daemon_stream_specs(
        mode=config.mode,
        symbols=symbols,
        max_streams=config.max_streams,
    )
    _print_event(
        "full_universe_streams_started",
        symbols=len(symbols),
        streams=len(specs),
    )
    await asyncio.gather(
        *[
            _consume_full_universe_stream_url(
                spec=spec,
                store=store,
                rest_client=rest_client,
                aggregator=aggregator,
                service=service,
                entry_gate=entry_gate,
                stop_event=stop_event,
                counter=counter,
                config=config,
                recovery_lock=recovery_lock,
                recover_before_connect=False,
            )
            for spec in specs
        ]
    )


async def _poll_closed_1m_loop(
    *,
    store: LiveStore,
    rest_client: BinanceRestClient,
    aggregator: MultiIntervalAggregator,
    service: RealtimeStrategyService,
    entry_gate: EntryHealthGate,
    symbols: list[str],
    intervals: tuple[str, ...],
    stop_event: asyncio.Event,
    counter: "_EventCounter",
    config: StreamDaemonConfig,
    recovery_lock: asyncio.Lock,
) -> None:
    while not stop_event.is_set():
        try:
            await _poll_closed_1m_once_async(
                recovery_lock=recovery_lock,
                store=store,
                rest_client=rest_client,
                aggregator=aggregator,
                service=service,
                entry_gate=entry_gate,
                symbols=symbols,
                intervals=intervals,
                poll_sleep_ms=config.closed_poll_sleep_ms,
                fetch_limit=config.closed_poll_fetch_limit,
            )
            _print_event("closed_bar_poll_finished", symbols=len(symbols))
            if counter.incremented_past_limit():
                stop_event.set()
                return
            await asyncio.sleep(
                _seconds_until_next_closed_1m_poll(
                    datetime.now(timezone.utc),
                    grace_seconds=config.closed_poll_grace_seconds,
                )
            )
        except Exception as exc:  # noqa: BLE001
            entry_gate.mark_stream_error(str(exc))
            _print_event("closed_bar_poll_error", error=str(exc), entry_gate=entry_gate.snapshot())
            await asyncio.sleep(_stream_error_backoff_seconds(exc, config))


async def _active_position_stream_manager(
    *,
    service: RealtimeStrategyService,
    entry_gate: EntryHealthGate,
    stop_event: asyncio.Event,
    counter: "_EventCounter",
    config: StreamDaemonConfig,
) -> None:
    current_symbols: tuple[str, ...] = ()
    stream_stop: asyncio.Event | None = None
    tasks: list[asyncio.Task] = []
    try:
        while not stop_event.is_set():
            service.refresh_active_symbols()
            active_symbols = service.active_symbols()
            if active_symbols != current_symbols:
                if stream_stop is not None:
                    stream_stop.set()
                for task in tasks:
                    task.cancel()
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                tasks = []
                stream_stop = None
                current_symbols = active_symbols
                if active_symbols:
                    stream_stop = asyncio.Event()
                    specs = build_daemon_stream_specs(
                        mode=config.mode,
                        symbols=list(active_symbols),
                        max_streams=config.max_streams,
                    )
                    tasks = [
                        asyncio.create_task(
                            _consume_active_position_stream_url(
                                spec=spec,
                                service=service,
                                entry_gate=entry_gate,
                                stop_event=stream_stop,
                                counter=counter,
                                config=config,
                            )
                        )
                        for spec in specs
                    ]
                    _print_event(
                        "active_streams_updated",
                        symbols=len(active_symbols),
                        streams=len(specs),
                    )
                else:
                    _print_event("active_streams_updated", symbols=0, streams=0)
            await asyncio.sleep(1.0)
    finally:
        if stream_stop is not None:
            stream_stop.set()
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)


async def _consume_active_position_stream_url(
    *,
    spec: StreamUrlSpec,
    service: RealtimeStrategyService,
    entry_gate: EntryHealthGate,
    stop_event: asyncio.Event,
    counter: "_EventCounter",
    config: StreamDaemonConfig,
) -> None:
    import websockets

    while not stop_event.is_set():
        try:
            async with websockets.connect(spec.url, ping_interval=180, ping_timeout=600) as websocket:
                _print_event(
                    "stream_connected",
                    url=spec.url.split("?streams=", 1)[0],
                    symbols=len(spec.symbols),
                    purpose="active_positions",
                )
                rotation_deadline = _stream_rotation_deadline(spec, config=config)
                stream_iterator = websocket.__aiter__()
                while not stop_event.is_set():
                    try:
                        message = await _next_stream_message_or_rotate(
                            stream_iterator,
                            rotation_deadline=rotation_deadline,
                        )
                    except StopAsyncIteration:
                        break
                    except _StreamRotationDue:
                        _print_event(
                            "stream_rotation_due",
                            url=spec.url.split("?streams=", 1)[0],
                            symbols=len(spec.symbols),
                            purpose="active_positions",
                        )
                        break
                    if stop_event.is_set():
                        return
                    payload = json.loads(message)
                    if not _should_parse_stream_payload(payload, service):
                        if counter.incremented_past_limit():
                            stop_event.set()
                            return
                        continue
                    event = parse_kline_stream_event(payload)
                    result = service.process_price_event(event, allow_pyramid_add=True)
                    _print_strategy_action(event=event, result=result)
                    if counter.incremented_past_limit():
                        stop_event.set()
                        return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            entry_gate.mark_stream_error(str(exc))
            _print_event("stream_error", error=str(exc), entry_gate=entry_gate.snapshot())
            await asyncio.sleep(_stream_error_backoff_seconds(exc, config))


async def _consume_full_universe_stream_url(
    *,
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

    router = MarketEventRouter(service=service)
    closed_worker = ClosedBarBatchWorker(store=store, aggregator=aggregator, service=service)
    needs_recovery = recover_before_connect
    while not stop_event.is_set():
        try:
            if needs_recovery:
                recovery_symbols = _active_recovery_symbols(service, list(spec.symbols))
                if recovery_symbols:
                    await _recover_symbols_1m_gap_async(
                        recovery_lock=recovery_lock,
                        store=store,
                        rest_client=rest_client,
                        aggregator=aggregator,
                        service=service,
                        symbols=recovery_symbols,
                        intervals=config.intervals,
                        recovery_sleep_ms=config.recovery_sleep_ms,
                        fetch_limit=config.closed_poll_fetch_limit,
                    )
                else:
                    _print_event(
                        "market_gap_recovery_skipped",
                        reason="no_active_positions",
                        symbols=len(spec.symbols),
                        purpose="full_universe_market_data",
                    )
            needs_recovery = True
            async with websockets.connect(spec.url, ping_interval=180, ping_timeout=600) as websocket:
                _print_event(
                    "stream_connected",
                    url=spec.url.split("?streams=", 1)[0],
                    symbols=len(spec.symbols),
                    purpose="full_universe_market_data",
                )
                rotation_deadline = _stream_rotation_deadline(spec, config=config)
                stream_iterator = websocket.__aiter__()
                while not stop_event.is_set():
                    try:
                        message = await _next_stream_message_or_rotate(
                            stream_iterator,
                            rotation_deadline=rotation_deadline,
                        )
                    except StopAsyncIteration:
                        break
                    except _StreamRotationDue:
                        _print_event(
                            "stream_rotation_due",
                            url=spec.url.split("?streams=", 1)[0],
                            symbols=len(spec.symbols),
                            purpose="full_universe_market_data",
                        )
                        break
                    if stop_event.is_set():
                        return
                    payload = json.loads(message)
                    event = parse_kline_stream_event(payload)
                    router.route(event)
                    if event.is_closed:
                        closed_worker.process_many(
                            _drain_closed_queue(router),
                            allow_entry=entry_gate.allow_entries,
                        )
                    if counter.incremented_past_limit():
                        stop_event.set()
                        return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001
            entry_gate.mark_stream_error(str(exc))
            _print_event("stream_error", error=str(exc), entry_gate=entry_gate.snapshot())
            await asyncio.sleep(_stream_error_backoff_seconds(exc, config))


def _drain_closed_queue(router: MarketEventRouter) -> list:
    events = []
    while not router.closed_queue.empty():
        events.append(router.closed_queue.get())
    return events


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
                recovery_symbols = _active_recovery_symbols(service, list(spec.symbols))
                if recovery_symbols:
                    await _recover_symbols_1m_gap_async(
                        recovery_lock=recovery_lock,
                        store=store,
                        rest_client=rest_client,
                        aggregator=aggregator,
                        service=service,
                        symbols=recovery_symbols,
                        intervals=config.intervals,
                        recovery_sleep_ms=config.recovery_sleep_ms,
                        fetch_limit=config.closed_poll_fetch_limit,
                    )
                else:
                    _print_event(
                        "market_gap_recovery_skipped",
                        reason="no_active_positions",
                        symbols=len(spec.symbols),
                    )
            needs_recovery = True
            async with websockets.connect(spec.url, ping_interval=180, ping_timeout=600) as websocket:
                _print_event("stream_connected", url=spec.url.split("?streams=", 1)[0])
                rotation_deadline = _stream_rotation_deadline(spec, config=config)
                stream_iterator = websocket.__aiter__()
                while not stop_event.is_set():
                    try:
                        message = await _next_stream_message_or_rotate(
                            stream_iterator,
                            rotation_deadline=rotation_deadline,
                        )
                    except StopAsyncIteration:
                        break
                    except _StreamRotationDue:
                        _print_event(
                            "stream_rotation_due",
                            url=spec.url.split("?streams=", 1)[0],
                            symbols=len(spec.symbols),
                        )
                        break
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
        ban_backoff = _binance_ban_backoff_seconds(str(exc))
        if ban_backoff is not None:
            return max(
                config.reconnect_backoff_seconds,
                config.rate_limit_backoff_seconds,
                ban_backoff,
            )
        return max(config.reconnect_backoff_seconds, config.rate_limit_backoff_seconds)
    text = str(exc)
    if "-1003" in text or " 429 " in text:
        ban_backoff = _binance_ban_backoff_seconds(text)
        if ban_backoff is not None:
            return max(
                config.reconnect_backoff_seconds,
                config.rate_limit_backoff_seconds,
                ban_backoff,
            )
        return max(config.reconnect_backoff_seconds, config.rate_limit_backoff_seconds)
    return config.reconnect_backoff_seconds


async def _retry_startup_step_after_rate_limit(
    *,
    step: str,
    config: StreamDaemonConfig,
    entry_gate: EntryHealthGate,
    action: Callable[[], Awaitable[_T]],
    sleep_func=asyncio.sleep,
) -> _T:
    while True:
        try:
            return await action()
        except Exception as exc:  # noqa: BLE001
            if not _is_rate_limit_exception(exc):
                raise
            entry_gate.mark_stream_error(str(exc))
            backoff = _stream_error_backoff_seconds(exc, config)
            _print_event(
                "startup_rate_limited",
                step=step,
                error=str(exc),
                backoff_seconds=backoff,
                entry_gate=entry_gate.snapshot(),
            )
            await sleep_func(backoff)


def _binance_ban_backoff_seconds(text: str) -> float | None:
    match = re.search(r"banned until (\d{13})", text)
    if match is None:
        return None
    ban_until_seconds = int(match.group(1)) / 1000
    return round(max(ban_until_seconds - time.time() + 1.0, 0.0), 3)


def _is_rate_limit_error(exc: BinanceApiError) -> bool:
    return exc.status == 429 or exc.code == -1003


def _is_rate_limit_exception(exc: Exception) -> bool:
    if isinstance(exc, BinanceApiError):
        return _is_rate_limit_error(exc)
    text = str(exc)
    return "-1003" in text or " 429 " in text or " 418 " in text


def _seconds_until_next_closed_1m_poll(now: datetime, *, grace_seconds: float) -> float:
    minute_start = now.astimezone(timezone.utc).replace(second=0, microsecond=0)
    next_target = minute_start + timedelta(minutes=1, seconds=grace_seconds)
    return max((next_target - now.astimezone(timezone.utc)).total_seconds(), 0.0)


def _recover_symbols_1m_gap(
    *,
    store: LiveStore,
    rest_client: BinanceRestClient,
    aggregator: MultiIntervalAggregator,
    service: RealtimeStrategyService,
    symbols: list[str],
    intervals: tuple[str, ...],
    recovery_sleep_ms: int,
    fetch_limit: int = 99,
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
            limit=fetch_limit,
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
    allow_entries: bool = False,
    allow_pyramid_add: bool = False,
    fetch_limit: int = 99,
    sleep_func=asyncio.sleep,
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
                limit=fetch_limit,
            )
            result = replay_closed_1m_events(
                store=store,
                aggregator=aggregator,
                sink=service,
                events=[event_from_market_bar(row) for row in rows],
                allow_entries=allow_entries,
                allow_pyramid_add=allow_pyramid_add,
            )
            recovered_source += result.source_bars
            recovered_aggregates += result.aggregated_bars
            if recovery_sleep_ms > 0:
                await sleep_func(recovery_sleep_ms / 1000)
        if recovered_source or recovered_aggregates:
            _print_event(
                "market_gap_recovered",
                symbols=len(symbols),
                source_bars=recovered_source,
                aggregated_bars=recovered_aggregates,
            )


async def _poll_closed_1m_once_async(
    *,
    recovery_lock: asyncio.Lock,
    store: LiveStore,
    rest_client: BinanceRestClient,
    aggregator: MultiIntervalAggregator,
    service: RealtimeStrategyService,
    entry_gate: EntryHealthGate,
    symbols: list[str],
    intervals: tuple[str, ...],
    poll_sleep_ms: int,
    fetch_limit: int = 99,
    sleep_func=asyncio.sleep,
    fetch_range=fetch_closed_klines_range,
) -> None:
    await _recover_symbols_1m_gap_async(
        recovery_lock=recovery_lock,
        store=store,
        rest_client=rest_client,
        aggregator=aggregator,
        service=service,
        symbols=symbols,
        intervals=intervals,
        recovery_sleep_ms=poll_sleep_ms,
        allow_entries=entry_gate.allow_entries,
        allow_pyramid_add=True,
        fetch_limit=fetch_limit,
        sleep_func=sleep_func,
        fetch_range=fetch_range,
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
    service: RealtimeStrategyService | None = None,
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
        if service is not None:
            service.refresh_active_symbols()
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
