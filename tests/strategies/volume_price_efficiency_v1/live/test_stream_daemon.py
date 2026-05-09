import asyncio
from datetime import datetime, timezone
import json
import sys
import threading
import time
import types

from xsignal.strategies.volume_price_efficiency_v1.live.binance_rest import BinanceApiError
from xsignal.strategies.volume_price_efficiency_v1.live.realtime import RealtimeEventResult
from xsignal.strategies.volume_price_efficiency_v1.live import stream_daemon as stream_daemon_module
from xsignal.strategies.volume_price_efficiency_v1.live.stream_daemon import (
    StreamDaemonConfig,
    StreamUrlSpec,
    _consume_stream_url,
    _build_user_data_tasks,
    _next_stream_message_or_rotate,
    _poll_closed_1m_once_async,
    _process_closed_1m_event,
    _recover_symbols_1m_gap,
    _recover_symbols_1m_gap_async,
    _should_parse_stream_message,
    _should_parse_stream_payload,
    _stream_error_backoff_seconds,
    _stream_rotation_deadline,
    _StreamRotationDue,
    build_daemon_stream_urls,
    build_daemon_stream_specs,
    seed_rolling_buffers_from_store,
    seed_rolling_buffers,
    ws_base_url_for_mode,
)
from xsignal.strategies.volume_price_efficiency_v1.live.ws_market import KlineStreamEvent


def _kline(open_ms: int, close_ms: int, close: str = "105"):
    return [
        open_ms,
        "100",
        "110",
        "90",
        close,
        "12.5",
        close_ms,
        "1250.5",
        42,
        "1",
        "2",
        "0",
    ]


class FakeSeedClient:
    def __init__(self) -> None:
        self.calls = []

    def request(self, method, path, *, signed=False, params=None):
        self.calls.append((method, path, signed, params or {}))
        return [_kline(1778313600000, 1778327999999)]


class RateLimitedSeedClient:
    def __init__(self) -> None:
        self.calls = 0

    def request(self, method, path, *, signed=False, params=None):
        self.calls += 1
        if self.calls == 1:
            raise BinanceApiError(status=429, code=-1003, message="too many requests")
        return [_kline(1778313600000, 1778327999999)]


class FakeStore:
    def __init__(self) -> None:
        self.bars = []
        self.cursors = []
        self.commits = 0
        self.connection = self

    def commit(self) -> None:
        self.commits += 1

    def upsert_market_bar(self, row, *, commit=True) -> None:
        self.bars.append((row, commit))

    def advance_market_cursor(self, *, symbol, interval, open_time, commit=True) -> None:
        self.cursors.append((symbol, interval, open_time, commit))

    def get_market_cursor(self, *, symbol, interval):
        return datetime(2026, 5, 9, 8, tzinfo=timezone.utc)


class FakeRecoveryRestClient:
    def __init__(self) -> None:
        self.calls = []

    def request(self, method, path, *, signed=False, params=None):
        self.calls.append((method, path, signed, params or {}))
        if path == "/fapi/v1/time":
            return {"serverTime": 1778330000000}
        return []


class FakeRecoveryStore:
    def get_market_cursor(self, *, symbol, interval):
        return None

    def upsert_market_bar(self, row, *, commit=True) -> None:
        raise AssertionError("no closed 1m rows should be replayed in this test")

    def advance_market_cursor(self, *, symbol, interval, open_time, commit=True) -> None:
        raise AssertionError("no cursor should advance in this test")


class ThreadCheckingConnection:
    def __init__(self, main_thread_id: int) -> None:
        self.main_thread_id = main_thread_id
        self.commit_threads = []

    def commit(self) -> None:
        self.commit_threads.append(threading.get_ident())
        assert threading.get_ident() == self.main_thread_id


class ThreadCheckingRecoveryStore:
    def __init__(self, main_thread_id: int) -> None:
        self.main_thread_id = main_thread_id
        self.connection = ThreadCheckingConnection(main_thread_id)
        self.cursor_threads = []
        self.upsert_threads = []
        self.advance_threads = []

    def get_market_cursor(self, *, symbol, interval):
        self.cursor_threads.append(threading.get_ident())
        assert threading.get_ident() == self.main_thread_id
        return datetime(2026, 5, 9, 8, tzinfo=timezone.utc)

    def upsert_market_bar(self, row, *, commit=True) -> None:
        self.upsert_threads.append(threading.get_ident())
        assert threading.get_ident() == self.main_thread_id

    def advance_market_cursor(self, *, symbol, interval, open_time, commit=True) -> None:
        self.advance_threads.append(threading.get_ident())
        assert threading.get_ident() == self.main_thread_id


class FakeAggregator:
    def __init__(self, aggregate: KlineStreamEvent) -> None:
        self.aggregate = aggregate

    def apply_1m_event(self, event):
        return [self.aggregate]


class EmptyAggregator:
    def apply_1m_event(self, event):
        return []


class FakeService:
    def __init__(self) -> None:
        self.price_calls = []
        self.closed_calls = []
        self.closed_batches = []

    def process_price_event(self, event, *, allow_pyramid_add, allow_stop_replace=True):
        self.price_calls.append((event, allow_pyramid_add, allow_stop_replace))
        return RealtimeEventResult(closed_signal_checked=False)

    def process_closed_bar(
        self,
        event,
        *,
        allow_entry,
        allow_pyramid_add,
        allow_stop_replace=True,
    ):
        self.closed_calls.append((event, allow_entry, allow_pyramid_add, allow_stop_replace))
        return RealtimeEventResult(closed_signal_checked=True)

    def process_closed_bar_batch(
        self,
        events,
        *,
        allow_entry,
        allow_pyramid_add,
        allow_stop_replace=True,
    ):
        self.closed_batches.append((tuple(events), allow_entry, allow_pyramid_add, allow_stop_replace))
        return RealtimeEventResult(closed_signal_checked=True)


class ClosedEntryGate:
    allow_entries = False

    def mark_stream_error(self, error: str) -> None:
        raise AssertionError(f"unexpected stream error: {error}")

    def snapshot(self):
        return {"allow_entries": False, "reasons": []}


class ActiveSymbolService:
    def __init__(self, symbols: set[str]) -> None:
        self.symbols = symbols

    def has_active_symbol_position(self, symbol: str) -> bool:
        return symbol in self.symbols

    def process_price_event(self, event, *, allow_pyramid_add, allow_stop_replace=True):
        return RealtimeEventResult(closed_signal_checked=False)

    def active_symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self.symbols))

    def refresh_active_symbols(self) -> None:
        return None


def _stream_payload(*, symbol: str = "BTCUSDT", closed: bool = False):
    return {
        "stream": f"{symbol.lower()}@kline_1m",
        "data": {
            "e": "kline",
            "E": 1778318492123,
            "s": symbol,
            "k": {
                "t": 1778313600000,
                "T": 1778313659999,
                "s": symbol,
                "i": "1m",
                "o": "100.0",
                "c": "105.5",
                "h": "108.0",
                "l": "99.0",
                "v": "12.3",
                "q": "1298.0",
                "x": closed,
            },
        },
    }


def test_ws_base_url_for_mode_uses_testnet_and_live_hosts():
    assert ws_base_url_for_mode("testnet") == "wss://stream.binancefuture.com/stream"
    assert ws_base_url_for_mode("live") == "wss://fstream.binance.com/market/stream"


def test_stream_daemon_config_defaults_to_realtime_intervals():
    config = StreamDaemonConfig(mode="testnet", db_path="live.sqlite")
    assert config.intervals == ("1h", "4h", "1d")
    assert config.lookback_bars == 120
    assert config.max_streams == 200
    assert config.stream_max_lifetime_seconds == 82_800
    assert config.stream_rotation_jitter_seconds == 1_800
    assert config.seed_sleep_ms == 20
    assert config.recovery_sleep_ms == 500
    assert config.closed_poll_fetch_limit == 99
    assert config.rate_limit_backoff_seconds == 60.0


def test_build_daemon_stream_urls_subscribes_only_to_1m_source_streams():
    urls = build_daemon_stream_urls(
        mode="testnet",
        symbols=["BTCUSDT", "ETHUSDT"],
        intervals=["1h", "4h"],
        max_streams=3,
    )

    assert urls == [
        "wss://stream.binancefuture.com/stream?streams=btcusdt@kline_1m/ethusdt@kline_1m",
    ]


def test_build_daemon_stream_specs_keeps_symbol_chunks_for_gap_recovery():
    specs = build_daemon_stream_specs(
        mode="testnet",
        symbols=["BTCUSDT", "ETHUSDT", "SOLUSDT"],
        max_streams=2,
    )

    assert [spec.symbols for spec in specs] == [
        ("BTCUSDT", "ETHUSDT"),
        ("SOLUSDT",),
    ]
    assert specs[0].url.endswith("btcusdt@kline_1m/ethusdt@kline_1m")
    assert specs[1].url.endswith("solusdt@kline_1m")


def test_seed_rolling_buffers_fetches_each_interval_and_symbol():
    client = FakeSeedClient()

    buffers = seed_rolling_buffers(
        client,
        symbols=["BTCUSDT", "ETHUSDT"],
        intervals=["1h", "4h"],
        lookback_bars=120,
        server_time_ms=1778330000000,
        max_bars=120,
    )

    assert sorted(buffers) == ["1h", "4h"]
    assert buffers["1h"].to_arrays().symbols == ("BTCUSDT", "ETHUSDT")
    assert buffers["4h"].to_arrays().open_times[0] == datetime(
        2026, 5, 9, 8, tzinfo=timezone.utc
    )
    assert len(client.calls) == 4
    assert {call[3]["interval"] for call in client.calls} == {"1h", "4h"}


def test_seed_rolling_buffers_backs_off_and_retries_rate_limits(monkeypatch):
    client = RateLimitedSeedClient()
    sleeps = []
    monkeypatch.setattr(stream_daemon_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    buffers = seed_rolling_buffers(
        client,
        symbols=["BTCUSDT"],
        intervals=["1h"],
        lookback_bars=120,
        server_time_ms=1778330000000,
        max_bars=120,
        rate_limit_backoff_seconds=7,
    )

    assert client.calls == 2
    assert sleeps == [7]
    assert buffers["1h"].to_arrays().symbols == ("BTCUSDT",)


def test_seed_rolling_buffers_from_store_avoids_rest_seed_requests():
    calls = []

    class StoreBackedSeed:
        def list_recent_market_bars(self, *, symbol, interval, limit):
            calls.append((symbol, interval, limit))
            return [
                {
                    "symbol": symbol,
                    "interval": interval,
                    "open_time": datetime(2026, 5, 9, 8, tzinfo=timezone.utc),
                    "open": 100.0,
                    "high": 101.0,
                    "low": 99.0,
                    "close": 100.5,
                    "quote_volume": 10.0,
                    "is_complete": True,
                }
            ]

    buffers = seed_rolling_buffers_from_store(
        StoreBackedSeed(),
        symbols=["BTCUSDT"],
        intervals=["1h"],
        lookback_bars=120,
        max_bars=120,
    )

    assert calls == [("BTCUSDT", "1h", 120)]
    assert buffers["1h"].to_arrays().symbols == ("BTCUSDT",)


def test_recover_symbols_1m_gap_uses_slower_recovery_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(stream_daemon_module.time, "sleep", lambda seconds: sleeps.append(seconds))

    _recover_symbols_1m_gap(
        store=FakeRecoveryStore(),
        rest_client=FakeRecoveryRestClient(),
        aggregator=EmptyAggregator(),
        service=None,
        symbols=["BTCUSDT", "ETHUSDT"],
        intervals=("1h",),
        recovery_sleep_ms=100,
    )

    assert sleeps == [0.1, 0.1]


def test_stream_error_backoff_uses_longer_delay_for_rate_limits():
    config = StreamDaemonConfig(
        mode="testnet",
        db_path="live.sqlite",
        reconnect_backoff_seconds=5.0,
        rate_limit_backoff_seconds=60.0,
    )

    assert (
        _stream_error_backoff_seconds(
            BinanceApiError(status=429, code=-1003, message="too many requests"),
            config,
        )
        == 60.0
    )
    assert _stream_error_backoff_seconds(RuntimeError("closed"), config) == 5.0


def test_stream_error_backoff_honors_binance_ban_until_timestamp(monkeypatch):
    config = StreamDaemonConfig(
        mode="testnet",
        db_path="live.sqlite",
        reconnect_backoff_seconds=5.0,
        rate_limit_backoff_seconds=60.0,
    )
    monkeypatch.setattr(stream_daemon_module.time, "time", lambda: 1_778_353_000.0)

    assert (
        _stream_error_backoff_seconds(
            BinanceApiError(
                status=418,
                code=-1003,
                message="Way too many requests; IP banned until 1778353379518.",
            ),
            config,
        )
        == 380.518
    )


def test_stream_rotation_deadline_spreads_connections_before_binance_hard_cutoff():
    config = StreamDaemonConfig(
        mode="testnet",
        db_path="live.sqlite",
        stream_max_lifetime_seconds=100,
        stream_rotation_jitter_seconds=30,
    )

    first = _stream_rotation_deadline(
        StreamUrlSpec(url="wss://example", symbols=("BTCUSDT", "ETHUSDT")),
        config=config,
        now_monotonic=1_000,
    )
    second = _stream_rotation_deadline(
        StreamUrlSpec(url="wss://example", symbols=("SOLUSDT", "XRPUSDT")),
        config=config,
        now_monotonic=1_000,
    )

    assert 1_070 <= first <= 1_100
    assert 1_070 <= second <= 1_100
    assert first != second


def test_next_stream_message_raises_rotation_due_when_lifetime_expires():
    class QuietIterator:
        def __aiter__(self):
            return self

        async def __anext__(self):
            await asyncio.sleep(60)
            raise AssertionError("rotation should happen before the next message")

    async def run() -> None:
        try:
            await _next_stream_message_or_rotate(
                QuietIterator(),
                rotation_deadline=asyncio.get_running_loop().time() + 0.01,
                monotonic=asyncio.get_running_loop().time,
            )
        except _StreamRotationDue:
            return
        raise AssertionError("expected stream rotation")

    asyncio.run(run())


def test_startup_step_retries_rate_limit_without_crashing_systemd_loop(monkeypatch):
    config = StreamDaemonConfig(
        mode="testnet",
        db_path="live.sqlite",
        reconnect_backoff_seconds=5.0,
        rate_limit_backoff_seconds=60.0,
    )
    monkeypatch.setattr(stream_daemon_module.time, "time", lambda: 1_778_353_000.0)
    attempts = []
    sleeps = []

    class Gate:
        def __init__(self) -> None:
            self.errors = []

        def mark_stream_error(self, error):
            self.errors.append(error)

        def snapshot(self):
            return {"allow_entries": False, "reasons": ["rest_rate_limited"]}

    async def action():
        attempts.append("try")
        if len(attempts) == 1:
            raise BinanceApiError(
                status=418,
                code=-1003,
                message="Way too many requests; IP banned until 1778353379518.",
            )
        return "ok"

    async def sleep(seconds):
        sleeps.append(seconds)

    async def run():
        gate = Gate()
        result = await stream_daemon_module._retry_startup_step_after_rate_limit(
            step="reconcile",
            config=config,
            entry_gate=gate,
            action=action,
            sleep_func=sleep,
        )
        return result, gate

    result, gate = asyncio.run(run())

    assert result == "ok"
    assert attempts == ["try", "try"]
    assert sleeps == [380.518]
    assert len(gate.errors) == 1


def test_async_recovery_keeps_sqlite_work_on_event_loop_thread():
    main_thread_id = threading.get_ident()
    store = ThreadCheckingRecoveryStore(main_thread_id)
    rest_thread_ids = []
    fetch_thread_ids = []

    class ThreadCheckingRestClient:
        def request(self, method, path, *, signed=False, params=None):
            rest_thread_ids.append(threading.get_ident())
            assert threading.get_ident() != main_thread_id
            return {"serverTime": 1778330000000}

    def fetch_range(rest_client, **kwargs):
        fetch_thread_ids.append(threading.get_ident())
        assert threading.get_ident() != main_thread_id
        return [
            {
                "symbol": kwargs["symbol"],
                "interval": "1m",
                "open_time": datetime(2026, 5, 9, 8, 1, tzinfo=timezone.utc),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "quote_volume": 10.0,
                "is_complete": True,
            }
        ]

    async def run() -> None:
        recovery_lock = asyncio.Lock()
        await _recover_symbols_1m_gap_async(
            recovery_lock=recovery_lock,
            store=store,
            rest_client=ThreadCheckingRestClient(),
            aggregator=EmptyAggregator(),
            service=None,
            symbols=["BTCUSDT"],
            intervals=("1h",),
            recovery_sleep_ms=0,
            fetch_range=fetch_range,
        )

    asyncio.run(run())

    assert rest_thread_ids
    assert fetch_thread_ids
    assert store.cursor_threads == [main_thread_id]
    assert store.upsert_threads == [main_thread_id]
    assert store.advance_threads == [main_thread_id]
    assert store.connection.commit_threads == [main_thread_id]


def test_async_recovery_serializes_fetches_but_keeps_event_loop_responsive():
    events = []

    def fetch_range(rest_client, **kwargs):
        events.append(f"start-{kwargs['symbol']}")
        time.sleep(0.05)
        events.append(f"end-{kwargs['symbol']}")
        return []

    class TimeClient:
        def request(self, method, path, *, signed=False, params=None):
            return {"serverTime": 1778330000000}

    async def tick() -> None:
        await asyncio.sleep(0.01)
        events.append("tick")

    async def run() -> None:
        recovery_lock = asyncio.Lock()
        first = asyncio.create_task(
            _recover_symbols_1m_gap_async(
                recovery_lock=recovery_lock,
                store=FakeRecoveryStore(),
                rest_client=TimeClient(),
                aggregator=EmptyAggregator(),
                service=None,
                symbols=["BTCUSDT"],
                intervals=("1h",),
                recovery_sleep_ms=0,
                fetch_range=fetch_range,
            )
        )
        await asyncio.sleep(0)
        second = asyncio.create_task(
            _recover_symbols_1m_gap_async(
                recovery_lock=recovery_lock,
                store=FakeRecoveryStore(),
                rest_client=TimeClient(),
                aggregator=EmptyAggregator(),
                service=None,
                symbols=["ETHUSDT"],
                intervals=("1h",),
                recovery_sleep_ms=0,
                fetch_range=fetch_range,
            )
        )
        ticker = asyncio.create_task(tick())
        await asyncio.gather(first, second, ticker)

    asyncio.run(run())

    assert events.index("tick") < events.index("end-BTCUSDT")
    assert events.index("start-ETHUSDT") > events.index("end-BTCUSDT")


def test_closed_bar_poll_allows_fresh_entries_and_paces_fetches():
    sleeps = []
    limits = []

    def fetch_range(rest_client, **kwargs):
        limits.append(kwargs["limit"])
        return [
            {
                "symbol": kwargs["symbol"],
                "interval": "1m",
                "open_time": datetime(2026, 5, 9, 8, 1, tzinfo=timezone.utc),
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.5,
                "quote_volume": 10.0,
                "is_complete": True,
            }
        ]

    class TimeClient:
        def request(self, method, path, *, signed=False, params=None):
            return {"serverTime": 1778330000000}

    aggregate = KlineStreamEvent(
        symbol="BTCUSDT",
        interval="1h",
        event_time=datetime(2026, 5, 9, 8, tzinfo=timezone.utc),
        open_time=datetime(2026, 5, 9, 8, tzinfo=timezone.utc),
        close_time=datetime(2026, 5, 9, 8, 59, 59, tzinfo=timezone.utc),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        quote_volume=10.0,
        is_closed=True,
    )
    service = FakeService()

    async def run() -> None:
        async def record_sleep(seconds):
            sleeps.append(seconds)

        await _poll_closed_1m_once_async(
            recovery_lock=asyncio.Lock(),
            store=FakeStore(),
            rest_client=TimeClient(),
            aggregator=FakeAggregator(aggregate),
            service=service,
            entry_gate=type("OpenGate", (), {"allow_entries": True})(),
            symbols=["BTCUSDT"],
            intervals=("1h",),
            poll_sleep_ms=25,
            fetch_limit=99,
            sleep_func=record_sleep,
            fetch_range=fetch_range,
        )

    asyncio.run(run())

    assert sleeps == [0.025]
    assert limits == [99]
    assert len(service.price_calls) == 1
    assert service.price_calls[0][1:] == (True, False)
    assert service.closed_calls == [(aggregate, True, True, False)]


def test_stream_daemon_starts_full_universe_stream_without_closed_poll_task(monkeypatch):
    created = []

    async def fake_full_universe(**kwargs):
        created.append(("full_universe_ws", tuple(kwargs["symbols"])))
        await kwargs["stop_event"].wait()

    async def fake_closed_poll(**kwargs):
        created.append(("closed_poll",))

    monkeypatch.setattr(stream_daemon_module, "_full_universe_stream_manager", fake_full_universe)
    monkeypatch.setattr(stream_daemon_module, "_poll_closed_1m_loop", fake_closed_poll)

    async def run() -> list[asyncio.Task]:
        stop_event = asyncio.Event()
        tasks = stream_daemon_module._build_market_data_tasks(
            store=object(),
            rest_client=object(),
            aggregator=object(),
            service=object(),
            entry_gate=object(),
            symbols=["BTCUSDT"],
            stop_event=stop_event,
            counter=stream_daemon_module._EventCounter(limit=1),
            config=StreamDaemonConfig(mode="testnet", db_path="live.sqlite"),
            recovery_lock=asyncio.Lock(),
        )
        await asyncio.sleep(0)
        stop_event.set()
        await asyncio.gather(*tasks, return_exceptions=True)
        return tasks

    tasks = asyncio.run(run())

    assert len(tasks) == 1
    assert created == [("full_universe_ws", ("BTCUSDT",))]


def test_stream_daemon_starts_user_data_stream_task_by_default(monkeypatch):
    created = []

    async def fake_user_data_stream(**kwargs):
        created.append((kwargs["mode"], kwargs["keepalive_interval_seconds"]))
        kwargs["stop_event"].set()

    monkeypatch.setattr(stream_daemon_module, "run_user_data_stream", fake_user_data_stream)

    async def run() -> None:
        stop_event = asyncio.Event()
        tasks = _build_user_data_tasks(
            store=object(),
            broker=object(),
            service=object(),
            mode="testnet",
            stop_event=stop_event,
            config=StreamDaemonConfig(mode="testnet", db_path="live.sqlite"),
        )
        await asyncio.gather(*tasks)

    asyncio.run(run())

    assert created == [("testnet", 1800.0)]


def test_stream_daemon_can_disable_user_data_stream_task():
    async def run() -> list:
        return _build_user_data_tasks(
            store=object(),
            broker=object(),
            service=object(),
            mode="testnet",
            stop_event=asyncio.Event(),
            config=StreamDaemonConfig(
                mode="testnet",
                db_path="live.sqlite",
                enable_user_data_stream=False,
            ),
        )

    assert asyncio.run(run()) == []


def test_startup_recovery_only_recovers_active_position_symbols(monkeypatch, tmp_path):
    recovered = []

    class FakeBroker:
        rest_client = object()

        def list_trading_usdt_perpetual_metadata(self):
            return {
                "BTCUSDT": object(),
                "ETHUSDT": object(),
                "SOLUSDT": object(),
            }

        def get_account_snapshot(self, **kwargs):
            raise AssertionError("account snapshot is not needed for this test")

    class DaemonStore:
        def initialize(self) -> None:
            return None

        def list_recent_market_bars(self, *, symbol, interval, limit):
            return []

    class ServiceWithOneActiveSymbol:
        def __init__(self, **kwargs):
            self.refreshed = 0

        def refresh_active_symbols(self) -> None:
            self.refreshed += 1

        def active_symbols(self) -> tuple[str, ...]:
            return ("ETHUSDT",)

    async def fake_recover(**kwargs):
        recovered.append(tuple(kwargs["symbols"]))

    monkeypatch.setattr(stream_daemon_module, "build_usd_futures_broker", lambda **kwargs: FakeBroker())
    monkeypatch.setattr(
        stream_daemon_module.LiveStore,
        "open",
        lambda path: DaemonStore(),
    )
    monkeypatch.setattr(
        stream_daemon_module,
        "run_reconciliation_pass",
        lambda **kwargs: types.SimpleNamespace(error_count=0),
    )
    monkeypatch.setattr(stream_daemon_module, "RealtimeStrategyService", ServiceWithOneActiveSymbol)
    monkeypatch.setattr(stream_daemon_module, "_recover_symbols_1m_gap_async", fake_recover)
    monkeypatch.setattr(stream_daemon_module, "_build_market_data_tasks", lambda **kwargs: [])

    result = asyncio.run(
        stream_daemon_module.run_stream_daemon_async(
                config=StreamDaemonConfig(
                    mode="testnet",
                    db_path=tmp_path / "live.sqlite",
                    reconcile_interval_seconds=0,
                    enable_user_data_stream=False,
                ),
                credentials=object(),
            )
    )

    assert result == 0
    assert recovered == [("ETHUSDT",)]


def test_consume_stream_url_can_skip_initial_recovery_after_startup(monkeypatch):
    events = []

    async def fake_recover(**kwargs):
        events.append("recover")

    class OneMessageSocket:
        def __init__(self) -> None:
            self.sent = False

        async def __aenter__(self):
            events.append("connect")
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return (
                '{"stream":"ethusdt@kline_1m","data":{"e":"kline","E":1778318492123,'
                '"s":"ETHUSDT","k":{"t":1778313600000,"s":"ETHUSDT","i":"1m","x":false}}}'
            )

    monkeypatch.setattr(stream_daemon_module, "_recover_symbols_1m_gap_async", fake_recover)
    monkeypatch.setitem(
        sys.modules,
        "websockets",
        types.SimpleNamespace(connect=lambda *args, **kwargs: OneMessageSocket()),
    )

    async def run() -> None:
        await _consume_stream_url(
            StreamUrlSpec(
                url="wss://stream.binancefuture.com/stream?streams=ethusdt@kline_1m",
                symbols=("ETHUSDT",),
            ),
            FakeRecoveryStore(),
            FakeRecoveryRestClient(),
            EmptyAggregator(),
            ActiveSymbolService(set()),
            ClosedEntryGate(),
            asyncio.Event(),
            stream_daemon_module._EventCounter(limit=1),
            StreamDaemonConfig(mode="testnet", db_path="live.sqlite", stop_after_events=1),
            asyncio.Lock(),
            recover_before_connect=False,
        )

    asyncio.run(run())

    assert events == ["connect"]


def test_full_universe_stream_recovers_gaps_before_connect(monkeypatch):
    events = []

    async def fake_recover(**kwargs):
        events.append(("recover", tuple(kwargs["symbols"])))

    class OneMessageSocket:
        def __init__(self):
            self.sent = False

        async def __aenter__(self):
            events.append(("connect",))
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return (
                '{"stream":"btcusdt@kline_1m","data":{"e":"kline","E":1778318492123,'
                '"s":"BTCUSDT","k":{"t":1778313600000,"T":1778313659999,"s":"BTCUSDT",'
                '"i":"1m","o":"100","c":"101","h":"102","l":"99","q":"10","x":false}}}'
            )

    monkeypatch.setattr(stream_daemon_module, "_recover_symbols_1m_gap_async", fake_recover)
    monkeypatch.setitem(
        sys.modules,
        "websockets",
        types.SimpleNamespace(connect=lambda *args, **kwargs: OneMessageSocket()),
    )

    async def run():
        stop_event = asyncio.Event()
        counter = stream_daemon_module._EventCounter(limit=1)
        await stream_daemon_module._consume_full_universe_stream_url(
            spec=StreamUrlSpec(url="wss://example", symbols=("BTCUSDT",)),
            store=object(),
            rest_client=object(),
            aggregator=object(),
            service=ActiveSymbolService({"BTCUSDT"}),
            entry_gate=ClosedEntryGate(),
            stop_event=stop_event,
            counter=counter,
            config=StreamDaemonConfig(mode="testnet", db_path="live.sqlite"),
            recovery_lock=asyncio.Lock(),
            recover_before_connect=True,
        )

    asyncio.run(run())

    assert events[0] == ("recover", ("BTCUSDT",))
    assert events[1] == ("connect",)


def test_full_universe_stream_recovers_only_active_symbol_gaps_before_connect(monkeypatch):
    events = []

    async def fake_recover(**kwargs):
        events.append(("recover", tuple(kwargs["symbols"])))

    class OneMessageSocket:
        def __init__(self):
            self.sent = False

        async def __aenter__(self):
            events.append(("connect",))
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return (
                '{"stream":"ethusdt@kline_1m","data":{"e":"kline","E":1778318492123,'
                '"s":"ETHUSDT","k":{"t":1778313600000,"T":1778313659999,"s":"ETHUSDT",'
                '"i":"1m","o":"100","c":"101","h":"102","l":"99","q":"10","x":false}}}'
            )

    monkeypatch.setattr(stream_daemon_module, "_recover_symbols_1m_gap_async", fake_recover)
    monkeypatch.setitem(
        sys.modules,
        "websockets",
        types.SimpleNamespace(connect=lambda *args, **kwargs: OneMessageSocket()),
    )

    async def run():
        stop_event = asyncio.Event()
        counter = stream_daemon_module._EventCounter(limit=1)
        await stream_daemon_module._consume_full_universe_stream_url(
            spec=StreamUrlSpec(
                url="wss://example",
                symbols=("BTCUSDT", "ETHUSDT", "SOLUSDT"),
            ),
            store=object(),
            rest_client=object(),
            aggregator=object(),
            service=ActiveSymbolService({"ETHUSDT"}),
            entry_gate=ClosedEntryGate(),
            stop_event=stop_event,
            counter=counter,
            config=StreamDaemonConfig(mode="testnet", db_path="live.sqlite"),
            recovery_lock=asyncio.Lock(),
            recover_before_connect=True,
        )

    asyncio.run(run())

    assert events[0] == ("recover", ("ETHUSDT",))
    assert events[1] == ("connect",)


def test_full_universe_stream_routes_closed_events_through_batch_worker(monkeypatch):
    class OneClosedMessageSocket:
        def __init__(self):
            self.sent = False

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return json.dumps(_stream_payload(symbol="BTCUSDT", closed=True))

    monkeypatch.setitem(
        sys.modules,
        "websockets",
        types.SimpleNamespace(connect=lambda *args, **kwargs: OneClosedMessageSocket()),
    )
    event = KlineStreamEvent(
        symbol="BTCUSDT",
        interval="1h",
        event_time=datetime(2026, 5, 9, 8, tzinfo=timezone.utc),
        open_time=datetime(2026, 5, 9, 8, tzinfo=timezone.utc),
        close_time=datetime(2026, 5, 9, 8, 59, 59, tzinfo=timezone.utc),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        quote_volume=10.0,
        is_closed=True,
    )
    store = FakeStore()
    service = FakeService()

    async def run():
        await stream_daemon_module._consume_full_universe_stream_url(
            spec=StreamUrlSpec(url="wss://example", symbols=("BTCUSDT",)),
            store=store,
            rest_client=object(),
            aggregator=FakeAggregator(event),
            service=service,
            entry_gate=ClosedEntryGate(),
            stop_event=asyncio.Event(),
            counter=stream_daemon_module._EventCounter(limit=1),
            config=StreamDaemonConfig(mode="testnet", db_path="live.sqlite"),
            recovery_lock=asyncio.Lock(),
            recover_before_connect=False,
        )

    asyncio.run(run())

    assert store.commits == 1
    assert service.closed_calls == []
    assert service.closed_batches == [((event,), False, True, True)]


def test_should_parse_stream_payload_skips_inactive_unclosed_updates():
    service = ActiveSymbolService({"BTCUSDT"})

    assert not _should_parse_stream_payload(
        _stream_payload(symbol="ETHUSDT", closed=False),
        service,
    )
    assert _should_parse_stream_payload(
        _stream_payload(symbol="BTCUSDT", closed=False),
        service,
    )
    assert _should_parse_stream_payload(
        _stream_payload(symbol="ETHUSDT", closed=True),
        service,
    )


def test_should_parse_stream_message_fast_skips_inactive_unclosed_updates():
    service = ActiveSymbolService({"BTCUSDT"})
    inactive_unclosed = (
        '{"stream":"ethusdt@kline_1m","data":{"e":"kline","E":1778318492123,'
        '"s":"ETHUSDT","k":{"t":1778313600000,"s":"ETHUSDT","i":"1m","x":false}}}'
    )
    active_unclosed = inactive_unclosed.replace("ethusdt", "btcusdt").replace("ETHUSDT", "BTCUSDT")
    inactive_closed = inactive_unclosed.replace('"x":false', '"x":true')

    assert not _should_parse_stream_message(inactive_unclosed, service)
    assert _should_parse_stream_message(active_unclosed, service)
    assert _should_parse_stream_message(inactive_closed, service)
    assert _should_parse_stream_message(b"{not-json", service)


def test_process_closed_1m_event_respects_entry_gate_for_aggregates():
    event = KlineStreamEvent(
        symbol="BTCUSDT",
        interval="1m",
        event_time=datetime(2026, 5, 9, 8, tzinfo=timezone.utc),
        open_time=datetime(2026, 5, 9, 8, tzinfo=timezone.utc),
        close_time=datetime(2026, 5, 9, 8, 0, 59, tzinfo=timezone.utc),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        quote_volume=10.0,
        is_closed=True,
    )
    aggregate = KlineStreamEvent(
        symbol="BTCUSDT",
        interval="1h",
        event_time=datetime(2026, 5, 9, 8, tzinfo=timezone.utc),
        open_time=datetime(2026, 5, 9, 8, tzinfo=timezone.utc),
        close_time=datetime(2026, 5, 9, 8, 59, 59, tzinfo=timezone.utc),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        quote_volume=10.0,
        is_closed=True,
    )
    store = FakeStore()
    service = FakeService()

    _process_closed_1m_event(
        store=store,
        aggregator=FakeAggregator(aggregate),
        service=service,
        event=event,
        entry_gate=ClosedEntryGate(),
    )

    assert [bar["interval"] for bar, commit in store.bars] == ["1m", "1h"]
    assert [commit for bar, commit in store.bars] == [False, False]
    assert store.cursors == [("BTCUSDT", "1m", event.open_time, False)]
    assert store.commits == 1
    assert service.price_calls == [(event, True, True)]
    assert service.closed_calls == [(aggregate, False, True, True)]
