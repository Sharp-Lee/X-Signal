import asyncio
from datetime import datetime, timezone
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
    _poll_closed_1m_once_async,
    _process_closed_1m_event,
    _recover_symbols_1m_gap,
    _recover_symbols_1m_gap_async,
    _should_parse_stream_message,
    _should_parse_stream_payload,
    _stream_error_backoff_seconds,
    build_daemon_stream_urls,
    build_daemon_stream_specs,
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


class ClosedEntryGate:
    allow_entries = False


class ActiveSymbolService:
    def __init__(self, symbols: set[str]) -> None:
        self.symbols = symbols

    def has_active_symbol_position(self, symbol: str) -> bool:
        return symbol in self.symbols


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
    assert config.max_streams == 25
    assert config.seed_sleep_ms == 20
    assert config.recovery_sleep_ms == 500
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

    def fetch_range(rest_client, **kwargs):
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
            sleep_func=record_sleep,
            fetch_range=fetch_range,
        )

    asyncio.run(run())

    assert sleeps == [0.025]
    assert len(service.price_calls) == 1
    assert service.price_calls[0][1:] == (True, False)
    assert service.closed_calls == [(aggregate, True, True, False)]


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
