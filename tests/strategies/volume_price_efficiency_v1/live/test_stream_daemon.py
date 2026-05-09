from datetime import datetime, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.binance_rest import BinanceApiError
from xsignal.strategies.volume_price_efficiency_v1.live.realtime import RealtimeEventResult
from xsignal.strategies.volume_price_efficiency_v1.live import stream_daemon as stream_daemon_module
from xsignal.strategies.volume_price_efficiency_v1.live.stream_daemon import (
    StreamDaemonConfig,
    _process_closed_1m_event,
    _recover_symbols_1m_gap,
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


class FakeStore:
    def __init__(self) -> None:
        self.bars = []
        self.cursors = []

    def upsert_market_bar(self, row) -> None:
        self.bars.append(row)

    def advance_market_cursor(self, *, symbol, interval, open_time) -> None:
        self.cursors.append((symbol, interval, open_time))


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

    def process_price_event(self, event, *, allow_pyramid_add):
        self.price_calls.append((event, allow_pyramid_add))
        return RealtimeEventResult(closed_signal_checked=False)

    def process_closed_bar(self, event, *, allow_entry, allow_pyramid_add):
        self.closed_calls.append((event, allow_entry, allow_pyramid_add))
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
    assert config.max_streams == 200
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

    assert [bar["interval"] for bar in store.bars] == ["1m", "1h"]
    assert store.cursors == [("BTCUSDT", "1m", event.open_time)]
    assert service.price_calls == [(event, True)]
    assert service.closed_calls == [(aggregate, False, True)]
