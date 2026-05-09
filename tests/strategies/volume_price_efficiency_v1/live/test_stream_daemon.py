from datetime import datetime, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.stream_daemon import (
    StreamDaemonConfig,
    build_daemon_stream_urls,
    seed_rolling_buffers,
    ws_base_url_for_mode,
)


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


def test_ws_base_url_for_mode_uses_testnet_and_live_hosts():
    assert ws_base_url_for_mode("testnet") == "wss://stream.binancefuture.com/stream"
    assert ws_base_url_for_mode("live") == "wss://fstream.binance.com/market/stream"


def test_stream_daemon_config_defaults_to_realtime_intervals():
    config = StreamDaemonConfig(mode="testnet", db_path="live.sqlite")
    assert config.intervals == ("1h", "4h", "1d")
    assert config.lookback_bars == 120


def test_build_daemon_stream_urls_chunks_all_symbols_and_intervals():
    urls = build_daemon_stream_urls(
        mode="testnet",
        symbols=["BTCUSDT", "ETHUSDT"],
        intervals=["1h", "4h"],
        max_streams=3,
    )

    assert urls == [
        "wss://stream.binancefuture.com/stream?streams=btcusdt@kline_1h/ethusdt@kline_1h/btcusdt@kline_4h",
        "wss://stream.binancefuture.com/stream?streams=ethusdt@kline_4h",
    ]


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
