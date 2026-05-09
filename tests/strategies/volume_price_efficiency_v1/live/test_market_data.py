from datetime import datetime, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.market_data import (
    build_arrays_from_klines,
    build_arrays_from_daily_klines,
    fetch_closed_klines,
    fetch_closed_daily_klines,
    load_recent_arrays,
    load_recent_daily_arrays,
    parse_kline,
    parse_daily_kline,
)


class FakeMarketRestClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def request(self, method, path, *, signed=False, params=None):
        self.calls.append((method, path, signed, params or {}))
        return self.payload


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


def test_parse_daily_kline_maps_binance_payload_to_closed_bar_row():
    row = parse_daily_kline("BTCUSDT", _kline(1778198400000, 1778284799999))

    assert row["symbol"] == "BTCUSDT"
    assert row["open_time"] == datetime(2026, 5, 8, tzinfo=timezone.utc)
    assert row["open"] == 100.0
    assert row["high"] == 110.0
    assert row["low"] == 90.0
    assert row["close"] == 105.0
    assert row["quote_volume"] == 1250.5
    assert row["is_complete"]


def test_parse_kline_supports_intraday_intervals():
    row = parse_kline("BTCUSDT", _kline(1778313600000, 1778327999999), interval="4h")

    assert row["symbol"] == "BTCUSDT"
    assert row["interval"] == "4h"
    assert row["open_time"] == datetime(2026, 5, 9, 8, tzinfo=timezone.utc)
    assert row["is_complete"]


def test_fetch_closed_klines_uses_requested_interval_and_excludes_forming_bar():
    closed = _kline(1778313600000, 1778327999999)
    forming = _kline(1778328000000, 1778342399999, close="106")
    client = FakeMarketRestClient([closed, forming])

    rows = fetch_closed_klines(
        client,
        symbol="BTCUSDT",
        interval="4h",
        limit=2,
        server_time_ms=1778330000000,
    )

    assert [row["open_time"] for row in rows] == [
        datetime(2026, 5, 9, 8, tzinfo=timezone.utc)
    ]
    assert client.calls == [
        (
            "GET",
            "/fapi/v1/klines",
            False,
            {"symbol": "BTCUSDT", "interval": "4h", "limit": 2},
        )
    ]


def test_fetch_closed_daily_klines_excludes_currently_forming_daily_bar():
    closed = _kline(1778198400000, 1778284799999)
    forming = _kline(1778284800000, 1778371199999, close="106")
    client = FakeMarketRestClient([closed, forming])

    rows = fetch_closed_daily_klines(
        client,
        symbol="BTCUSDT",
        limit=2,
        server_time_ms=1778310000000,
    )

    assert [row["open_time"] for row in rows] == [datetime(2026, 5, 8, tzinfo=timezone.utc)]
    assert client.calls == [
        (
            "GET",
            "/fapi/v1/klines",
            False,
            {"symbol": "BTCUSDT", "interval": "1d", "limit": 2},
        )
    ]


def test_build_arrays_from_daily_klines_pivots_symbols_and_times():
    rows = [
        parse_daily_kline("ETHUSDT", _kline(1778198400000, 1778284799999, close="205")),
        parse_daily_kline("BTCUSDT", _kline(1778198400000, 1778284799999, close="105")),
        parse_daily_kline("BTCUSDT", _kline(1778284800000, 1778371199999, close="106")),
    ]

    arrays = build_arrays_from_daily_klines(rows)

    assert arrays.symbols == ("BTCUSDT", "ETHUSDT")
    assert arrays.open.shape == (2, 2)
    assert arrays.close[0, 0] == 105.0
    assert arrays.close[0, 1] == 205.0
    assert arrays.quality[1, 0]
    assert not arrays.quality[1, 1]


def test_build_arrays_from_klines_keeps_interval_rows():
    rows = [
        parse_kline("ETHUSDT", _kline(1778313600000, 1778327999999, close="205"), interval="4h"),
        parse_kline("BTCUSDT", _kline(1778313600000, 1778327999999, close="105"), interval="4h"),
        parse_kline("BTCUSDT", _kline(1778328000000, 1778342399999, close="106"), interval="4h"),
    ]

    arrays = build_arrays_from_klines(rows)

    assert arrays.symbols == ("BTCUSDT", "ETHUSDT")
    assert arrays.open.shape == (2, 2)
    assert arrays.close[0, 0] == 105.0
    assert arrays.close[0, 1] == 205.0
    assert arrays.quality[1, 0]
    assert not arrays.quality[1, 1]


def test_load_recent_daily_arrays_fetches_all_symbols():
    class MultiSymbolClient:
        def __init__(self) -> None:
            self.calls = []

        def request(self, method, path, *, signed=False, params=None):
            self.calls.append((method, path, signed, params or {}))
            return [_kline(1778198400000, 1778284799999)]

    client = MultiSymbolClient()

    arrays = load_recent_daily_arrays(
        client,
        symbols=["BTCUSDT", "ETHUSDT"],
        limit=1,
        server_time_ms=1778310000000,
    )

    assert arrays.symbols == ("BTCUSDT", "ETHUSDT")
    assert len(client.calls) == 2


def test_load_recent_arrays_fetches_all_symbols_for_interval():
    class MultiSymbolClient:
        def __init__(self) -> None:
            self.calls = []

        def request(self, method, path, *, signed=False, params=None):
            self.calls.append((method, path, signed, params or {}))
            return [_kline(1778313600000, 1778327999999)]

    client = MultiSymbolClient()

    arrays = load_recent_arrays(
        client,
        symbols=["BTCUSDT", "ETHUSDT"],
        interval="4h",
        limit=1,
        server_time_ms=1778330000000,
    )

    assert arrays.symbols == ("BTCUSDT", "ETHUSDT")
    assert {call[3]["interval"] for call in client.calls} == {"4h"}
