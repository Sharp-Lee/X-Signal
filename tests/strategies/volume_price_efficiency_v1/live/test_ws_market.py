from datetime import datetime, timezone

import pytest

from xsignal.strategies.volume_price_efficiency_v1.live.ws_market import (
    BINANCE_KLINE_INTERVALS,
    KlineStreamEvent,
    build_combined_stream_urls,
    chunk_stream_names,
    parse_kline_stream_event,
    stream_name,
    validate_interval,
)


def _payload(*, closed: bool = True, interval: str = "4h"):
    return {
        "stream": f"btcusdt@kline_{interval}",
        "data": {
            "e": "kline",
            "E": 1778318492123,
            "s": "BTCUSDT",
            "k": {
                "t": 1778313600000,
                "T": 1778327999999,
                "s": "BTCUSDT",
                "i": interval,
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


def test_validate_interval_accepts_all_binance_kline_intervals():
    assert validate_interval("1m") == "1m"
    assert validate_interval("4h") == "4h"
    assert validate_interval("1M") == "1M"
    assert "1d" in BINANCE_KLINE_INTERVALS


def test_validate_interval_rejects_unknown_interval():
    with pytest.raises(ValueError, match="unsupported Binance kline interval"):
        validate_interval("7h")


def test_stream_name_is_lowercase_symbol_with_interval():
    assert stream_name("BTCUSDT", "4h") == "btcusdt@kline_4h"


def test_chunk_stream_names_splits_streams_without_overflow():
    chunks = chunk_stream_names([f"s{i}@kline_1m" for i in range(5)], max_streams=2)
    assert chunks == [
        ("s0@kline_1m", "s1@kline_1m"),
        ("s2@kline_1m", "s3@kline_1m"),
        ("s4@kline_1m",),
    ]


def test_build_combined_stream_urls_chunks_symbols_and_intervals():
    urls = build_combined_stream_urls(
        symbols=["BTCUSDT", "ETHUSDT"],
        intervals=["1h", "4h"],
        base_url="wss://example.test/stream",
        max_streams=3,
    )

    assert urls == [
        "wss://example.test/stream?streams=btcusdt@kline_1h/ethusdt@kline_1h/btcusdt@kline_4h",
        "wss://example.test/stream?streams=ethusdt@kline_4h",
    ]


def test_parse_kline_stream_event_maps_closed_payload():
    event = parse_kline_stream_event(_payload(closed=True, interval="4h"))

    assert event == KlineStreamEvent(
        symbol="BTCUSDT",
        interval="4h",
        event_time=datetime.fromtimestamp(1778318492123 / 1000, tz=timezone.utc),
        open_time=datetime.fromtimestamp(1778313600000 / 1000, tz=timezone.utc),
        close_time=datetime.fromtimestamp(1778327999999 / 1000, tz=timezone.utc),
        open=100.0,
        high=108.0,
        low=99.0,
        close=105.5,
        quote_volume=1298.0,
        is_closed=True,
    )


def test_parse_kline_stream_event_keeps_unclosed_realtime_updates():
    event = parse_kline_stream_event(_payload(closed=False, interval="1h"))

    assert event.interval == "1h"
    assert event.high == 108.0
    assert event.close == 105.5
    assert not event.is_closed
