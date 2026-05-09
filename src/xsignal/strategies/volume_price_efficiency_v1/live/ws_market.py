from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


BINANCE_USD_FUTURES_LIVE_WS_BASE_URL = "wss://fstream.binance.com/market/stream"
BINANCE_USD_FUTURES_TESTNET_WS_BASE_URL = "wss://stream.binancefuture.com/stream"
BINANCE_MAX_COMBINED_STREAMS = 1024
BINANCE_KLINE_INTERVALS = (
    "1m",
    "3m",
    "5m",
    "15m",
    "30m",
    "1h",
    "2h",
    "4h",
    "6h",
    "8h",
    "12h",
    "1d",
    "3d",
    "1w",
    "1M",
)


@dataclass(frozen=True)
class KlineStreamEvent:
    symbol: str
    interval: str
    event_time: datetime
    open_time: datetime
    close_time: datetime
    open: float
    high: float
    low: float
    close: float
    quote_volume: float
    is_closed: bool


def validate_interval(interval: str) -> str:
    if interval not in BINANCE_KLINE_INTERVALS:
        supported = ", ".join(BINANCE_KLINE_INTERVALS)
        raise ValueError(f"unsupported Binance kline interval {interval}; supported: {supported}")
    return interval


def stream_name(symbol: str, interval: str) -> str:
    return f"{symbol.lower()}@kline_{validate_interval(interval)}"


def chunk_stream_names(
    streams: list[str],
    *,
    max_streams: int = BINANCE_MAX_COMBINED_STREAMS,
) -> list[tuple[str, ...]]:
    if max_streams <= 0:
        raise ValueError("max_streams must be positive")
    return [tuple(streams[index : index + max_streams]) for index in range(0, len(streams), max_streams)]


def build_combined_stream_urls(
    *,
    symbols: list[str],
    intervals: list[str],
    base_url: str,
    max_streams: int = BINANCE_MAX_COMBINED_STREAMS,
) -> list[str]:
    streams = [
        stream_name(symbol, interval)
        for interval in intervals
        for symbol in symbols
    ]
    return [
        f"{base_url.rstrip('/')}?streams={'/'.join(chunk)}"
        for chunk in chunk_stream_names(streams, max_streams=max_streams)
    ]


def parse_kline_stream_event(payload: dict[str, Any]) -> KlineStreamEvent:
    data = payload.get("data", payload)
    if data.get("e") != "kline":
        raise ValueError("payload is not a kline event")
    kline = data["k"]
    interval = validate_interval(str(kline["i"]))
    symbol = str(kline.get("s") or data["s"])
    return KlineStreamEvent(
        symbol=symbol,
        interval=interval,
        event_time=_dt_ms(data["E"]),
        open_time=_dt_ms(kline["t"]),
        close_time=_dt_ms(kline["T"]),
        open=float(kline["o"]),
        high=float(kline["h"]),
        low=float(kline["l"]),
        close=float(kline["c"]),
        quote_volume=float(kline["q"]),
        is_closed=bool(kline["x"]),
    )


def _dt_ms(value: int | str) -> datetime:
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)
