from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.live.ws_market import validate_interval


DAY_MS = 24 * 60 * 60 * 1000


def parse_kline(symbol: str, payload: list[Any], *, interval: str) -> dict[str, object]:
    validate_interval(interval)
    open_time_ms = int(payload[0])
    close_time_ms = int(payload[6])
    return {
        "symbol": symbol,
        "interval": interval,
        "open_time": datetime.fromtimestamp(open_time_ms / 1000, tz=timezone.utc),
        "open": float(payload[1]),
        "high": float(payload[2]),
        "low": float(payload[3]),
        "close": float(payload[4]),
        "quote_volume": float(payload[7]),
        "is_complete": close_time_ms + 1 <= open_time_ms + DAY_MS,
    }


def parse_daily_kline(symbol: str, payload: list[Any]) -> dict[str, object]:
    return parse_kline(symbol, payload, interval="1d")


def fetch_closed_klines(
    rest_client,
    *,
    symbol: str,
    interval: str,
    limit: int,
    server_time_ms: int,
) -> list[dict[str, object]]:
    validate_interval(interval)
    payload = rest_client.request(
        "GET",
        "/fapi/v1/klines",
        params={"symbol": symbol, "interval": interval, "limit": limit},
    )
    rows = []
    for item in payload:
        close_time_ms = int(item[6])
        if close_time_ms >= server_time_ms:
            continue
        rows.append(parse_kline(symbol, item, interval=interval))
    return rows


def fetch_closed_daily_klines(
    rest_client,
    *,
    symbol: str,
    limit: int,
    server_time_ms: int,
) -> list[dict[str, object]]:
    return fetch_closed_klines(
        rest_client,
        symbol=symbol,
        interval="1d",
        limit=limit,
        server_time_ms=server_time_ms,
    )


def load_recent_arrays(
    rest_client,
    *,
    symbols: list[str],
    interval: str,
    limit: int,
    server_time_ms: int,
) -> OhlcvArrays:
    validate_interval(interval)
    rows: list[dict[str, object]] = []
    for symbol in symbols:
        rows.extend(
            fetch_closed_klines(
                rest_client,
                symbol=symbol,
                interval=interval,
                limit=limit,
                server_time_ms=server_time_ms,
            )
        )
    return build_arrays_from_klines(rows)


def load_recent_daily_arrays(
    rest_client,
    *,
    symbols: list[str],
    limit: int,
    server_time_ms: int,
) -> OhlcvArrays:
    return load_recent_arrays(
        rest_client,
        symbols=symbols,
        interval="1d",
        limit=limit,
        server_time_ms=server_time_ms,
    )


def build_arrays_from_klines(rows: list[dict[str, object]]) -> OhlcvArrays:
    symbols = tuple(sorted({str(row["symbol"]) for row in rows}))
    open_times = tuple(sorted({row["open_time"] for row in rows}))
    shape = (len(open_times), len(symbols))
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    time_index = {open_time: index for index, open_time in enumerate(open_times)}
    arrays = {
        name: np.full(shape, np.nan, dtype=np.float64)
        for name in ("open", "high", "low", "close", "quote_volume")
    }
    quality = np.zeros(shape, dtype=bool)
    for row in rows:
        t_index = time_index[row["open_time"]]
        s_index = symbol_index[str(row["symbol"])]
        for name in arrays:
            arrays[name][t_index, s_index] = float(row[name])
        quality[t_index, s_index] = bool(row["is_complete"]) and _valid_price_row(row)
    return OhlcvArrays(
        symbols=symbols,
        open_times=np.array(open_times, dtype=object),
        open=arrays["open"],
        high=arrays["high"],
        low=arrays["low"],
        close=arrays["close"],
        quote_volume=arrays["quote_volume"],
        quality=quality,
    )


def build_arrays_from_daily_klines(rows: list[dict[str, object]]) -> OhlcvArrays:
    return build_arrays_from_klines(rows)


def _valid_price_row(row: dict[str, object]) -> bool:
    open_ = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    quote_volume = float(row["quote_volume"])
    return (
        open_ > 0
        and high > 0
        and low > 0
        and close > 0
        and high >= max(open_, close)
        and low <= min(open_, close)
        and quote_volume > 0
    )
