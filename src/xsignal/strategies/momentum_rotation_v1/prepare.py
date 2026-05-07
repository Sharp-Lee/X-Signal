from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np

from xsignal.strategies.momentum_rotation_v1.data import CanonicalBarTable


@dataclass(frozen=True)
class PreparedArrays:
    symbols: tuple[str, ...]
    rebalance_times: np.ndarray
    close_1h: np.ndarray
    close_4h: np.ndarray
    close_1d: np.ndarray
    quote_volume_1d: np.ndarray
    complete_1h: np.ndarray
    complete_4h: np.ndarray
    complete_1d: np.ndarray
    quality_1h_24h: np.ndarray
    quality_4h_7d: np.ndarray
    quality_1d_30d: np.ndarray


def _rows_by_symbol_close_time(
    table: CanonicalBarTable,
    close_delta: timedelta,
) -> dict[tuple[str, object], dict]:
    rows = {}
    for row in table.table.to_pylist():
        close_time = row["open_time"] + close_delta
        rows[(row["symbol"], close_time)] = row
    return rows


def _is_good(row: dict | None) -> bool:
    return bool(
        row
        and row["is_complete"]
        and not row["has_synthetic"]
        and row["bar_count"] == row["expected_1m_count"]
        and row["close"] is not None
        and float(row["close"]) > 0
    )


def _has_price(row: dict | None) -> bool:
    return bool(row and row["close"] is not None and float(row["close"]) > 0)


def _window_quality(
    rows: dict[tuple[str, object], dict],
    symbol: str,
    rebalance_time: object,
    step: timedelta,
    steps_back: int,
) -> bool:
    for offset in range(steps_back + 1):
        close_time = rebalance_time - step * offset
        if not _is_good(rows.get((symbol, close_time))):
            return False
    return True


def _forward_fill_prices(values: np.ndarray) -> None:
    for column_index in range(values.shape[1]):
        last_price = np.nan
        for row_index in range(values.shape[0]):
            value = values[row_index, column_index]
            if np.isfinite(value):
                last_price = value
            elif np.isfinite(last_price):
                values[row_index, column_index] = last_price


def prepare_daily_arrays(
    *,
    bars_1h: CanonicalBarTable,
    bars_4h: CanonicalBarTable,
    bars_1d: CanonicalBarTable,
) -> PreparedArrays:
    if bars_1h.timeframe != "1h" or bars_4h.timeframe != "4h" or bars_1d.timeframe != "1d":
        raise ValueError("expected 1h, 4h, and 1d canonical tables")
    symbols = tuple(sorted(set(bars_1d.table.column("symbol").to_pylist())))
    if not symbols:
        raise ValueError("no symbols in daily canonical table")
    daily_rows = _rows_by_symbol_close_time(bars_1d, timedelta(days=1))
    hourly_rows = _rows_by_symbol_close_time(bars_1h, timedelta(hours=1))
    four_hour_rows = _rows_by_symbol_close_time(bars_4h, timedelta(hours=4))
    rebalance_times = tuple(sorted({key[1] for key in daily_rows}))
    shape = (len(rebalance_times), len(symbols))
    close_1h = np.full(shape, np.nan, dtype=np.float64)
    close_4h = np.full(shape, np.nan, dtype=np.float64)
    close_1d = np.full(shape, np.nan, dtype=np.float64)
    quote_volume_1d = np.full(shape, np.nan, dtype=np.float64)
    complete_1h = np.zeros(shape, dtype=bool)
    complete_4h = np.zeros(shape, dtype=bool)
    complete_1d = np.zeros(shape, dtype=bool)
    quality_1h_24h = np.zeros(shape, dtype=bool)
    quality_4h_7d = np.zeros(shape, dtype=bool)
    quality_1d_30d = np.zeros(shape, dtype=bool)
    for t_index, rebalance_time in enumerate(rebalance_times):
        for s_index, symbol in enumerate(symbols):
            h_row = hourly_rows.get((symbol, rebalance_time))
            h4_row = four_hour_rows.get((symbol, rebalance_time))
            d_row = daily_rows.get((symbol, rebalance_time))
            if _has_price(h_row):
                close_1h[t_index, s_index] = float(h_row["close"])
            if _is_good(h_row):
                complete_1h[t_index, s_index] = True
            if _has_price(h4_row):
                close_4h[t_index, s_index] = float(h4_row["close"])
            if _is_good(h4_row):
                complete_4h[t_index, s_index] = True
            if _has_price(d_row):
                close_1d[t_index, s_index] = float(d_row["close"])
                quote_volume_1d[t_index, s_index] = float(d_row["quote_volume"])
            if _is_good(d_row):
                complete_1d[t_index, s_index] = True
            quality_1h_24h[t_index, s_index] = _window_quality(
                hourly_rows,
                symbol,
                rebalance_time,
                timedelta(hours=1),
                24,
            )
            quality_4h_7d[t_index, s_index] = _window_quality(
                four_hour_rows,
                symbol,
                rebalance_time,
                timedelta(hours=4),
                42,
            )
            quality_1d_30d[t_index, s_index] = _window_quality(
                daily_rows,
                symbol,
                rebalance_time,
                timedelta(days=1),
                30,
            )
    _forward_fill_prices(close_1d)
    return PreparedArrays(
        symbols=symbols,
        rebalance_times=np.array(rebalance_times, dtype=object),
        close_1h=close_1h,
        close_4h=close_4h,
        close_1d=close_1d,
        quote_volume_1d=quote_volume_1d,
        complete_1h=complete_1h,
        complete_4h=complete_4h,
        complete_1d=complete_1d,
        quality_1h_24h=quality_1h_24h,
        quality_4h_7d=quality_4h_7d,
        quality_1d_30d=quality_1d_30d,
    )


def save_prepared_arrays(cache_dir: Path, arrays: PreparedArrays) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "symbols.json").write_text(json.dumps(list(arrays.symbols), indent=2) + "\n")
    np.save(cache_dir / "times_1d.npy", arrays.rebalance_times, allow_pickle=True)
    np.save(cache_dir / "close_1h.npy", arrays.close_1h)
    np.save(cache_dir / "close_4h.npy", arrays.close_4h)
    np.save(cache_dir / "close_1d.npy", arrays.close_1d)
    np.save(cache_dir / "quote_volume_1d.npy", arrays.quote_volume_1d)
    np.save(cache_dir / "complete_1h.npy", arrays.complete_1h)
    np.save(cache_dir / "complete_4h.npy", arrays.complete_4h)
    np.save(cache_dir / "complete_1d.npy", arrays.complete_1d)
    np.save(cache_dir / "quality_1h_24h.npy", arrays.quality_1h_24h)
    np.save(cache_dir / "quality_4h_7d.npy", arrays.quality_4h_7d)
    np.save(cache_dir / "quality_1d_30d.npy", arrays.quality_1d_30d)


def load_prepared_arrays(cache_dir: Path) -> PreparedArrays:
    symbols = tuple(json.loads((cache_dir / "symbols.json").read_text()))
    return PreparedArrays(
        symbols=symbols,
        rebalance_times=np.load(cache_dir / "times_1d.npy", allow_pickle=True),
        close_1h=np.load(cache_dir / "close_1h.npy"),
        close_4h=np.load(cache_dir / "close_4h.npy"),
        close_1d=np.load(cache_dir / "close_1d.npy"),
        quote_volume_1d=np.load(cache_dir / "quote_volume_1d.npy"),
        complete_1h=np.load(cache_dir / "complete_1h.npy"),
        complete_4h=np.load(cache_dir / "complete_4h.npy"),
        complete_1d=np.load(cache_dir / "complete_1d.npy"),
        quality_1h_24h=np.load(cache_dir / "quality_1h_24h.npy"),
        quality_4h_7d=np.load(cache_dir / "quality_4h_7d.npy"),
        quality_1d_30d=np.load(cache_dir / "quality_1d_30d.npy"),
    )
