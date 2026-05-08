from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


REQUIRED_CANONICAL_COLUMNS = (
    "symbol",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "quote_volume",
    "bar_count",
    "expected_1m_count",
    "is_complete",
    "has_synthetic",
    "fill_policy",
)


@dataclass(frozen=True)
class CanonicalOhlcvTable:
    timeframe: str
    fill_policy: str
    manifest_path: Path
    parquet_path: Path
    table: pa.Table


@dataclass(frozen=True)
class OhlcvArrays:
    symbols: tuple[str, ...]
    open_times: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    quote_volume: np.ndarray
    quality: np.ndarray


def _normalize_open_time_column(table: pa.Table) -> pa.Table:
    field_index = table.schema.get_field_index("open_time")
    open_time = table["open_time"]
    target_type = pa.timestamp("s", tz="UTC")
    if pa.types.is_timestamp(open_time.type):
        normalized = pc.cast(open_time, target_type)
    elif pa.types.is_integer(open_time.type):
        normalized = pa.chunked_array(
            [chunk.cast(pa.int64()).view(target_type) for chunk in open_time.chunks]
        )
    else:
        raise ValueError(f"unsupported open_time type: {open_time.type}")
    return table.set_column(field_index, "open_time", normalized)


def load_manifested_table(
    manifest_path: Path,
    *,
    timeframe: str = "1d",
    fill_policy: str = "raw",
) -> CanonicalOhlcvTable:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("timeframe") != timeframe:
        raise ValueError("manifest timeframe does not match request")
    if manifest.get("fill_policy") != fill_policy:
        raise ValueError("manifest fill_policy does not match request")
    parquet_path = Path(manifest["parquet_path"])
    table = pq.ParquetFile(parquet_path).read()
    missing = sorted(set(REQUIRED_CANONICAL_COLUMNS) - set(table.column_names))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    table = table.select(list(REQUIRED_CANONICAL_COLUMNS))
    table = _normalize_open_time_column(table)
    if table.num_rows != int(manifest["row_count"]):
        raise ValueError("manifest row_count does not match parquet")
    return CanonicalOhlcvTable(
        timeframe=timeframe,
        fill_policy=fill_policy,
        manifest_path=manifest_path,
        parquet_path=parquet_path,
        table=table,
    )


def collect_offline_manifest_paths(root: Path, *, fill_policy: str = "raw") -> tuple[Path, ...]:
    base = root / "canonical_bars" / "timeframe=1d" / f"fill_policy={fill_policy}"
    manifest_paths = tuple(sorted(base.glob("year=*/manifest.json")))
    if not manifest_paths:
        raise ValueError("offline canonical manifests missing for timeframe=1d")
    return manifest_paths


def load_offline_ohlcv_table(
    root: Path,
    *,
    fill_policy: str = "raw",
) -> tuple[CanonicalOhlcvTable, tuple[Path, ...]]:
    manifest_paths = collect_offline_manifest_paths(root, fill_policy=fill_policy)
    tables = [
        load_manifested_table(path, timeframe="1d", fill_policy=fill_policy).table
        for path in manifest_paths
    ]
    return (
        CanonicalOhlcvTable(
            timeframe="1d",
            fill_policy=fill_policy,
            manifest_path=manifest_paths[0],
            parquet_path=Path("multiple-partitions"),
            table=pa.concat_tables(tables, promote_options="default"),
        ),
        manifest_paths,
    )


def _as_bool(value) -> bool:
    return bool(value)


def _is_quality_row(row: dict) -> bool:
    open_ = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    quote_volume = float(row["quote_volume"])
    return bool(
        _as_bool(row["is_complete"])
        and not _as_bool(row["has_synthetic"])
        and row["bar_count"] == row["expected_1m_count"]
        and open_ > 0
        and high > 0
        and low > 0
        and close > 0
        and high >= max(open_, close)
        and low <= min(open_, close)
        and quote_volume > 0
    )


def prepare_ohlcv_arrays(table: CanonicalOhlcvTable) -> OhlcvArrays:
    rows = table.table.to_pylist()
    symbols = tuple(sorted({row["symbol"] for row in rows}))
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
        s_index = symbol_index[row["symbol"]]
        for name in arrays:
            arrays[name][t_index, s_index] = float(row[name])
        quality[t_index, s_index] = _is_quality_row(row)
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
