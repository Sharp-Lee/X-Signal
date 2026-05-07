from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_CANONICAL_COLUMNS = (
    "symbol",
    "open_time",
    "close",
    "quote_volume",
    "bar_count",
    "expected_1m_count",
    "is_complete",
    "has_synthetic",
    "fill_policy",
)


@dataclass(frozen=True)
class CanonicalBarTable:
    timeframe: str
    fill_policy: str
    manifest_path: Path
    parquet_path: Path
    table: pa.Table


def load_manifested_table(
    manifest_path: Path,
    *,
    timeframe: str,
    fill_policy: str,
) -> CanonicalBarTable:
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
    if table.num_rows != int(manifest["row_count"]):
        raise ValueError("manifest row_count does not match parquet")
    return CanonicalBarTable(
        timeframe=timeframe,
        fill_policy=fill_policy,
        manifest_path=manifest_path,
        parquet_path=parquet_path,
        table=table,
    )
