from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xsignal.strategies.momentum_rotation_v1.data import (
    REQUIRED_CANONICAL_COLUMNS,
    CanonicalBarTable,
    load_manifested_table,
)


def write_manifested_partition(
    root: Path,
    timeframe: str = "1d",
    fill_policy: str = "raw",
) -> Path:
    partition_dir = (
        root
        / "canonical_bars"
        / f"timeframe={timeframe}"
        / f"fill_policy={fill_policy}"
        / "year=2026"
    )
    parquet_path = partition_dir / "bars.abc.parquet"
    partition_dir.mkdir(parents=True)
    table = pa.table(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "open_time": pa.array(
                [
                    datetime(2026, 1, 1, tzinfo=timezone.utc),
                    datetime(2026, 1, 1, tzinfo=timezone.utc),
                ],
                type=pa.timestamp("us", tz="UTC"),
            ),
            "close": [100.0, 50.0],
            "quote_volume": [10_000.0, 8_000.0],
            "bar_count": [1440, 1440],
            "expected_1m_count": [1440, 1440],
            "is_complete": [True, True],
            "has_synthetic": [False, False],
            "fill_policy": [fill_policy, fill_policy],
        }
    )
    pq.write_table(table.select(list(REQUIRED_CANONICAL_COLUMNS)), parquet_path)
    manifest = {
        "timeframe": timeframe,
        "fill_policy": fill_policy,
        "parquet_path": str(parquet_path),
        "row_count": 2,
    }
    (partition_dir / "manifest.json").write_text(json.dumps(manifest))
    return partition_dir / "manifest.json"


def test_load_manifested_table_validates_identity_and_columns(tmp_path):
    manifest_path = write_manifested_partition(tmp_path)

    loaded = load_manifested_table(manifest_path, timeframe="1d", fill_policy="raw")

    assert isinstance(loaded, CanonicalBarTable)
    assert loaded.timeframe == "1d"
    assert loaded.fill_policy == "raw"
    assert loaded.manifest_path == manifest_path
    assert loaded.table.num_rows == 2
    assert loaded.table.column_names == list(REQUIRED_CANONICAL_COLUMNS)


def test_load_manifested_table_rejects_fill_policy_mismatch(tmp_path):
    manifest_path = write_manifested_partition(tmp_path, fill_policy="prev_close_zero_volume")

    with pytest.raises(ValueError, match="fill_policy"):
        load_manifested_table(manifest_path, timeframe="1d", fill_policy="raw")


def test_load_manifested_table_rejects_missing_required_column(tmp_path):
    manifest_path = write_manifested_partition(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    pq.write_table(pa.table({"symbol": ["BTCUSDT"]}), Path(manifest["parquet_path"]))

    with pytest.raises(ValueError, match="missing required columns"):
        load_manifested_table(manifest_path, timeframe="1d", fill_policy="raw")
