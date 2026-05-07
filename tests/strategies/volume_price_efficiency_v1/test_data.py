from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xsignal.strategies.volume_price_efficiency_v1.data import (
    REQUIRED_CANONICAL_COLUMNS,
    CanonicalOhlcvTable,
    OhlcvArrays,
    collect_offline_manifest_paths,
    load_manifested_table,
    load_offline_ohlcv_table,
    prepare_ohlcv_arrays,
)


def _row(
    symbol: str,
    open_time,
    *,
    open_: float = 100.0,
    high: float = 105.0,
    low: float = 95.0,
    close: float = 103.0,
    quote_volume: float = 1_000_000.0,
    bar_count: int = 240,
    expected_1m_count: int = 240,
    is_complete=True,
    has_synthetic=False,
    fill_policy: str = "raw",
) -> dict:
    return {
        "symbol": symbol,
        "open_time": open_time,
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
        "quote_volume": quote_volume,
        "bar_count": bar_count,
        "expected_1m_count": expected_1m_count,
        "is_complete": is_complete,
        "has_synthetic": has_synthetic,
        "fill_policy": fill_policy,
    }


def write_4h_manifested_partition(
    tmp_path: Path,
    rows: list[dict],
    *,
    year: int = 2026,
    month: int = 1,
    timeframe: str = "4h",
    fill_policy: str = "raw",
) -> Path:
    partition_dir = (
        tmp_path
        / "canonical_bars"
        / f"timeframe={timeframe}"
        / f"fill_policy={fill_policy}"
        / f"year={year:04d}"
        / f"month={month:02d}"
    )
    partition_dir.mkdir(parents=True)
    parquet_path = partition_dir / "bars.abc.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet_path)
    manifest_path = partition_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "timeframe": timeframe,
                "fill_policy": fill_policy,
                "parquet_path": str(parquet_path),
                "row_count": len(rows),
            }
        )
    )
    return manifest_path


def test_load_manifested_table_validates_identity_and_columns(tmp_path):
    first_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    manifest_path = write_4h_manifested_partition(
        tmp_path,
        [_row("BTCUSDT", first_time), _row("ETHUSDT", first_time)],
    )

    loaded = load_manifested_table(manifest_path)

    assert isinstance(loaded, CanonicalOhlcvTable)
    assert loaded.timeframe == "4h"
    assert loaded.fill_policy == "raw"
    assert loaded.manifest_path == manifest_path
    assert loaded.table.num_rows == 2
    assert loaded.table.column_names == list(REQUIRED_CANONICAL_COLUMNS)


def test_load_manifested_table_normalizes_epoch_seconds_open_time(tmp_path):
    first_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    manifest_path = write_4h_manifested_partition(
        tmp_path,
        [_row("BTCUSDT", int(first_time.timestamp()))],
    )

    loaded = load_manifested_table(manifest_path)

    assert loaded.table["open_time"].type == pa.timestamp("s", tz="UTC")
    assert loaded.table["open_time"].to_pylist() == [first_time]


def test_load_manifested_table_rejects_missing_required_ohlcv_column(tmp_path):
    manifest_path = write_4h_manifested_partition(
        tmp_path,
        [_row("BTCUSDT", datetime(2026, 1, 1, tzinfo=timezone.utc))],
    )
    manifest = json.loads(manifest_path.read_text())
    pq.write_table(pa.table({"symbol": ["BTCUSDT"]}), Path(manifest["parquet_path"]))

    with pytest.raises(ValueError, match="missing required columns"):
        load_manifested_table(manifest_path)


def test_load_manifested_table_rejects_manifest_mismatch(tmp_path):
    manifest_path = write_4h_manifested_partition(
        tmp_path,
        [_row("BTCUSDT", datetime(2026, 1, 1, tzinfo=timezone.utc))],
        timeframe="1h",
    )

    with pytest.raises(ValueError, match="timeframe"):
        load_manifested_table(manifest_path)


def test_prepare_ohlcv_arrays_aligns_symbols_times_and_quality(tmp_path):
    first_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    second_time = datetime(2026, 1, 1, 4, tzinfo=timezone.utc)
    manifest_path = write_4h_manifested_partition(
        tmp_path,
        [
            _row("ETHUSDT", second_time, open_=200.0, high=210.0, low=195.0, close=205.0),
            _row("BTCUSDT", first_time, close=101.0),
            _row("ETHUSDT", first_time, open_=200.0, high=205.0, low=195.0, close=201.0),
            _row("BTCUSDT", second_time, close=102.0, is_complete=False),
        ],
    )
    table = load_manifested_table(manifest_path)

    arrays = prepare_ohlcv_arrays(table)

    assert isinstance(arrays, OhlcvArrays)
    assert arrays.symbols == ("BTCUSDT", "ETHUSDT")
    assert arrays.open_times.tolist() == [first_time, second_time]
    assert arrays.close.tolist() == [[101.0, 201.0], [102.0, 205.0]]
    assert arrays.open.shape == (2, 2)
    assert arrays.high.shape == (2, 2)
    assert arrays.low.shape == (2, 2)
    assert arrays.quote_volume.shape == (2, 2)
    assert arrays.quality.tolist() == [[True, True], [False, True]]


def test_prepare_ohlcv_arrays_marks_invalid_ohlc_as_bad_quality(tmp_path):
    first_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    manifest_path = write_4h_manifested_partition(
        tmp_path,
        [
            _row("BTCUSDT", first_time, high=99.0, low=95.0, open_=100.0, close=98.0),
            _row("ETHUSDT", first_time, high=105.0, low=106.0, open_=100.0, close=103.0),
            _row("SOLUSDT", first_time, quote_volume=0.0),
        ],
    )
    table = load_manifested_table(manifest_path)

    arrays = prepare_ohlcv_arrays(table)

    assert arrays.symbols == ("BTCUSDT", "ETHUSDT", "SOLUSDT")
    assert arrays.quality.tolist() == [[False, False, False]]


def test_load_offline_ohlcv_table_reads_all_4h_manifests(tmp_path):
    first_time = datetime(2026, 1, 1, tzinfo=timezone.utc)
    second_time = datetime(2026, 2, 1, tzinfo=timezone.utc)
    first_manifest = write_4h_manifested_partition(
        tmp_path,
        [_row("BTCUSDT", first_time)],
        month=1,
    )
    second_manifest = write_4h_manifested_partition(
        tmp_path,
        [_row("BTCUSDT", second_time)],
        month=2,
    )

    manifest_paths = collect_offline_manifest_paths(tmp_path)
    loaded, loaded_paths = load_offline_ohlcv_table(tmp_path)

    assert manifest_paths == (first_manifest, second_manifest)
    assert loaded_paths == (first_manifest, second_manifest)
    assert loaded.table.num_rows == 2


def test_collect_offline_manifest_paths_rejects_missing_4h_data(tmp_path):
    with pytest.raises(ValueError, match="4h"):
        collect_offline_manifest_paths(tmp_path)
