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
    StrategyCanonicalInputs,
    collect_offline_strategy_inputs,
    collect_strategy_inputs,
    load_manifested_table,
)
from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
from xsignal.data.canonical_bars import Partition
from xsignal.data.canonical_export import CanonicalDataset


def write_manifested_partition(
    root: Path,
    timeframe: str = "1d",
    fill_policy: str = "raw",
    *,
    year: int = 2026,
    month: int | None = None,
    open_time: datetime | None = None,
) -> Path:
    open_time = open_time or datetime(year, month or 1, 1, tzinfo=timezone.utc)
    partition_dir = (
        root
        / "canonical_bars"
        / f"timeframe={timeframe}"
        / f"fill_policy={fill_policy}"
        / f"year={year:04d}"
    )
    if month is not None:
        partition_dir = partition_dir / f"month={month:02d}"
    parquet_path = partition_dir / "bars.abc.parquet"
    partition_dir.mkdir(parents=True)
    table = pa.table(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "open_time": pa.array(
                [
                    open_time,
                    open_time,
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


def test_load_manifested_table_normalizes_epoch_seconds_open_time(tmp_path):
    manifest_path = write_manifested_partition(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    first_open = datetime(2026, 1, 1, tzinfo=timezone.utc)
    second_open = datetime(2026, 1, 2, tzinfo=timezone.utc)
    table = pa.table(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "open_time": pa.array(
                [int(first_open.timestamp()), int(second_open.timestamp())],
                type=pa.uint32(),
            ),
            "close": [100.0, 50.0],
            "quote_volume": [10_000.0, 8_000.0],
            "bar_count": [1440, 1440],
            "expected_1m_count": [1440, 1440],
            "is_complete": [True, True],
            "has_synthetic": [False, False],
            "fill_policy": ["raw", "raw"],
        }
    )
    pq.write_table(table.select(list(REQUIRED_CANONICAL_COLUMNS)), Path(manifest["parquet_path"]))

    loaded = load_manifested_table(manifest_path, timeframe="1d", fill_policy="raw")

    assert loaded.table["open_time"].type == pa.timestamp("s", tz="UTC")
    assert loaded.table["open_time"].to_pylist() == [first_open, second_open]


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


def test_collect_strategy_inputs_calls_export_layer_for_required_timeframes(tmp_path):
    calls = []

    def fake_ensure(request, paths, partitions, exporter):
        calls.append((request, paths, partitions, exporter))
        expected_counts = {"1h": 60, "4h": 240, "1d": 1440}
        for partition in partitions:
            manifest_path = paths.manifest_path(partition)
            parquet_path = paths.published_parquet_path(partition, "abc123")
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.table(
                {
                    "symbol": ["BTCUSDT"],
                    "open_time": pa.array(
                        [datetime(2026, 1, 1, tzinfo=timezone.utc)],
                        type=pa.timestamp("us", tz="UTC"),
                    ),
                    "close": [100.0],
                    "quote_volume": [1_000_000.0],
                    "bar_count": [expected_counts[request.timeframe]],
                    "expected_1m_count": [expected_counts[request.timeframe]],
                    "is_complete": [True],
                    "has_synthetic": [False],
                    "fill_policy": ["raw"],
                }
            )
            pq.write_table(table.select(list(REQUIRED_CANONICAL_COLUMNS)), parquet_path)
            manifest_path.write_text(
                json.dumps(
                    {
                        "timeframe": request.timeframe,
                        "fill_policy": request.fill_policy,
                        "parquet_path": str(parquet_path),
                        "row_count": 1,
                    }
                )
            )
        return CanonicalDataset(request=request, root=paths.base, partitions=partitions)

    def fake_bounds(_client):
        return (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 2, 1, tzinfo=timezone.utc),
        )

    result = collect_strategy_inputs(
        root=tmp_path,
        config=MomentumRotationConfig(),
        ensure=fake_ensure,
        discover_bounds=fake_bounds,
        exporter_factory=lambda: object(),
    )

    assert isinstance(result, StrategyCanonicalInputs)
    assert [call[0].timeframe for call in calls] == ["1h", "4h", "1d"]
    assert [call[0].fill_policy for call in calls] == ["raw", "raw", "raw"]
    assert all(call[2] for call in calls)
    assert Partition(timeframe="1h", year=2026, month=1) in calls[0][2]


def test_collect_offline_strategy_inputs_reads_manifests_without_clickhouse_or_export(tmp_path):
    write_manifested_partition(tmp_path, timeframe="1h", year=2026, month=1)
    write_manifested_partition(tmp_path, timeframe="4h", year=2026, month=1)
    write_manifested_partition(tmp_path, timeframe="1d", year=2026)

    result = collect_strategy_inputs(
        root=tmp_path,
        config=MomentumRotationConfig(),
        offline=True,
        ensure=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no export")),
        discover_bounds=lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("no bounds")),
        clickhouse_client_factory=lambda: (_ for _ in ()).throw(AssertionError("no clickhouse")),
    )

    assert isinstance(result, StrategyCanonicalInputs)
    assert result.bars_1h.table.num_rows == 2
    assert result.bars_4h.table.num_rows == 2
    assert result.bars_1d.table.num_rows == 2
    assert [path.name for path in result.manifest_paths] == [
        "manifest.json",
        "manifest.json",
        "manifest.json",
    ]


def test_collect_offline_strategy_inputs_rejects_missing_required_timeframe(tmp_path):
    write_manifested_partition(tmp_path, timeframe="1h", year=2026, month=1)
    write_manifested_partition(tmp_path, timeframe="1d", year=2026)

    with pytest.raises(ValueError, match="offline canonical manifests missing.*4h"):
        collect_offline_strategy_inputs(root=tmp_path, config=MomentumRotationConfig())


def test_collect_offline_strategy_inputs_orders_monthly_manifests(tmp_path):
    write_manifested_partition(
        tmp_path,
        timeframe="1h",
        year=2026,
        month=2,
        open_time=datetime(2026, 2, 1, tzinfo=timezone.utc),
    )
    write_manifested_partition(
        tmp_path,
        timeframe="1h",
        year=2026,
        month=1,
        open_time=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    write_manifested_partition(tmp_path, timeframe="4h", year=2026, month=1)
    write_manifested_partition(tmp_path, timeframe="1d", year=2026)

    result = collect_offline_strategy_inputs(root=tmp_path, config=MomentumRotationConfig())

    open_times = result.bars_1h.table["open_time"].to_pylist()
    assert open_times == [
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 1, 1, tzinfo=timezone.utc),
        datetime(2026, 2, 1, tzinfo=timezone.utc),
        datetime(2026, 2, 1, tzinfo=timezone.utc),
    ]


def test_collect_offline_strategy_inputs_rejects_partition_gaps(tmp_path):
    write_manifested_partition(tmp_path, timeframe="1h", year=2026, month=1)
    write_manifested_partition(tmp_path, timeframe="1h", year=2026, month=3)
    write_manifested_partition(tmp_path, timeframe="4h", year=2026, month=1)
    write_manifested_partition(tmp_path, timeframe="1d", year=2026)

    with pytest.raises(ValueError, match="offline canonical manifests are not contiguous.*1h"):
        collect_offline_strategy_inputs(root=tmp_path, config=MomentumRotationConfig())
