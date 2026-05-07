from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from xsignal.data.canonical_bars import CanonicalRequest
from xsignal.data.canonical_export import (
    ClickHouseExporter,
    discover_full_history_bounds,
    ensure_canonical_bars,
    partitions_for_full_history,
)
from xsignal.data.clickhouse import ClickHouseClient, ClickHouseConfig
from xsignal.data.paths import CanonicalPaths
from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig


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


def _normalize_open_time_column(table: pa.Table) -> pa.Table:
    field_index = table.schema.get_field_index("open_time")
    open_time = table["open_time"]
    target_type = pa.timestamp("s", tz="UTC")
    if pa.types.is_timestamp(open_time.type):
        normalized = pc.cast(open_time, target_type)
    elif pa.types.is_integer(open_time.type):
        # ClickHouse DateTime can arrive through Arrow as epoch seconds.
        normalized = pa.chunked_array(
            [chunk.cast(pa.int64()).view(target_type) for chunk in open_time.chunks]
        )
    else:
        raise ValueError(f"unsupported open_time type: {open_time.type}")
    return table.set_column(field_index, "open_time", normalized)


@dataclass(frozen=True)
class CanonicalBarTable:
    timeframe: str
    fill_policy: str
    manifest_path: Path
    parquet_path: Path
    table: pa.Table


@dataclass(frozen=True)
class StrategyCanonicalInputs:
    bars_1h: CanonicalBarTable
    bars_4h: CanonicalBarTable
    bars_1d: CanonicalBarTable
    manifest_paths: tuple[Path, ...]


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
    table = _normalize_open_time_column(table)
    if table.num_rows != int(manifest["row_count"]):
        raise ValueError("manifest row_count does not match parquet")
    return CanonicalBarTable(
        timeframe=timeframe,
        fill_policy=fill_policy,
        manifest_path=manifest_path,
        parquet_path=parquet_path,
        table=table,
    )


def collect_strategy_inputs(
    *,
    root: Path,
    config: MomentumRotationConfig,
    ensure: Callable = ensure_canonical_bars,
    discover_bounds: Callable = discover_full_history_bounds,
    exporter_factory: Callable[[], object] | None = None,
) -> StrategyCanonicalInputs:
    client = None if exporter_factory else ClickHouseClient(ClickHouseConfig())
    start, end = discover_bounds(client)
    exporter = exporter_factory() if exporter_factory else ClickHouseExporter(client)
    loaded: dict[str, CanonicalBarTable] = {}
    manifests: list[Path] = []
    for timeframe in config.timeframes:
        request = CanonicalRequest(timeframe=timeframe, fill_policy=config.fill_policy)
        paths = CanonicalPaths(root=root, fill_policy=config.fill_policy)
        partitions = partitions_for_full_history(timeframe, start, end)
        dataset = ensure(request, paths, partitions, exporter)
        timeframe_manifests: list[Path] = []
        for partition in dataset.partitions:
            manifest_path = paths.manifest_path(partition)
            if manifest_path.exists():
                timeframe_manifests.append(manifest_path)
        if not timeframe_manifests:
            raise ValueError(f"no manifests found for timeframe={timeframe}")
        manifests.extend(timeframe_manifests)
        timeframe_tables = [
            load_manifested_table(
                manifest_path,
                timeframe=timeframe,
                fill_policy=config.fill_policy,
            ).table
            for manifest_path in timeframe_manifests
        ]
        if not timeframe_tables:
            raise ValueError(f"no canonical tables loaded for timeframe={timeframe}")
        loaded[timeframe] = CanonicalBarTable(
            timeframe=timeframe,
            fill_policy=config.fill_policy,
            manifest_path=timeframe_manifests[0],
            parquet_path=Path("multiple-partitions"),
            table=pa.concat_tables(timeframe_tables, promote_options="default"),
        )
    return StrategyCanonicalInputs(
        bars_1h=loaded["1h"],
        bars_4h=loaded["4h"],
        bars_1d=loaded["1d"],
        manifest_paths=tuple(manifests),
    )
