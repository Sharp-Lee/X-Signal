from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import ValidationError

from xsignal.data.canonical_bars import CANONICAL_BAR_COLUMNS, Partition, partition_bounds
from xsignal.data.locks import ExportLock, atomic_publish
from xsignal.data.paths import CanonicalPaths
from xsignal.data.query_templates import CLICKHOUSE_SOURCE_TABLE, build_aggregate_query, query_hash
from xsignal.runs.manifest import ExportManifest


class PartitionStatus(StrEnum):
    COMPLETE = "complete"
    MISSING = "missing"
    STALE = "stale"


class Catalog:
    def __init__(self, paths: CanonicalPaths) -> None:
        self.paths = paths

    def _lock_path(self, timeframe: str) -> Path:
        catalog_path = self.paths.catalog_path(timeframe)
        return catalog_path.with_name(f"{catalog_path.name}.lock")

    def load_manifest(self, partition: Partition) -> ExportManifest | None:
        manifest_path = self.paths.manifest_path(partition)
        if not manifest_path.exists():
            return None
        try:
            return ExportManifest.model_validate_json(manifest_path.read_text())
        except (OSError, json.JSONDecodeError, ValidationError, ValueError):
            return None

    def status(self, partition: Partition, dataset_version: str) -> PartitionStatus:
        manifest_path = self.paths.manifest_path(partition)
        if not manifest_path.exists():
            return PartitionStatus.MISSING

        manifest = self.load_manifest(partition)
        if manifest is None:
            return PartitionStatus.STALE

        return self.status_for_manifest(partition, dataset_version, manifest)

    def status_for_manifest(
        self,
        partition: Partition,
        dataset_version: str,
        manifest: ExportManifest,
    ) -> PartitionStatus:
        parquet_path = Path(manifest.parquet_path)
        if not parquet_path.exists() or not parquet_path.is_file():
            return PartitionStatus.STALE
        if manifest.dataset_version != dataset_version:
            return PartitionStatus.STALE
        if manifest.timeframe != partition.timeframe:
            return PartitionStatus.STALE
        if manifest.partition_key != partition.key:
            return PartitionStatus.STALE
        if manifest.row_count <= 0:
            return PartitionStatus.STALE
        if manifest.source_table != CLICKHOUSE_SOURCE_TABLE:
            return PartitionStatus.STALE
        if manifest.deduplication_mode != "FINAL":
            return PartitionStatus.STALE
        if manifest.aggregation_semantics_version != "ohlcv-v1":
            return PartitionStatus.STALE
        start, end = partition_bounds(partition)
        expected_query_hash = query_hash(build_aggregate_query(partition.timeframe, start, end))
        if manifest.query_hash != expected_query_hash:
            return PartitionStatus.STALE
        try:
            import pyarrow.parquet as pq

            parquet_metadata = pq.read_metadata(parquet_path)
        except Exception:
            return PartitionStatus.STALE
        if parquet_metadata.num_rows != manifest.row_count:
            return PartitionStatus.STALE
        if tuple(parquet_metadata.schema.names) != CANONICAL_BAR_COLUMNS:
            return PartitionStatus.STALE
        return PartitionStatus.COMPLETE

    def mark_complete(self, manifest: ExportManifest) -> None:
        catalog_path = self.paths.catalog_path(manifest.timeframe)
        with ExportLock(self._lock_path(manifest.timeframe)):
            catalog_path.parent.mkdir(parents=True, exist_ok=True)
            if catalog_path.exists():
                catalog = json.loads(catalog_path.read_text())
            else:
                catalog = {"timeframe": manifest.timeframe, "partitions": {}}
            entry = {
                "dataset_version": manifest.dataset_version,
                "row_count": manifest.row_count,
                "query_hash": manifest.query_hash,
                "parquet_path": manifest.parquet_path,
                "exported_at": manifest.exported_at,
            }
            if catalog["partitions"].get(manifest.partition_key) == entry:
                return
            catalog["partitions"][manifest.partition_key] = entry
            temp_path = catalog_path.with_name(f".{catalog_path.name}.tmp")
            temp_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
            atomic_publish(temp_path, catalog_path)
