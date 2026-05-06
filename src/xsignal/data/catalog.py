from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path

from pydantic import ValidationError

from xsignal.data.canonical_bars import Partition
from xsignal.data.locks import ExportLock, atomic_publish
from xsignal.data.paths import CanonicalPaths
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
        parquet_path = self.paths.parquet_path(partition)
        manifest_path = self.paths.manifest_path(partition)
        if not parquet_path.exists() or not manifest_path.exists():
            return PartitionStatus.MISSING
        if not parquet_path.is_file():
            return PartitionStatus.STALE

        manifest = self.load_manifest(partition)
        if manifest is None:
            return PartitionStatus.STALE

        if manifest.dataset_version != dataset_version:
            return PartitionStatus.STALE
        if manifest.timeframe != partition.timeframe:
            return PartitionStatus.STALE
        if manifest.partition_key != partition.key:
            return PartitionStatus.STALE
        if manifest.parquet_path != str(parquet_path):
            return PartitionStatus.STALE
        if manifest.row_count <= 0:
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
            catalog["partitions"][manifest.partition_key] = {
                "dataset_version": manifest.dataset_version,
                "row_count": manifest.row_count,
                "query_hash": manifest.query_hash,
                "parquet_path": manifest.parquet_path,
                "exported_at": manifest.exported_at,
            }
            temp_path = catalog_path.with_name(f".{catalog_path.name}.tmp")
            temp_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
            atomic_publish(temp_path, catalog_path)
