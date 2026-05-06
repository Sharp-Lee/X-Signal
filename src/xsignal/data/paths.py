from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xsignal.data.canonical_bars import Partition, validate_timeframe


def _validate_run_id(run_id: str) -> str:
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError("run_id must be non-empty and must not contain path separators or '..'")
    return run_id


@dataclass(frozen=True)
class CanonicalPaths:
    root: Path

    @property
    def base(self) -> Path:
        return self.root / "canonical_bars"

    def partition_dir(self, partition: Partition) -> Path:
        path = self.base / f"timeframe={partition.timeframe}" / f"year={partition.year:04d}"
        if partition.month is not None:
            path = path / f"month={partition.month:02d}"
        return path

    def parquet_path(self, partition: Partition) -> Path:
        return self.partition_dir(partition) / "bars.parquet"

    def temp_parquet_path(self, partition: Partition, run_id: str) -> Path:
        run_id = _validate_run_id(run_id)
        return self.partition_dir(partition) / f".bars.{run_id}.tmp.parquet"

    def manifest_path(self, partition: Partition) -> Path:
        return self.partition_dir(partition) / "manifest.json"

    def temp_manifest_path(self, partition: Partition, run_id: str) -> Path:
        run_id = _validate_run_id(run_id)
        return self.partition_dir(partition) / f".manifest.{run_id}.tmp.json"

    def lock_path(self, partition: Partition) -> Path:
        lock_name = partition.key.replace("/", "__") + ".lock"
        return self.base / "_locks" / lock_name

    def catalog_path(self, timeframe: str) -> Path:
        validate_timeframe(timeframe)
        return self.base / "_catalog" / f"timeframe={timeframe}.json"
