from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xsignal.data.canonical_bars import Partition, validate_fill_policy, validate_timeframe


def _validate_run_id(run_id: str) -> str:
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError("run_id must be non-empty and must not contain path separators or '..'")
    return run_id


@dataclass(frozen=True)
class CanonicalPaths:
    root: Path
    fill_policy: str = "raw"

    def __post_init__(self) -> None:
        validate_fill_policy(self.fill_policy)

    @property
    def base(self) -> Path:
        return self.root / "canonical_bars"

    def partition_dir(self, partition: Partition) -> Path:
        path = (
            self.base
            / f"timeframe={partition.timeframe}"
            / f"fill_policy={self.fill_policy}"
            / f"year={partition.year:04d}"
        )
        if partition.month is not None:
            path = path / f"month={partition.month:02d}"
        return path

    def parquet_path(self, partition: Partition) -> Path:
        return self.partition_dir(partition) / "bars.parquet"

    def published_parquet_path(self, partition: Partition, run_id: str) -> Path:
        run_id = _validate_run_id(run_id)
        return self.partition_dir(partition) / f"bars.{run_id}.parquet"

    def temp_parquet_path(self, partition: Partition, run_id: str) -> Path:
        run_id = _validate_run_id(run_id)
        return self.partition_dir(partition) / f".bars.{run_id}.tmp.parquet"

    def manifest_path(self, partition: Partition) -> Path:
        return self.partition_dir(partition) / "manifest.json"

    def temp_manifest_path(self, partition: Partition, run_id: str) -> Path:
        run_id = _validate_run_id(run_id)
        return self.partition_dir(partition) / f".manifest.{run_id}.tmp.json"

    def lock_path(self, partition: Partition) -> Path:
        lock_parts = [
            f"timeframe={partition.timeframe}",
            f"fill_policy={self.fill_policy}",
            f"year={partition.year:04d}",
        ]
        if partition.month is not None:
            lock_parts.append(f"month={partition.month:02d}")
        lock_name = "__".join(lock_parts) + ".lock"
        return self.base / "_locks" / lock_name

    def catalog_path(self, timeframe: str) -> Path:
        validate_timeframe(timeframe)
        return self.base / "_catalog" / f"timeframe={timeframe}" / f"fill_policy={self.fill_policy}.json"
