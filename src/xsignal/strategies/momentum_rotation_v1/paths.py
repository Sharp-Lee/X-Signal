from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _validate_run_id(run_id: str) -> str:
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError("run_id must be non-empty and must not contain path separators or '..'")
    return run_id


@dataclass(frozen=True)
class MomentumRotationPaths:
    root: Path

    @property
    def base(self) -> Path:
        return self.root / "strategies" / "momentum_rotation_v1"

    @property
    def cache(self) -> Path:
        return self.base / "cache"

    @property
    def runs(self) -> Path:
        return self.base / "runs"

    def cache_file(self, name: str) -> Path:
        if "/" in name or "\\" in name or not name:
            raise ValueError("cache file name must be a plain file name")
        return self.cache / name

    def run_dir(self, run_id: str) -> Path:
        return self.runs / _validate_run_id(run_id)
