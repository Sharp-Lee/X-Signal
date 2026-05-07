from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _validate_plain_id(value: str, field_name: str) -> str:
    if not value or "/" in value or "\\" in value or ".." in value:
        raise ValueError(
            f"{field_name} must be non-empty and must not contain path separators or '..'"
        )
    return value


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

    @property
    def scans(self) -> Path:
        return self.base / "scans"

    def cache_file(self, name: str) -> Path:
        if "/" in name or "\\" in name or not name:
            raise ValueError("cache file name must be a plain file name")
        return self.cache / name

    def run_dir(self, run_id: str) -> Path:
        return self.runs / _validate_plain_id(run_id, "run_id")

    def scan_dir(self, scan_id: str) -> Path:
        return self.scans / _validate_plain_id(scan_id, "scan_id")
