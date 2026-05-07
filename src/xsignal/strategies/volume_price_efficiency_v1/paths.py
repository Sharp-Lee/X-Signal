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
class VolumePriceEfficiencyPaths:
    root: Path

    @property
    def base(self) -> Path:
        return self.root / "strategies" / "volume_price_efficiency_v1"

    @property
    def runs(self) -> Path:
        return self.base / "runs"

    @property
    def scans(self) -> Path:
        return self.base / "scans"

    @property
    def trailing_runs(self) -> Path:
        return self.base / "trailing_runs"

    @property
    def trailing_scans(self) -> Path:
        return self.base / "trailing_scans"

    def run_dir(self, run_id: str) -> Path:
        return self.runs / _validate_plain_id(run_id, "run_id")

    def scan_dir(self, scan_id: str) -> Path:
        return self.scans / _validate_plain_id(scan_id, "scan_id")

    def trailing_run_dir(self, run_id: str) -> Path:
        return self.trailing_runs / _validate_plain_id(run_id, "run_id")

    def trailing_scan_dir(self, scan_id: str) -> Path:
        return self.trailing_scans / _validate_plain_id(scan_id, "scan_id")
