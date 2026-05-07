from __future__ import annotations

import pytest

from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)


def test_paths_are_strategy_scoped(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)

    assert paths.base == tmp_path / "strategies" / "volume_price_efficiency_v1"
    assert paths.runs == paths.base / "runs"
    assert paths.scans == paths.base / "scans"
    assert paths.trailing_runs == paths.base / "trailing_runs"
    assert paths.trailing_scans == paths.base / "trailing_scans"
    assert paths.trailing_diagnostics == paths.base / "trailing_diagnostics"
    assert paths.run_dir("run123") == paths.runs / "run123"
    assert paths.scan_dir("scan123") == paths.scans / "scan123"
    assert paths.trailing_run_dir("trail123") == paths.trailing_runs / "trail123"
    assert paths.trailing_scan_dir("trailscan123") == paths.trailing_scans / "trailscan123"
    assert (
        paths.trailing_diagnostic_dir("diag123")
        == paths.trailing_diagnostics / "diag123"
    )


def test_run_id_rejects_path_traversal(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)

    for bad_run_id in ["", "../abc", "abc/def", "abc\\def"]:
        with pytest.raises(ValueError, match="run_id"):
            paths.run_dir(bad_run_id)


def test_scan_id_rejects_path_traversal(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)

    for bad_scan_id in ["", "../abc", "abc/def", "abc\\def"]:
        with pytest.raises(ValueError, match="scan_id"):
            paths.scan_dir(bad_scan_id)


def test_trailing_run_id_rejects_path_traversal(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)

    for bad_run_id in ["", "../abc", "abc/def", "abc\\def"]:
        with pytest.raises(ValueError, match="run_id"):
            paths.trailing_run_dir(bad_run_id)


def test_trailing_scan_id_rejects_path_traversal(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)

    for bad_scan_id in ["", "../abc", "abc/def", "abc\\def"]:
        with pytest.raises(ValueError, match="scan_id"):
            paths.trailing_scan_dir(bad_scan_id)


def test_trailing_diagnostic_id_rejects_path_traversal(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)

    for bad_diagnostic_id in ["", "../abc", "abc/def", "abc\\def"]:
        with pytest.raises(ValueError, match="diagnostic_id"):
            paths.trailing_diagnostic_dir(bad_diagnostic_id)
