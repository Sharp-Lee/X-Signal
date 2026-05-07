from __future__ import annotations

import pytest

from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)


def test_paths_are_strategy_scoped(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)

    assert paths.base == tmp_path / "strategies" / "volume_price_efficiency_v1"
    assert paths.runs == paths.base / "runs"
    assert paths.run_dir("run123") == paths.runs / "run123"


def test_run_id_rejects_path_traversal(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)

    for bad_run_id in ["", "../abc", "abc/def", "abc\\def"]:
        with pytest.raises(ValueError, match="run_id"):
            paths.run_dir(bad_run_id)
