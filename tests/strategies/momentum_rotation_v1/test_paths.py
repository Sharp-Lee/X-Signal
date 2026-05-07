from __future__ import annotations

from xsignal.strategies.momentum_rotation_v1.paths import MomentumRotationPaths


def test_strategy_paths_are_under_strategy_owned_data_dir(tmp_path):
    paths = MomentumRotationPaths(root=tmp_path)

    assert paths.base == tmp_path / "strategies" / "momentum_rotation_v1"
    assert paths.cache == paths.base / "cache"
    assert paths.runs == paths.base / "runs"
    assert paths.cache_file("close_1d.npy") == paths.cache / "close_1d.npy"
    assert paths.run_dir("abc123") == paths.runs / "abc123"


def test_run_id_rejects_path_traversal(tmp_path):
    paths = MomentumRotationPaths(root=tmp_path)

    for bad_run_id in ["", "../abc", "abc/def", "abc\\def"]:
        try:
            paths.run_dir(bad_run_id)
        except ValueError as exc:
            assert "run_id" in str(exc)
        else:
            raise AssertionError(f"run_id {bad_run_id!r} should fail")
