from __future__ import annotations

import json

import numpy as np
import pytest

from xsignal.strategies.momentum_rotation_v1.artifacts import write_run_artifacts
from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
from xsignal.strategies.momentum_rotation_v1.kernel import BacktestResult
from xsignal.strategies.momentum_rotation_v1.paths import MomentumRotationPaths


def test_write_run_artifacts_creates_manifest_summary_equity_and_positions(tmp_path):
    paths = MomentumRotationPaths(root=tmp_path)
    result = BacktestResult(
        equity=np.array([1.0, 1.1]),
        period_returns=np.array([0.1]),
        weights=np.array([[1.0, 0.0], [0.0, 1.0]]),
        turnover=np.array([1.0, 2.0]),
        costs=np.array([0.001, 0.002]),
    )

    run_dir = write_run_artifacts(
        paths=paths,
        run_id="run123",
        config=MomentumRotationConfig(top_n=1),
        symbols=("BTCUSDT", "ETHUSDT"),
        rebalance_times=np.array(["2026-01-02", "2026-01-03"], dtype=object),
        result=result,
        canonical_manifests=["manifest-1h.json", "manifest-4h.json", "manifest-1d.json"],
        git_commit="abc123",
        runtime_seconds=1.25,
    )

    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "equity_curve.parquet").exists()
    assert (run_dir / "daily_positions.parquet").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert manifest["strategy_name"] == "momentum_rotation_v1"
    assert manifest["config_hash"] == MomentumRotationConfig(top_n=1).config_hash()
    assert manifest["canonical_manifests"] == [
        "manifest-1h.json",
        "manifest-4h.json",
        "manifest-1d.json",
    ]
    assert summary["final_equity"] == 1.1
    assert summary["total_return"] == pytest.approx(0.1)
