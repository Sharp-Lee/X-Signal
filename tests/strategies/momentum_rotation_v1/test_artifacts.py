from __future__ import annotations

import json

import numpy as np
import pytest

from xsignal.strategies.momentum_rotation_v1.artifacts import (
    build_backtest_summary,
    write_run_artifacts,
    write_scan_artifacts,
)
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


def test_build_backtest_summary_handles_empty_period_returns():
    result = BacktestResult(
        equity=np.array([1.0]),
        period_returns=np.array([]),
        weights=np.array([[1.0]]),
        turnover=np.array([1.0]),
        costs=np.array([0.001]),
    )

    summary = build_backtest_summary(result)

    assert summary["period_count"] == 0
    assert summary["mean_period_return"] == 0.0
    assert summary["total_cost"] == pytest.approx(0.001)


def test_write_scan_artifacts_creates_manifest_json_and_csv(tmp_path):
    paths = MomentumRotationPaths(root=tmp_path)
    rows = [
        {
            "scan_id": "scan123",
            "config_hash": "hash1",
            "top_n": 1,
            "fee_bps": 5.0,
            "slippage_bps": 1.0,
            "min_rolling_7d_quote_volume": 0.0,
            "initial_equity": 1.0,
            "final_equity": 1.01,
            "total_return": 0.01,
            "period_count": 1,
            "mean_period_return": 0.01,
            "total_cost": 0.001,
        },
        {
            "scan_id": "scan123",
            "config_hash": "hash2",
            "top_n": 2,
            "fee_bps": 5.0,
            "slippage_bps": 1.0,
            "min_rolling_7d_quote_volume": 0.0,
            "initial_equity": 1.0,
            "final_equity": 1.02,
            "total_return": 0.02,
            "period_count": 1,
            "mean_period_return": 0.02,
            "total_cost": 0.001,
        },
    ]

    scan_dir = write_scan_artifacts(
        paths=paths,
        scan_id="scan123",
        base_config=MomentumRotationConfig(),
        rows=rows,
        canonical_manifests=["manifest-1h.json", "manifest-4h.json", "manifest-1d.json"],
        git_commit="abc123",
        runtime_seconds=2.5,
        symbol_count=300,
    )

    assert (scan_dir / "manifest.json").exists()
    assert (scan_dir / "summary.json").exists()
    assert (scan_dir / "summary.csv").exists()
    manifest = json.loads((scan_dir / "manifest.json").read_text())
    summary = json.loads((scan_dir / "summary.json").read_text())
    assert manifest["strategy_name"] == "momentum_rotation_v1"
    assert manifest["scan_id"] == "scan123"
    assert manifest["symbol_count"] == 300
    assert manifest["canonical_manifests"] == [
        "manifest-1h.json",
        "manifest-4h.json",
        "manifest-1d.json",
    ]
    assert summary["combination_count"] == 2
    assert summary["best_final_equity"] == 1.02
    assert "hash1" in (scan_dir / "summary.csv").read_text()
