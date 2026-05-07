from __future__ import annotations

import json

import numpy as np

from xsignal.strategies.momentum_rotation_v1.cli import main
from xsignal.strategies.momentum_rotation_v1.kernel import BacktestResult
from xsignal.strategies.momentum_rotation_v1.prepare import PreparedArrays
from xsignal.strategies.momentum_rotation_v1.signals import SignalArrays


def test_cli_run_writes_artifacts_with_injected_pipeline(tmp_path, monkeypatch):
    arrays = PreparedArrays(
        symbols=("BTCUSDT",),
        rebalance_times=np.array(["2026-01-02", "2026-01-03"], dtype=object),
        close_1h=np.array([[100.0], [101.0]]),
        close_4h=np.array([[100.0], [101.0]]),
        close_1d=np.array([[100.0], [101.0]]),
        quote_volume_1d=np.ones((2, 1)),
        complete_1h=np.ones((2, 1), dtype=bool),
        complete_4h=np.ones((2, 1), dtype=bool),
        complete_1d=np.ones((2, 1), dtype=bool),
        quality_1h_24h=np.ones((2, 1), dtype=bool),
        quality_4h_7d=np.ones((2, 1), dtype=bool),
        quality_1d_30d=np.ones((2, 1), dtype=bool),
    )
    signals = SignalArrays(score=np.array([[1.0], [1.0]]), tradable_mask=np.ones((2, 1), dtype=bool))
    result = BacktestResult(
        equity=np.array([1.0, 1.01]),
        period_returns=np.array([0.01]),
        weights=np.array([[1.0], [1.0]]),
        turnover=np.array([1.0, 0.0]),
        costs=np.array([0.0, 0.0]),
    )

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.prepare_from_canonical",
        lambda *_args, **_kwargs: (arrays, ["manifest.json"]),
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.compute_momentum_signals",
        lambda *_args, **_kwargs: signals,
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.run_backtest",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr("xsignal.strategies.momentum_rotation_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(["run", "--root", str(tmp_path), "--run-id", "testrun", "--top-n", "1"])

    assert exit_code == 0
    manifest_path = (
        tmp_path / "strategies" / "momentum_rotation_v1" / "runs" / "testrun" / "manifest.json"
    )
    assert json.loads(manifest_path.read_text())["git_commit"] == "abc123"
