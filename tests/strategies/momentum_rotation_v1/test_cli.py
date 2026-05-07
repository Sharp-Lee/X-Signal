from __future__ import annotations

import csv
import json
from datetime import date

import numpy as np
import pytest

from xsignal.strategies.momentum_rotation_v1.cli import main
from xsignal.strategies.momentum_rotation_v1.kernel import BacktestResult
from xsignal.strategies.momentum_rotation_v1.prepare import PreparedArrays, save_prepared_arrays
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


def test_cli_run_passes_offline_mode_to_canonical_preparation(tmp_path, monkeypatch):
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
    calls = []

    def fake_prepare(root, config, *, offline, use_cache):
        calls.append((root, config, offline, use_cache))
        return arrays, ["manifest.json"]

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.prepare_from_canonical",
        fake_prepare,
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

    exit_code = main(
        ["run", "--root", str(tmp_path), "--run-id", "offline-run", "--top-n", "1", "--offline"]
    )

    assert exit_code == 0
    assert calls[0][0] == tmp_path
    assert calls[0][2] is True
    assert calls[0][3] is True


def test_cli_run_passes_no_cache_to_canonical_preparation(tmp_path, monkeypatch):
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
    calls = []

    def fake_prepare(root, config, *, offline, use_cache):
        calls.append((root, config, offline, use_cache))
        return arrays, ["manifest.json"]

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.prepare_from_canonical",
        fake_prepare,
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

    exit_code = main(
        ["run", "--root", str(tmp_path), "--run-id", "nocache-run", "--top-n", "1", "--no-cache"]
    )

    assert exit_code == 0
    assert calls[0][3] is False


def test_cli_run_filters_rebalance_dates_before_signals_and_backtest(tmp_path, monkeypatch):
    arrays = PreparedArrays(
        symbols=("BTCUSDT",),
        rebalance_times=np.array(["2026-01-01", "2026-01-02", "2026-01-03"], dtype=object),
        close_1h=np.array([[100.0], [101.0], [102.0]]),
        close_4h=np.array([[100.0], [101.0], [102.0]]),
        close_1d=np.array([[100.0], [101.0], [102.0]]),
        quote_volume_1d=np.ones((3, 1)),
        complete_1h=np.ones((3, 1), dtype=bool),
        complete_4h=np.ones((3, 1), dtype=bool),
        complete_1d=np.ones((3, 1), dtype=bool),
        quality_1h_24h=np.ones((3, 1), dtype=bool),
        quality_4h_7d=np.ones((3, 1), dtype=bool),
        quality_1d_30d=np.ones((3, 1), dtype=bool),
    )
    signals = SignalArrays(score=np.array([[1.0], [1.0]]), tradable_mask=np.ones((2, 1), dtype=bool))
    signal_shapes = []

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.prepare_from_canonical",
        lambda *_args, **_kwargs: (arrays, ["manifest.json"]),
    )

    def fake_signals(filtered, _config):
        signal_shapes.append(filtered.close_1d.shape)
        return signals

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.compute_momentum_signals",
        fake_signals,
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.run_backtest",
        lambda *_args, **_kwargs: BacktestResult(
            equity=np.array([1.0, 1.01]),
            period_returns=np.array([0.01]),
            weights=np.array([[1.0], [1.0]]),
            turnover=np.array([1.0, 0.0]),
            costs=np.array([0.0, 0.0]),
        ),
    )
    monkeypatch.setattr("xsignal.strategies.momentum_rotation_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        [
            "run",
            "--root",
            str(tmp_path),
            "--run-id",
            "filtered-run",
            "--start-date",
            "2026-01-02",
            "--end-date",
            "2026-01-04",
        ]
    )

    assert exit_code == 0
    assert signal_shapes == [(2, 1)]


def test_cli_scan_writes_summary_for_parameter_grid_with_single_prepare(tmp_path, monkeypatch):
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
    prepare_calls = []
    signal_min_volumes = []
    backtest_configs = []

    def fake_prepare(root, config, *, offline, use_cache):
        prepare_calls.append((root, config, offline, use_cache))
        return arrays, ["manifest.json"]

    def fake_signals(_arrays, config):
        signal_min_volumes.append(config.min_rolling_7d_quote_volume)
        return signals

    def fake_backtest(_arrays, _signals, config):
        backtest_configs.append(config)
        final_equity = 1.0 + config.top_n / 100.0 + config.slippage_bps / 10_000.0
        return BacktestResult(
            equity=np.array([1.0, final_equity]),
            period_returns=np.array([final_equity - 1.0]),
            weights=np.array([[1.0], [1.0]]),
            turnover=np.array([1.0, 0.0]),
            costs=np.array([config.fee_bps / 10_000.0, 0.0]),
        )

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.prepare_from_canonical",
        fake_prepare,
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.compute_momentum_signals",
        fake_signals,
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.run_backtest",
        fake_backtest,
    )
    monkeypatch.setattr("xsignal.strategies.momentum_rotation_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        [
            "scan",
            "--root",
            str(tmp_path),
            "--scan-id",
            "scan123",
            "--offline",
            "--top-n",
            "1,2",
            "--fee-bps",
            "5",
            "--slippage-bps",
            "1,2",
            "--min-rolling-7d-quote-volume",
            "0",
            "--holdout-days",
            "0",
        ]
    )

    scan_dir = tmp_path / "strategies" / "momentum_rotation_v1" / "scans" / "scan123"
    rows = list(csv.DictReader((scan_dir / "summary.csv").open()))
    manifest = json.loads((scan_dir / "manifest.json").read_text())
    summary = json.loads((scan_dir / "summary.json").read_text())

    assert exit_code == 0
    assert len(prepare_calls) == 1
    assert prepare_calls[0][0] == tmp_path
    assert prepare_calls[0][2] is True
    assert prepare_calls[0][3] is True
    assert signal_min_volumes == [0.0]
    assert [config.slippage_bps for config in backtest_configs] == [1.0, 2.0, 1.0, 2.0]
    assert len(rows) == 4
    assert rows[0]["scan_id"] == "scan123"
    assert rows[0]["top_n"] == "1"
    assert rows[0]["fee_bps"] == "5.0"
    assert rows[0]["slippage_bps"] == "1.0"
    assert float(rows[0]["final_equity"]) == 1.0101
    assert manifest["git_commit"] == "abc123"
    assert manifest["canonical_manifests"] == ["manifest.json"]
    assert summary["combination_count"] == 4


def test_cli_scan_defaults_to_research_split_and_records_holdout_metadata(tmp_path, monkeypatch):
    arrays = PreparedArrays(
        symbols=("BTCUSDT",),
        rebalance_times=np.array(
            [date(2026, 1, 1), date(2026, 1, 2), date(2026, 1, 3), date(2026, 1, 4)],
            dtype=object,
        ),
        close_1h=np.array([[100.0], [101.0], [102.0], [103.0]]),
        close_4h=np.array([[100.0], [101.0], [102.0], [103.0]]),
        close_1d=np.array([[100.0], [101.0], [102.0], [103.0]]),
        quote_volume_1d=np.ones((4, 1)),
        complete_1h=np.ones((4, 1), dtype=bool),
        complete_4h=np.ones((4, 1), dtype=bool),
        complete_1d=np.ones((4, 1), dtype=bool),
        quality_1h_24h=np.ones((4, 1), dtype=bool),
        quality_4h_7d=np.ones((4, 1), dtype=bool),
        quality_1d_30d=np.ones((4, 1), dtype=bool),
    )
    signals = SignalArrays(score=np.array([[1.0], [1.0]]), tradable_mask=np.ones((2, 1), dtype=bool))
    signal_shapes = []

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.prepare_from_canonical",
        lambda *_args, **_kwargs: (arrays, ["manifest.json"]),
    )

    def fake_signals(filtered, _config):
        signal_shapes.append(filtered.close_1d.shape)
        return signals

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.compute_momentum_signals",
        fake_signals,
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.run_backtest",
        lambda *_args, **_kwargs: BacktestResult(
            equity=np.array([1.0, 1.01]),
            period_returns=np.array([0.01]),
            weights=np.array([[1.0], [1.0]]),
            turnover=np.array([1.0, 0.0]),
            costs=np.array([0.0, 0.0]),
        ),
    )
    monkeypatch.setattr("xsignal.strategies.momentum_rotation_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        [
            "scan",
            "--root",
            str(tmp_path),
            "--scan-id",
            "holdout-scan",
            "--top-n",
            "1",
            "--holdout-days",
            "1",
        ]
    )

    manifest_path = (
        tmp_path
        / "strategies"
        / "momentum_rotation_v1"
        / "scans"
        / "holdout-scan"
        / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text())

    assert exit_code == 0
    assert signal_shapes == [(2, 1)]
    assert manifest["data_split"] == {
        "holdout_days": 1,
        "research_start": "2026-01-01T00:00:00Z",
        "research_end": "2026-01-02T00:00:00Z",
        "holdout_start": "2026-01-03T00:00:00Z",
        "holdout_end": "2026-01-04T00:00:00Z",
    }


def test_cli_scan_passes_no_cache_to_canonical_preparation(tmp_path, monkeypatch):
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
    calls = []

    def fake_prepare(root, config, *, offline, use_cache):
        calls.append((root, config, offline, use_cache))
        return arrays, ["manifest.json"]

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.prepare_from_canonical",
        fake_prepare,
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.compute_momentum_signals",
        lambda *_args, **_kwargs: signals,
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.run_backtest",
        lambda *_args, **_kwargs: BacktestResult(
            equity=np.array([1.0, 1.01]),
            period_returns=np.array([0.01]),
            weights=np.array([[1.0], [1.0]]),
            turnover=np.array([1.0, 0.0]),
            costs=np.array([0.0, 0.0]),
        ),
    )
    monkeypatch.setattr("xsignal.strategies.momentum_rotation_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        [
            "scan",
            "--root",
            str(tmp_path),
            "--scan-id",
            "scan-nocache",
            "--top-n",
            "1",
            "--no-cache",
            "--holdout-days",
            "0",
        ]
    )

    assert exit_code == 0
    assert calls[0][3] is False


def test_cli_select_writes_selection_and_holdout_run_command(tmp_path):
    scan_dir = tmp_path / "strategies" / "momentum_rotation_v1" / "scans" / "scan123"
    scan_dir.mkdir(parents=True)
    (scan_dir / "manifest.json").write_text(
        json.dumps(
            {
                "scan_id": "scan123",
                "data_split": {
                    "holdout_days": 180,
                    "research_start": "2020-01-02T00:00:00Z",
                    "research_end": "2025-11-08T00:00:00Z",
                    "holdout_start": "2025-11-09T00:00:00Z",
                    "holdout_end": "2026-05-08T00:00:00Z",
                },
            }
        )
    )
    with (scan_dir / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "scan_id",
                "config_hash",
                "top_n",
                "fee_bps",
                "slippage_bps",
                "min_rolling_7d_quote_volume",
                "final_equity",
                "total_return",
                "max_drawdown",
                "missing_weighted_return_count",
                "missing_weighted_return_weight",
            ],
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "scan_id": "scan123",
                    "config_hash": "high-return-high-risk",
                    "top_n": 5,
                    "fee_bps": 5,
                    "slippage_bps": 5,
                    "min_rolling_7d_quote_volume": 0,
                    "final_equity": 3.0,
                    "total_return": 2.0,
                    "max_drawdown": 0.9,
                    "missing_weighted_return_count": 0,
                    "missing_weighted_return_weight": 0.0,
                },
                {
                    "scan_id": "scan123",
                    "config_hash": "balanced",
                    "top_n": 10,
                    "fee_bps": 5,
                    "slippage_bps": 5,
                    "min_rolling_7d_quote_volume": 1000,
                    "final_equity": 2.0,
                    "total_return": 1.0,
                    "max_drawdown": 0.1,
                    "missing_weighted_return_count": 0,
                    "missing_weighted_return_weight": 0.0,
                },
            ]
        )

    exit_code = main(
        [
            "select",
            "--root",
            str(tmp_path),
            "--scan-id",
            "scan123",
            "--selection-id",
            "pick1",
            "--drawdown-penalty",
            "2.0",
        ]
    )

    selection_path = scan_dir / "selections" / "pick1.json"
    selection = json.loads(selection_path.read_text())

    assert exit_code == 0
    assert selection["selected"]["config_hash"] == "balanced"
    assert selection["selected"]["score"] == 0.8
    assert selection["holdout_run_command"] == (
        "xsignal-momentum-v1 run --root data --offline --run-id pick1-holdout "
        "--top-n 10 --fee-bps 5.0 --slippage-bps 5.0 "
        "--min-rolling-7d-quote-volume 1000.0 --start-date 2025-11-09"
    )


def _write_selection_scan(tmp_path, scan_id, rows):
    scan_dir = tmp_path / "strategies" / "momentum_rotation_v1" / "scans" / scan_id
    scan_dir.mkdir(parents=True)
    (scan_dir / "manifest.json").write_text(
        json.dumps(
            {
                "scan_id": scan_id,
                "data_split": {
                    "holdout_days": 180,
                    "research_start": "2020-01-02T00:00:00Z",
                    "research_end": "2025-11-08T00:00:00Z",
                    "holdout_start": "2025-11-09T00:00:00Z",
                    "holdout_end": "2026-05-08T00:00:00Z",
                },
            }
        )
    )
    with (scan_dir / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=[
                "scan_id",
                "config_hash",
                "top_n",
                "fee_bps",
                "slippage_bps",
                "min_rolling_7d_quote_volume",
                "final_equity",
                "total_return",
                "period_count",
                "max_drawdown",
                "missing_weighted_return_count",
                "missing_weighted_return_weight",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return scan_dir


def test_cli_select_applies_hard_filters_before_scoring(tmp_path):
    scan_dir = _write_selection_scan(
        tmp_path,
        "filter-scan",
        [
            {
                "scan_id": "filter-scan",
                "config_hash": "high-return-high-drawdown",
                "top_n": 5,
                "fee_bps": 5,
                "slippage_bps": 5,
                "min_rolling_7d_quote_volume": 0,
                "final_equity": 11.0,
                "total_return": 10.0,
                "period_count": 365,
                "max_drawdown": 0.8,
                "missing_weighted_return_count": 0,
                "missing_weighted_return_weight": 0.0,
            },
            {
                "scan_id": "filter-scan",
                "config_hash": "high-return-missing-prices",
                "top_n": 10,
                "fee_bps": 5,
                "slippage_bps": 5,
                "min_rolling_7d_quote_volume": 0,
                "final_equity": 9.0,
                "total_return": 8.0,
                "period_count": 365,
                "max_drawdown": 0.1,
                "missing_weighted_return_count": 2,
                "missing_weighted_return_weight": 0.3,
            },
            {
                "scan_id": "filter-scan",
                "config_hash": "high-return-too-short",
                "top_n": 15,
                "fee_bps": 5,
                "slippage_bps": 5,
                "min_rolling_7d_quote_volume": 0,
                "final_equity": 8.0,
                "total_return": 7.0,
                "period_count": 20,
                "max_drawdown": 0.1,
                "missing_weighted_return_count": 0,
                "missing_weighted_return_weight": 0.0,
            },
            {
                "scan_id": "filter-scan",
                "config_hash": "eligible-balanced",
                "top_n": 20,
                "fee_bps": 5,
                "slippage_bps": 5,
                "min_rolling_7d_quote_volume": 1000,
                "final_equity": 1.6,
                "total_return": 0.6,
                "period_count": 200,
                "max_drawdown": 0.15,
                "missing_weighted_return_count": 1,
                "missing_weighted_return_weight": 0.05,
            },
        ],
    )

    exit_code = main(
        [
            "select",
            "--root",
            str(tmp_path),
            "--scan-id",
            "filter-scan",
            "--selection-id",
            "filtered-pick",
            "--max-drawdown-lte",
            "0.2",
            "--missing-weight-lte",
            "0.1",
            "--min-periods",
            "100",
        ]
    )

    selection = json.loads((scan_dir / "selections" / "filtered-pick.json").read_text())

    assert exit_code == 0
    assert selection["candidate_count"] == 4
    assert selection["eligible_count"] == 1
    assert selection["filters"] == {
        "max_drawdown_lte": 0.2,
        "missing_weight_lte": 0.1,
        "min_periods": 100,
    }
    assert selection["selected"]["config_hash"] == "eligible-balanced"


def test_cli_select_raises_when_hard_filters_remove_all_rows(tmp_path):
    _write_selection_scan(
        tmp_path,
        "empty-filter-scan",
        [
            {
                "scan_id": "empty-filter-scan",
                "config_hash": "too-deep",
                "top_n": 5,
                "fee_bps": 5,
                "slippage_bps": 5,
                "min_rolling_7d_quote_volume": 0,
                "final_equity": 2.0,
                "total_return": 1.0,
                "period_count": 365,
                "max_drawdown": 0.8,
                "missing_weighted_return_count": 0,
                "missing_weighted_return_weight": 0.0,
            }
        ],
    )

    with pytest.raises(ValueError, match="no scan rows passed selection filters"):
        main(
            [
                "select",
                "--root",
                str(tmp_path),
                "--scan-id",
                "empty-filter-scan",
                "--selection-id",
                "empty-pick",
                "--max-drawdown-lte",
                "0.2",
            ]
        )


def test_cli_select_does_not_pass_rows_with_missing_filter_metrics(tmp_path):
    _write_selection_scan(
        tmp_path,
        "missing-filter-metrics-scan",
        [
            {
                "scan_id": "missing-filter-metrics-scan",
                "config_hash": "missing-safety-metrics",
                "top_n": 5,
                "fee_bps": 5,
                "slippage_bps": 5,
                "min_rolling_7d_quote_volume": 0,
                "final_equity": 2.0,
                "total_return": 1.0,
                "period_count": 365,
                "max_drawdown": "",
                "missing_weighted_return_count": 0,
                "missing_weighted_return_weight": "",
            }
        ],
    )

    with pytest.raises(ValueError, match="no scan rows passed selection filters"):
        main(
            [
                "select",
                "--root",
                str(tmp_path),
                "--scan-id",
                "missing-filter-metrics-scan",
                "--selection-id",
                "missing-filter-metrics-pick",
                "--max-drawdown-lte",
                "0.2",
                "--missing-weight-lte",
                "0.1",
            ]
        )


def test_cli_prepare_from_canonical_uses_prepared_cache(tmp_path, monkeypatch):
    from xsignal.strategies.momentum_rotation_v1.cli import prepare_from_canonical
    from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
    from xsignal.strategies.momentum_rotation_v1.data import StrategyCanonicalInputs

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
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"row_count": 2}, sort_keys=True))
    cache_dir = (
        tmp_path
        / "strategies"
        / "momentum_rotation_v1"
        / "cache"
        / "prepared"
        / "cached-key"
    )
    save_prepared_arrays(cache_dir, arrays)
    (cache_dir / "cache_manifest.json").write_text(
        json.dumps(
            {
                "cache_key": "cached-key",
                "cache_version": "momentum-rotation-prepared-v3",
                "canonical_manifests": [str(manifest_path)],
            }
        )
    )

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli._prepared_cache_key",
        lambda *_args, **_kwargs: "cached-key",
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.collect_strategy_inputs",
        lambda *_args, **_kwargs: StrategyCanonicalInputs(
            bars_1h=None,
            bars_4h=None,
            bars_1d=None,
            manifest_paths=(manifest_path,),
        ),
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.prepare_daily_arrays",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )

    loaded, manifests = prepare_from_canonical(tmp_path, MomentumRotationConfig())

    assert loaded.symbols == arrays.symbols
    assert manifests == [str(manifest_path)]


def test_cli_prepare_from_canonical_can_disable_prepared_cache(tmp_path, monkeypatch):
    from xsignal.strategies.momentum_rotation_v1.cli import prepare_from_canonical
    from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
    from xsignal.strategies.momentum_rotation_v1.data import StrategyCanonicalInputs

    arrays = PreparedArrays(
        symbols=("BTCUSDT",),
        rebalance_times=np.array(["2026-01-02"], dtype=object),
        close_1h=np.array([[100.0]]),
        close_4h=np.array([[100.0]]),
        close_1d=np.array([[100.0]]),
        quote_volume_1d=np.ones((1, 1)),
        complete_1h=np.ones((1, 1), dtype=bool),
        complete_4h=np.ones((1, 1), dtype=bool),
        complete_1d=np.ones((1, 1), dtype=bool),
        quality_1h_24h=np.ones((1, 1), dtype=bool),
        quality_4h_7d=np.ones((1, 1), dtype=bool),
        quality_1d_30d=np.ones((1, 1), dtype=bool),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.collect_strategy_inputs",
        lambda *_args, **_kwargs: StrategyCanonicalInputs(
            bars_1h=object(),
            bars_4h=object(),
            bars_1d=object(),
            manifest_paths=(manifest_path,),
        ),
    )
    calls = []

    def fake_prepare_daily_arrays(**_kwargs):
        calls.append(True)
        return arrays

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.prepare_daily_arrays",
        fake_prepare_daily_arrays,
    )

    loaded, manifests = prepare_from_canonical(
        tmp_path,
        MomentumRotationConfig(),
        use_cache=False,
    )

    assert loaded is arrays
    assert manifests == [str(manifest_path)]
    assert calls == [True]
