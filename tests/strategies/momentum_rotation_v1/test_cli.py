from __future__ import annotations

import csv
import json
from datetime import date

import numpy as np

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
