from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pyarrow.parquet as pq

from xsignal.strategies.volume_price_efficiency_v1.cli import main
from xsignal.strategies.volume_price_efficiency_v1.data import (
    CanonicalOhlcvTable,
    OhlcvArrays,
)
from xsignal.strategies.volume_price_efficiency_v1.features import FeatureArrays


def _arrays(row_count: int = 4) -> OhlcvArrays:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    values = np.arange(row_count, dtype=np.float64).reshape(row_count, 1) + 100.0
    return OhlcvArrays(
        symbols=("BTCUSDT",),
        open_times=np.array([start + timedelta(hours=4 * i) for i in range(row_count)], dtype=object),
        open=values.copy(),
        high=values.copy() + 2.0,
        low=values.copy() - 2.0,
        close=values.copy() + 1.0,
        quote_volume=values.copy() * 1000.0,
        quality=np.ones((row_count, 1), dtype=bool),
    )


def _features(row_count: int = 4) -> FeatureArrays:
    values = np.arange(row_count, dtype=np.float64).reshape(row_count, 1) + 1.0
    signal = np.zeros((row_count, 1), dtype=bool)
    signal[0, 0] = True
    return FeatureArrays(
        true_range=values.copy(),
        atr=np.full((row_count, 1), 2.0),
        move_unit=values.copy(),
        volume_baseline=values.copy(),
        volume_unit=values.copy(),
        efficiency=values.copy(),
        efficiency_threshold=values.copy() - 1.0,
        close_position=np.full((row_count, 1), 0.8),
        body_ratio=np.full((row_count, 1), 0.5),
        signal=signal,
    )


def test_cli_trail_runs_on_reserved_holdout_and_writes_artifacts(tmp_path, monkeypatch):
    all_arrays = _arrays(8)
    holdout_arrays = _arrays(5)
    captured = {"split": [], "features": [], "simulate": []}

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("4h", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
            (tmp_path / "manifest.json",),
        ),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.prepare_ohlcv_arrays",
        lambda _table: all_arrays,
    )

    def fake_split(arrays, *, holdout_days):
        captured["split"].append((arrays, holdout_days))
        return (
            _arrays(3),
            holdout_arrays,
            {
                "holdout_days": holdout_days,
                "research_start": "2026-01-01T00:00:00Z",
                "research_end": "2026-01-01T12:00:00Z",
                "holdout_start": "2026-01-02T00:00:00Z",
                "holdout_end": "2026-01-02T12:00:00Z",
            },
        )

    def fake_compute_features(arrays, config):
        captured["features"].append((arrays, config))
        return _features(arrays.open.shape[0])

    def fake_simulate(arrays, features, config, *, atr_multiplier):
        captured["simulate"].append((arrays, features, config, atr_multiplier))
        from xsignal.strategies.volume_price_efficiency_v1.trailing import TrailingStopResult

        return TrailingStopResult(
            trades=[],
            equity=np.ones(arrays.open.shape[0], dtype=np.float64),
            period_returns=np.zeros(arrays.open.shape[0] - 1, dtype=np.float64),
            positions=np.zeros(arrays.open.shape, dtype=bool),
            stop_prices=np.full(arrays.open.shape, np.nan),
        )

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.split_research_and_holdout",
        fake_split,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.holdout_mask_for_open_times",
        lambda _open_times, *, holdout_days: np.array(
            [False, False, False, True, True, True, True, True],
            dtype=bool,
        ),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.compute_features",
        fake_compute_features,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.simulate_trailing_stop",
        fake_simulate,
    )
    monkeypatch.setattr("xsignal.strategies.volume_price_efficiency_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        [
            "trail",
            "--root",
            str(tmp_path),
            "--run-id",
            "trailrun",
            "--offline",
            "--holdout-days",
            "7",
            "--atr-multiplier",
            "2.0",
        ]
    )

    run_dir = (
        tmp_path
        / "strategies"
        / "volume_price_efficiency_v1"
        / "trailing_runs"
        / "trailrun"
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert exit_code == 0
    assert captured["split"] == [(all_arrays, 7)]
    assert captured["features"][0][0] is all_arrays
    assert captured["simulate"][0][0] is holdout_arrays
    assert captured["simulate"][0][1].signal.shape == holdout_arrays.open.shape
    assert captured["simulate"][0][3] == 2.0
    assert manifest["run_type"] == "trailing_stop_holdout"
    assert manifest["data_split"]["holdout_days"] == 7
    assert (run_dir / "trades.parquet").exists()
    assert (run_dir / "equity_curve.parquet").exists()
    assert (run_dir / "daily_positions.parquet").exists()


def test_cli_trail_refuses_non_offline_mode(tmp_path):
    try:
        main(["trail", "--root", str(tmp_path), "--run-id", "online"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("non-offline trail should fail")


def test_cli_trail_requires_fixed_two_atr_multiplier(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("4h", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
            (),
        ),
    )

    try:
        main(
            [
                "trail",
                "--root",
                str(tmp_path),
                "--run-id",
                "bad-multiplier",
                "--offline",
                "--atr-multiplier",
                "3.0",
            ]
        )
    except ValueError as exc:
        assert "atr_multiplier must stay fixed at 2.0" in str(exc)
    else:
        raise AssertionError("trail should reject non-2.0 ATR multiplier")


def test_cli_trail_scan_runs_on_research_window_and_writes_artifacts(tmp_path, monkeypatch):
    arrays = _arrays(8)
    research_arrays = _arrays(5)
    captured = {"split": [], "features": [], "simulate": []}

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("4h", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
            (tmp_path / "manifest.json",),
        ),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.prepare_ohlcv_arrays",
        lambda _table: arrays,
    )

    def fake_split(split_arrays, *, holdout_days):
        captured["split"].append((split_arrays, holdout_days))
        return (
            research_arrays,
            _arrays(3),
            {
                "holdout_days": holdout_days,
                "research_start": "2026-01-01T00:00:00Z",
                "research_end": "2026-01-01T16:00:00Z",
                "holdout_start": "2026-01-01T20:00:00Z",
                "holdout_end": "2026-01-02T04:00:00Z",
            },
        )

    def fake_compute_features(feature_arrays, config):
        captured["features"].append((feature_arrays, config))
        return _features(feature_arrays.open.shape[0])

    def fake_simulate(sim_arrays, features, config, *, atr_multiplier):
        captured["simulate"].append((sim_arrays, features, config, atr_multiplier))
        from xsignal.strategies.volume_price_efficiency_v1.trailing import TrailingStopResult

        return TrailingStopResult(
            trades=[
                {
                    "symbol": "BTCUSDT",
                    "realized_return": 0.04,
                    "net_realized_return": 0.038,
                    "holding_bars": 2,
                    "ignored_signal_count": 0,
                }
            ],
            equity=np.array([1.0, 1.04], dtype=np.float64),
            period_returns=np.array([0.04], dtype=np.float64),
            positions=np.zeros((2, 1), dtype=bool),
            stop_prices=np.full((2, 1), np.nan),
        )

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.split_research_and_holdout",
        fake_split,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.compute_features",
        fake_compute_features,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.simulate_trailing_stop",
        fake_simulate,
    )
    monkeypatch.setattr("xsignal.strategies.volume_price_efficiency_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        [
            "trail-scan",
            "--root",
            str(tmp_path),
            "--scan-id",
            "trailscan",
            "--offline",
            "--holdout-days",
            "7",
            "--efficiency-percentile",
            "0.9,0.95",
            "--min-move-unit",
            "1.2",
            "--min-volume-unit",
            "1.5",
            "--min-close-position",
            "0.94",
            "--min-body-ratio",
            "0.85",
            "--top-k",
            "1",
            "--min-trades",
            "1",
        ]
    )

    scan_dir = (
        tmp_path
        / "strategies"
        / "volume_price_efficiency_v1"
        / "trailing_scans"
        / "trailscan"
    )
    manifest = json.loads((scan_dir / "manifest.json").read_text())
    top_configs = json.loads((scan_dir / "top_configs.json").read_text())
    assert exit_code == 0
    assert captured["split"] == [(arrays, 7)]
    assert [config.efficiency_percentile for _arrays, config in captured["features"]] == [0.9, 0.95]
    assert [call[0] for call in captured["simulate"]] == [research_arrays, research_arrays]
    assert manifest["run_type"] == "trailing_stop_research_scan"
    assert manifest["data_split"]["holdout_days"] == 7
    assert manifest["combination_count"] == 2
    assert len(top_configs) == 1
    assert (scan_dir / "summary.csv").exists()


def test_cli_trail_scan_refuses_non_offline_mode(tmp_path):
    try:
        main(["trail-scan", "--root", str(tmp_path), "--scan-id", "online"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("non-offline trail-scan should fail")


def test_cli_trail_diagnose_writes_research_and_holdout_diagnostics(tmp_path, monkeypatch):
    all_arrays = _arrays(8)
    research_arrays = _arrays(5)
    holdout_arrays = _arrays(3)
    captured = {"split": [], "simulate": []}

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("4h", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
            (tmp_path / "manifest.json",),
        ),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.prepare_ohlcv_arrays",
        lambda _table: all_arrays,
    )

    def fake_split(split_arrays, *, holdout_days):
        captured["split"].append((split_arrays, holdout_days))
        return (
            research_arrays,
            holdout_arrays,
            {
                "holdout_days": holdout_days,
                "research_start": "2026-01-01T00:00:00Z",
                "research_end": "2026-01-01T16:00:00Z",
                "holdout_start": "2026-01-01T20:00:00Z",
                "holdout_end": "2026-01-02T04:00:00Z",
            },
        )

    def fake_simulate(sim_arrays, features, config, *, atr_multiplier):
        captured["simulate"].append((sim_arrays, features, config, atr_multiplier))
        from xsignal.strategies.volume_price_efficiency_v1.trailing import TrailingStopResult

        return TrailingStopResult(
            trades=[
                {
                    "symbol": "BTCUSDT",
                    "signal_open_time": sim_arrays.open_times[0].isoformat(),
                    "realized_return": 0.04,
                    "net_realized_return": 0.038,
                    "holding_bars": 2,
                    "ignored_signal_count": 0,
                    "move_unit": 1.2,
                }
            ],
            equity=np.array([1.0, 1.04], dtype=np.float64),
            period_returns=np.array([0.04], dtype=np.float64),
            positions=np.zeros((2, 1), dtype=bool),
            stop_prices=np.full((2, 1), np.nan),
        )

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.split_research_and_holdout",
        fake_split,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.holdout_mask_for_open_times",
        lambda _open_times, *, holdout_days: np.array(
            [False, False, False, False, False, True, True, True],
            dtype=bool,
        ),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.compute_features",
        lambda feature_arrays, _config: _features(feature_arrays.open.shape[0]),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.simulate_trailing_stop",
        fake_simulate,
    )
    monkeypatch.setattr("xsignal.strategies.volume_price_efficiency_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        [
            "trail-diagnose",
            "--root",
            str(tmp_path),
            "--diagnostic-id",
            "diag",
            "--offline",
            "--holdout-days",
            "7",
            "--lookback-bars",
            "2",
        ]
    )

    diagnostic_dir = (
        tmp_path
        / "strategies"
        / "volume_price_efficiency_v1"
        / "trailing_diagnostics"
        / "diag"
    )
    manifest = json.loads((diagnostic_dir / "manifest.json").read_text())
    assert exit_code == 0
    assert captured["split"] == [(all_arrays, 7)]
    assert [call[0] for call in captured["simulate"]] == [research_arrays, holdout_arrays]
    assert manifest["run_type"] == "trailing_stop_diagnostics"
    assert manifest["lookback_bars"] == 2
    assert (diagnostic_dir / "time_summary.parquet").exists()
    assert (diagnostic_dir / "bucket_summary.parquet").exists()


def test_cli_trail_diagnose_refuses_non_offline_mode(tmp_path):
    try:
        main(["trail-diagnose", "--root", str(tmp_path), "--diagnostic-id", "online"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("non-offline trail-diagnose should fail")


def test_cli_trail_walk_forward_uses_research_only_prior_train_windows(
    tmp_path,
    monkeypatch,
):
    all_arrays = _arrays(14)
    research_arrays = _arrays(12)
    holdout_arrays = _arrays(2)
    captured = {"split": [], "feature_shapes": [], "simulate_shapes": []}

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("4h", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
            (tmp_path / "manifest.json",),
        ),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.prepare_ohlcv_arrays",
        lambda _table: all_arrays,
    )

    def fake_split(split_arrays, *, holdout_days):
        captured["split"].append((split_arrays, holdout_days))
        return (
            research_arrays,
            holdout_arrays,
            {
                "holdout_days": holdout_days,
                "research_start": "2026-01-01T00:00:00Z",
                "research_end": "2026-01-02T20:00:00Z",
                "holdout_start": "2026-01-03T00:00:00Z",
                "holdout_end": "2026-01-03T04:00:00Z",
            },
        )

    def fake_compute_features(feature_arrays, _config):
        captured["feature_shapes"].append(feature_arrays.open.shape)
        return _features(feature_arrays.open.shape[0])

    def fake_simulate(sim_arrays, features, config, *, atr_multiplier):
        captured["simulate_shapes"].append(sim_arrays.open.shape)
        from xsignal.strategies.volume_price_efficiency_v1.trailing import TrailingStopResult

        trade_count = 2 if config.efficiency_percentile == 0.9 else 1
        return TrailingStopResult(
            trades=[
                {
                    "symbol": "BTCUSDT",
                    "signal_open_time": sim_arrays.open_times[0].isoformat(),
                    "realized_return": 0.04,
                    "net_realized_return": 0.038,
                    "holding_bars": 2,
                    "ignored_signal_count": 0,
                    "move_unit": 1.2,
                }
                for _ in range(trade_count)
            ],
            equity=np.array([1.0, 1.04 + trade_count / 100.0], dtype=np.float64),
            period_returns=np.array([0.04], dtype=np.float64),
            positions=np.zeros(features.signal.shape, dtype=bool),
            stop_prices=np.full(features.signal.shape, np.nan),
        )

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.split_research_and_holdout",
        fake_split,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.compute_features",
        fake_compute_features,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.simulate_trailing_stop",
        fake_simulate,
    )
    monkeypatch.setattr("xsignal.strategies.volume_price_efficiency_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        [
            "trail-walk-forward",
            "--root",
            str(tmp_path),
            "--walk-forward-id",
            "wf",
            "--offline",
            "--holdout-days",
            "7",
            "--train-days",
            "1",
            "--test-days",
            "1",
            "--step-days",
            "1",
            "--efficiency-percentile",
            "0.9,0.95",
            "--min-move-unit",
            "1.2",
            "--min-volume-unit",
            "1.0",
            "--min-close-position",
            "0.94",
            "--min-body-ratio",
            "0.85",
            "--min-trades",
            "1",
            "--top-k",
            "1",
        ]
    )

    output_dir = (
        tmp_path
        / "strategies"
        / "volume_price_efficiency_v1"
        / "trailing_walk_forwards"
        / "wf"
    )
    manifest = json.loads((output_dir / "manifest.json").read_text())
    fold_summary = pq.read_table(output_dir / "fold_summary.parquet").to_pylist()
    selection_summary = pq.read_table(output_dir / "selection_summary.parquet").to_pylist()
    assert exit_code == 0
    assert captured["split"] == [(all_arrays, 7)]
    assert captured["feature_shapes"] == [((12, 1)), ((12, 1))]
    assert captured["simulate_shapes"] == [((6, 1)), ((6, 1)), ((6, 1))]
    assert manifest["run_type"] == "trailing_stop_research_walk_forward"
    assert manifest["data_scope"] == "research_only"
    assert manifest["data_split"]["holdout_days"] == 7
    assert manifest["train_days"] == 1
    assert manifest["test_days"] == 1
    assert manifest["fold_count"] == 1
    assert len(fold_summary) == 1
    assert len(selection_summary) == 2
    assert fold_summary[0]["selected_config_hash"] == selection_summary[0]["config_hash"]


def test_cli_trail_walk_forward_refuses_non_offline_mode(tmp_path):
    try:
        main(["trail-walk-forward", "--root", str(tmp_path), "--walk-forward-id", "online"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("non-offline trail-walk-forward should fail")


def test_cli_trail_regime_scan_runs_on_research_only_and_writes_artifacts(
    tmp_path,
    monkeypatch,
):
    all_arrays = _arrays(8)
    research_arrays = _arrays(5)
    holdout_arrays = _arrays(3)
    captured = {"split": [], "simulate_shapes": [], "signal_counts": []}

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("4h", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
            (tmp_path / "manifest.json",),
        ),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.prepare_ohlcv_arrays",
        lambda _table: all_arrays,
    )

    def fake_split(split_arrays, *, holdout_days):
        captured["split"].append((split_arrays, holdout_days))
        return (
            research_arrays,
            holdout_arrays,
            {
                "holdout_days": holdout_days,
                "research_start": "2026-01-01T00:00:00Z",
                "research_end": "2026-01-01T16:00:00Z",
                "holdout_start": "2026-01-01T20:00:00Z",
                "holdout_end": "2026-01-02T04:00:00Z",
            },
        )

    def fake_simulate(sim_arrays, features, config, *, atr_multiplier):
        captured["simulate_shapes"].append(sim_arrays.open.shape)
        signal_count = int(np.count_nonzero(features.signal))
        captured["signal_counts"].append(signal_count)
        from xsignal.strategies.volume_price_efficiency_v1.trailing import TrailingStopResult

        return TrailingStopResult(
            trades=[
                {
                    "symbol": "BTCUSDT",
                    "realized_return": 0.04,
                    "net_realized_return": 0.038,
                    "holding_bars": 2,
                    "ignored_signal_count": 0,
                }
                for _ in range(signal_count)
            ],
            equity=np.ones(sim_arrays.open.shape[0], dtype=np.float64),
            period_returns=np.zeros(max(sim_arrays.open.shape[0] - 1, 0), dtype=np.float64),
            positions=np.zeros(features.signal.shape, dtype=bool),
            stop_prices=np.full(features.signal.shape, np.nan),
        )

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.split_research_and_holdout",
        fake_split,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.compute_features",
        lambda feature_arrays, _config: _features(feature_arrays.open.shape[0]),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.simulate_trailing_stop",
        fake_simulate,
    )
    monkeypatch.setattr("xsignal.strategies.volume_price_efficiency_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        [
            "trail-regime-scan",
            "--root",
            str(tmp_path),
            "--regime-scan-id",
            "regime",
            "--offline",
            "--holdout-days",
            "7",
            "--feature-name",
            "move_unit",
            "--quantile",
            "0.5",
            "--min-trades",
            "1",
            "--top-k",
            "1",
        ]
    )

    output_dir = (
        tmp_path
        / "strategies"
        / "volume_price_efficiency_v1"
        / "trailing_regime_scans"
        / "regime"
    )
    manifest = json.loads((output_dir / "manifest.json").read_text())
    rows = pq.read_table(output_dir / "summary.parquet").to_pylist()
    top_filters = json.loads((output_dir / "top_filters.json").read_text())
    assert exit_code == 0
    assert captured["split"] == [(all_arrays, 7)]
    assert captured["simulate_shapes"] == [(research_arrays.open.shape)] * 3
    assert captured["signal_counts"] == [4, 2, 2]
    assert manifest["run_type"] == "trailing_stop_research_regime_scan"
    assert manifest["data_scope"] == "research_only"
    assert manifest["threshold_scope"] == "full_research_signal_distribution_diagnostic_only"
    assert manifest["data_split"]["holdout_days"] == 7
    assert manifest["feature_names"] == ["move_unit"]
    assert manifest["quantiles"] == [0.5]
    assert [row["rule_id"] for row in rows] == [
        "unfiltered",
        "move_unit_gte_p50",
        "move_unit_lt_p50",
    ]
    assert [row["rule_id"] for row in top_filters] == ["move_unit_gte_p50"]


def test_cli_trail_regime_scan_refuses_non_offline_mode(tmp_path):
    try:
        main(["trail-regime-scan", "--root", str(tmp_path), "--regime-scan-id", "online"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("non-offline trail-regime-scan should fail")
