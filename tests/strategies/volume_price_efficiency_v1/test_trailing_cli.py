from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pyarrow as pa
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
            CanonicalOhlcvTable("1d", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
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


def test_cli_trail_accepts_explicit_non_default_atr_multiplier(tmp_path, monkeypatch):
    all_arrays = _arrays(4)
    holdout_arrays = _arrays(2)
    captured = {"simulate": []}

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("1d", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
            (tmp_path / "manifest.json",),
        ),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.prepare_ohlcv_arrays",
        lambda _table: all_arrays,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.split_research_and_holdout",
        lambda _split_arrays, *, holdout_days: (
            _arrays(2),
            holdout_arrays,
            {
                "holdout_days": holdout_days,
                "research_start": "2026-01-01T00:00:00Z",
                "research_end": "2026-01-01T04:00:00Z",
                "holdout_start": "2026-01-01T08:00:00Z",
                "holdout_end": "2026-01-01T12:00:00Z",
            },
        ),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.holdout_mask_for_open_times",
        lambda _open_times, *, holdout_days: np.array([False, False, True, True], dtype=bool),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.compute_features",
        lambda feature_arrays, _config: _features(feature_arrays.open.shape[0]),
    )

    def fake_simulate(sim_arrays, features, config, *, atr_multiplier):
        captured["simulate"].append(atr_multiplier)
        from xsignal.strategies.volume_price_efficiency_v1.trailing import TrailingStopResult

        return TrailingStopResult(
            trades=[],
            equity=np.ones(sim_arrays.open.shape[0], dtype=np.float64),
            period_returns=np.zeros(max(sim_arrays.open.shape[0] - 1, 0), dtype=np.float64),
            positions=np.zeros(features.signal.shape, dtype=bool),
            stop_prices=np.full(features.signal.shape, np.nan),
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
            "custom-multiplier",
            "--offline",
            "--atr-multiplier",
            "3.0",
        ]
    )

    run_dir = (
        tmp_path
        / "strategies"
        / "volume_price_efficiency_v1"
        / "trailing_runs"
        / "custom-multiplier"
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert exit_code == 0
    assert captured["simulate"] == [3.0]
    assert manifest["atr_multiplier"] == 3.0


def test_cli_trail_regime_holdout_trains_rule_on_research_then_runs_holdout(
    tmp_path,
    monkeypatch,
):
    all_arrays = _arrays(8)
    research_arrays = _arrays(4)
    holdout_arrays = _arrays(4)
    captured = {"split": [], "features": [], "simulate_signal_indices": []}

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("1d", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
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
            research_arrays,
            holdout_arrays,
            {
                "holdout_days": holdout_days,
                "research_start": "2026-01-01T00:00:00Z",
                "research_end": "2026-01-01T12:00:00Z",
                "holdout_start": "2026-01-01T16:00:00Z",
                "holdout_end": "2026-01-02T04:00:00Z",
            },
        )

    signal = np.ones((8, 1), dtype=bool)
    move = np.array([[0.0], [1.0], [2.0], [3.0], [0.0], [1.0], [2.0], [3.0]])

    def fake_features(feature_arrays, config):
        captured["features"].append((feature_arrays, config))
        row_count = feature_arrays.open.shape[0]
        features = _features(row_count)
        features = FeatureArrays(
            true_range=features.true_range,
            atr=features.atr,
            move_unit=move[:row_count].copy(),
            volume_baseline=features.volume_baseline,
            volume_unit=features.volume_unit,
            efficiency=features.efficiency,
            efficiency_threshold=features.efficiency_threshold,
            close_position=features.close_position,
            body_ratio=features.body_ratio,
            signal=signal[:row_count].copy(),
        )
        return features

    def fake_simulate(sim_arrays, features, config, *, atr_multiplier):
        signal_indices = tuple(int(index) for index in np.flatnonzero(features.signal[:, 0]))
        captured["simulate_signal_indices"].append(signal_indices)
        from xsignal.strategies.volume_price_efficiency_v1.trailing import TrailingStopResult

        final_equity = 1.2 if signal_indices == (2, 3) else 1.0 + len(signal_indices) / 100.0
        equity = np.linspace(1.0, final_equity, sim_arrays.open.shape[0], dtype=np.float64)
        return TrailingStopResult(
            trades=[
                {
                    "symbol": "BTCUSDT",
                    "signal_open_time": sim_arrays.open_times[index].isoformat(),
                    "realized_return": 0.04,
                    "net_realized_return": 0.038,
                    "holding_bars": 2,
                    "ignored_signal_count": 0,
                }
                for index in signal_indices
            ],
            equity=equity,
            period_returns=equity[1:] / equity[:-1] - 1.0,
            positions=np.zeros(features.signal.shape, dtype=bool),
            stop_prices=np.full(features.signal.shape, np.nan),
        )

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.split_research_and_holdout",
        fake_split,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.holdout_mask_for_open_times",
        lambda _open_times, *, holdout_days: np.array(
            [False, False, False, False, True, True, True, True],
            dtype=bool,
        ),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.compute_features",
        fake_features,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.build_signal_mask",
        lambda arrays, _features, _config: signal[: arrays.open.shape[0]],
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.simulate_trailing_stop",
        fake_simulate,
    )
    monkeypatch.setattr("xsignal.strategies.volume_price_efficiency_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        [
            "trail-regime-holdout",
            "--root",
            str(tmp_path),
            "--run-id",
            "regime-holdout",
            "--offline",
            "--holdout-days",
            "7",
            "--atr-multiplier",
            "2.0",
            "--feature-name",
            "move_unit",
            "--quantile",
            "0.5",
            "--min-trades",
            "1",
        ]
    )

    run_dir = (
        tmp_path
        / "strategies"
        / "volume_price_efficiency_v1"
        / "trailing_runs"
        / "regime-holdout"
    )
    manifest = json.loads((run_dir / "manifest.json").read_text())
    selected_filter = json.loads((run_dir / "selected_filter.json").read_text())
    assert exit_code == 0
    assert captured["split"] == [(all_arrays, 7)]
    assert captured["features"][0][0] is research_arrays
    assert captured["features"][1][0] is all_arrays
    assert captured["simulate_signal_indices"][-1] == (2, 3)
    assert manifest["run_type"] == "trailing_stop_regime_holdout"
    assert manifest["selection_scope"] == "research_only"
    assert manifest["threshold_scope"] == "full_research_signal_distribution"
    assert manifest["selected_rule_id"] == "move_unit_gte_p50"
    assert selected_filter["rule_id"] == "move_unit_gte_p50"
    assert selected_filter["threshold"] == 1.5
    assert pq.read_table(run_dir / "selection_summary.parquet").num_rows == 2


def test_cli_trail_scan_runs_on_research_window_and_writes_artifacts(tmp_path, monkeypatch):
    arrays = _arrays(8)
    research_arrays = _arrays(5)
    captured = {"split": [], "features": [], "simulate": []}

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("1d", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
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
            "--atr-multiplier",
            "1.5,3.0",
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
    assert [call[0] for call in captured["simulate"]] == [
        research_arrays,
        research_arrays,
        research_arrays,
        research_arrays,
    ]
    assert [call[3] for call in captured["simulate"]] == [1.5, 3.0, 1.5, 3.0]
    assert manifest["run_type"] == "trailing_stop_research_scan"
    assert manifest["data_split"]["holdout_days"] == 7
    assert manifest["atr_multipliers"] == [1.5, 3.0]
    assert manifest["combination_count"] == 4
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
            CanonicalOhlcvTable("1d", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
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
    captured = {"split": [], "feature_shapes": [], "simulate_shapes": [], "atr_multipliers": []}

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("1d", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
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
        captured["atr_multipliers"].append(atr_multiplier)
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
            "--atr-multiplier",
            "1.5,3.0",
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
    assert captured["simulate_shapes"] == [((6, 1)), ((6, 1)), ((6, 1)), ((6, 1)), ((6, 1))]
    assert captured["atr_multipliers"] == [1.5, 3.0, 1.5, 3.0, 1.5]
    assert manifest["run_type"] == "trailing_stop_research_walk_forward"
    assert manifest["data_scope"] == "research_only"
    assert manifest["data_split"]["holdout_days"] == 7
    assert manifest["atr_multipliers"] == [1.5, 3.0]
    assert manifest["train_days"] == 1
    assert manifest["test_days"] == 1
    assert manifest["fold_count"] == 1
    assert len(fold_summary) == 1
    assert len(selection_summary) == 4
    assert fold_summary[0]["atr_multiplier"] == 1.5
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
    captured = {"split": [], "simulate_shapes": [], "signal_counts": [], "atr_multipliers": []}

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("1d", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
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
        captured["atr_multipliers"].append(atr_multiplier)
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
            "--atr-multiplier",
            "1.5,3.0",
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
    assert captured["simulate_shapes"] == [(research_arrays.open.shape)] * 6
    assert captured["signal_counts"] == [4, 2, 2, 4, 2, 2]
    assert captured["atr_multipliers"] == [1.5, 1.5, 1.5, 3.0, 3.0, 3.0]
    assert manifest["run_type"] == "trailing_stop_research_regime_scan"
    assert manifest["data_scope"] == "research_only"
    assert manifest["threshold_scope"] == "full_research_signal_distribution_diagnostic_only"
    assert manifest["data_split"]["holdout_days"] == 7
    assert manifest["atr_multipliers"] == [1.5, 3.0]
    assert manifest["feature_names"] == ["move_unit"]
    assert manifest["quantiles"] == [0.5]
    assert [row["rule_id"] for row in rows] == [
        "unfiltered",
        "move_unit_gte_p50",
        "move_unit_lt_p50",
        "unfiltered",
        "move_unit_gte_p50",
        "move_unit_lt_p50",
    ]
    assert [row["atr_multiplier"] for row in rows] == [1.5, 1.5, 1.5, 3.0, 3.0, 3.0]
    assert [row["rule_id"] for row in top_filters] == ["move_unit_gte_p50"]


def test_cli_trail_regime_scan_refuses_non_offline_mode(tmp_path):
    try:
        main(["trail-regime-scan", "--root", str(tmp_path), "--regime-scan-id", "online"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("non-offline trail-regime-scan should fail")


def test_cli_trail_regime_walk_forward_uses_train_fold_thresholds(
    tmp_path,
    monkeypatch,
):
    all_arrays = _arrays(14)
    research_arrays = _arrays(12)
    holdout_arrays = _arrays(2)
    captured = {"split": [], "simulate_signal_counts": [], "atr_multipliers": []}

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("1d", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
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

    def fake_features(row_count, signal):
        values = np.zeros((row_count, 1), dtype=np.float64)
        values[:12, 0] = [0.0, 1.0, 2.0, 5.0, 6.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0, 0.0]
        return FeatureArrays(
            true_range=np.ones((row_count, 1), dtype=np.float64),
            atr=np.ones((row_count, 1), dtype=np.float64),
            move_unit=values,
            volume_baseline=np.ones((row_count, 1), dtype=np.float64),
            volume_unit=np.ones((row_count, 1), dtype=np.float64),
            efficiency=values,
            efficiency_threshold=np.zeros((row_count, 1), dtype=np.float64),
            close_position=np.full((row_count, 1), 0.9),
            body_ratio=np.full((row_count, 1), 0.9),
            signal=signal,
        )

    signal = np.zeros((12, 1), dtype=bool)
    signal[[1, 2, 3, 4, 7, 8, 9, 10], 0] = True

    def fake_simulate(sim_arrays, features, config, *, atr_multiplier):
        signal_count = int(np.count_nonzero(features.signal))
        captured["simulate_signal_counts"].append(signal_count)
        captured["atr_multipliers"].append(atr_multiplier)
        from xsignal.strategies.volume_price_efficiency_v1.trailing import TrailingStopResult

        final_equity = 1.0 + signal_count / 100.0
        signal_indices = [int(index) for index in np.flatnonzero(features.signal[:, 0])]
        return TrailingStopResult(
            trades=[
                {
                    "symbol": "BTCUSDT",
                    "signal_open_time": sim_arrays.open_times[index].isoformat(),
                    "realized_return": 0.04,
                    "net_realized_return": 0.038,
                    "holding_bars": 2,
                    "ignored_signal_count": 0,
                }
                for index in signal_indices
            ],
            equity=np.array([1.0, final_equity], dtype=np.float64),
            period_returns=np.array([final_equity - 1.0], dtype=np.float64),
            positions=np.zeros(features.signal.shape, dtype=bool),
            stop_prices=np.full(features.signal.shape, np.nan),
        )

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.split_research_and_holdout",
        fake_split,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.compute_features",
        lambda feature_arrays, _config: fake_features(feature_arrays.open.shape[0], signal),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.build_signal_mask",
        lambda _arrays, _features, _config: signal,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.simulate_trailing_stop",
        fake_simulate,
    )
    monkeypatch.setattr("xsignal.strategies.volume_price_efficiency_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        [
            "trail-regime-walk-forward",
            "--root",
            str(tmp_path),
            "--regime-walk-forward-id",
            "rwf",
            "--offline",
            "--holdout-days",
            "7",
            "--train-days",
            "1",
            "--test-days",
            "1",
            "--step-days",
            "1",
            "--feature-name",
            "move_unit",
            "--quantile",
            "0.5",
            "--atr-multiplier",
            "1.5,3.0",
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
        / "trailing_regime_walk_forwards"
        / "rwf"
    )
    manifest = json.loads((output_dir / "manifest.json").read_text())
    fold_rows = pq.read_table(output_dir / "fold_summary.parquet").to_pylist()
    selection_rows = pq.read_table(output_dir / "selection_summary.parquet").to_pylist()
    validation_trades = pq.read_table(output_dir / "validation_trades.parquet").to_pylist()
    assert exit_code == 0
    assert captured["split"] == [(all_arrays, 7)]
    assert captured["simulate_signal_counts"] == [2, 2, 2, 2, 4]
    assert captured["atr_multipliers"] == [1.5, 3.0, 1.5, 3.0, 1.5]
    assert manifest["run_type"] == "trailing_stop_research_regime_walk_forward"
    assert manifest["data_scope"] == "research_only"
    assert manifest["threshold_scope"] == "per_fold_train_signal_distribution"
    assert manifest["data_split"]["holdout_days"] == 7
    assert manifest["atr_multipliers"] == [1.5, 3.0]
    assert len(fold_rows) == 1
    assert fold_rows[0]["threshold"] == 3.5
    assert fold_rows[0]["atr_multiplier"] == 1.5
    assert fold_rows[0]["validation_filtered_signal_count"] == 4
    assert {row["threshold"] for row in selection_rows} == {3.5}
    assert {row["atr_multiplier"] for row in selection_rows} == {1.5, 3.0}
    assert len(validation_trades) == 4
    assert validation_trades[0]["fold_index"] == 0
    assert validation_trades[0]["rule_id"] == "move_unit_gte_p50"
    assert validation_trades[0]["threshold_source"] == "train_fold_signal_distribution"
    assert validation_trades[0]["atr_multiplier"] == 1.5
    assert validation_trades[0]["move_unit"] == 4.0
    assert manifest["outputs"]["validation_trades"].endswith("validation_trades.parquet")


def test_cli_trail_regime_walk_forward_can_select_by_train_stability(
    tmp_path,
    monkeypatch,
):
    all_arrays = _arrays(14)
    research_arrays = _arrays(12)
    holdout_arrays = _arrays(2)
    captured = {"split": [], "simulate_signal_counts": [], "stable_select": []}

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("1d", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
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

    values = np.zeros((12, 1), dtype=np.float64)
    values[:12, 0] = [0.0, 1.0, 2.0, 5.0, 6.0, 0.0, 0.0, 4.0, 4.0, 4.0, 4.0, 0.0]
    signal = np.zeros((12, 1), dtype=bool)
    signal[[1, 2, 3, 4, 7, 8, 9, 10], 0] = True

    def fake_features(row_count):
        return FeatureArrays(
            true_range=np.ones((row_count, 1), dtype=np.float64),
            atr=np.ones((row_count, 1), dtype=np.float64),
            move_unit=values[:row_count].copy(),
            volume_baseline=np.ones((row_count, 1), dtype=np.float64),
            volume_unit=np.ones((row_count, 1), dtype=np.float64),
            efficiency=values[:row_count].copy(),
            efficiency_threshold=np.zeros((row_count, 1), dtype=np.float64),
            close_position=np.full((row_count, 1), 0.9),
            body_ratio=np.full((row_count, 1), 0.9),
            signal=signal[:row_count].copy(),
        )

    def fake_simulate(sim_arrays, features, config, *, atr_multiplier):
        signal_count = int(np.count_nonzero(features.signal))
        captured["simulate_signal_counts"].append(signal_count)
        from xsignal.strategies.volume_price_efficiency_v1.trailing import TrailingStopResult

        final_equity = 1.0 + signal_count / 100.0
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
            equity=np.array([1.0, final_equity], dtype=np.float64),
            period_returns=np.array([final_equity - 1.0], dtype=np.float64),
            positions=np.zeros(features.signal.shape, dtype=bool),
            stop_prices=np.full(features.signal.shape, np.nan),
        )

    def fake_stable_select(rows, *, top_k, min_trades, stability_min_positive_splits):
        captured["stable_select"].append(
            {
                "top_k": top_k,
                "min_trades": min_trades,
                "stability_min_positive_splits": stability_min_positive_splits,
                "rule_ids": [row["rule_id"] for row in rows],
                "split_counts": [row.get("stability_split_count") for row in rows],
            }
        )
        return [rows[-1]]

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.split_research_and_holdout",
        fake_split,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.compute_features",
        lambda feature_arrays, _config: fake_features(feature_arrays.open.shape[0]),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.build_signal_mask",
        lambda _arrays, _features, _config: signal,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.simulate_trailing_stop",
        fake_simulate,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.select_stable_regime_filters",
        fake_stable_select,
        raising=False,
    )
    monkeypatch.setattr("xsignal.strategies.volume_price_efficiency_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        [
            "trail-regime-walk-forward",
            "--root",
            str(tmp_path),
            "--regime-walk-forward-id",
            "stable-rwf",
            "--offline",
            "--holdout-days",
            "7",
            "--train-days",
            "1",
            "--test-days",
            "1",
            "--step-days",
            "1",
            "--feature-name",
            "move_unit",
            "--quantile",
            "0.5",
            "--min-trades",
            "1",
            "--top-k",
            "1",
            "--stability-splits",
            "2",
            "--stability-min-trades",
            "1",
            "--stability-min-positive-splits",
            "2",
        ]
    )

    output_dir = (
        tmp_path
        / "strategies"
        / "volume_price_efficiency_v1"
        / "trailing_regime_walk_forwards"
        / "stable-rwf"
    )
    manifest = json.loads((output_dir / "manifest.json").read_text())
    fold_rows = pq.read_table(output_dir / "fold_summary.parquet").to_pylist()
    selection_rows = pq.read_table(output_dir / "selection_summary.parquet").to_pylist()
    assert exit_code == 0
    assert captured["split"] == [(all_arrays, 7)]
    assert captured["simulate_signal_counts"] == [2, 0, 2, 2, 2, 0, 0]
    assert captured["stable_select"][0]["rule_ids"] == [
        "move_unit_gte_p50",
        "move_unit_lt_p50",
    ]
    assert captured["stable_select"][0]["split_counts"] == [2, 2]
    assert captured["stable_select"][0]["stability_min_positive_splits"] == 2
    assert manifest["selection_method"] == "train_stability"
    assert manifest["stability_splits"] == 2
    assert manifest["stability_min_trades"] == 1
    assert manifest["stability_min_positive_splits"] == 2
    assert fold_rows[0]["selection_method"] == "train_stability"
    assert fold_rows[0]["rule_id"] == "move_unit_lt_p50"
    assert fold_rows[0]["validation_filtered_signal_count"] == 0
    assert {row["stability_split_count"] for row in selection_rows} == {2}


def test_cli_trail_regime_walk_forward_uses_causal_contraction_feature_thresholds(
    tmp_path,
    monkeypatch,
):
    all_arrays = _arrays(14)
    research_arrays = _arrays(12)
    holdout_arrays = _arrays(2)
    captured = {"split": [], "simulate_signal_counts": []}

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("1d", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
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

    signal = np.zeros((12, 1), dtype=bool)
    signal[[2, 3, 4, 7, 8, 9, 10], 0] = True
    atr = np.array(
        [[3.0], [1.0], [1.0], [3.0], [3.0], [3.0], [1.0], [1.5], [3.5], [14.0], [14.0], [14.0]],
        dtype=np.float64,
    )

    def fake_features(row_count):
        values = np.ones((row_count, 1), dtype=np.float64)
        return FeatureArrays(
            true_range=atr[:row_count].copy(),
            atr=atr[:row_count].copy(),
            move_unit=values,
            volume_baseline=values,
            volume_unit=values,
            efficiency=values,
            efficiency_threshold=np.zeros((row_count, 1), dtype=np.float64),
            close_position=np.full((row_count, 1), 0.9),
            body_ratio=np.full((row_count, 1), 0.9),
            signal=signal[:row_count].copy(),
        )

    def fake_simulate(sim_arrays, features, config, *, atr_multiplier):
        signal_count = int(np.count_nonzero(features.signal))
        captured["simulate_signal_counts"].append(signal_count)
        from xsignal.strategies.volume_price_efficiency_v1.trailing import TrailingStopResult

        final_equity = 1.0 + signal_count / 100.0
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
            equity=np.array([1.0, final_equity], dtype=np.float64),
            period_returns=np.array([final_equity - 1.0], dtype=np.float64),
            positions=np.zeros(features.signal.shape, dtype=bool),
            stop_prices=np.full(features.signal.shape, np.nan),
        )

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.split_research_and_holdout",
        fake_split,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.compute_features",
        lambda feature_arrays, _config: fake_features(feature_arrays.open.shape[0]),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.build_signal_mask",
        lambda _arrays, _features, _config: signal,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.simulate_trailing_stop",
        fake_simulate,
    )
    monkeypatch.setattr("xsignal.strategies.volume_price_efficiency_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        [
            "trail-regime-walk-forward",
            "--root",
            str(tmp_path),
            "--regime-walk-forward-id",
            "rwf-contraction",
            "--offline",
            "--holdout-days",
            "7",
            "--train-days",
            "1",
            "--test-days",
            "1",
            "--step-days",
            "1",
            "--lookback-bars",
            "2",
            "--feature-name",
            "pre_signal_atr_contraction",
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
        / "trailing_regime_walk_forwards"
        / "rwf-contraction"
    )
    fold_rows = pq.read_table(output_dir / "fold_summary.parquet").to_pylist()
    selection_rows = pq.read_table(output_dir / "selection_summary.parquet").to_pylist()
    assert exit_code == 0
    assert captured["split"] == [(all_arrays, 7)]
    assert captured["simulate_signal_counts"] == [2, 1, 3]
    assert fold_rows[0]["rule_id"] == "pre_signal_atr_contraction_gte_p50"
    assert fold_rows[0]["threshold"] == 1.0
    assert fold_rows[0]["threshold_source"] == "train_fold_signal_distribution"
    assert fold_rows[0]["validation_filtered_signal_count"] == 3
    assert {row["threshold"] for row in selection_rows} == {1.0}


def test_cli_trail_regime_walk_forward_can_select_train_fold_combo_rule(
    tmp_path,
    monkeypatch,
):
    all_arrays = _arrays(14)
    research_arrays = _arrays(12)
    holdout_arrays = _arrays(2)
    captured = {"split": [], "simulate_signal_indices": []}

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("1d", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
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

    signal = np.zeros((12, 1), dtype=bool)
    signal[[1, 2, 3, 4, 7, 8, 9, 10], 0] = True
    move = np.zeros((12, 1), dtype=np.float64)
    move[:12, 0] = [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0]
    volume = np.zeros((12, 1), dtype=np.float64)
    volume[:12, 0] = [0.0, 1.0, 4.0, 1.0, 4.0, 0.0, 0.0, 1.0, 4.0, 1.0, 4.0, 0.0]

    def fake_features(row_count):
        return FeatureArrays(
            true_range=np.ones((row_count, 1), dtype=np.float64),
            atr=np.ones((row_count, 1), dtype=np.float64),
            move_unit=move[:row_count].copy(),
            volume_baseline=np.ones((row_count, 1), dtype=np.float64),
            volume_unit=volume[:row_count].copy(),
            efficiency=move[:row_count].copy(),
            efficiency_threshold=np.zeros((row_count, 1), dtype=np.float64),
            close_position=np.full((row_count, 1), 0.9),
            body_ratio=np.full((row_count, 1), 0.9),
            signal=signal[:row_count].copy(),
        )

    def fake_simulate(sim_arrays, features, config, *, atr_multiplier):
        signal_indices = tuple(int(index) for index in np.flatnonzero(features.signal[:, 0]))
        captured["simulate_signal_indices"].append(signal_indices)
        from xsignal.strategies.volume_price_efficiency_v1.trailing import TrailingStopResult

        final_equity = 1.2 if signal_indices == (4,) else 1.0 + len(signal_indices) / 100.0
        return TrailingStopResult(
            trades=[
                {
                    "symbol": "BTCUSDT",
                    "signal_open_time": sim_arrays.open_times[index].isoformat(),
                    "realized_return": 0.04,
                    "net_realized_return": 0.038,
                    "holding_bars": 2,
                    "ignored_signal_count": 0,
                }
                for index in signal_indices
            ],
            equity=np.array([1.0, final_equity], dtype=np.float64),
            period_returns=np.array([final_equity - 1.0], dtype=np.float64),
            positions=np.zeros(features.signal.shape, dtype=bool),
            stop_prices=np.full(features.signal.shape, np.nan),
        )

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.split_research_and_holdout",
        fake_split,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.compute_features",
        lambda feature_arrays, _config: fake_features(feature_arrays.open.shape[0]),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.build_signal_mask",
        lambda _arrays, _features, _config: signal,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.simulate_trailing_stop",
        fake_simulate,
    )
    monkeypatch.setattr("xsignal.strategies.volume_price_efficiency_v1.cli._git_commit", lambda: "abc123")

    base_args = [
        "trail-regime-walk-forward",
        "--root",
        str(tmp_path),
        "--offline",
        "--holdout-days",
        "7",
        "--train-days",
        "1",
        "--test-days",
        "1",
        "--step-days",
        "1",
        "--feature-name",
        "move_unit,volume_unit",
        "--quantile",
        "0.5",
        "--min-trades",
        "1",
        "--top-k",
        "1",
        "--max-rule-size",
        "2",
        "--combo-seed-top-k",
        "4",
    ]

    default_exit_code = main(
        [
            *base_args,
            "--regime-walk-forward-id",
            "combo-rwf-default",
        ]
    )

    default_output_dir = (
        tmp_path
        / "strategies"
        / "volume_price_efficiency_v1"
        / "trailing_regime_walk_forwards"
        / "combo-rwf-default"
    )
    default_manifest = json.loads((default_output_dir / "manifest.json").read_text())
    default_fold_rows = pq.read_table(default_output_dir / "fold_summary.parquet").to_pylist()
    default_selection_rows = pq.read_table(default_output_dir / "selection_summary.parquet").to_pylist()
    assert default_exit_code == 0
    assert len(default_selection_rows) == 8
    assert default_manifest["allow_combo_selection"] is False
    assert default_fold_rows[0]["component_count"] == 1
    assert "__and__" not in default_fold_rows[0]["rule_id"]

    exit_code = main(
        [
            *base_args,
            "--regime-walk-forward-id",
            "combo-rwf",
            "--allow-combo-selection",
        ]
    )

    output_dir = (
        tmp_path
        / "strategies"
        / "volume_price_efficiency_v1"
        / "trailing_regime_walk_forwards"
        / "combo-rwf"
    )
    manifest = json.loads((output_dir / "manifest.json").read_text())
    fold_rows = pq.read_table(output_dir / "fold_summary.parquet").to_pylist()
    selection_rows = pq.read_table(output_dir / "selection_summary.parquet").to_pylist()
    assert exit_code == 0
    assert captured["split"] == [(all_arrays, 7), (all_arrays, 7)]
    assert len(selection_rows) == 8
    assert manifest["max_rule_size"] == 2
    assert manifest["combo_seed_top_k"] == 4
    assert manifest["allow_combo_selection"] is True
    assert fold_rows[0]["rule_id"] == "move_unit_gte_p50__and__volume_unit_gte_p50"
    assert fold_rows[0]["component_count"] == 2
    assert fold_rows[0]["validation_filtered_signal_count"] == 1


def test_cli_trail_regime_walk_forward_combo_gate_can_reject_train_fold_combo_rule(
    tmp_path,
    monkeypatch,
):
    all_arrays = _arrays(14)
    research_arrays = _arrays(12)
    holdout_arrays = _arrays(2)

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("1d", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
            (tmp_path / "manifest.json",),
        ),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.prepare_ohlcv_arrays",
        lambda _table: all_arrays,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.split_research_and_holdout",
        lambda split_arrays, *, holdout_days: (
            research_arrays,
            holdout_arrays,
            {
                "holdout_days": holdout_days,
                "research_start": "2026-01-01T00:00:00Z",
                "research_end": "2026-01-02T20:00:00Z",
                "holdout_start": "2026-01-03T00:00:00Z",
                "holdout_end": "2026-01-03T04:00:00Z",
            },
        ),
    )

    signal = np.zeros((12, 1), dtype=bool)
    signal[[1, 2, 3, 4, 7, 8, 9, 10], 0] = True
    move = np.zeros((12, 1), dtype=np.float64)
    move[:12, 0] = [0.0, 1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 0.0]
    volume = np.zeros((12, 1), dtype=np.float64)
    volume[:12, 0] = [0.0, 1.0, 4.0, 1.0, 4.0, 0.0, 0.0, 1.0, 4.0, 1.0, 4.0, 0.0]

    def fake_features(row_count):
        return FeatureArrays(
            true_range=np.ones((row_count, 1), dtype=np.float64),
            atr=np.ones((row_count, 1), dtype=np.float64),
            move_unit=move[:row_count].copy(),
            volume_baseline=np.ones((row_count, 1), dtype=np.float64),
            volume_unit=volume[:row_count].copy(),
            efficiency=move[:row_count].copy(),
            efficiency_threshold=np.zeros((row_count, 1), dtype=np.float64),
            close_position=np.full((row_count, 1), 0.9),
            body_ratio=np.full((row_count, 1), 0.9),
            signal=signal[:row_count].copy(),
        )

    def fake_simulate(sim_arrays, features, config, *, atr_multiplier):
        signal_indices = tuple(int(index) for index in np.flatnonzero(features.signal[:, 0]))
        from xsignal.strategies.volume_price_efficiency_v1.trailing import TrailingStopResult

        final_equity = 1.2 if signal_indices == (4,) else 1.0 + len(signal_indices) / 100.0
        return TrailingStopResult(
            trades=[
                {
                    "symbol": "BTCUSDT",
                    "signal_open_time": sim_arrays.open_times[index].isoformat(),
                    "realized_return": 0.04,
                    "net_realized_return": 0.038,
                    "holding_bars": 2,
                    "ignored_signal_count": 0,
                }
                for index in signal_indices
            ],
            equity=np.array([1.0, final_equity], dtype=np.float64),
            period_returns=np.array([final_equity - 1.0], dtype=np.float64),
            positions=np.zeros(features.signal.shape, dtype=bool),
            stop_prices=np.full(features.signal.shape, np.nan),
        )

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.compute_features",
        lambda feature_arrays, _config: fake_features(feature_arrays.open.shape[0]),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.build_signal_mask",
        lambda _arrays, _features, _config: signal,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.simulate_trailing_stop",
        fake_simulate,
    )
    monkeypatch.setattr("xsignal.strategies.volume_price_efficiency_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        [
            "trail-regime-walk-forward",
            "--root",
            str(tmp_path),
            "--offline",
            "--regime-walk-forward-id",
            "combo-gated-rwf",
            "--holdout-days",
            "7",
            "--train-days",
            "1",
            "--test-days",
            "1",
            "--step-days",
            "1",
            "--feature-name",
            "move_unit,volume_unit",
            "--quantile",
            "0.5",
            "--min-trades",
            "1",
            "--top-k",
            "1",
            "--max-rule-size",
            "2",
            "--combo-seed-top-k",
            "4",
            "--allow-combo-selection",
            "--combo-min-score-lift-vs-best-single",
            "0.5",
        ]
    )

    output_dir = (
        tmp_path
        / "strategies"
        / "volume_price_efficiency_v1"
        / "trailing_regime_walk_forwards"
        / "combo-gated-rwf"
    )
    manifest = json.loads((output_dir / "manifest.json").read_text())
    fold_rows = pq.read_table(output_dir / "fold_summary.parquet").to_pylist()
    assert exit_code == 0
    assert manifest["combo_min_score_lift_vs_best_single"] == 0.5
    assert fold_rows[0]["component_count"] == 1
    assert "__and__" not in fold_rows[0]["rule_id"]


def test_cli_trail_regime_walk_forward_refuses_non_offline_mode(tmp_path):
    try:
        main(["trail-regime-walk-forward", "--root", str(tmp_path), "--regime-walk-forward-id", "online"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("non-offline trail-regime-walk-forward should fail")


def test_cli_trail_regime_diagnose_writes_combo_diagnostics(tmp_path, monkeypatch):
    base = tmp_path / "strategies" / "volume_price_efficiency_v1" / "trailing_regime_walk_forwards"
    source_dir = base / "default-single"
    combo_dir = base / "allow-combo"
    source_dir.mkdir(parents=True)
    combo_dir.mkdir(parents=True)
    (source_dir / "manifest.json").write_text(json.dumps({"regime_walk_forward_id": "default-single"}))
    (combo_dir / "manifest.json").write_text(json.dumps({"regime_walk_forward_id": "allow-combo"}))
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "fold_index": 0,
                    "rule_id": "move_unit_gte_p50",
                    "component_count": 1,
                    "score": 0.03,
                    "trade_count": 120,
                },
                {
                    "fold_index": 0,
                    "rule_id": "volume_unit_gte_p50",
                    "component_count": 1,
                    "score": 0.02,
                    "trade_count": 100,
                },
                {
                    "fold_index": 0,
                    "rule_id": "move_unit_gte_p50__and__volume_unit_gte_p50",
                    "component_count": 2,
                    "component_rule_ids": json.dumps(
                        ["move_unit_gte_p50", "volume_unit_gte_p50"]
                    ),
                    "score": 0.04,
                    "trade_count": 70,
                },
            ]
        ),
        source_dir / "selection_summary.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "fold_index": 0,
                    "rule_id": "move_unit_gte_p50",
                    "component_count": 1,
                    "train_score": 0.03,
                    "validation_score": 0.01,
                    "validation_total_return": 0.02,
                    "validation_max_drawdown": 0.01,
                    "validation_trade_count": 12,
                }
            ]
        ),
        source_dir / "fold_summary.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "fold_index": 0,
                    "rule_id": "move_unit_gte_p50__and__volume_unit_gte_p50",
                    "component_count": 2,
                    "component_rule_ids": json.dumps(
                        ["move_unit_gte_p50", "volume_unit_gte_p50"]
                    ),
                    "train_score": 0.05,
                    "validation_score": -0.02,
                    "validation_total_return": -0.01,
                    "validation_max_drawdown": 0.03,
                    "validation_trade_count": 8,
                }
            ]
        ),
        combo_dir / "fold_summary.parquet",
    )
    monkeypatch.setattr("xsignal.strategies.volume_price_efficiency_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        [
            "trail-regime-diagnose",
            "--root",
            str(tmp_path),
            "--diagnostic-id",
            "diag",
            "--regime-walk-forward-id",
            "default-single",
            "--combo-selection-regime-walk-forward-id",
            "allow-combo",
        ]
    )

    output_dir = (
        tmp_path
        / "strategies"
        / "volume_price_efficiency_v1"
        / "trailing_regime_diagnostics"
        / "diag"
    )
    manifest = json.loads((output_dir / "manifest.json").read_text())
    candidate_rows = pq.read_table(output_dir / "combo_candidate_diagnostics.parquet").to_pylist()
    comparison_rows = pq.read_table(output_dir / "selected_rule_comparison.parquet").to_pylist()
    assert exit_code == 0
    assert manifest["source_regime_walk_forward_id"] == "default-single"
    assert manifest["combo_selection_regime_walk_forward_id"] == "allow-combo"
    assert candidate_rows[0]["score_lift_vs_best_component"] == 0.01
    assert comparison_rows[0]["validation_score_delta"] == -0.03


def test_cli_trail_regime_gate_sweep_writes_artifact_only_summary(tmp_path, monkeypatch):
    base = tmp_path / "strategies" / "volume_price_efficiency_v1" / "trailing_regime_walk_forwards"
    baseline_dir = base / "default-single"
    gated_dir = base / "gated"
    baseline_dir.mkdir(parents=True)
    gated_dir.mkdir(parents=True)
    (baseline_dir / "manifest.json").write_text(
        json.dumps(
            {
                "regime_walk_forward_id": "default-single",
                "allow_combo_selection": False,
                "max_rule_size": 1,
            }
        )
    )
    (gated_dir / "manifest.json").write_text(
        json.dumps(
            {
                "regime_walk_forward_id": "gated",
                "allow_combo_selection": True,
                "max_rule_size": 2,
                "combo_min_score_lift_vs_best_single": 0.005,
            }
        )
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "fold_index": 0,
                    "rule_id": "move_unit_gte_p50",
                    "component_count": 1,
                    "validation_score": 0.01,
                    "validation_total_return": 0.02,
                    "validation_trade_count": 12,
                },
                {
                    "fold_index": 1,
                    "rule_id": "volume_unit_gte_p50",
                    "component_count": 1,
                    "validation_score": 0.03,
                    "validation_total_return": 0.04,
                    "validation_trade_count": 10,
                },
            ]
        ),
        baseline_dir / "fold_summary.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "fold_index": 0,
                    "rule_id": "move_unit_gte_p50",
                    "component_count": 1,
                }
            ]
        ),
        baseline_dir / "selection_summary.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "fold_index": 0,
                    "rule_id": "move_unit_gte_p50__and__volume_unit_gte_p50",
                    "component_count": 2,
                    "component_rule_ids": json.dumps(
                        ["move_unit_gte_p50", "volume_unit_gte_p50"]
                    ),
                    "validation_score": -0.02,
                    "validation_total_return": -0.01,
                    "validation_trade_count": 8,
                },
                {
                    "fold_index": 1,
                    "rule_id": "volume_unit_gte_p50",
                    "component_count": 1,
                    "validation_score": 0.05,
                    "validation_total_return": 0.06,
                    "validation_trade_count": 11,
                },
            ]
        ),
        gated_dir / "fold_summary.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "fold_index": 0,
                    "rule_id": "move_unit_gte_p50__and__volume_unit_gte_p50",
                    "component_count": 2,
                    "combo_gate_passed": True,
                },
                {
                    "fold_index": 0,
                    "rule_id": "move_unit_gte_p50__and__close_position_gte_p50",
                    "component_count": 2,
                    "combo_gate_passed": False,
                },
            ]
        ),
        gated_dir / "selection_summary.parquet",
    )
    monkeypatch.setattr("xsignal.strategies.volume_price_efficiency_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        [
            "trail-regime-gate-sweep",
            "--root",
            str(tmp_path),
            "--diagnostic-id",
            "sweep",
            "--baseline-regime-walk-forward-id",
            "default-single",
            "--experiment-regime-walk-forward-id",
            "gated",
        ]
    )

    output_dir = (
        tmp_path
        / "strategies"
        / "volume_price_efficiency_v1"
        / "trailing_regime_diagnostics"
        / "sweep"
    )
    manifest = json.loads((output_dir / "manifest.json").read_text())
    summary_rows = pq.read_table(output_dir / "gate_sweep_summary.parquet").to_pylist()
    comparison_rows = pq.read_table(output_dir / "gate_sweep_fold_comparison.parquet").to_pylist()
    assert exit_code == 0
    assert manifest["run_type"] == "trailing_stop_research_regime_gate_sweep"
    assert manifest["baseline_regime_walk_forward_id"] == "default-single"
    assert manifest["experiment_regime_walk_forward_ids"] == ["gated"]
    assert summary_rows[1]["regime_walk_forward_id"] == "gated"
    assert summary_rows[1]["selected_combo_fold_count"] == 1
    assert summary_rows[1]["combo_gate_failed_count"] == 1
    assert comparison_rows[0]["validation_total_return_delta"] == -0.03
