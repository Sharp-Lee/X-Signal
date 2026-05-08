from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing import (
    TrailingStopResult,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_artifacts import (
    build_trailing_summary,
    write_trailing_scan_artifacts,
    write_trailing_regime_holdout_artifacts,
    write_trailing_run_artifacts,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_regime_scan import (
    RegimeFilterRule,
)


def _result() -> TrailingStopResult:
    return TrailingStopResult(
        trades=[
            {
                "symbol": "BTCUSDT",
                "signal_open_time": "2026-01-01T00:00:00+00:00",
                "decision_time": "2026-01-01T04:00:00+00:00",
                "entry_open_time": "2026-01-01T04:00:00+00:00",
                "exit_time": "2026-01-01T08:00:00+00:00",
                "entry_price": 100.0,
                "exit_price": 108.0,
                "stop_price_at_exit": 108.0,
                "atr_at_entry": 5.0,
                "atr_at_exit": 1.0,
                "highest_high": 110.0,
                "realized_return": 0.08,
                "net_realized_return": 0.078,
                "holding_bars": 2,
                "ignored_signal_count": 1,
            },
            {
                "symbol": "ETHUSDT",
                "signal_open_time": "2026-01-02T00:00:00+00:00",
                "decision_time": "2026-01-02T04:00:00+00:00",
                "entry_open_time": "2026-01-02T04:00:00+00:00",
                "exit_time": "2026-01-02T08:00:00+00:00",
                "entry_price": 100.0,
                "exit_price": 95.0,
                "stop_price_at_exit": 95.0,
                "atr_at_entry": 5.0,
                "atr_at_exit": 5.0,
                "highest_high": 105.0,
                "realized_return": -0.05,
                "net_realized_return": -0.052,
                "holding_bars": 2,
                "ignored_signal_count": 0,
            },
        ],
        equity=np.array([1.0, 1.08, 1.026], dtype=np.float64),
        period_returns=np.array([0.08, -0.05], dtype=np.float64),
        positions=np.array([[False, False], [True, False], [False, True]], dtype=bool),
        stop_prices=np.array([[np.nan, np.nan], [90.0, np.nan], [np.nan, 95.0]]),
    )


def test_build_trailing_summary_reports_trade_metrics():
    summary = build_trailing_summary(_result())

    assert summary["trade_count"] == 2
    assert summary["winning_trade_count"] == 1
    assert summary["win_rate"] == 0.5
    assert summary["mean_realized_return"] == pytest.approx(0.015)
    assert summary["mean_net_realized_return"] == pytest.approx(0.013)
    assert summary["final_equity"] == pytest.approx(1.026)
    assert summary["total_return"] == pytest.approx(0.026)
    assert summary["max_drawdown"] == pytest.approx(0.05)
    assert summary["total_ignored_signal_count"] == 1
    assert summary["average_holding_bars"] == pytest.approx(2.0)


def test_write_trailing_run_artifacts_creates_manifest_summary_and_tables(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)
    config = VolumePriceEfficiencyConfig(fee_bps=5, slippage_bps=5)
    result = _result()

    run_dir = write_trailing_run_artifacts(
        paths=paths,
        run_id="trailrun",
        config=config,
        result=result,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_times=np.array(
            [
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T04:00:00+00:00",
                "2026-01-01T08:00:00+00:00",
            ],
            dtype=object,
        ),
        canonical_manifests=["manifest.json"],
        git_commit="abc123",
        runtime_seconds=1.5,
        data_split={"holdout_days": 180},
        atr_multiplier=2.0,
    )

    assert run_dir == paths.trailing_run_dir("trailrun")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert manifest["strategy_name"] == "volume_price_efficiency_v1"
    assert manifest["run_type"] == "trailing_stop_holdout"
    assert manifest["config_hash"] == config.config_hash()
    assert manifest["canonical_manifests"] == ["manifest.json"]
    assert manifest["data_split"]["holdout_days"] == 180
    assert manifest["atr_multiplier"] == 2.0
    assert set(manifest["outputs"]) == {
        "summary",
        "trades",
        "equity_curve",
        "daily_positions",
    }
    assert summary["trade_count"] == 2

    trades = pq.read_table(run_dir / "trades.parquet")
    equity = pq.read_table(run_dir / "equity_curve.parquet")
    positions = pq.read_table(run_dir / "daily_positions.parquet")
    assert trades.num_rows == 2
    assert set(
        [
            "symbol",
            "signal_open_time",
            "entry_open_time",
            "exit_time",
            "exit_price",
            "stop_price_at_exit",
            "realized_return",
            "net_realized_return",
        ]
    ).issubset(set(trades.column_names))
    assert equity.num_rows == 3
    assert positions.num_rows == 2


def test_write_trailing_run_artifacts_writes_empty_trade_table(tmp_path):
    result = TrailingStopResult(
        trades=[],
        equity=np.array([1.0], dtype=np.float64),
        period_returns=np.array([], dtype=np.float64),
        positions=np.zeros((1, 1), dtype=bool),
        stop_prices=np.full((1, 1), np.nan),
    )

    run_dir = write_trailing_run_artifacts(
        paths=VolumePriceEfficiencyPaths(root=tmp_path),
        run_id="empty",
        config=VolumePriceEfficiencyConfig(),
        result=result,
        symbols=("BTCUSDT",),
        open_times=np.array(["2026-01-01T00:00:00+00:00"], dtype=object),
        canonical_manifests=[],
        git_commit="abc123",
        runtime_seconds=0.1,
        data_split={"holdout_days": 180},
        atr_multiplier=2.0,
    )

    assert pq.read_table(run_dir / "trades.parquet").num_rows == 0
    assert pq.read_table(run_dir / "daily_positions.parquet").num_rows == 0


def test_write_trailing_regime_holdout_artifacts_records_research_selection(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)
    config = VolumePriceEfficiencyConfig()
    result = _result()
    selected_rule = RegimeFilterRule(
        rule_id="move_unit_gte_p50",
        feature_name="move_unit",
        direction="gte",
        quantile=0.5,
        threshold=2.0,
    )

    run_dir = write_trailing_regime_holdout_artifacts(
        paths=paths,
        run_id="regime-holdout",
        config=config,
        result=result,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_times=np.array(
            [
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T04:00:00+00:00",
                "2026-01-01T08:00:00+00:00",
            ],
            dtype=object,
        ),
        canonical_manifests=["manifest.json"],
        git_commit="abc123",
        runtime_seconds=1.5,
        data_split={"holdout_days": 180},
        atr_multiplier=2.0,
        lookback_bars=30,
        quantiles=(0.5,),
        feature_names=("move_unit",),
        min_trades=10,
        selected_rule=selected_rule,
        selected_train_row={
            "rule_id": "move_unit_gte_p50",
            "score": 0.04,
            "trade_count": 22,
            "base_signal_count": 80,
            "filtered_signal_count": 40,
        },
        selection_rows=[
            {
                "rule_id": "move_unit_gte_p50",
                "score": 0.04,
                "trade_count": 22,
            }
        ],
        holdout_base_signal_count=30,
        holdout_filtered_signal_count=12,
    )

    manifest = json.loads((run_dir / "manifest.json").read_text())
    selected_filter = json.loads((run_dir / "selected_filter.json").read_text())
    assert run_dir == paths.trailing_run_dir("regime-holdout")
    assert manifest["run_type"] == "trailing_stop_regime_holdout"
    assert manifest["data_scope"] == "holdout_only_final_production_test"
    assert manifest["selection_scope"] == "research_only"
    assert manifest["threshold_scope"] == "full_research_signal_distribution"
    assert manifest["selected_rule_id"] == "move_unit_gte_p50"
    assert manifest["holdout_base_signal_count"] == 30
    assert manifest["holdout_filtered_signal_count"] == 12
    assert manifest["holdout_signal_keep_rate"] == 0.4
    assert set(manifest["outputs"]) == {
        "summary",
        "trades",
        "equity_curve",
        "daily_positions",
        "selected_filter",
        "selection_summary",
        "selection_summary_csv",
    }
    assert selected_filter["rule_id"] == "move_unit_gte_p50"
    assert selected_filter["threshold"] == 2.0
    assert selected_filter["train_score"] == 0.04
    assert pq.read_table(run_dir / "selection_summary.parquet").to_pylist() == [
        {"rule_id": "move_unit_gte_p50", "score": 0.04, "trade_count": 22}
    ]


def test_write_trailing_scan_artifacts_creates_manifest_summary_csv_and_top_configs(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)
    config = VolumePriceEfficiencyConfig()
    rows = [
        {
            "scan_id": "trailscan",
            "config_hash": "hash1",
            "efficiency_percentile": 0.9,
            "min_move_unit": 1.2,
            "min_volume_unit": 1.5,
            "min_close_position": 0.94,
            "min_body_ratio": 0.85,
            "fee_bps": 5.0,
            "slippage_bps": 5.0,
            "baseline_seed": 17,
            "atr_multiplier": 2.0,
            "symbol_count": 2,
            "trade_count": 20,
            "win_rate": 0.45,
            "mean_net_realized_return": 0.01,
            "average_holding_bars": 5.0,
            "total_ignored_signal_count": 3,
            "final_equity": 1.08,
            "total_return": 0.08,
            "max_drawdown": 0.02,
            "score": 0.06,
        }
    ]

    scan_dir = write_trailing_scan_artifacts(
        paths=paths,
        scan_id="trailscan",
        base_config=config,
        rows=rows,
        top_configs=rows,
        canonical_manifests=["manifest.json"],
        git_commit="abc123",
        runtime_seconds=2.5,
        symbol_count=2,
        data_split={"holdout_days": 180},
        atr_multiplier=2.0,
        min_trades=10,
    )

    assert scan_dir == paths.trailing_scan_dir("trailscan")
    manifest = json.loads((scan_dir / "manifest.json").read_text())
    summary = json.loads((scan_dir / "summary.json").read_text())
    top = json.loads((scan_dir / "top_configs.json").read_text())
    assert manifest["run_type"] == "trailing_stop_research_scan"
    assert manifest["data_split"]["holdout_days"] == 180
    assert manifest["atr_multiplier"] == 2.0
    assert manifest["min_trades"] == 10
    assert set(manifest["outputs"]) == {"summary", "summary_csv", "top_configs"}
    assert summary["combination_count"] == 1
    assert summary["eligible_combination_count"] == 1
    assert summary["best_score"] == 0.06
    assert top == rows
    assert "hash1" in (scan_dir / "summary.csv").read_text()
