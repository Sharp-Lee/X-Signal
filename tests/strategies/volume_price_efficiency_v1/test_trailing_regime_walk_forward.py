from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pyarrow.parquet as pq

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_regime_scan import (
    RegimeFilterRule,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_regime_walk_forward import (
    build_regime_validation_trade_rows,
    build_regime_walk_forward_fold_row,
    build_regime_stability_summary,
    filter_combo_regime_candidates,
    select_stable_regime_filters,
    write_trailing_regime_walk_forward_artifacts,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_walk_forward import (
    build_walk_forward_folds,
)


def _open_times(day_count: int):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [start + timedelta(days=index) for index in range(day_count)]


def test_build_regime_walk_forward_fold_row_records_train_threshold_and_validation_metrics():
    config = VolumePriceEfficiencyConfig(min_move_unit=1.2)
    rule = RegimeFilterRule(
        rule_id="market_lookback_return_lt_p80",
        feature_name="market_lookback_return",
        direction="lt",
        quantile=0.8,
        threshold=0.12,
    )

    row = build_regime_walk_forward_fold_row(
        regime_walk_forward_id="rwf",
        fold_index=1,
        fold=build_walk_forward_folds(_open_times(6), train_days=3, test_days=2)[0],
        selected_train_row={
            "rule_id": rule.rule_id,
            "score": 0.04,
            "trade_count": 22,
            "total_return": 0.05,
            "max_drawdown": 0.01,
            "filtered_signal_count": 30,
            "base_signal_count": 60,
            "selection_method": "train_stability",
            "stability_split_count": 3,
            "stability_min_trades": 5,
            "stability_eligible_split_count": 3,
            "stability_positive_split_count": 2,
            "stability_mean_score": 0.0123456789012,
            "stability_min_score": -0.003,
        },
        validation_row={
            "score": -0.02,
            "trade_count": 8,
            "total_return": -0.01,
            "max_drawdown": 0.01,
            "win_rate": 0.375,
            "mean_net_realized_return": -0.002,
            "median_net_realized_return": -0.001,
            "average_holding_bars": 5.0,
            "total_ignored_signal_count": 2,
            "final_equity": 0.99,
            "filtered_signal_count": 9,
            "base_signal_count": 18,
        },
        selected_rule=rule,
        selected_config=config,
    )

    assert row["regime_walk_forward_id"] == "rwf"
    assert row["fold_index"] == 1
    assert row["rule_id"] == "market_lookback_return_lt_p80"
    assert row["threshold"] == 0.12
    assert row["threshold_source"] == "train_fold_signal_distribution"
    assert row["train_score"] == 0.04
    assert row["train_signal_keep_rate"] == 0.5
    assert row["selection_method"] == "train_stability"
    assert row["train_stability_split_count"] == 3
    assert row["train_stability_min_trades"] == 5
    assert row["train_stability_eligible_split_count"] == 3
    assert row["train_stability_positive_split_count"] == 2
    assert row["train_stability_mean_score"] == 0.012345678901
    assert row["train_stability_min_score"] == -0.003
    assert row["validation_score"] == -0.02
    assert row["validation_signal_keep_rate"] == 0.5
    assert row["min_move_unit"] == 1.2


def test_build_regime_stability_summary_counts_eligible_and_positive_segments():
    summary = build_regime_stability_summary(
        [
            {"trade_count": 8, "score": 0.03},
            {"trade_count": 7, "score": -0.01},
            {"trade_count": 2, "score": 0.50},
        ],
        split_count=3,
        min_trades=5,
    )

    assert summary == {
        "selection_method": "train_stability",
        "stability_split_count": 3,
        "stability_min_trades": 5,
        "stability_evaluated_split_count": 3,
        "stability_eligible_split_count": 2,
        "stability_positive_split_count": 1,
        "stability_mean_score": 0.01,
        "stability_min_score": -0.01,
    }


def test_build_regime_stability_summary_can_include_segment_score_values():
    summary = build_regime_stability_summary(
        [
            {"trade_count": 8, "score": 0.03},
            {"trade_count": 7, "score": -0.01},
            {"trade_count": 2, "score": 0.50},
        ],
        split_count=3,
        min_trades=5,
        include_values=True,
    )

    assert summary["stability_score_values"] == json.dumps([0.03, -0.01, 0.5])
    assert summary["stability_trade_count_values"] == json.dumps([8, 7, 2])


def test_filter_combo_regime_candidates_applies_train_only_combo_gates():
    rows = [
        {
            "rule_id": "move_unit_gte_p50",
            "component_count": 1,
            "score": 0.03,
            "trade_count": 100,
            "stability_score_values": json.dumps([0.01, 0.02, 0.01]),
        },
        {
            "rule_id": "volume_unit_gte_p50",
            "component_count": 1,
            "score": 0.02,
            "trade_count": 90,
            "stability_score_values": json.dumps([0.0, 0.01, 0.0]),
        },
        {
            "rule_id": "move_unit_gte_p50__and__volume_unit_gte_p50",
            "component_count": 2,
            "component_rule_ids": json.dumps(["move_unit_gte_p50", "volume_unit_gte_p50"]),
            "score": 0.05,
            "trade_count": 80,
            "stability_score_values": json.dumps([0.02, 0.01, 0.03]),
        },
        {
            "rule_id": "move_unit_gte_p50__and__quality_gte_p50",
            "component_count": 2,
            "component_rule_ids": json.dumps(["move_unit_gte_p50", "quality_gte_p50"]),
            "score": 0.031,
            "trade_count": 75,
            "stability_score_values": json.dumps([0.02, 0.01, 0.005]),
        },
    ]

    filtered = filter_combo_regime_candidates(
        rows,
        allow_combo_selection=True,
        min_score_lift_vs_best_component=0.005,
        min_score_lift_vs_best_single=0.01,
        min_component_outperformance_splits=2,
        min_single_outperformance_splits=2,
    )

    assert [row["rule_id"] for row in filtered] == [
        "move_unit_gte_p50",
        "volume_unit_gte_p50",
        "move_unit_gte_p50__and__volume_unit_gte_p50",
    ]
    assert filtered[2]["combo_best_component_rule_id"] == "move_unit_gte_p50"
    assert filtered[2]["combo_score_lift_vs_best_component"] == 0.02
    assert filtered[2]["combo_score_lift_vs_best_single"] == 0.02
    assert filtered[2]["combo_component_outperformance_splits"] == 2
    assert filtered[2]["combo_single_outperformance_splits"] == 2
    assert filtered[2]["combo_gate_passed"] is True
    assert rows[3]["combo_gate_passed"] is False
    assert rows[3]["combo_score_lift_vs_best_single"] == 0.001


def test_select_stable_regime_filters_prefers_consistency_over_raw_train_score():
    rows = [
        {
            "rule_id": "high_train_unstable",
            "trade_count": 100,
            "score": 0.50,
            "stability_split_count": 3,
            "stability_eligible_split_count": 3,
            "stability_positive_split_count": 1,
            "stability_mean_score": 0.08,
            "stability_min_score": -0.20,
        },
        {
            "rule_id": "lower_train_stable",
            "trade_count": 80,
            "score": 0.20,
            "stability_split_count": 3,
            "stability_eligible_split_count": 3,
            "stability_positive_split_count": 3,
            "stability_mean_score": 0.04,
            "stability_min_score": 0.02,
        },
        {
            "rule_id": "unfiltered",
            "trade_count": 500,
            "score": 1.0,
            "stability_split_count": 3,
            "stability_eligible_split_count": 3,
            "stability_positive_split_count": 3,
            "stability_mean_score": 1.0,
            "stability_min_score": 1.0,
        },
    ]

    assert select_stable_regime_filters(
        rows,
        top_k=1,
        min_trades=10,
        stability_min_positive_splits=2,
    ) == [rows[1]]


def test_build_regime_validation_trade_rows_adds_fold_rule_and_feature_context():
    rule = RegimeFilterRule(
        rule_id="pre_signal_atr_contraction_gte_p50",
        feature_name="pre_signal_atr_contraction",
        direction="gte",
        quantile=0.5,
        threshold=0.96,
    )

    rows = build_regime_validation_trade_rows(
        regime_walk_forward_id="rwf",
        fold_index=15,
        fold=build_walk_forward_folds(_open_times(20), train_days=10, test_days=5)[0],
        selected_train_row={
            "score": 0.024,
            "trade_count": 100,
            "signal_keep_rate": 0.5,
        },
        validation_row={
            "score": -0.0068,
            "trade_count": 1,
            "signal_keep_rate": 0.4,
        },
        selected_rule=rule,
        trades=[
            {
                "symbol": "BTCUSDT",
                "signal_open_time": "2026-01-11T00:00:00+00:00",
                "entry_open_time": "2026-01-11T04:00:00+00:00",
                "exit_time": "2026-01-12T00:00:00+00:00",
                "net_realized_return": -0.012,
                "holding_bars": 5,
                "atr_at_entry": 2.0,
            }
        ],
        trade_feature_rows=[
            {
                "pre_signal_atr_contraction": 1.08,
                "market_lookback_return": -0.04,
            }
        ],
    )

    assert rows == [
        {
            "regime_walk_forward_id": "rwf",
            "fold_index": 15,
            "test_start": "2026-01-11T00:00:00+00:00",
            "test_end": "2026-01-16T00:00:00+00:00",
            "rule_id": "pre_signal_atr_contraction_gte_p50",
            "feature_name": "pre_signal_atr_contraction",
            "direction": "gte",
            "quantile": 0.5,
            "threshold": 0.96,
            "threshold_source": "train_fold_signal_distribution",
            "train_score": 0.024,
            "train_trade_count": 100,
            "train_signal_keep_rate": 0.5,
            "validation_score": -0.0068,
            "validation_trade_count": 1,
            "validation_signal_keep_rate": 0.4,
            "symbol": "BTCUSDT",
            "signal_open_time": "2026-01-11T00:00:00+00:00",
            "entry_open_time": "2026-01-11T04:00:00+00:00",
            "exit_time": "2026-01-12T00:00:00+00:00",
            "net_realized_return": -0.012,
            "holding_bars": 5,
            "atr_at_entry": 2.0,
            "pre_signal_atr_contraction": 1.08,
            "market_lookback_return": -0.04,
        }
    ]


def test_write_trailing_regime_walk_forward_artifacts(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)
    config = VolumePriceEfficiencyConfig()
    rule = RegimeFilterRule(
        rule_id="market_lookback_return_lt_p80",
        feature_name="market_lookback_return",
        direction="lt",
        quantile=0.8,
        threshold=0.12,
    )
    fold_rows = [
        build_regime_walk_forward_fold_row(
            regime_walk_forward_id="rwf",
            fold_index=0,
            fold=build_walk_forward_folds(_open_times(6), train_days=3, test_days=2)[0],
            selected_train_row={"score": 0.04, "trade_count": 22},
            validation_row={"score": -0.02, "trade_count": 8},
            selected_rule=rule,
            selected_config=config,
        )
    ]
    selection_rows = [
        {
            "regime_walk_forward_id": "rwf",
            "fold_index": 0,
            "rule_id": rule.rule_id,
            "score": 0.04,
            "trade_count": 22,
        },
        {
            "regime_walk_forward_id": "rwf",
            "fold_index": 0,
            "rule_id": "a__and__b",
            "score": 0.05,
            "trade_count": 18,
            "combo_gate_passed": False,
            "combo_score_lift_vs_best_single": 0.01,
        }
    ]
    validation_trade_rows = [
        {
            "regime_walk_forward_id": "rwf",
            "fold_index": 0,
            "rule_id": rule.rule_id,
            "symbol": "BTCUSDT",
            "net_realized_return": -0.01,
        }
    ]

    output = write_trailing_regime_walk_forward_artifacts(
        paths=paths,
        regime_walk_forward_id="rwf",
        config=config,
        fold_rows=fold_rows,
        selection_rows=selection_rows,
        validation_trade_rows=validation_trade_rows,
        top_filters=selection_rows,
        canonical_manifests=["manifest.json"],
        git_commit="abc123",
        runtime_seconds=1.5,
        symbol_count=5,
        data_split={"holdout_days": 180},
        atr_multiplier=2.0,
        pyramid_add_step_atr=1.0,
        pyramid_max_adds=1,
        lookback_bars=30,
        train_days=720,
        test_days=90,
        step_days=90,
        min_trades=50,
        stability_splits=3,
        stability_min_trades=10,
        stability_min_positive_splits=2,
        quantiles=(0.8,),
        feature_names=("market_lookback_return",),
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert output == paths.trailing_regime_walk_forward_dir("rwf")
    assert manifest["run_type"] == "trailing_stop_research_regime_walk_forward"
    assert manifest["data_scope"] == "research_only"
    assert manifest["threshold_scope"] == "per_fold_train_signal_distribution"
    assert manifest["selection_method"] == "train_stability"
    assert manifest["pyramid_add_step_atr"] == 1.0
    assert manifest["pyramid_max_adds"] == 1
    assert manifest["stability_splits"] == 3
    assert manifest["stability_min_trades"] == 10
    assert manifest["stability_min_positive_splits"] == 2
    assert set(manifest["outputs"]) == {
        "fold_summary",
        "fold_summary_csv",
        "selection_summary",
        "validation_trades",
        "validation_trades_csv",
        "top_filters",
    }
    assert pq.read_table(output / "fold_summary.parquet").num_rows == 1
    selection_table = pq.read_table(output / "selection_summary.parquet")
    assert selection_table.num_rows == 2
    assert "combo_gate_passed" in selection_table.column_names
    assert selection_table.to_pylist()[1]["combo_score_lift_vs_best_single"] == 0.01
    assert pq.read_table(output / "validation_trades.parquet").to_pylist() == validation_trade_rows
