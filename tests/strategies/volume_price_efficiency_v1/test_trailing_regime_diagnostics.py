from __future__ import annotations

import json

import pyarrow.parquet as pq

from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_regime_diagnostics import (
    build_combo_candidate_diagnostic_rows,
    build_selected_rule_comparison_rows,
    write_trailing_regime_diagnostic_artifacts,
)


def test_build_combo_candidate_diagnostic_rows_compares_combo_to_components_and_best_single():
    rows = build_combo_candidate_diagnostic_rows(
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
                "signal_keep_rate": 0.25,
                "stability_positive_split_count": 3,
                "stability_split_count": 4,
            },
        ]
    )

    assert rows == [
        {
            "fold_index": 0,
            "rule_id": "move_unit_gte_p50__and__volume_unit_gte_p50",
            "component_count": 2,
            "component_rule_ids": json.dumps(
                ["move_unit_gte_p50", "volume_unit_gte_p50"]
            ),
            "train_score": 0.04,
            "train_trade_count": 70,
            "train_signal_keep_rate": 0.25,
            "best_component_rule_id": "move_unit_gte_p50",
            "best_component_score": 0.03,
            "best_component_trade_count": 120,
            "score_lift_vs_best_component": 0.01,
            "best_single_rule_id": "move_unit_gte_p50",
            "best_single_score": 0.03,
            "score_lift_vs_best_single": 0.01,
            "stability_positive_split_count": 3,
            "stability_split_count": 4,
        }
    ]


def test_build_selected_rule_comparison_rows_compares_combo_selection_to_baseline():
    rows = build_selected_rule_comparison_rows(
        baseline_fold_rows=[
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
        ],
        experiment_fold_rows=[
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
        ],
        baseline_label="default_single",
        experiment_label="allow_combo",
    )

    assert rows == [
        {
            "fold_index": 0,
            "baseline_label": "default_single",
            "experiment_label": "allow_combo",
            "baseline_rule_id": "move_unit_gte_p50",
            "experiment_rule_id": "move_unit_gte_p50__and__volume_unit_gte_p50",
            "experiment_component_count": 2,
            "experiment_component_rule_ids": json.dumps(
                ["move_unit_gte_p50", "volume_unit_gte_p50"]
            ),
            "experiment_is_combo": True,
            "train_score_delta": 0.02,
            "validation_score_delta": -0.03,
            "validation_total_return_delta": -0.03,
            "validation_max_drawdown_delta": 0.02,
            "validation_trade_count_delta": -4,
        }
    ]


def test_write_trailing_regime_diagnostic_artifacts(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)
    output = write_trailing_regime_diagnostic_artifacts(
        paths=paths,
        diagnostic_id="diag",
        source_regime_walk_forward_id="default-single",
        combo_selection_regime_walk_forward_id="allow-combo",
        candidate_rows=[{"fold_index": 0, "rule_id": "a__and__b"}],
        comparison_rows=[{"fold_index": 0, "validation_score_delta": -0.02}],
        git_commit="abc123",
        runtime_seconds=1.25,
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert output == paths.trailing_regime_diagnostic_dir("diag")
    assert manifest["run_type"] == "trailing_regime_combo_diagnostics"
    assert manifest["data_scope"] == "artifact_only_research_summary"
    assert manifest["source_regime_walk_forward_id"] == "default-single"
    assert manifest["combo_selection_regime_walk_forward_id"] == "allow-combo"
    assert pq.read_table(output / "combo_candidate_diagnostics.parquet").to_pylist() == [
        {"fold_index": 0, "rule_id": "a__and__b"}
    ]
    assert pq.read_table(output / "selected_rule_comparison.parquet").to_pylist() == [
        {"fold_index": 0, "validation_score_delta": -0.02}
    ]
