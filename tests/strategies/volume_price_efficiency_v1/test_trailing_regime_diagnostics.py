from __future__ import annotations

import json

import pyarrow.parquet as pq

from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_regime_diagnostics import (
    build_gate_sweep_fold_comparison_rows,
    build_gate_sweep_summary_rows,
    build_combo_candidate_diagnostic_rows,
    build_selected_rule_comparison_rows,
    write_trailing_regime_gate_sweep_artifacts,
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


def test_build_gate_sweep_summary_rows_marks_combo_counts_and_best_worst_deltas():
    baseline_fold_rows = [
        {
            "fold_index": 0,
            "rule_id": "move_unit_gte_p50",
            "component_count": 1,
            "validation_total_return": 0.10,
        },
        {
            "fold_index": 1,
            "rule_id": "volume_unit_gte_p50",
            "component_count": 1,
            "validation_total_return": -0.05,
        },
    ]
    experiment_fold_rows = [
        {
            "fold_index": 0,
            "rule_id": "move_unit_gte_p50__and__volume_unit_gte_p50",
            "component_count": 2,
            "validation_total_return": 0.08,
        },
        {
            "fold_index": 1,
            "rule_id": "volume_unit_gte_p50",
            "component_count": 1,
            "validation_total_return": -0.02,
        },
    ]

    rows = build_gate_sweep_summary_rows(
        baseline_regime_walk_forward_id="baseline",
        run_artifacts=[
            {
                "regime_walk_forward_id": "baseline",
                "manifest": {"allow_combo_selection": False, "max_rule_size": 1},
                "fold_rows": baseline_fold_rows,
                "selection_rows": [
                    {"fold_index": 0, "rule_id": "move_unit_gte_p50", "component_count": 1}
                ],
            },
            {
                "regime_walk_forward_id": "gated",
                "manifest": {
                    "allow_combo_selection": True,
                    "max_rule_size": 2,
                    "combo_min_score_lift_vs_best_single": 0.005,
                },
                "fold_rows": experiment_fold_rows,
                "selection_rows": [
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
                ],
            },
        ],
    )

    assert rows[0]["regime_walk_forward_id"] == "baseline"
    assert rows[0]["is_baseline"] is True
    assert rows[0]["compound_validation_return"] == 0.045
    assert rows[0]["compound_validation_return_delta_vs_baseline"] == 0.0
    assert rows[1]["regime_walk_forward_id"] == "gated"
    assert rows[1]["is_baseline"] is False
    assert rows[1]["allow_combo_selection"] is True
    assert rows[1]["max_rule_size"] == 2
    assert rows[1]["combo_min_score_lift_vs_best_single"] == 0.005
    assert rows[1]["compound_validation_return"] == 0.0584
    assert rows[1]["compound_validation_return_delta_vs_baseline"] == 0.0134
    assert rows[1]["selected_combo_fold_count"] == 1
    assert rows[1]["combo_candidate_count"] == 2
    assert rows[1]["combo_gate_passed_count"] == 1
    assert rows[1]["combo_gate_failed_count"] == 1
    assert rows[1]["worst_delta_fold_index_vs_baseline"] == 0
    assert rows[1]["worst_validation_total_return_delta_vs_baseline"] == -0.02
    assert rows[1]["best_delta_fold_index_vs_baseline"] == 1
    assert rows[1]["best_validation_total_return_delta_vs_baseline"] == 0.03


def test_build_gate_sweep_fold_comparison_rows_keeps_raw_values_and_deltas():
    rows = build_gate_sweep_fold_comparison_rows(
        baseline_regime_walk_forward_id="baseline",
        experiment_regime_walk_forward_id="gated",
        baseline_fold_rows=[
            {
                "fold_index": 0,
                "rule_id": "move_unit_gte_p50",
                "component_count": 1,
                "train_score": 0.03,
                "validation_score": 0.01,
                "validation_total_return": 0.02,
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
                "validation_trade_count": 8,
            }
        ],
    )

    assert rows == [
        {
            "fold_index": 0,
            "baseline_regime_walk_forward_id": "baseline",
            "experiment_regime_walk_forward_id": "gated",
            "baseline_rule_id": "move_unit_gte_p50",
            "experiment_rule_id": "move_unit_gte_p50__and__volume_unit_gte_p50",
            "experiment_component_count": 2,
            "experiment_component_rule_ids": json.dumps(
                ["move_unit_gte_p50", "volume_unit_gte_p50"]
            ),
            "experiment_is_combo": True,
            "baseline_train_score": 0.03,
            "experiment_train_score": 0.05,
            "train_score_delta": 0.02,
            "baseline_validation_score": 0.01,
            "experiment_validation_score": -0.02,
            "validation_score_delta": -0.03,
            "baseline_validation_total_return": 0.02,
            "experiment_validation_total_return": -0.01,
            "validation_total_return_delta": -0.03,
            "baseline_validation_max_drawdown": None,
            "experiment_validation_max_drawdown": None,
            "validation_max_drawdown_delta": None,
            "baseline_validation_trade_count": 12,
            "experiment_validation_trade_count": 8,
            "validation_trade_count_delta": -4,
        }
    ]


def test_write_trailing_regime_gate_sweep_artifacts(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)
    output = write_trailing_regime_gate_sweep_artifacts(
        paths=paths,
        diagnostic_id="sweep",
        baseline_regime_walk_forward_id="baseline",
        experiment_regime_walk_forward_ids=["gated"],
        summary_rows=[{"regime_walk_forward_id": "baseline", "compound_validation_return": 0.045}],
        comparison_rows=[{"fold_index": 0, "validation_total_return_delta": -0.03}],
        git_commit="abc123",
        runtime_seconds=1.25,
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert output == paths.trailing_regime_diagnostic_dir("sweep")
    assert manifest["run_type"] == "trailing_stop_research_regime_gate_sweep"
    assert manifest["data_scope"] == "artifact_only_research_summary"
    assert manifest["baseline_regime_walk_forward_id"] == "baseline"
    assert manifest["experiment_regime_walk_forward_ids"] == ["gated"]
    assert pq.read_table(output / "gate_sweep_summary.parquet").to_pylist() == [
        {"regime_walk_forward_id": "baseline", "compound_validation_return": 0.045}
    ]
    assert pq.read_table(output / "gate_sweep_fold_comparison.parquet").to_pylist() == [
        {"fold_index": 0, "validation_total_return_delta": -0.03}
    ]
