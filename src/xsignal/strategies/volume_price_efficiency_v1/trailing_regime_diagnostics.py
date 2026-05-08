from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_safe) + "\n")


def _rows_table(rows: list[dict[str, Any]]) -> pa.Table:
    if rows:
        return pa.Table.from_pylist(rows)
    return pa.table({})


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _rounded(value: Any) -> float | None:
    return None if value is None else round(float(value), 12)


def _int_or_none(value: Any) -> int | None:
    return None if value is None else int(value)


def _json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    return [str(item) for item in json.loads(str(value))]


def _component_count(row: dict[str, Any]) -> int:
    return int(row.get("component_count") or 0)


def _score(row: dict[str, Any], field: str = "score") -> float | None:
    value = row.get(field)
    return None if value is None else float(value)


def _best_by_score(rows: list[dict[str, Any]], *, score_field: str = "score") -> dict[str, Any] | None:
    scored = [row for row in rows if _score(row, score_field) is not None]
    if not scored:
        return None
    return sorted(
        scored,
        key=lambda row: (
            -float(row[score_field]),
            str(row.get("rule_id")),
        ),
    )[0]


def build_combo_candidate_diagnostic_rows(
    selection_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fold_indices = sorted({int(row["fold_index"]) for row in selection_rows})
    for fold_index in fold_indices:
        fold_rows = [row for row in selection_rows if int(row["fold_index"]) == fold_index]
        row_by_rule_id = {str(row["rule_id"]): row for row in fold_rows}
        single_rows = [row for row in fold_rows if _component_count(row) <= 1]
        best_single = _best_by_score(single_rows)
        for row in fold_rows:
            if _component_count(row) <= 1:
                continue
            component_rule_ids = _json_list(row.get("component_rule_ids"))
            if not component_rule_ids:
                component_rule_ids = str(row["rule_id"]).split("__and__")
            component_rows = [
                row_by_rule_id[rule_id]
                for rule_id in component_rule_ids
                if rule_id in row_by_rule_id
            ]
            best_component = _best_by_score(component_rows)
            train_score = _score(row)
            best_component_score = _score(best_component) if best_component is not None else None
            best_single_score = _score(best_single) if best_single is not None else None
            rows.append(
                {
                    "fold_index": fold_index,
                    "rule_id": str(row["rule_id"]),
                    "component_count": _component_count(row),
                    "component_rule_ids": json.dumps(component_rule_ids),
                    "train_score": _rounded(train_score),
                    "train_trade_count": _int_or_none(row.get("trade_count")),
                    "train_signal_keep_rate": _rounded(row.get("signal_keep_rate")),
                    "best_component_rule_id": (
                        str(best_component["rule_id"]) if best_component is not None else None
                    ),
                    "best_component_score": _rounded(best_component_score),
                    "best_component_trade_count": (
                        _int_or_none(best_component.get("trade_count"))
                        if best_component is not None
                        else None
                    ),
                    "score_lift_vs_best_component": _rounded(
                        None
                        if train_score is None or best_component_score is None
                        else train_score - best_component_score
                    ),
                    "best_single_rule_id": str(best_single["rule_id"]) if best_single is not None else None,
                    "best_single_score": _rounded(best_single_score),
                    "score_lift_vs_best_single": _rounded(
                        None
                        if train_score is None or best_single_score is None
                        else train_score - best_single_score
                    ),
                    "stability_positive_split_count": _int_or_none(
                        row.get("stability_positive_split_count")
                    ),
                    "stability_split_count": _int_or_none(row.get("stability_split_count")),
                }
            )
    return rows


def build_selected_rule_comparison_rows(
    *,
    baseline_fold_rows: list[dict[str, Any]],
    experiment_fold_rows: list[dict[str, Any]],
    baseline_label: str,
    experiment_label: str,
) -> list[dict[str, Any]]:
    baseline_by_fold = {int(row["fold_index"]): row for row in baseline_fold_rows}
    experiment_by_fold = {int(row["fold_index"]): row for row in experiment_fold_rows}
    rows: list[dict[str, Any]] = []
    for fold_index in sorted(set(baseline_by_fold) & set(experiment_by_fold)):
        baseline = baseline_by_fold[fold_index]
        experiment = experiment_by_fold[fold_index]
        experiment_component_count = _component_count(experiment)
        baseline_train_score = _score(baseline, "train_score")
        experiment_train_score = _score(experiment, "train_score")
        baseline_validation_score = _score(baseline, "validation_score")
        experiment_validation_score = _score(experiment, "validation_score")
        baseline_total_return = _score(baseline, "validation_total_return")
        experiment_total_return = _score(experiment, "validation_total_return")
        baseline_max_drawdown = _score(baseline, "validation_max_drawdown")
        experiment_max_drawdown = _score(experiment, "validation_max_drawdown")
        baseline_trade_count = baseline.get("validation_trade_count")
        experiment_trade_count = experiment.get("validation_trade_count")
        rows.append(
            {
                "fold_index": fold_index,
                "baseline_label": baseline_label,
                "experiment_label": experiment_label,
                "baseline_rule_id": str(baseline.get("rule_id")),
                "experiment_rule_id": str(experiment.get("rule_id")),
                "experiment_component_count": experiment_component_count,
                "experiment_component_rule_ids": json.dumps(
                    _json_list(experiment.get("component_rule_ids"))
                ),
                "experiment_is_combo": experiment_component_count > 1,
                "train_score_delta": _rounded(
                    None
                    if baseline_train_score is None or experiment_train_score is None
                    else experiment_train_score - baseline_train_score
                ),
                "validation_score_delta": _rounded(
                    None
                    if baseline_validation_score is None or experiment_validation_score is None
                    else experiment_validation_score - baseline_validation_score
                ),
                "validation_total_return_delta": _rounded(
                    None
                    if baseline_total_return is None or experiment_total_return is None
                    else experiment_total_return - baseline_total_return
                ),
                "validation_max_drawdown_delta": _rounded(
                    None
                    if baseline_max_drawdown is None or experiment_max_drawdown is None
                    else experiment_max_drawdown - baseline_max_drawdown
                ),
                "validation_trade_count_delta": (
                    None
                    if baseline_trade_count is None or experiment_trade_count is None
                    else int(experiment_trade_count) - int(baseline_trade_count)
                ),
            }
        )
    return rows


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    parsed = float(value)
    return parsed if math.isfinite(parsed) else None


def _validation_return_values(fold_rows: list[dict[str, Any]]) -> list[tuple[int, float]]:
    values = []
    for row in fold_rows:
        value = _finite_float_or_none(row.get("validation_total_return"))
        if value is not None:
            values.append((int(row["fold_index"]), value))
    return values


def _compound_validation_return(fold_rows: list[dict[str, Any]]) -> float | None:
    values = _validation_return_values(fold_rows)
    if not values:
        return None
    compound = 1.0
    for _fold_index, value in values:
        compound *= 1.0 + value
    return _rounded(compound - 1.0)


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _ratio(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _gate_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
    return bool(value)


def build_gate_sweep_fold_comparison_rows(
    *,
    baseline_regime_walk_forward_id: str,
    experiment_regime_walk_forward_id: str,
    baseline_fold_rows: list[dict[str, Any]],
    experiment_fold_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_by_fold = {int(row["fold_index"]): row for row in baseline_fold_rows}
    experiment_by_fold = {int(row["fold_index"]): row for row in experiment_fold_rows}
    rows: list[dict[str, Any]] = []
    for fold_index in sorted(set(baseline_by_fold) & set(experiment_by_fold)):
        baseline = baseline_by_fold[fold_index]
        experiment = experiment_by_fold[fold_index]
        baseline_train_score = _score(baseline, "train_score")
        experiment_train_score = _score(experiment, "train_score")
        baseline_validation_score = _score(baseline, "validation_score")
        experiment_validation_score = _score(experiment, "validation_score")
        baseline_total_return = _score(baseline, "validation_total_return")
        experiment_total_return = _score(experiment, "validation_total_return")
        baseline_max_drawdown = _score(baseline, "validation_max_drawdown")
        experiment_max_drawdown = _score(experiment, "validation_max_drawdown")
        baseline_trade_count = baseline.get("validation_trade_count")
        experiment_trade_count = experiment.get("validation_trade_count")
        experiment_component_count = _component_count(experiment)
        rows.append(
            {
                "fold_index": fold_index,
                "baseline_regime_walk_forward_id": baseline_regime_walk_forward_id,
                "experiment_regime_walk_forward_id": experiment_regime_walk_forward_id,
                "baseline_rule_id": str(baseline.get("rule_id")),
                "experiment_rule_id": str(experiment.get("rule_id")),
                "experiment_component_count": experiment_component_count,
                "experiment_component_rule_ids": json.dumps(
                    _json_list(experiment.get("component_rule_ids"))
                ),
                "experiment_is_combo": experiment_component_count > 1,
                "baseline_train_score": _rounded(baseline_train_score),
                "experiment_train_score": _rounded(experiment_train_score),
                "train_score_delta": _rounded(
                    None
                    if baseline_train_score is None or experiment_train_score is None
                    else experiment_train_score - baseline_train_score
                ),
                "baseline_validation_score": _rounded(baseline_validation_score),
                "experiment_validation_score": _rounded(experiment_validation_score),
                "validation_score_delta": _rounded(
                    None
                    if baseline_validation_score is None or experiment_validation_score is None
                    else experiment_validation_score - baseline_validation_score
                ),
                "baseline_validation_total_return": _rounded(baseline_total_return),
                "experiment_validation_total_return": _rounded(experiment_total_return),
                "validation_total_return_delta": _rounded(
                    None
                    if baseline_total_return is None or experiment_total_return is None
                    else experiment_total_return - baseline_total_return
                ),
                "baseline_validation_max_drawdown": _rounded(baseline_max_drawdown),
                "experiment_validation_max_drawdown": _rounded(experiment_max_drawdown),
                "validation_max_drawdown_delta": _rounded(
                    None
                    if baseline_max_drawdown is None or experiment_max_drawdown is None
                    else experiment_max_drawdown - baseline_max_drawdown
                ),
                "baseline_validation_trade_count": _int_or_none(baseline_trade_count),
                "experiment_validation_trade_count": _int_or_none(experiment_trade_count),
                "validation_trade_count_delta": (
                    None
                    if baseline_trade_count is None or experiment_trade_count is None
                    else int(experiment_trade_count) - int(baseline_trade_count)
                ),
            }
        )
    return rows


def build_gate_sweep_summary_rows(
    *,
    baseline_regime_walk_forward_id: str,
    run_artifacts: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    baseline_artifact = next(
        (
            artifact
            for artifact in run_artifacts
            if str(artifact["regime_walk_forward_id"]) == baseline_regime_walk_forward_id
        ),
        None,
    )
    if baseline_artifact is None:
        raise ValueError("baseline_regime_walk_forward_id is not present in run_artifacts")

    baseline_fold_rows = list(baseline_artifact.get("fold_rows") or [])
    baseline_compound_return = _compound_validation_return(baseline_fold_rows)
    rows = []
    for artifact in run_artifacts:
        regime_walk_forward_id = str(artifact["regime_walk_forward_id"])
        manifest = dict(artifact.get("manifest") or {})
        fold_rows = list(artifact.get("fold_rows") or [])
        selection_rows = list(artifact.get("selection_rows") or [])
        validation_values = [value for _fold_index, value in _validation_return_values(fold_rows)]
        selected_combo_fold_count = sum(_component_count(row) > 1 for row in fold_rows)
        combo_candidate_rows = [row for row in selection_rows if _component_count(row) > 1]
        combo_gate_values = [_gate_bool(row.get("combo_gate_passed")) for row in combo_candidate_rows]
        comparison_rows = (
            []
            if regime_walk_forward_id == baseline_regime_walk_forward_id
            else build_gate_sweep_fold_comparison_rows(
                baseline_regime_walk_forward_id=baseline_regime_walk_forward_id,
                experiment_regime_walk_forward_id=regime_walk_forward_id,
                baseline_fold_rows=baseline_fold_rows,
                experiment_fold_rows=fold_rows,
            )
        )
        deltas = [
            row
            for row in comparison_rows
            if row.get("validation_total_return_delta") is not None
        ]
        worst_delta = min(deltas, key=lambda row: float(row["validation_total_return_delta"])) if deltas else None
        best_delta = max(deltas, key=lambda row: float(row["validation_total_return_delta"])) if deltas else None
        compound_return = _compound_validation_return(fold_rows)
        compound_delta = (
            None
            if compound_return is None or baseline_compound_return is None
            else compound_return - baseline_compound_return
        )
        rows.append(
            {
                "regime_walk_forward_id": regime_walk_forward_id,
                "is_baseline": regime_walk_forward_id == baseline_regime_walk_forward_id,
                "run_type": manifest.get("run_type"),
                "data_scope": manifest.get("data_scope"),
                "allow_combo_selection": manifest.get("allow_combo_selection"),
                "max_rule_size": manifest.get("max_rule_size"),
                "combo_seed_top_k": manifest.get("combo_seed_top_k"),
                "combo_min_score_lift_vs_best_component": manifest.get(
                    "combo_min_score_lift_vs_best_component"
                ),
                "combo_min_score_lift_vs_best_single": manifest.get(
                    "combo_min_score_lift_vs_best_single"
                ),
                "combo_min_component_outperformance_splits": manifest.get(
                    "combo_min_component_outperformance_splits"
                ),
                "combo_min_single_outperformance_splits": manifest.get(
                    "combo_min_single_outperformance_splits"
                ),
                "stability_splits": manifest.get("stability_splits"),
                "stability_min_trades": manifest.get("stability_min_trades"),
                "stability_min_positive_splits": manifest.get(
                    "stability_min_positive_splits"
                ),
                "min_trades": manifest.get("min_trades"),
                "train_days": manifest.get("train_days"),
                "test_days": manifest.get("test_days"),
                "step_days": manifest.get("step_days"),
                "fold_count": len(fold_rows),
                "validation_return_fold_count": len(validation_values),
                "compound_validation_return": _rounded(compound_return),
                "compound_validation_return_delta_vs_baseline": _rounded(compound_delta),
                "mean_validation_total_return": _rounded(_mean(validation_values)),
                "min_validation_total_return": _rounded(
                    min(validation_values) if validation_values else None
                ),
                "max_validation_total_return": _rounded(
                    max(validation_values) if validation_values else None
                ),
                "selected_combo_fold_count": selected_combo_fold_count,
                "selected_single_fold_count": len(fold_rows) - selected_combo_fold_count,
                "selected_combo_fold_rate": _rounded(
                    _ratio(selected_combo_fold_count, len(fold_rows))
                ),
                "selection_candidate_count": len(selection_rows),
                "combo_candidate_count": len(combo_candidate_rows),
                "combo_gate_passed_count": sum(value is True for value in combo_gate_values),
                "combo_gate_failed_count": sum(value is False for value in combo_gate_values),
                "combo_gate_missing_count": sum(value is None for value in combo_gate_values),
                "worst_delta_fold_index_vs_baseline": (
                    None if worst_delta is None else int(worst_delta["fold_index"])
                ),
                "worst_validation_total_return_delta_vs_baseline": _rounded(
                    None if worst_delta is None else worst_delta["validation_total_return_delta"]
                ),
                "best_delta_fold_index_vs_baseline": (
                    None if best_delta is None else int(best_delta["fold_index"])
                ),
                "best_validation_total_return_delta_vs_baseline": _rounded(
                    None if best_delta is None else best_delta["validation_total_return_delta"]
                ),
            }
        )
    return rows


def write_trailing_regime_gate_sweep_artifacts(
    *,
    paths: VolumePriceEfficiencyPaths,
    diagnostic_id: str,
    baseline_regime_walk_forward_id: str,
    experiment_regime_walk_forward_ids: list[str],
    summary_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    git_commit: str,
    runtime_seconds: float,
) -> Path:
    output_dir = paths.trailing_regime_diagnostic_dir(diagnostic_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "strategy_name": "volume_price_efficiency",
        "strategy_version": "v1",
        "run_type": "trailing_stop_research_regime_gate_sweep",
        "data_scope": "artifact_only_research_summary",
        "diagnostic_id": diagnostic_id,
        "baseline_regime_walk_forward_id": baseline_regime_walk_forward_id,
        "experiment_regime_walk_forward_ids": experiment_regime_walk_forward_ids,
        "git_commit": git_commit,
        "runtime_seconds": runtime_seconds,
        "run_count": len(summary_rows),
        "comparison_count": len(comparison_rows),
        "outputs": {
            "gate_sweep_summary": str(output_dir / "gate_sweep_summary.parquet"),
            "gate_sweep_summary_csv": str(output_dir / "gate_sweep_summary.csv"),
            "gate_sweep_fold_comparison": str(
                output_dir / "gate_sweep_fold_comparison.parquet"
            ),
            "gate_sweep_fold_comparison_csv": str(
                output_dir / "gate_sweep_fold_comparison.csv"
            ),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    pq.write_table(_rows_table(summary_rows), output_dir / "gate_sweep_summary.parquet")
    pq.write_table(
        _rows_table(comparison_rows),
        output_dir / "gate_sweep_fold_comparison.parquet",
    )
    _write_csv(output_dir / "gate_sweep_summary.csv", summary_rows)
    _write_csv(output_dir / "gate_sweep_fold_comparison.csv", comparison_rows)
    return output_dir


def write_trailing_regime_diagnostic_artifacts(
    *,
    paths: VolumePriceEfficiencyPaths,
    diagnostic_id: str,
    source_regime_walk_forward_id: str,
    combo_selection_regime_walk_forward_id: str | None,
    candidate_rows: list[dict[str, Any]],
    comparison_rows: list[dict[str, Any]],
    git_commit: str,
    runtime_seconds: float,
) -> Path:
    output_dir = paths.trailing_regime_diagnostic_dir(diagnostic_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "strategy_name": "volume_price_efficiency",
        "strategy_version": "v1",
        "run_type": "trailing_regime_combo_diagnostics",
        "data_scope": "artifact_only_research_summary",
        "diagnostic_id": diagnostic_id,
        "source_regime_walk_forward_id": source_regime_walk_forward_id,
        "combo_selection_regime_walk_forward_id": combo_selection_regime_walk_forward_id,
        "git_commit": git_commit,
        "runtime_seconds": runtime_seconds,
        "candidate_count": len(candidate_rows),
        "comparison_count": len(comparison_rows),
        "outputs": {
            "combo_candidate_diagnostics": str(output_dir / "combo_candidate_diagnostics.parquet"),
            "combo_candidate_diagnostics_csv": str(output_dir / "combo_candidate_diagnostics.csv"),
            "selected_rule_comparison": str(output_dir / "selected_rule_comparison.parquet"),
            "selected_rule_comparison_csv": str(output_dir / "selected_rule_comparison.csv"),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    pq.write_table(_rows_table(candidate_rows), output_dir / "combo_candidate_diagnostics.parquet")
    pq.write_table(_rows_table(comparison_rows), output_dir / "selected_rule_comparison.parquet")
    _write_csv(output_dir / "combo_candidate_diagnostics.csv", candidate_rows)
    _write_csv(output_dir / "selected_rule_comparison.csv", comparison_rows)
    return output_dir
