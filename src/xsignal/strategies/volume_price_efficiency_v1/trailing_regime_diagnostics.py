from __future__ import annotations

import csv
import json
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
