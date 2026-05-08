from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
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
from xsignal.strategies.volume_price_efficiency_v1.trailing_walk_forward import (
    WalkForwardFold,
)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_safe) + "\n")


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 12)


def _safe_keep_rate(row: dict[str, Any]) -> float | None:
    base = row.get("base_signal_count")
    filtered = row.get("filtered_signal_count")
    if base is None or filtered is None or int(base) == 0:
        return None
    return float(int(filtered) / int(base))


def _score_or_none(row: dict[str, Any]) -> float | None:
    value = row.get("score")
    return None if value is None else float(value)


def _json_float_values(value: Any) -> list[float | None]:
    if value is None:
        return []
    parsed = value if isinstance(value, list) else json.loads(str(value))
    return [None if item is None else float(item) for item in parsed]


def _json_rule_ids(value: Any, fallback_rule_id: str) -> list[str]:
    if value is None:
        return fallback_rule_id.split("__and__") if "__and__" in fallback_rule_id else []
    parsed = value if isinstance(value, list) else json.loads(str(value))
    rule_ids = [str(item) for item in parsed]
    return rule_ids or (fallback_rule_id.split("__and__") if "__and__" in fallback_rule_id else [])


def _component_count(row: dict[str, Any]) -> int:
    return int(row.get("component_count") or 0)


def _signal_keep_rate(row: dict[str, Any]) -> float | None:
    if row.get("signal_keep_rate") is not None:
        return float(row["signal_keep_rate"])
    return _safe_keep_rate(row)


def _rows_table(rows: list[dict[str, Any]]) -> pa.Table:
    if rows:
        fieldnames: list[str] = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
        normalized_rows = [{key: row.get(key) for key in fieldnames} for row in rows]
        return pa.Table.from_pylist(normalized_rows)
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


def _rule_component_payload(rule: RegimeFilterRule | None) -> dict[str, Any]:
    if rule is None:
        return {
            "component_count": 0,
            "component_rule_ids": "[]",
            "component_feature_names": "[]",
            "component_directions": "[]",
            "component_quantiles": "[]",
            "component_thresholds": "[]",
        }
    parts = rule.parts
    return {
        "component_count": len(parts),
        "component_rule_ids": json.dumps([part.rule_id for part in parts]),
        "component_feature_names": json.dumps([part.feature_name for part in parts]),
        "component_directions": json.dumps([part.direction for part in parts]),
        "component_quantiles": json.dumps([part.quantile for part in parts]),
        "component_thresholds": json.dumps([_rounded(part.threshold) for part in parts]),
    }


def build_regime_stability_summary(
    segment_rows: list[dict[str, Any]],
    *,
    split_count: int,
    min_trades: int,
    include_values: bool = False,
) -> dict[str, Any]:
    if split_count <= 0:
        raise ValueError("split_count must be positive")
    if min_trades < 0:
        raise ValueError("min_trades must be non-negative")

    eligible_scores = [
        score
        for row in segment_rows
        if int(row.get("trade_count") or 0) >= min_trades
        for score in [_score_or_none(row)]
        if score is not None
    ]
    mean_score = sum(eligible_scores) / len(eligible_scores) if eligible_scores else None
    min_score = min(eligible_scores) if eligible_scores else None
    summary = {
        "selection_method": "train_stability",
        "stability_split_count": split_count,
        "stability_min_trades": min_trades,
        "stability_evaluated_split_count": len(segment_rows),
        "stability_eligible_split_count": len(eligible_scores),
        "stability_positive_split_count": sum(score > 0.0 for score in eligible_scores),
        "stability_mean_score": _rounded(mean_score),
        "stability_min_score": _rounded(min_score),
    }
    if include_values:
        summary.update(
            {
                "stability_score_values": json.dumps(
                    [_rounded(_score_or_none(row)) for row in segment_rows]
                ),
                "stability_trade_count_values": json.dumps(
                    [int(row.get("trade_count") or 0) for row in segment_rows]
                ),
            }
        )
    return summary


def _best_by_score(rows: list[dict[str, Any]]) -> dict[str, Any] | None:
    scored = [row for row in rows if _score_or_none(row) is not None]
    if not scored:
        return None
    return sorted(
        scored,
        key=lambda row: (
            -float(row["score"]),
            str(row.get("rule_id")),
        ),
    )[0]


def _outperformance_count(
    candidate_scores: list[float | None],
    benchmark_scores: list[float | None],
) -> int:
    count = 0
    for candidate, benchmark in zip(candidate_scores, benchmark_scores, strict=False):
        if candidate is not None and benchmark is not None and candidate > benchmark:
            count += 1
    return count


def _best_segment_scores(rows: list[dict[str, Any]]) -> list[float | None]:
    score_lists = [_json_float_values(row.get("stability_score_values")) for row in rows]
    max_len = max((len(scores) for scores in score_lists), default=0)
    best_scores: list[float | None] = []
    for index in range(max_len):
        values = [scores[index] for scores in score_lists if index < len(scores)]
        finite_values = [value for value in values if value is not None]
        best_scores.append(max(finite_values) if finite_values else None)
    return best_scores


def _atr_multiplier_key(row: dict[str, Any]) -> float:
    return float(row.get("atr_multiplier") or 2.0)


def filter_combo_regime_candidates(
    rows: list[dict[str, Any]],
    *,
    allow_combo_selection: bool,
    min_score_lift_vs_best_component: float = 0.0,
    min_score_lift_vs_best_single: float = 0.0,
    min_component_outperformance_splits: int = 0,
    min_single_outperformance_splits: int = 0,
) -> list[dict[str, Any]]:
    if not allow_combo_selection:
        return [row for row in rows if _component_count(row) <= 1]
    if min_score_lift_vs_best_component < 0.0:
        raise ValueError("min_score_lift_vs_best_component must be non-negative")
    if min_score_lift_vs_best_single < 0.0:
        raise ValueError("min_score_lift_vs_best_single must be non-negative")
    if min_component_outperformance_splits < 0:
        raise ValueError("min_component_outperformance_splits must be non-negative")
    if min_single_outperformance_splits < 0:
        raise ValueError("min_single_outperformance_splits must be non-negative")

    row_by_rule_and_atr = {
        (str(row.get("rule_id")), _atr_multiplier_key(row)): row
        for row in rows
    }
    single_rows = [row for row in rows if _component_count(row) <= 1]

    filtered: list[dict[str, Any]] = []
    for row in rows:
        if _component_count(row) <= 1:
            filtered.append(row)
            continue

        candidate_score = _score_or_none(row)
        rule_id = str(row.get("rule_id"))
        atr_multiplier = _atr_multiplier_key(row)
        component_rule_ids = _json_rule_ids(row.get("component_rule_ids"), rule_id)
        component_rows = [
            row_by_rule_and_atr[(component_rule_id, atr_multiplier)]
            for component_rule_id in component_rule_ids
            if (component_rule_id, atr_multiplier) in row_by_rule_and_atr
        ]
        single_rows_for_atr = [
            single_row
            for single_row in single_rows
            if _atr_multiplier_key(single_row) == atr_multiplier
        ]
        best_single = _best_by_score(single_rows_for_atr)
        best_single_score = _score_or_none(best_single) if best_single is not None else None
        best_single_segment_scores = _best_segment_scores(single_rows_for_atr)
        best_component = _best_by_score(component_rows)
        best_component_score = _score_or_none(best_component) if best_component is not None else None
        if candidate_score is None or best_component_score is None or best_single_score is None:
            row["combo_gate_passed"] = False
            continue

        score_lift_vs_best_component = candidate_score - best_component_score
        score_lift_vs_best_single = candidate_score - best_single_score
        candidate_segment_scores = _json_float_values(row.get("stability_score_values"))
        best_component_segment_scores = _best_segment_scores(component_rows)
        component_outperformance_splits = _outperformance_count(
            candidate_segment_scores,
            best_component_segment_scores,
        )
        single_outperformance_splits = _outperformance_count(
            candidate_segment_scores,
            best_single_segment_scores,
        )
        gate_passed = (
            score_lift_vs_best_component >= min_score_lift_vs_best_component
            and score_lift_vs_best_single >= min_score_lift_vs_best_single
            and component_outperformance_splits >= min_component_outperformance_splits
            and single_outperformance_splits >= min_single_outperformance_splits
        )
        row.update(
            {
                "combo_best_component_rule_id": str(best_component.get("rule_id")),
                "combo_best_single_rule_id": str(best_single.get("rule_id")),
                "combo_score_lift_vs_best_component": _rounded(score_lift_vs_best_component),
                "combo_score_lift_vs_best_single": _rounded(score_lift_vs_best_single),
                "combo_component_outperformance_splits": component_outperformance_splits,
                "combo_single_outperformance_splits": single_outperformance_splits,
                "combo_gate_passed": gate_passed,
            }
        )
        if gate_passed:
            filtered.append(row)
    return filtered


def select_stable_regime_filters(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    min_trades: int,
    stability_min_positive_splits: int,
) -> list[dict[str, Any]]:
    if top_k <= 0:
        return []
    required_stable_splits = max(1, stability_min_positive_splits)
    eligible = []
    for row in rows:
        if row.get("rule_id") == "unfiltered" or int(row.get("trade_count") or 0) < min_trades:
            continue
        split_count = int(row.get("stability_split_count") or 0)
        eligible_split_count = int(row.get("stability_eligible_split_count") or 0)
        positive_split_count = int(row.get("stability_positive_split_count") or 0)
        if split_count <= 0 or eligible_split_count < required_stable_splits:
            continue
        if positive_split_count < required_stable_splits:
            continue
        if row.get("stability_min_score") is None or row.get("stability_mean_score") is None:
            continue
        eligible.append(row)
    return sorted(
        eligible,
        key=lambda row: (
            -(float(row["score"]) if row.get("score") is not None else float("-inf")),
            -float(row["stability_mean_score"]),
            -float(row["stability_min_score"]),
            str(row["rule_id"]),
        ),
    )[:top_k]


def build_regime_validation_trade_rows(
    *,
    regime_walk_forward_id: str,
    fold_index: int,
    fold: WalkForwardFold,
    selected_train_row: dict[str, Any],
    validation_row: dict[str, Any],
    selected_rule: RegimeFilterRule,
    trades: list[dict[str, Any]],
    trade_feature_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    for trade_index, trade in enumerate(trades):
        row = {
            "regime_walk_forward_id": regime_walk_forward_id,
            "fold_index": fold_index,
            "test_start": _json_safe(fold.test_start),
            "test_end": _json_safe(fold.test_end),
            "rule_id": selected_rule.rule_id,
            "feature_name": selected_rule.feature_name,
            "direction": selected_rule.direction,
            "quantile": selected_rule.quantile,
            "threshold": _rounded(selected_rule.threshold),
            "threshold_source": "train_fold_signal_distribution",
            "train_score": _rounded(selected_train_row.get("score")),
            "train_trade_count": selected_train_row.get("trade_count"),
            "train_signal_keep_rate": _rounded(_signal_keep_rate(selected_train_row)),
            "validation_score": _rounded(validation_row.get("score")),
            "validation_trade_count": validation_row.get("trade_count"),
            "validation_signal_keep_rate": _rounded(_signal_keep_rate(validation_row)),
        }
        if selected_train_row.get("atr_multiplier") is not None:
            row["atr_multiplier"] = selected_train_row["atr_multiplier"]
        row.update(trade)
        if trade_index < len(trade_feature_rows):
            row.update(trade_feature_rows[trade_index])
        rows.append(row)
    return rows


def build_regime_walk_forward_fold_row(
    *,
    regime_walk_forward_id: str,
    fold_index: int,
    fold: WalkForwardFold,
    selected_train_row: dict[str, Any] | None,
    validation_row: dict[str, Any] | None,
    selected_rule: RegimeFilterRule | None,
    selected_config: VolumePriceEfficiencyConfig,
) -> dict[str, Any]:
    train = selected_train_row or {}
    validation = validation_row or {}
    row = {
        "regime_walk_forward_id": regime_walk_forward_id,
        "fold_index": fold_index,
        "train_start": _json_safe(fold.train_start),
        "train_end": _json_safe(fold.train_end),
        "test_start": _json_safe(fold.test_start),
        "test_end": _json_safe(fold.test_end),
        "train_bar_count": len(fold.train_indices),
        "validation_bar_count": len(fold.test_indices),
        "rule_id": selected_rule.rule_id if selected_rule is not None else None,
        "feature_name": selected_rule.feature_name if selected_rule is not None else None,
        "direction": selected_rule.direction if selected_rule is not None else None,
        "quantile": selected_rule.quantile if selected_rule is not None else None,
        "threshold": _rounded(selected_rule.threshold if selected_rule is not None else None),
        **_rule_component_payload(selected_rule),
        "threshold_source": "train_fold_signal_distribution",
        "selection_method": train.get("selection_method", "train_total_score"),
        "train_score": _rounded(train.get("score")),
        "train_trade_count": train.get("trade_count"),
        "train_total_return": _rounded(train.get("total_return")),
        "train_max_drawdown": _rounded(train.get("max_drawdown")),
        "train_base_signal_count": train.get("base_signal_count"),
        "train_filtered_signal_count": train.get("filtered_signal_count"),
        "train_signal_keep_rate": _rounded(_safe_keep_rate(train)),
        "train_stability_split_count": train.get("stability_split_count"),
        "train_stability_min_trades": train.get("stability_min_trades"),
        "train_stability_eligible_split_count": train.get("stability_eligible_split_count"),
        "train_stability_positive_split_count": train.get("stability_positive_split_count"),
        "train_stability_mean_score": _rounded(train.get("stability_mean_score")),
        "train_stability_min_score": _rounded(train.get("stability_min_score")),
        "validation_score": _rounded(validation.get("score")),
        "validation_trade_count": validation.get("trade_count"),
        "validation_win_rate": _rounded(validation.get("win_rate")),
        "validation_mean_net_realized_return": _rounded(
            validation.get("mean_net_realized_return")
        ),
        "validation_median_net_realized_return": _rounded(
            validation.get("median_net_realized_return")
        ),
        "validation_average_holding_bars": _rounded(validation.get("average_holding_bars")),
        "validation_total_ignored_signal_count": validation.get("total_ignored_signal_count"),
        "validation_final_equity": _rounded(validation.get("final_equity")),
        "validation_total_return": _rounded(validation.get("total_return")),
        "validation_max_drawdown": _rounded(validation.get("max_drawdown")),
        "validation_base_signal_count": validation.get("base_signal_count"),
        "validation_filtered_signal_count": validation.get("filtered_signal_count"),
        "validation_signal_keep_rate": _rounded(_safe_keep_rate(validation)),
        "atr_multiplier": train.get("atr_multiplier"),
        "efficiency_percentile": selected_config.efficiency_percentile,
        "min_move_unit": selected_config.min_move_unit,
        "min_volume_unit": selected_config.min_volume_unit,
        "min_close_position": selected_config.min_close_position,
        "min_body_ratio": selected_config.min_body_ratio,
        "fee_bps": selected_config.fee_bps,
        "slippage_bps": selected_config.slippage_bps,
        "baseline_seed": selected_config.baseline_seed,
    }
    return row


def write_trailing_regime_walk_forward_artifacts(
    *,
    paths: VolumePriceEfficiencyPaths,
    regime_walk_forward_id: str,
    config: VolumePriceEfficiencyConfig,
    fold_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    validation_trade_rows: list[dict[str, Any]],
    top_filters: list[dict[str, Any]],
    canonical_manifests: list[str],
    git_commit: str,
    runtime_seconds: float,
    symbol_count: int,
    data_split: dict[str, Any],
    atr_multiplier: float,
    lookback_bars: int,
    train_days: int,
    test_days: int,
    step_days: int,
    min_trades: int,
    quantiles: tuple[float, ...],
    feature_names: tuple[str, ...],
    stability_splits: int = 1,
    stability_min_trades: int = 0,
    stability_min_positive_splits: int = 0,
    max_rule_size: int = 1,
    combo_seed_top_k: int = 0,
    allow_combo_selection: bool = False,
    combo_min_score_lift_vs_best_component: float = 0.0,
    combo_min_score_lift_vs_best_single: float = 0.0,
    combo_min_component_outperformance_splits: int = 0,
    combo_min_single_outperformance_splits: int = 0,
    atr_multipliers: tuple[float, ...] | None = None,
    pyramid_add_step_atr: float | None = None,
    pyramid_max_adds: int = 0,
) -> Path:
    output_dir = paths.trailing_regime_walk_forward_dir(regime_walk_forward_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "strategy_name": config.strategy_name,
        "strategy_version": "v1",
        "run_type": "trailing_stop_research_regime_walk_forward",
        "data_scope": "research_only",
        "threshold_scope": "per_fold_train_signal_distribution",
        "regime_walk_forward_id": regime_walk_forward_id,
        "git_commit": git_commit,
        "config": config.model_dump(mode="json"),
        "config_hash": config.config_hash(),
        "atr_multiplier": atr_multiplier,
        "atr_multipliers": list(atr_multipliers or (atr_multiplier,)),
        "pyramid_add_step_atr": pyramid_add_step_atr,
        "pyramid_max_adds": pyramid_max_adds,
        "lookback_bars": lookback_bars,
        "train_days": train_days,
        "test_days": test_days,
        "step_days": step_days,
        "min_trades": min_trades,
        "selection_method": "train_stability" if stability_splits > 1 else "train_total_score",
        "stability_splits": stability_splits,
        "stability_min_trades": stability_min_trades,
        "stability_min_positive_splits": stability_min_positive_splits,
        "max_rule_size": max_rule_size,
        "combo_seed_top_k": combo_seed_top_k,
        "allow_combo_selection": allow_combo_selection,
        "combo_min_score_lift_vs_best_component": combo_min_score_lift_vs_best_component,
        "combo_min_score_lift_vs_best_single": combo_min_score_lift_vs_best_single,
        "combo_min_component_outperformance_splits": combo_min_component_outperformance_splits,
        "combo_min_single_outperformance_splits": combo_min_single_outperformance_splits,
        "quantiles": list(quantiles),
        "feature_names": list(feature_names),
        "canonical_manifests": canonical_manifests,
        "symbol_count": symbol_count,
        "runtime_seconds": runtime_seconds,
        "fold_count": len(fold_rows),
        "selection_count": len(selection_rows),
        "data_split": data_split,
        "outputs": {
            "fold_summary": str(output_dir / "fold_summary.parquet"),
            "fold_summary_csv": str(output_dir / "fold_summary.csv"),
            "selection_summary": str(output_dir / "selection_summary.parquet"),
            "validation_trades": str(output_dir / "validation_trades.parquet"),
            "validation_trades_csv": str(output_dir / "validation_trades.csv"),
            "top_filters": str(output_dir / "top_filters.json"),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "top_filters.json", top_filters)
    pq.write_table(_rows_table(fold_rows), output_dir / "fold_summary.parquet")
    pq.write_table(_rows_table(selection_rows), output_dir / "selection_summary.parquet")
    pq.write_table(_rows_table(validation_trade_rows), output_dir / "validation_trades.parquet")
    _write_csv(output_dir / "fold_summary.csv", fold_rows)
    _write_csv(output_dir / "validation_trades.csv", validation_trade_rows)
    return output_dir
