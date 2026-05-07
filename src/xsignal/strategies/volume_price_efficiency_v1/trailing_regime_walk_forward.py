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
        "threshold_source": "train_fold_signal_distribution",
        "train_score": _rounded(train.get("score")),
        "train_trade_count": train.get("trade_count"),
        "train_total_return": _rounded(train.get("total_return")),
        "train_max_drawdown": _rounded(train.get("max_drawdown")),
        "train_base_signal_count": train.get("base_signal_count"),
        "train_filtered_signal_count": train.get("filtered_signal_count"),
        "train_signal_keep_rate": _rounded(_safe_keep_rate(train)),
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
        "lookback_bars": lookback_bars,
        "train_days": train_days,
        "test_days": test_days,
        "step_days": step_days,
        "min_trades": min_trades,
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
            "top_filters": str(output_dir / "top_filters.json"),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "top_filters.json", top_filters)
    pq.write_table(_rows_table(fold_rows), output_dir / "fold_summary.parquet")
    pq.write_table(_rows_table(selection_rows), output_dir / "selection_summary.parquet")
    _write_csv(output_dir / "fold_summary.csv", fold_rows)
    return output_dir
