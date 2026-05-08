from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing import (
    TrailingStopResult,
)


TRADE_COLUMNS = (
    "symbol",
    "signal_open_time",
    "decision_time",
    "entry_open_time",
    "exit_time",
    "entry_price",
    "exit_price",
    "stop_price_at_exit",
    "atr_at_entry",
    "atr_at_exit",
    "highest_high",
    "realized_return",
    "net_realized_return",
    "holding_bars",
    "ignored_signal_count",
    "move_unit",
    "volume_unit",
    "efficiency",
    "efficiency_threshold",
    "close_position",
    "body_ratio",
    "quote_volume",
    "volume_baseline",
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


def _float_values(rows: list[dict[str, Any]], key: str) -> np.ndarray:
    return np.array(
        [float(row[key]) for row in rows if row.get(key) is not None],
        dtype=np.float64,
    )


def _metric(values: np.ndarray, fn) -> float | None:
    if values.size == 0:
        return None
    return float(fn(values))


def build_trailing_summary(result: TrailingStopResult) -> dict[str, float | int | str | None]:
    returns = _float_values(result.trades, "realized_return")
    net_returns = _float_values(result.trades, "net_realized_return")
    holding_bars = _float_values(result.trades, "holding_bars")
    running_peak = np.maximum.accumulate(result.equity) if result.equity.size else np.array([])
    drawdown = result.equity / running_peak - 1.0 if running_peak.size else np.array([])
    return {
        "trade_count": len(result.trades),
        "winning_trade_count": int(np.count_nonzero(returns > 0.0)),
        "win_rate": _rounded(float(np.mean(returns > 0.0)) if returns.size else None),
        "mean_realized_return": _rounded(_metric(returns, np.mean)),
        "median_realized_return": _rounded(_metric(returns, np.median)),
        "mean_net_realized_return": _rounded(_metric(net_returns, np.mean)),
        "median_net_realized_return": _rounded(_metric(net_returns, np.median)),
        "average_holding_bars": _rounded(_metric(holding_bars, np.mean)),
        "total_ignored_signal_count": int(
            sum(int(row.get("ignored_signal_count") or 0) for row in result.trades)
        ),
        "equity_method": "equal_symbol_sleeves_closed_trade",
        "initial_equity": float(result.equity[0]) if result.equity.size else None,
        "final_equity": float(result.equity[-1]) if result.equity.size else None,
        "total_return": float(result.equity[-1] / result.equity[0] - 1.0)
        if result.equity.size
        else None,
        "period_count": int(result.period_returns.shape[0]),
        "mean_period_return": float(result.period_returns.mean())
        if result.period_returns.size
        else 0.0,
        "max_drawdown": float(abs(drawdown.min())) if drawdown.size else None,
    }


def _score_values(rows: list[dict[str, Any]]) -> list[float]:
    return [float(row["score"]) for row in rows if row.get("score") is not None]


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
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


def _rows_table(rows: list[dict[str, Any]]) -> pa.Table:
    if not rows:
        return pa.table({})
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    normalized_rows = [{key: row.get(key) for key in fieldnames} for row in rows]
    return pa.Table.from_pylist(normalized_rows)


def _signal_keep_rate(base_signal_count: int, filtered_signal_count: int) -> float | None:
    if base_signal_count == 0:
        return None
    return filtered_signal_count / base_signal_count


def _atr_multiplier_grid(
    *,
    atr_multiplier: float,
    atr_multipliers: tuple[float, ...] | None = None,
) -> list[float]:
    return [float(value) for value in (atr_multipliers or (atr_multiplier,))]


def _rule_parts(rule: Any) -> tuple[Any, ...]:
    parts = getattr(rule, "parts", None)
    if parts is None:
        return (rule,)
    return tuple(parts)


def _rule_payload(rule: Any) -> dict[str, Any]:
    parts = _rule_parts(rule)
    return {
        "rule_id": getattr(rule, "rule_id"),
        "feature_name": getattr(rule, "feature_name"),
        "direction": getattr(rule, "direction"),
        "quantile": getattr(rule, "quantile"),
        "threshold": _rounded(getattr(rule, "threshold")),
        "component_count": len(parts),
        "component_rule_ids": json.dumps([getattr(part, "rule_id") for part in parts]),
        "component_feature_names": json.dumps([getattr(part, "feature_name") for part in parts]),
        "component_directions": json.dumps([getattr(part, "direction") for part in parts]),
        "component_quantiles": json.dumps([getattr(part, "quantile") for part in parts]),
        "component_thresholds": json.dumps(
            [_rounded(getattr(part, "threshold")) for part in parts]
        ),
    }


def _selected_filter_payload(rule: Any, selected_train_row: dict[str, Any]) -> dict[str, Any]:
    payload = _rule_payload(rule)
    payload.update(
        {
            "threshold_source": "full_research_signal_distribution",
            "selection_scope": "research_only",
            "train_score": _rounded(selected_train_row.get("score")),
            "train_trade_count": selected_train_row.get("trade_count"),
            "train_base_signal_count": selected_train_row.get("base_signal_count"),
            "train_filtered_signal_count": selected_train_row.get("filtered_signal_count"),
            "train_signal_keep_rate": _rounded(selected_train_row.get("signal_keep_rate")),
            "train_total_return": _rounded(selected_train_row.get("total_return")),
            "train_max_drawdown": _rounded(selected_train_row.get("max_drawdown")),
        }
    )
    return payload


def _trade_schema() -> pa.Schema:
    return pa.schema(
        [
            pa.field("symbol", pa.string()),
            pa.field("signal_open_time", pa.string()),
            pa.field("decision_time", pa.string()),
            pa.field("entry_open_time", pa.string()),
            pa.field("exit_time", pa.string()),
            pa.field("entry_price", pa.float64()),
            pa.field("exit_price", pa.float64()),
            pa.field("stop_price_at_exit", pa.float64()),
            pa.field("atr_at_entry", pa.float64()),
            pa.field("atr_at_exit", pa.float64()),
            pa.field("highest_high", pa.float64()),
            pa.field("realized_return", pa.float64()),
            pa.field("net_realized_return", pa.float64()),
            pa.field("holding_bars", pa.int64()),
            pa.field("ignored_signal_count", pa.int64()),
            pa.field("move_unit", pa.float64()),
            pa.field("volume_unit", pa.float64()),
            pa.field("efficiency", pa.float64()),
            pa.field("efficiency_threshold", pa.float64()),
            pa.field("close_position", pa.float64()),
            pa.field("body_ratio", pa.float64()),
            pa.field("quote_volume", pa.float64()),
            pa.field("volume_baseline", pa.float64()),
        ]
    )


def _trades_table(rows: list[dict[str, Any]]) -> pa.Table:
    schema = _trade_schema()
    if not rows:
        return pa.Table.from_arrays(
            [pa.array([], type=field.type) for field in schema],
            schema=schema,
        )
    normalized = [{name: row.get(name) for name in schema.names} for row in rows]
    return pa.Table.from_pylist(normalized, schema=schema)


def _equity_table(open_times: np.ndarray, result: TrailingStopResult) -> pa.Table:
    period_returns = np.full(result.equity.shape[0], np.nan, dtype=np.float64)
    if result.period_returns.size:
        period_returns[1:] = result.period_returns
    return pa.table(
        {
            "open_time": [_json_safe(value) for value in open_times],
            "equity": result.equity.tolist(),
            "period_return": period_returns.tolist(),
            "open_position_count": result.positions.sum(axis=1).astype(np.int64).tolist(),
        }
    )


def _positions_table(
    symbols: tuple[str, ...],
    open_times: np.ndarray,
    result: TrailingStopResult,
) -> pa.Table:
    rows = []
    for t_index, open_time in enumerate(open_times):
        for s_index, symbol in enumerate(symbols):
            if result.positions[t_index, s_index]:
                rows.append(
                    {
                        "open_time": _json_safe(open_time),
                        "symbol": symbol,
                        "stop_price": float(result.stop_prices[t_index, s_index])
                        if np.isfinite(result.stop_prices[t_index, s_index])
                        else None,
                    }
                )
    if rows:
        return pa.Table.from_pylist(rows)
    return pa.table(
        {
            "open_time": pa.array([], type=pa.string()),
            "symbol": pa.array([], type=pa.string()),
            "stop_price": pa.array([], type=pa.float64()),
        }
    )


def write_trailing_run_artifacts(
    *,
    paths: VolumePriceEfficiencyPaths,
    run_id: str,
    config: VolumePriceEfficiencyConfig,
    result: TrailingStopResult,
    symbols: tuple[str, ...],
    open_times: np.ndarray,
    canonical_manifests: list[str],
    git_commit: str,
    runtime_seconds: float,
    data_split: dict[str, Any],
    atr_multiplier: float,
) -> Path:
    run_dir = paths.trailing_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = build_trailing_summary(result)
    manifest = {
        "strategy_name": config.strategy_name,
        "strategy_version": "v1",
        "run_type": "trailing_stop_holdout",
        "git_commit": git_commit,
        "config": config.model_dump(mode="json"),
        "config_hash": config.config_hash(),
        "atr_multiplier": atr_multiplier,
        "canonical_manifests": canonical_manifests,
        "symbol_count": len(symbols),
        "symbols": list(symbols),
        "runtime_seconds": runtime_seconds,
        "data_split": data_split,
        "outputs": {
            "summary": str(run_dir / "summary.json"),
            "trades": str(run_dir / "trades.parquet"),
            "equity_curve": str(run_dir / "equity_curve.parquet"),
            "daily_positions": str(run_dir / "daily_positions.parquet"),
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "summary.json", summary)
    pq.write_table(_trades_table(result.trades), run_dir / "trades.parquet")
    pq.write_table(_equity_table(open_times, result), run_dir / "equity_curve.parquet")
    pq.write_table(_positions_table(symbols, open_times, result), run_dir / "daily_positions.parquet")
    return run_dir


def write_trailing_regime_holdout_artifacts(
    *,
    paths: VolumePriceEfficiencyPaths,
    run_id: str,
    config: VolumePriceEfficiencyConfig,
    result: TrailingStopResult,
    symbols: tuple[str, ...],
    open_times: np.ndarray,
    canonical_manifests: list[str],
    git_commit: str,
    runtime_seconds: float,
    data_split: dict[str, Any],
    atr_multiplier: float,
    lookback_bars: int,
    quantiles: tuple[float, ...],
    feature_names: tuple[str, ...],
    min_trades: int,
    selected_rule: Any,
    selected_train_row: dict[str, Any],
    selection_rows: list[dict[str, Any]],
    holdout_base_signal_count: int,
    holdout_filtered_signal_count: int,
) -> Path:
    run_dir = paths.trailing_run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = build_trailing_summary(result)
    selected_filter = _selected_filter_payload(selected_rule, selected_train_row)
    holdout_signal_keep_rate = _signal_keep_rate(
        holdout_base_signal_count,
        holdout_filtered_signal_count,
    )
    manifest = {
        "strategy_name": config.strategy_name,
        "strategy_version": "v1",
        "run_type": "trailing_stop_regime_holdout",
        "data_scope": "holdout_only_final_production_test",
        "selection_scope": "research_only",
        "threshold_scope": "full_research_signal_distribution",
        "run_id": run_id,
        "git_commit": git_commit,
        "config": config.model_dump(mode="json"),
        "config_hash": config.config_hash(),
        "atr_multiplier": atr_multiplier,
        "lookback_bars": lookback_bars,
        "quantiles": list(quantiles),
        "feature_names": list(feature_names),
        "min_trades": min_trades,
        "selected_rule_id": selected_filter["rule_id"],
        "selected_filter": selected_filter,
        "holdout_base_signal_count": holdout_base_signal_count,
        "holdout_filtered_signal_count": holdout_filtered_signal_count,
        "holdout_signal_keep_rate": _rounded(holdout_signal_keep_rate),
        "canonical_manifests": canonical_manifests,
        "symbol_count": len(symbols),
        "symbols": list(symbols),
        "runtime_seconds": runtime_seconds,
        "data_split": data_split,
        "outputs": {
            "summary": str(run_dir / "summary.json"),
            "trades": str(run_dir / "trades.parquet"),
            "equity_curve": str(run_dir / "equity_curve.parquet"),
            "daily_positions": str(run_dir / "daily_positions.parquet"),
            "selected_filter": str(run_dir / "selected_filter.json"),
            "selection_summary": str(run_dir / "selection_summary.parquet"),
            "selection_summary_csv": str(run_dir / "selection_summary.csv"),
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "summary.json", summary)
    _write_json(run_dir / "selected_filter.json", selected_filter)
    pq.write_table(_trades_table(result.trades), run_dir / "trades.parquet")
    pq.write_table(_equity_table(open_times, result), run_dir / "equity_curve.parquet")
    pq.write_table(_positions_table(symbols, open_times, result), run_dir / "daily_positions.parquet")
    pq.write_table(_rows_table(selection_rows), run_dir / "selection_summary.parquet")
    _write_summary_csv(run_dir / "selection_summary.csv", selection_rows)
    return run_dir


def write_trailing_scan_artifacts(
    *,
    paths: VolumePriceEfficiencyPaths,
    scan_id: str,
    base_config: VolumePriceEfficiencyConfig,
    rows: list[dict[str, Any]],
    top_configs: list[dict[str, Any]],
    canonical_manifests: list[str],
    git_commit: str,
    runtime_seconds: float,
    symbol_count: int,
    data_split: dict[str, Any],
    atr_multiplier: float,
    min_trades: int,
    atr_multipliers: tuple[float, ...] | None = None,
) -> Path:
    scan_dir = paths.trailing_scan_dir(scan_id)
    scan_dir.mkdir(parents=True, exist_ok=True)
    scores = _score_values(rows)
    eligible_scores = _score_values(top_configs)
    eligible_count = sum(1 for row in rows if int(row.get("trade_count") or 0) >= min_trades)
    summary = {
        "scan_id": scan_id,
        "combination_count": len(rows),
        "eligible_combination_count": eligible_count,
        "runtime_seconds": runtime_seconds,
        "best_score": _rounded(max(eligible_scores or scores) if rows else None),
    }
    manifest = {
        "strategy_name": base_config.strategy_name,
        "strategy_version": "v1",
        "run_type": "trailing_stop_research_scan",
        "scan_id": scan_id,
        "git_commit": git_commit,
        "base_config": base_config.model_dump(mode="json"),
        "base_config_hash": base_config.config_hash(),
        "atr_multiplier": atr_multiplier,
        "atr_multipliers": _atr_multiplier_grid(
            atr_multiplier=atr_multiplier,
            atr_multipliers=atr_multipliers,
        ),
        "min_trades": min_trades,
        "canonical_manifests": canonical_manifests,
        "symbol_count": symbol_count,
        "runtime_seconds": runtime_seconds,
        "combination_count": len(rows),
        "data_split": data_split,
        "outputs": {
            "summary": str(scan_dir / "summary.json"),
            "summary_csv": str(scan_dir / "summary.csv"),
            "top_configs": str(scan_dir / "top_configs.json"),
        },
    }
    _write_json(scan_dir / "manifest.json", manifest)
    _write_json(scan_dir / "summary.json", summary)
    _write_json(scan_dir / "top_configs.json", top_configs)
    _write_summary_csv(scan_dir / "summary.csv", rows)
    return scan_dir
