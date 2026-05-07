from __future__ import annotations

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
