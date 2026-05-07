from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
from xsignal.strategies.momentum_rotation_v1.kernel import BacktestResult
from xsignal.strategies.momentum_rotation_v1.paths import MomentumRotationPaths


def _json_safe(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_safe) + "\n")


def write_run_artifacts(
    *,
    paths: MomentumRotationPaths,
    run_id: str,
    config: MomentumRotationConfig,
    symbols: tuple[str, ...],
    rebalance_times,
    result: BacktestResult,
    canonical_manifests: list[str],
    git_commit: str,
    runtime_seconds: float,
) -> Path:
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "initial_equity": float(result.equity[0]),
        "final_equity": float(result.equity[-1]),
        "total_return": float(result.equity[-1] / result.equity[0] - 1.0),
        "period_count": int(result.period_returns.shape[0]),
        "mean_period_return": float(result.period_returns.mean())
        if result.period_returns.size
        else 0.0,
        "total_cost": float(result.costs.sum()),
    }
    manifest = {
        "strategy_name": config.strategy_name,
        "strategy_version": "v1",
        "git_commit": git_commit,
        "config": config.model_dump(mode="json"),
        "config_hash": config.config_hash(),
        "canonical_manifests": canonical_manifests,
        "symbol_count": len(symbols),
        "symbols": list(symbols),
        "runtime_seconds": runtime_seconds,
        "outputs": {
            "summary": str(run_dir / "summary.json"),
            "equity_curve": str(run_dir / "equity_curve.parquet"),
            "daily_positions": str(run_dir / "daily_positions.parquet"),
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "summary.json", summary)
    equity_table = pa.table(
        {
            "rebalance_time": [_json_safe(value) for value in rebalance_times],
            "equity": result.equity.tolist(),
            "turnover": result.turnover.tolist(),
            "cost": result.costs.tolist(),
        }
    )
    pq.write_table(equity_table, run_dir / "equity_curve.parquet")
    position_rows = []
    for t_index, rebalance_time in enumerate(rebalance_times):
        for s_index, symbol in enumerate(symbols):
            weight = float(result.weights[t_index, s_index])
            if weight != 0.0:
                position_rows.append(
                    {
                        "rebalance_time": _json_safe(rebalance_time),
                        "symbol": symbol,
                        "weight": weight,
                    }
                )
    pq.write_table(pa.Table.from_pylist(position_rows), run_dir / "daily_positions.parquet")
    return run_dir
