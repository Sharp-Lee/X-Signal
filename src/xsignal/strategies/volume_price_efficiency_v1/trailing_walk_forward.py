from __future__ import annotations

import csv
import json
from dataclasses import dataclass, replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.features import FeatureArrays
from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)


@dataclass(frozen=True)
class WalkForwardFold:
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def slice_ohlcv_arrays(arrays: OhlcvArrays, indices: tuple[int, ...] | np.ndarray) -> OhlcvArrays:
    indexer = indices if isinstance(indices, np.ndarray) else np.array(indices, dtype=np.int64)
    return replace(
        arrays,
        open_times=arrays.open_times[indexer],
        open=arrays.open[indexer],
        high=arrays.high[indexer],
        low=arrays.low[indexer],
        close=arrays.close[indexer],
        quote_volume=arrays.quote_volume[indexer],
        quality=arrays.quality[indexer],
    )


def slice_feature_arrays(features: FeatureArrays, indices: tuple[int, ...] | np.ndarray) -> FeatureArrays:
    indexer = indices if isinstance(indices, np.ndarray) else np.array(indices, dtype=np.int64)
    return replace(
        features,
        true_range=features.true_range[indexer],
        atr=features.atr[indexer],
        move_unit=features.move_unit[indexer],
        volume_baseline=features.volume_baseline[indexer],
        volume_unit=features.volume_unit[indexer],
        efficiency=features.efficiency[indexer],
        efficiency_threshold=features.efficiency_threshold[indexer],
        close_position=features.close_position[indexer],
        body_ratio=features.body_ratio[indexer],
        signal=features.signal[indexer],
    )


def _as_utc_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, np.datetime64):
        seconds = value.astype("datetime64[s]").astype("int64")
        return datetime.fromtimestamp(int(seconds), tz=timezone.utc)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    raise TypeError(f"unsupported open_time type: {type(value)!r}")


def _json_safe(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_safe) + "\n")


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 12)


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


def build_walk_forward_folds(
    open_times: list[object] | tuple[object, ...] | np.ndarray,
    *,
    train_days: int,
    test_days: int,
    step_days: int | None = None,
) -> list[WalkForwardFold]:
    if train_days <= 0:
        raise ValueError("train_days must be positive")
    if test_days <= 0:
        raise ValueError("test_days must be positive")
    if step_days is None:
        step_days = test_days
    if step_days <= 0:
        raise ValueError("step_days must be positive")
    if len(open_times) == 0:
        raise ValueError("cannot build walk-forward folds from empty open_times")

    times = tuple(_as_utc_datetime(value) for value in open_times)
    train_delta = timedelta(days=train_days)
    test_delta = timedelta(days=test_days)
    step_delta = timedelta(days=step_days)
    first_time = times[0]
    last_time = times[-1]
    test_start = first_time + train_delta
    folds: list[WalkForwardFold] = []

    while test_start <= last_time:
        train_start = test_start - train_delta
        test_end = test_start + test_delta
        train_indices = tuple(
            index for index, open_time in enumerate(times) if train_start <= open_time < test_start
        )
        test_indices = tuple(
            index for index, open_time in enumerate(times) if test_start <= open_time < test_end
        )
        if train_indices and test_indices:
            folds.append(
                WalkForwardFold(
                    train_start=train_start,
                    train_end=times[train_indices[-1]],
                    test_start=test_start,
                    test_end=test_end,
                    train_indices=train_indices,
                    test_indices=test_indices,
                )
            )
        test_start += step_delta
    return folds


def build_walk_forward_fold_row(
    *,
    walk_forward_id: str,
    fold_index: int,
    fold: WalkForwardFold,
    selected_train_row: dict[str, Any] | None,
    validation_row: dict[str, Any] | None,
    selected_config: VolumePriceEfficiencyConfig | None,
) -> dict[str, Any]:
    train = selected_train_row or {}
    validation = validation_row or {}
    row = {
        "walk_forward_id": walk_forward_id,
        "fold_index": fold_index,
        "train_start": _json_safe(fold.train_start),
        "train_end": _json_safe(fold.train_end),
        "test_start": _json_safe(fold.test_start),
        "test_end": _json_safe(fold.test_end),
        "train_bar_count": len(fold.train_indices),
        "validation_bar_count": len(fold.test_indices),
        "selected_config_hash": train.get("config_hash"),
        "train_score": _rounded(train.get("score")),
        "train_trade_count": train.get("trade_count"),
        "train_total_return": _rounded(train.get("total_return")),
        "train_max_drawdown": _rounded(train.get("max_drawdown")),
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
    }
    if selected_config is not None:
        row.update(
            {
                "efficiency_percentile": selected_config.efficiency_percentile,
                "min_move_unit": selected_config.min_move_unit,
                "min_volume_unit": selected_config.min_volume_unit,
                "min_close_position": selected_config.min_close_position,
                "min_body_ratio": selected_config.min_body_ratio,
                "fee_bps": selected_config.fee_bps,
                "slippage_bps": selected_config.slippage_bps,
                "baseline_seed": selected_config.baseline_seed,
            }
        )
    return row


def write_trailing_walk_forward_artifacts(
    *,
    paths: VolumePriceEfficiencyPaths,
    walk_forward_id: str,
    base_config: VolumePriceEfficiencyConfig,
    fold_rows: list[dict[str, Any]],
    selection_rows: list[dict[str, Any]],
    top_configs: list[dict[str, Any]],
    canonical_manifests: list[str],
    git_commit: str,
    runtime_seconds: float,
    symbol_count: int,
    data_split: dict[str, Any],
    atr_multiplier: float,
    train_days: int,
    test_days: int,
    step_days: int,
    min_trades: int,
) -> Path:
    output_dir = paths.trailing_walk_forward_dir(walk_forward_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "strategy_name": base_config.strategy_name,
        "strategy_version": "v1",
        "run_type": "trailing_stop_research_walk_forward",
        "data_scope": "research_only",
        "walk_forward_id": walk_forward_id,
        "git_commit": git_commit,
        "base_config": base_config.model_dump(mode="json"),
        "base_config_hash": base_config.config_hash(),
        "atr_multiplier": atr_multiplier,
        "train_days": train_days,
        "test_days": test_days,
        "step_days": step_days,
        "min_trades": min_trades,
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
            "top_configs": str(output_dir / "top_configs.json"),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "top_configs.json", top_configs)
    pq.write_table(_rows_table(fold_rows), output_dir / "fold_summary.parquet")
    pq.write_table(_rows_table(selection_rows), output_dir / "selection_summary.parquet")
    _write_csv(output_dir / "fold_summary.csv", fold_rows)
    return output_dir
