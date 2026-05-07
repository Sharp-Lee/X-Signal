from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.artifacts import (
    build_event_study_summary,
)
from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)


DEFAULT_BUCKET_FEATURES = (
    "efficiency",
    "move_unit",
    "volume_unit",
    "close_position",
    "body_ratio",
)


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 12)


def build_scan_configs(
    *,
    efficiency_percentiles: tuple[float, ...],
    min_move_units: tuple[float, ...],
    min_volume_units: tuple[float, ...],
    min_close_positions: tuple[float, ...],
    min_body_ratios: tuple[float, ...],
    fee_bps: float,
    slippage_bps: float,
    baseline_seed: int,
) -> list[VolumePriceEfficiencyConfig]:
    return [
        VolumePriceEfficiencyConfig(
            efficiency_percentile=efficiency_percentile,
            min_move_unit=min_move_unit,
            min_volume_unit=min_volume_unit,
            min_close_position=min_close_position,
            min_body_ratio=min_body_ratio,
            fee_bps=fee_bps,
            slippage_bps=slippage_bps,
            baseline_seed=baseline_seed,
        )
        for (
            efficiency_percentile,
            min_move_unit,
            min_volume_unit,
            min_close_position,
            min_body_ratio,
        ) in itertools.product(
            efficiency_percentiles,
            min_move_units,
            min_volume_units,
            min_close_positions,
            min_body_ratios,
        )
    ]


def build_scan_row(
    *,
    scan_id: str,
    config: VolumePriceEfficiencyConfig,
    events: list[dict],
    baseline_events: list[dict],
    symbols: tuple[str, ...],
    ranking_horizon: int,
) -> dict[str, Any]:
    summary = build_event_study_summary(events, baseline_events, config.horizons)
    row: dict[str, Any] = {
        "scan_id": scan_id,
        "config_hash": config.config_hash(),
        "efficiency_percentile": config.efficiency_percentile,
        "min_move_unit": config.min_move_unit,
        "min_volume_unit": config.min_volume_unit,
        "min_close_position": config.min_close_position,
        "min_body_ratio": config.min_body_ratio,
        "fee_bps": config.fee_bps,
        "slippage_bps": config.slippage_bps,
        "baseline_seed": config.baseline_seed,
        "ranking_horizon": ranking_horizon,
        "event_count": summary["event_count"],
        "baseline_event_count": summary["baseline_event_count"],
        "symbol_count": len(symbols),
    }
    for horizon in config.horizons:
        metrics = summary["horizons"][str(horizon)]
        prefix = f"h{horizon}"
        row[f"{prefix}_mean_return"] = metrics["mean_return"]
        row[f"{prefix}_net_mean_return"] = metrics["net_mean_return"]
        row[f"{prefix}_baseline_mean_return"] = metrics["baseline_mean_return"]
        row[f"{prefix}_event_minus_baseline_mean_return"] = metrics[
            "event_minus_baseline_mean_return"
        ]
        row[f"{prefix}_win_rate"] = metrics["win_rate"]
    row["score"] = row.get(f"h{ranking_horizon}_event_minus_baseline_mean_return")
    return row


def select_top_configs(rows: list[dict[str, Any]], *, top_k: int) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -(float(row["score"]) if row.get("score") is not None else float("-inf")),
            str(row["config_hash"]),
        ),
    )[:top_k]


def _metric(values: list[float]) -> float | None:
    if not values:
        return None
    return _rounded(float(np.mean(values)))


def _bucket_return_metrics(rows: list[dict], horizon: int) -> dict[str, float | int | None]:
    gross_values = [
        float(row[f"forward_return_{horizon}"])
        for row in rows
        if row.get(f"forward_return_{horizon}") is not None
    ]
    net_values = [
        float(row[f"net_forward_return_{horizon}"])
        for row in rows
        if row.get(f"net_forward_return_{horizon}") is not None
    ]
    return {
        f"h{horizon}_mean_return": _metric(gross_values),
        f"h{horizon}_net_mean_return": _metric(net_values),
        f"h{horizon}_win_rate": _rounded(float(np.mean(np.array(gross_values) > 0.0)))
        if gross_values
        else None,
    }


def build_bucket_summary_rows(
    *,
    config: VolumePriceEfficiencyConfig,
    events: list[dict],
    horizons: tuple[int, ...],
    feature_names: tuple[str, ...] = DEFAULT_BUCKET_FEATURES,
    bucket_count: int = 5,
) -> list[dict[str, Any]]:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")

    rows: list[dict[str, Any]] = []
    for feature_name in feature_names:
        feature_events = [
            row
            for row in events
            if row.get(feature_name) is not None and np.isfinite(float(row[feature_name]))
        ]
        feature_events.sort(key=lambda row: float(row[feature_name]))
        if not feature_events:
            continue
        split_events = np.array_split(np.array(feature_events, dtype=object), bucket_count)
        for bucket_index, bucket_array in enumerate(split_events):
            bucket_rows = list(bucket_array)
            if not bucket_rows:
                continue
            feature_values = [float(row[feature_name]) for row in bucket_rows]
            bucket: dict[str, Any] = {
                "config_hash": config.config_hash(),
                "feature_name": feature_name,
                "bucket_index": bucket_index,
                "bucket_count": bucket_count,
                "lower_bound": _rounded(min(feature_values)),
                "upper_bound": _rounded(max(feature_values)),
                "event_count": len(bucket_rows),
            }
            for horizon in horizons:
                bucket.update(_bucket_return_metrics(bucket_rows, horizon))
            rows.append(bucket)
    return rows
