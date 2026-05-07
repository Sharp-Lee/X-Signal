from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
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


EVENT_COLUMNS = (
    "symbol",
    "signal_open_time",
    "decision_time",
    "entry_open_time",
    "entry_price",
    "move_unit",
    "volume_unit",
    "efficiency",
    "efficiency_threshold",
    "close_position",
    "body_ratio",
    "quote_volume",
    "volume_baseline",
    "atr",
    "matched_signal_month",
    "matched_signal_count_for_symbol_month",
)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_safe) + "\n")


def _clean(values: list[Any]) -> np.ndarray:
    return np.array([float(value) for value in values if value is not None], dtype=np.float64)


def _metric_or_none(values: np.ndarray, fn) -> float | None:
    if values.size == 0:
        return None
    return float(fn(values))


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 12)


def _return_metrics(values: list[Any]) -> dict[str, float | int | None]:
    cleaned = _clean(values)
    return {
        "available_count": int(cleaned.size),
        "mean": _rounded(_metric_or_none(cleaned, np.mean)),
        "median": _rounded(_metric_or_none(cleaned, np.median)),
        "win_rate": _rounded(float(np.mean(cleaned > 0.0)) if cleaned.size else None),
        "p10": _rounded(_metric_or_none(cleaned, lambda array: np.percentile(array, 10))),
        "p25": _rounded(_metric_or_none(cleaned, lambda array: np.percentile(array, 25))),
        "p75": _rounded(_metric_or_none(cleaned, lambda array: np.percentile(array, 75))),
        "p90": _rounded(_metric_or_none(cleaned, lambda array: np.percentile(array, 90))),
    }


def _top_symbols_by_event_count(events: list[dict]) -> list[dict[str, Any]]:
    counts = Counter(row["symbol"] for row in events)
    return [
        {"symbol": symbol, "event_count": count}
        for symbol, count in sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:20]
    ]


def _top_symbols_by_average_return(events: list[dict], horizon: int) -> list[dict[str, Any]]:
    returns: dict[str, list[float]] = defaultdict(list)
    key = f"forward_return_{horizon}"
    for row in events:
        value = row.get(key)
        if value is not None:
            returns[row["symbol"]].append(float(value))
    ranked = [
        {
            "symbol": symbol,
            "event_count": len(values),
            "average_return": _rounded(float(np.mean(values))),
        }
        for symbol, values in returns.items()
        if values
    ]
    return sorted(
        ranked,
        key=lambda row: (-(row["average_return"] or 0.0), row["symbol"]),
    )[:20]


def build_event_study_summary(
    events: list[dict],
    baseline_events: list[dict],
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    signal_times = sorted(row["signal_open_time"] for row in events)
    summary: dict[str, Any] = {
        "event_count": len(events),
        "baseline_event_count": len(baseline_events),
        "symbol_count": len({row["symbol"] for row in events}),
        "first_signal_time": signal_times[0] if signal_times else None,
        "last_signal_time": signal_times[-1] if signal_times else None,
        "horizons": {},
        "top_symbols_by_event_count": _top_symbols_by_event_count(events),
        "top_symbols_by_average_return": _top_symbols_by_average_return(
            events,
            horizons[0],
        )
        if horizons
        else [],
    }
    for horizon in horizons:
        event_metrics = _return_metrics([row.get(f"forward_return_{horizon}") for row in events])
        net_metrics = _return_metrics([row.get(f"net_forward_return_{horizon}") for row in events])
        baseline_metrics = _return_metrics(
            [row.get(f"forward_return_{horizon}") for row in baseline_events]
        )
        mean_delta = None
        median_delta = None
        if event_metrics["mean"] is not None and baseline_metrics["mean"] is not None:
            mean_delta = _rounded(event_metrics["mean"] - baseline_metrics["mean"])
        if event_metrics["median"] is not None and baseline_metrics["median"] is not None:
            median_delta = _rounded(event_metrics["median"] - baseline_metrics["median"])
        summary["horizons"][str(horizon)] = {
            "available_event_count": event_metrics["available_count"],
            "mean_return": event_metrics["mean"],
            "median_return": event_metrics["median"],
            "win_rate": event_metrics["win_rate"],
            "p10": event_metrics["p10"],
            "p25": event_metrics["p25"],
            "p75": event_metrics["p75"],
            "p90": event_metrics["p90"],
            "net_mean_return": net_metrics["mean"],
            "net_median_return": net_metrics["median"],
            "net_win_rate": net_metrics["win_rate"],
            "baseline_mean_return": baseline_metrics["mean"],
            "baseline_median_return": baseline_metrics["median"],
            "baseline_win_rate": baseline_metrics["win_rate"],
            "event_minus_baseline_mean_return": mean_delta,
            "event_minus_baseline_median_return": median_delta,
        }
    return summary


def _event_schema(horizons: tuple[int, ...]) -> pa.Schema:
    fields = [
        pa.field("symbol", pa.string()),
        pa.field("signal_open_time", pa.string()),
        pa.field("decision_time", pa.string()),
        pa.field("entry_open_time", pa.string()),
        pa.field("entry_price", pa.float64()),
        pa.field("move_unit", pa.float64()),
        pa.field("volume_unit", pa.float64()),
        pa.field("efficiency", pa.float64()),
        pa.field("efficiency_threshold", pa.float64()),
        pa.field("close_position", pa.float64()),
        pa.field("body_ratio", pa.float64()),
        pa.field("quote_volume", pa.float64()),
        pa.field("volume_baseline", pa.float64()),
        pa.field("atr", pa.float64()),
        pa.field("matched_signal_month", pa.string()),
        pa.field("matched_signal_count_for_symbol_month", pa.int64()),
    ]
    for horizon in horizons:
        fields.append(pa.field(f"forward_return_{horizon}", pa.float64()))
        fields.append(pa.field(f"net_forward_return_{horizon}", pa.float64()))
    return pa.schema(fields)


def _event_table(rows: list[dict], horizons: tuple[int, ...]) -> pa.Table:
    schema = _event_schema(horizons)
    if not rows:
        return pa.Table.from_arrays(
            [pa.array([], type=field.type) for field in schema],
            schema=schema,
        )
    normalized = []
    for row in rows:
        normalized.append({name: row.get(name) for name in schema.names})
    return pa.Table.from_pylist(normalized, schema=schema)


def write_run_artifacts(
    *,
    paths: VolumePriceEfficiencyPaths,
    run_id: str,
    config: VolumePriceEfficiencyConfig,
    events: list[dict],
    baseline_events: list[dict],
    symbols: tuple[str, ...],
    canonical_manifests: list[str],
    git_commit: str,
    runtime_seconds: float,
) -> Path:
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = build_event_study_summary(events, baseline_events, config.horizons)
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
            "events": str(run_dir / "events.parquet"),
            "baseline_events": str(run_dir / "baseline_events.parquet"),
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "summary.json", summary)
    pq.write_table(_event_table(events, config.horizons), run_dir / "events.parquet")
    pq.write_table(
        _event_table(baseline_events, config.horizons),
        run_dir / "baseline_events.parquet",
    )
    return run_dir


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


def _bucket_table(rows: list[dict[str, Any]]) -> pa.Table:
    if rows:
        return pa.Table.from_pylist(rows)
    return pa.table(
        {
            "config_hash": pa.array([], type=pa.string()),
            "feature_name": pa.array([], type=pa.string()),
            "bucket_index": pa.array([], type=pa.int64()),
            "bucket_count": pa.array([], type=pa.int64()),
            "lower_bound": pa.array([], type=pa.float64()),
            "upper_bound": pa.array([], type=pa.float64()),
            "event_count": pa.array([], type=pa.int64()),
        }
    )


def write_scan_artifacts(
    *,
    paths: VolumePriceEfficiencyPaths,
    scan_id: str,
    base_config: VolumePriceEfficiencyConfig,
    rows: list[dict[str, Any]],
    top_configs: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    canonical_manifests: list[str],
    git_commit: str,
    runtime_seconds: float,
    symbol_count: int,
    data_split: dict[str, Any] | None = None,
) -> Path:
    scan_dir = paths.scan_dir(scan_id)
    scan_dir.mkdir(parents=True, exist_ok=True)
    scores = _score_values(rows)
    summary = {
        "scan_id": scan_id,
        "combination_count": len(rows),
        "runtime_seconds": runtime_seconds,
        "best_score": _rounded(max(scores) if scores else None),
    }
    manifest = {
        "strategy_name": base_config.strategy_name,
        "strategy_version": "v1",
        "scan_id": scan_id,
        "git_commit": git_commit,
        "base_config": base_config.model_dump(mode="json"),
        "base_config_hash": base_config.config_hash(),
        "canonical_manifests": canonical_manifests,
        "symbol_count": symbol_count,
        "runtime_seconds": runtime_seconds,
        "combination_count": len(rows),
        "outputs": {
            "summary": str(scan_dir / "summary.json"),
            "summary_csv": str(scan_dir / "summary.csv"),
            "top_configs": str(scan_dir / "top_configs.json"),
            "bucket_summary": str(scan_dir / "bucket_summary.parquet"),
        },
    }
    if data_split is not None:
        manifest["data_split"] = data_split
    _write_json(scan_dir / "manifest.json", manifest)
    _write_json(scan_dir / "summary.json", summary)
    _write_json(scan_dir / "top_configs.json", top_configs)
    _write_summary_csv(scan_dir / "summary.csv", rows)
    pq.write_table(_bucket_table(bucket_rows), scan_dir / "bucket_summary.parquet")
    return scan_dir
