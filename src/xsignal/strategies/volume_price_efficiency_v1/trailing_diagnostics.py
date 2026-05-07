from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)


DEFAULT_DIAGNOSTIC_FEATURES = (
    "btc_lookback_return",
    "market_lookback_return",
    "symbol_lookback_return",
    "signal_quote_volume",
    "efficiency",
    "move_unit",
    "volume_unit",
    "close_position",
    "body_ratio",
)


def _json_safe(value: Any) -> Any:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_safe) + "\n")


def _parse_time(value: object) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed.astimezone(timezone.utc) if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _rounded(value: float | None) -> float | None:
    return None if value is None else round(float(value), 12)


def _metric(values: list[float], fn) -> float | None:
    if not values:
        return None
    return float(fn(np.array(values, dtype=np.float64)))


def _summary_row(rows: list[dict[str, Any]]) -> dict[str, float | int | None]:
    returns = [float(row["net_realized_return"]) for row in rows]
    holding_bars = [
        float(row["holding_bars"])
        for row in rows
        if row.get("holding_bars") is not None
    ]
    return {
        "trade_count": len(rows),
        "win_rate": _rounded(float(np.mean(np.array(returns) > 0.0)) if returns else None),
        "mean_net_realized_return": _rounded(_metric(returns, np.mean)),
        "median_net_realized_return": _rounded(_metric(returns, np.median)),
        "average_holding_bars": _rounded(_metric(holding_bars, np.mean)),
        "total_ignored_signal_count": int(
            sum(int(row.get("ignored_signal_count") or 0) for row in rows)
        ),
    }


def _lookback_return(close: np.ndarray, end_index: int, start_index: int, symbol_index: int) -> float | None:
    end_value = close[end_index, symbol_index]
    start_value = close[start_index, symbol_index]
    if np.isfinite(end_value) and np.isfinite(start_value) and start_value > 0.0:
        return float(end_value / start_value - 1.0)
    return None


def _market_lookback_return(close: np.ndarray, end_index: int, start_index: int) -> float | None:
    end_values = close[end_index]
    start_values = close[start_index]
    valid = np.isfinite(end_values) & np.isfinite(start_values) & (start_values > 0.0)
    if not np.any(valid):
        return None
    return float(np.mean(end_values[valid] / start_values[valid] - 1.0))


def enrich_trades_with_market_context(
    trades: list[dict[str, Any]],
    arrays: OhlcvArrays,
    *,
    lookback_bars: int = 30,
    btc_symbol: str = "BTCUSDT",
) -> list[dict[str, Any]]:
    if lookback_bars <= 0:
        raise ValueError("lookback_bars must be positive")

    time_to_index = {_parse_time(value).isoformat(): index for index, value in enumerate(arrays.open_times)}
    symbol_to_index = {symbol: index for index, symbol in enumerate(arrays.symbols)}
    btc_index = symbol_to_index.get(btc_symbol)
    enriched: list[dict[str, Any]] = []
    for row in trades:
        item = dict(row)
        signal_time = _parse_time(row["signal_open_time"])
        signal_index = time_to_index.get(signal_time.isoformat())
        symbol_index = symbol_to_index.get(str(row["symbol"]))
        item["signal_year"] = signal_time.year
        item["signal_month"] = f"{signal_time.year:04d}-{signal_time.month:02d}"
        if signal_index is not None and signal_index >= lookback_bars:
            start_index = signal_index - lookback_bars
            if btc_index is not None:
                item["btc_lookback_return"] = _lookback_return(
                    arrays.close,
                    signal_index,
                    start_index,
                    btc_index,
                )
            if symbol_index is not None:
                item["symbol_lookback_return"] = _lookback_return(
                    arrays.close,
                    signal_index,
                    start_index,
                    symbol_index,
                )
                quote_volume = arrays.quote_volume[signal_index, symbol_index]
                item["signal_quote_volume"] = (
                    float(quote_volume) if np.isfinite(quote_volume) else None
                )
            item["market_lookback_return"] = _market_lookback_return(
                arrays.close,
                signal_index,
                start_index,
            )
        enriched.append(item)
    return enriched


def build_trailing_time_summary_rows(
    trades: list[dict[str, Any]],
    *,
    data_set: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in trades:
        signal_time = _parse_time(row["signal_open_time"])
        groups[("year", f"{signal_time.year:04d}")].append(row)
        groups[("month", f"{signal_time.year:04d}-{signal_time.month:02d}")].append(row)

    output = []
    for period_type, period in sorted(groups):
        summary = _summary_row(groups[(period_type, period)])
        output.append(
            {
                "data_set": data_set,
                "period_type": period_type,
                "period": period,
                **summary,
            }
        )
    return output


def build_trailing_bucket_summary_rows(
    trades: list[dict[str, Any]],
    *,
    data_set: str,
    feature_names: tuple[str, ...] = DEFAULT_DIAGNOSTIC_FEATURES,
    bucket_count: int = 5,
) -> list[dict[str, Any]]:
    if bucket_count <= 0:
        raise ValueError("bucket_count must be positive")

    output = []
    for feature_name in feature_names:
        feature_rows = [
            row
            for row in trades
            if row.get(feature_name) is not None and np.isfinite(float(row[feature_name]))
        ]
        feature_rows.sort(key=lambda row: float(row[feature_name]))
        if not feature_rows:
            continue
        for bucket_index, bucket_array in enumerate(
            np.array_split(np.array(feature_rows, dtype=object), bucket_count)
        ):
            bucket_rows = list(bucket_array)
            if not bucket_rows:
                continue
            feature_values = [float(row[feature_name]) for row in bucket_rows]
            output.append(
                {
                    "data_set": data_set,
                    "feature_name": feature_name,
                    "bucket_index": bucket_index,
                    "bucket_count": bucket_count,
                    "lower_bound": _rounded(min(feature_values)),
                    "upper_bound": _rounded(max(feature_values)),
                    **_summary_row(bucket_rows),
                }
            )
    return output


def _rows_table(rows: list[dict[str, Any]]) -> pa.Table:
    if rows:
        return pa.Table.from_pylist(rows)
    return pa.table({})


def write_trailing_diagnostic_artifacts(
    *,
    paths: VolumePriceEfficiencyPaths,
    diagnostic_id: str,
    config: VolumePriceEfficiencyConfig,
    time_rows: list[dict[str, Any]],
    bucket_rows: list[dict[str, Any]],
    canonical_manifests: list[str],
    git_commit: str,
    runtime_seconds: float,
    data_split: dict[str, Any],
    atr_multiplier: float,
    lookback_bars: int,
) -> Path:
    diagnostic_dir = paths.trailing_diagnostic_dir(diagnostic_id)
    diagnostic_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "strategy_name": config.strategy_name,
        "strategy_version": "v1",
        "run_type": "trailing_stop_diagnostics",
        "diagnostic_id": diagnostic_id,
        "git_commit": git_commit,
        "config": config.model_dump(mode="json"),
        "config_hash": config.config_hash(),
        "atr_multiplier": atr_multiplier,
        "lookback_bars": lookback_bars,
        "canonical_manifests": canonical_manifests,
        "runtime_seconds": runtime_seconds,
        "data_split": data_split,
        "outputs": {
            "time_summary": str(diagnostic_dir / "time_summary.parquet"),
            "bucket_summary": str(diagnostic_dir / "bucket_summary.parquet"),
        },
    }
    _write_json(diagnostic_dir / "manifest.json", manifest)
    pq.write_table(_rows_table(time_rows), diagnostic_dir / "time_summary.parquet")
    pq.write_table(_rows_table(bucket_rows), diagnostic_dir / "bucket_summary.parquet")
    return diagnostic_dir
