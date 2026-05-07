from __future__ import annotations

import csv
import json
from dataclasses import dataclass
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
from xsignal.strategies.volume_price_efficiency_v1.trailing import TrailingStopResult
from xsignal.strategies.volume_price_efficiency_v1.trailing_scan import (
    build_trailing_scan_row,
)


DEFAULT_REGIME_FEATURES = (
    "btc_lookback_return",
    "market_lookback_return",
    "symbol_lookback_return",
    "move_unit",
    "volume_unit",
)


@dataclass(frozen=True)
class RegimeFilterRule:
    rule_id: str
    feature_name: str
    direction: str
    quantile: float
    threshold: float


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


def _lookback_return_matrix(close: np.ndarray, lookback_bars: int) -> np.ndarray:
    output = np.full(close.shape, np.nan, dtype=np.float64)
    for index in range(lookback_bars, close.shape[0]):
        start = close[index - lookback_bars]
        end = close[index]
        valid = np.isfinite(start) & np.isfinite(end) & (start > 0.0)
        output[index] = np.divide(
            end,
            start,
            out=np.full(close.shape[1], np.nan, dtype=np.float64),
            where=valid,
        ) - 1.0
    return output


def _market_lookback_return_matrix(close: np.ndarray, lookback_bars: int) -> np.ndarray:
    symbol_returns = _lookback_return_matrix(close, lookback_bars)
    output = np.full(close.shape, np.nan, dtype=np.float64)
    for index in range(lookback_bars, close.shape[0]):
        row = symbol_returns[index]
        finite = row[np.isfinite(row)]
        if finite.size:
            output[index] = float(np.mean(finite))
    return output


def _lookback_high_low_position_matrix(
    close: np.ndarray,
    lookback_bars: int,
) -> tuple[np.ndarray, np.ndarray]:
    drawdown = np.full(close.shape, np.nan, dtype=np.float64)
    position = np.full(close.shape, np.nan, dtype=np.float64)
    for index in range(lookback_bars, close.shape[0]):
        window = close[index - lookback_bars : index + 1]
        finite = np.isfinite(window)
        if not np.any(finite):
            continue
        high = np.where(finite, window, -np.inf).max(axis=0)
        low = np.where(finite, window, np.inf).min(axis=0)
        current = close[index]
        valid_high = np.isfinite(high) & np.isfinite(current) & (high > 0.0)
        valid_range = np.isfinite(high) & np.isfinite(low) & (high > low)
        drawdown[index] = np.divide(
            current,
            high,
            out=np.full(close.shape[1], np.nan, dtype=np.float64),
            where=valid_high,
        ) - 1.0
        position[index] = np.divide(
            current - low,
            high - low,
            out=np.full(close.shape[1], np.nan, dtype=np.float64),
            where=valid_range,
        )
    return drawdown, position


def _lagged_ratio_to_rolling_median_matrix(values: np.ndarray, lookback_bars: int) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(lookback_bars, values.shape[0]):
        current = values[index - 1]
        window = values[index - lookback_bars : index]
        median = np.full(values.shape[1], np.nan, dtype=np.float64)
        for column_index in range(values.shape[1]):
            column = window[:, column_index]
            finite = column[np.isfinite(column)]
            if finite.size:
                median[column_index] = float(np.median(finite))
        valid = np.isfinite(current) & np.isfinite(median) & (median > 0.0)
        output[index] = np.divide(
            current,
            median,
            out=np.full(values.shape[1], np.nan, dtype=np.float64),
            where=valid,
        )
    return output


def _broadcast_row_mean_matrix(values: np.ndarray) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    for index in range(values.shape[0]):
        row = values[index]
        finite = row[np.isfinite(row)]
        if finite.size:
            output[index] = float(np.mean(finite))
    return output


def _broadcast_first_symbol_matrix(values: np.ndarray, symbol_index: int) -> np.ndarray:
    output = np.full(values.shape, np.nan, dtype=np.float64)
    column = values[:, symbol_index : symbol_index + 1]
    for index in range(values.shape[0]):
        value = column[index, 0]
        if np.isfinite(value):
            output[index] = float(value)
    return output


def _btc_lookback_return_matrix(
    arrays: OhlcvArrays,
    lookback_bars: int,
    *,
    btc_symbol: str = "BTCUSDT",
) -> np.ndarray:
    output = np.full(arrays.close.shape, np.nan, dtype=np.float64)
    try:
        btc_index = arrays.symbols.index(btc_symbol)
    except ValueError:
        return output
    btc_returns = _lookback_return_matrix(arrays.close, lookback_bars)
    return _broadcast_first_symbol_matrix(btc_returns, btc_index)


def build_regime_value_arrays(
    arrays: OhlcvArrays,
    features: FeatureArrays,
    *,
    lookback_bars: int,
) -> dict[str, np.ndarray]:
    if lookback_bars <= 0:
        raise ValueError("lookback_bars must be positive")
    symbol_drawdown, symbol_range_position = _lookback_high_low_position_matrix(
        arrays.close,
        lookback_bars,
    )
    try:
        btc_index = arrays.symbols.index("BTCUSDT")
    except ValueError:
        btc_drawdown = np.full(arrays.close.shape, np.nan, dtype=np.float64)
        btc_range_position = np.full(arrays.close.shape, np.nan, dtype=np.float64)
    else:
        btc_drawdown = _broadcast_first_symbol_matrix(symbol_drawdown, btc_index)
        btc_range_position = _broadcast_first_symbol_matrix(symbol_range_position, btc_index)

    return {
        "btc_lookback_return": _btc_lookback_return_matrix(arrays, lookback_bars),
        "market_lookback_return": _market_lookback_return_matrix(arrays.close, lookback_bars),
        "symbol_lookback_return": _lookback_return_matrix(arrays.close, lookback_bars),
        "btc_drawdown_from_lookback_high": btc_drawdown,
        "market_drawdown_from_lookback_high": _broadcast_row_mean_matrix(symbol_drawdown),
        "symbol_drawdown_from_lookback_high": symbol_drawdown,
        "btc_range_position": btc_range_position,
        "market_range_position": _broadcast_row_mean_matrix(symbol_range_position),
        "symbol_range_position": symbol_range_position,
        "pre_signal_atr_contraction": _lagged_ratio_to_rolling_median_matrix(
            features.atr,
            lookback_bars,
        ),
        "pre_signal_true_range_contraction": _lagged_ratio_to_rolling_median_matrix(
            features.true_range,
            lookback_bars,
        ),
        "pre_signal_volume_contraction": _lagged_ratio_to_rolling_median_matrix(
            arrays.quote_volume,
            lookback_bars,
        ),
        "signal_quote_volume": arrays.quote_volume,
        "efficiency": features.efficiency,
        "move_unit": features.move_unit,
        "volume_unit": features.volume_unit,
        "close_position": features.close_position,
        "body_ratio": features.body_ratio,
    }


def build_regime_filter_rules(
    signal: np.ndarray,
    values_by_feature: dict[str, np.ndarray],
    *,
    feature_names: tuple[str, ...] = DEFAULT_REGIME_FEATURES,
    quantiles: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8),
) -> tuple[RegimeFilterRule, ...]:
    if not quantiles:
        raise ValueError("quantiles must contain at least one value")
    if any(quantile <= 0.0 or quantile >= 1.0 for quantile in quantiles):
        raise ValueError("quantiles must be between 0 and 1")

    rules: list[RegimeFilterRule] = []
    seen: set[tuple[str, str, float]] = set()
    for feature_name in feature_names:
        values = values_by_feature[feature_name]
        signal_values = values[signal & np.isfinite(values)]
        if signal_values.size == 0:
            continue
        for quantile in quantiles:
            threshold = float(np.percentile(signal_values, quantile * 100.0))
            for direction in ("gte", "lt"):
                key = (feature_name, direction, threshold)
                if key in seen:
                    continue
                seen.add(key)
                rules.append(
                    RegimeFilterRule(
                        rule_id=f"{feature_name}_{direction}_p{int(quantile * 100):02d}",
                        feature_name=feature_name,
                        direction=direction,
                        quantile=quantile,
                        threshold=threshold,
                    )
                )
    return tuple(rules)


def apply_regime_filter_rule(
    signal: np.ndarray,
    values_by_feature: dict[str, np.ndarray],
    rule: RegimeFilterRule,
) -> np.ndarray:
    values = values_by_feature[rule.feature_name]
    if values.shape != signal.shape:
        raise ValueError("regime feature shape does not match signal shape")
    if rule.direction == "gte":
        keep = np.isfinite(values) & (values >= rule.threshold)
    elif rule.direction == "lt":
        keep = np.isfinite(values) & (values < rule.threshold)
    else:
        raise ValueError("rule direction must be gte or lt")
    return signal & keep


def build_regime_scan_row(
    *,
    regime_scan_id: str,
    config: VolumePriceEfficiencyConfig,
    result: TrailingStopResult,
    symbols: tuple[str, ...],
    rule: RegimeFilterRule | None,
    base_signal_count: int,
    filtered_signal_count: int,
    atr_multiplier: float = 2.0,
) -> dict[str, Any]:
    row = build_trailing_scan_row(
        scan_id=regime_scan_id,
        config=config,
        result=result,
        symbols=symbols,
        atr_multiplier=atr_multiplier,
    )
    row["regime_scan_id"] = row.pop("scan_id")
    row.update(
        {
            "rule_id": rule.rule_id if rule is not None else "unfiltered",
            "feature_name": rule.feature_name if rule is not None else None,
            "direction": rule.direction if rule is not None else None,
            "quantile": rule.quantile if rule is not None else None,
            "threshold": _rounded(rule.threshold if rule is not None else None),
            "base_signal_count": int(base_signal_count),
            "filtered_signal_count": int(filtered_signal_count),
            "signal_keep_rate": _rounded(
                float(filtered_signal_count / base_signal_count) if base_signal_count else None
            ),
        }
    )
    return row


def select_top_regime_filters(
    rows: list[dict[str, Any]],
    *,
    top_k: int,
    min_trades: int,
) -> list[dict[str, Any]]:
    eligible = [
        row
        for row in rows
        if row.get("rule_id") != "unfiltered" and int(row.get("trade_count") or 0) >= min_trades
    ]
    return sorted(
        eligible,
        key=lambda row: (
            -(float(row["score"]) if row.get("score") is not None else float("-inf")),
            str(row["rule_id"]),
        ),
    )[:top_k]


def write_trailing_regime_scan_artifacts(
    *,
    paths: VolumePriceEfficiencyPaths,
    regime_scan_id: str,
    config: VolumePriceEfficiencyConfig,
    rows: list[dict[str, Any]],
    top_filters: list[dict[str, Any]],
    canonical_manifests: list[str],
    git_commit: str,
    runtime_seconds: float,
    symbol_count: int,
    data_split: dict[str, Any],
    atr_multiplier: float,
    lookback_bars: int,
    quantiles: tuple[float, ...],
    feature_names: tuple[str, ...],
) -> Path:
    output_dir = paths.trailing_regime_scan_dir(regime_scan_id)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "strategy_name": config.strategy_name,
        "strategy_version": "v1",
        "run_type": "trailing_stop_research_regime_scan",
        "data_scope": "research_only",
        "threshold_scope": "full_research_signal_distribution_diagnostic_only",
        "regime_scan_id": regime_scan_id,
        "git_commit": git_commit,
        "config": config.model_dump(mode="json"),
        "config_hash": config.config_hash(),
        "atr_multiplier": atr_multiplier,
        "lookback_bars": lookback_bars,
        "quantiles": list(quantiles),
        "feature_names": list(feature_names),
        "canonical_manifests": canonical_manifests,
        "symbol_count": symbol_count,
        "runtime_seconds": runtime_seconds,
        "combination_count": len(rows),
        "data_split": data_split,
        "outputs": {
            "summary": str(output_dir / "summary.parquet"),
            "summary_csv": str(output_dir / "summary.csv"),
            "top_filters": str(output_dir / "top_filters.json"),
        },
    }
    _write_json(output_dir / "manifest.json", manifest)
    _write_json(output_dir / "top_filters.json", top_filters)
    pq.write_table(_rows_table(rows), output_dir / "summary.parquet")
    _write_csv(output_dir / "summary.csv", rows)
    return output_dir
