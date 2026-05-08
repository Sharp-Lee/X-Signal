from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays


@dataclass(frozen=True)
class FeatureArrays:
    true_range: np.ndarray
    atr: np.ndarray
    move_unit: np.ndarray
    volume_baseline: np.ndarray
    volume_unit: np.ndarray
    efficiency: np.ndarray
    efficiency_threshold: np.ndarray
    close_position: np.ndarray
    body_ratio: np.ndarray
    signal: np.ndarray


def _window_stat(values: np.ndarray, end_index: int, window: int, reducer) -> np.ndarray:
    start_index = end_index - window
    if start_index < 0:
        return np.full(values.shape[1], np.nan, dtype=np.float64)
    window_values = values[start_index:end_index]
    result = np.full(values.shape[1], np.nan, dtype=np.float64)
    for column_index in range(values.shape[1]):
        column = window_values[:, column_index]
        finite = column[np.isfinite(column)]
        if finite.size:
            result[column_index] = reducer(finite)
    return result


def build_signal_mask(
    arrays: OhlcvArrays,
    features: FeatureArrays,
    config: VolumePriceEfficiencyConfig,
) -> np.ndarray:
    if config.signal_mode == "seed_efficiency":
        return build_seed_efficiency_signal_mask(arrays, features, config)
    return build_classic_signal_mask(arrays, features, config)


def build_classic_signal_mask(
    arrays: OhlcvArrays,
    features: FeatureArrays,
    config: VolumePriceEfficiencyConfig,
) -> np.ndarray:
    signal = np.zeros(arrays.close.shape, dtype=bool)
    for index in range(1, arrays.close.shape[0]):
        signal[index] = (
            arrays.quality[index]
            & np.isfinite(features.efficiency[index])
            & np.isfinite(features.efficiency[index - 1])
            & np.isfinite(features.efficiency_threshold[index])
            & (features.efficiency[index] > features.efficiency[index - 1])
            & (features.efficiency[index] > features.efficiency_threshold[index])
            & (features.move_unit[index] >= config.min_move_unit)
            & (features.volume_unit[index] >= config.min_volume_unit)
            & (features.close_position[index] >= config.min_close_position)
            & (features.body_ratio[index] >= config.min_body_ratio)
        )
    return signal


def _previous_finite(values: np.ndarray, index: int, column_index: int, window: int) -> np.ndarray:
    start_index = index - window
    if start_index < 0:
        return np.array([], dtype=np.float64)
    window_values = values[start_index:index, column_index]
    return window_values[np.isfinite(window_values)]


def _price_range_position(
    arrays: OhlcvArrays,
    index: int,
    column_index: int,
    window: int,
) -> float:
    start_index = index - window
    if start_index < 0:
        return np.nan
    lows = arrays.low[start_index:index, column_index]
    highs = arrays.high[start_index:index, column_index]
    finite_lows = lows[np.isfinite(lows)]
    finite_highs = highs[np.isfinite(highs)]
    close = arrays.close[index, column_index]
    if finite_lows.size < window or finite_highs.size < window or not np.isfinite(close):
        return np.nan
    range_low = float(np.min(finite_lows))
    range_high = float(np.max(finite_highs))
    if range_high <= range_low:
        return np.nan
    return float((close - range_low) / (range_high - range_low))


def build_seed_efficiency_signal_mask(
    arrays: OhlcvArrays,
    features: FeatureArrays,
    config: VolumePriceEfficiencyConfig,
) -> np.ndarray:
    signal = np.zeros(arrays.close.shape, dtype=bool)
    for index in range(1, arrays.close.shape[0]):
        for column_index in range(arrays.close.shape[1]):
            if not arrays.quality[index, column_index]:
                continue
            current_efficiency = features.efficiency[index, column_index]
            recent_efficiency = _previous_finite(
                features.efficiency,
                index,
                column_index,
                config.seed_efficiency_lookback,
            )
            if recent_efficiency.size < config.seed_efficiency_lookback:
                continue
            price_range_position = _price_range_position(
                arrays,
                index,
                column_index,
                config.seed_bottom_lookback,
            )
            if not np.isfinite(price_range_position):
                continue
            recent_max = float(np.max(recent_efficiency))
            recent_mean = float(np.mean(recent_efficiency))
            signal[index, column_index] = bool(
                np.isfinite(current_efficiency)
                and current_efficiency > recent_max * config.seed_min_efficiency_ratio_to_max
                and current_efficiency > recent_mean * config.seed_min_efficiency_ratio_to_mean
                and features.move_unit[index, column_index] >= config.min_move_unit
                and features.volume_unit[index, column_index] >= config.min_volume_unit
                and features.volume_unit[index, column_index] <= config.seed_max_volume_unit
                and features.close_position[index, column_index] >= config.min_close_position
                and features.body_ratio[index, column_index] >= config.min_body_ratio
                and price_range_position <= config.seed_max_close_position_in_range
            )
    return signal


def compute_features(
    arrays: OhlcvArrays,
    config: VolumePriceEfficiencyConfig,
) -> FeatureArrays:
    shape = arrays.close.shape
    true_range = np.full(shape, np.nan, dtype=np.float64)
    atr = np.full(shape, np.nan, dtype=np.float64)
    move_unit = np.full(shape, np.nan, dtype=np.float64)
    volume_baseline = np.full(shape, np.nan, dtype=np.float64)
    volume_unit = np.full(shape, np.nan, dtype=np.float64)
    efficiency = np.full(shape, np.nan, dtype=np.float64)
    efficiency_threshold = np.full(shape, np.nan, dtype=np.float64)
    close_position = np.full(shape, np.nan, dtype=np.float64)
    body_ratio = np.full(shape, np.nan, dtype=np.float64)
    signal = np.zeros(shape, dtype=bool)

    high_low = arrays.high - arrays.low
    true_range[0] = high_low[0]
    for index in range(1, shape[0]):
        high_gap = np.abs(arrays.high[index] - arrays.close[index - 1])
        low_gap = np.abs(arrays.low[index] - arrays.close[index - 1])
        true_range[index] = np.maximum.reduce([high_low[index], high_gap, low_gap])

    for index in range(shape[0]):
        atr[index] = _window_stat(
            true_range,
            index + 1,
            config.atr_window,
            np.mean,
        )
        volume_baseline[index] = _window_stat(
            arrays.quote_volume,
            index,
            config.volume_window,
            np.median,
        )
        efficiency_threshold[index] = _window_stat(
            efficiency,
            index,
            config.efficiency_lookback,
            lambda values: np.percentile(values, config.efficiency_percentile * 100),
        )

        range_ = arrays.high[index] - arrays.low[index]
        valid_range = range_ > 0
        close_position[index] = np.divide(
            arrays.close[index] - arrays.low[index],
            range_,
            out=np.full(shape[1], np.nan, dtype=np.float64),
            where=valid_range,
        )
        body_ratio[index] = np.divide(
            np.abs(arrays.close[index] - arrays.open[index]),
            range_,
            out=np.full(shape[1], np.nan, dtype=np.float64),
            where=valid_range,
        )

        up_move = np.maximum(arrays.close[index] - arrays.open[index], 0.0)
        move_unit[index] = np.divide(
            up_move,
            atr[index],
            out=np.full(shape[1], np.nan, dtype=np.float64),
            where=atr[index] > 0,
        )
        volume_unit[index] = np.divide(
            arrays.quote_volume[index],
            volume_baseline[index],
            out=np.full(shape[1], np.nan, dtype=np.float64),
            where=volume_baseline[index] > 0,
        )
        efficiency[index] = np.divide(
            move_unit[index],
            np.maximum(volume_unit[index], config.volume_floor),
            out=np.full(shape[1], np.nan, dtype=np.float64),
            where=np.isfinite(volume_unit[index]),
        )

    features = FeatureArrays(
        true_range=true_range,
        atr=atr,
        move_unit=move_unit,
        volume_baseline=volume_baseline,
        volume_unit=volume_unit,
        efficiency=efficiency,
        efficiency_threshold=efficiency_threshold,
        close_position=close_position,
        body_ratio=body_ratio,
        signal=signal,
    )
    return FeatureArrays(
        true_range=features.true_range,
        atr=features.atr,
        move_unit=features.move_unit,
        volume_baseline=features.volume_baseline,
        volume_unit=features.volume_unit,
        efficiency=features.efficiency,
        efficiency_threshold=features.efficiency_threshold,
        close_position=features.close_position,
        body_ratio=features.body_ratio,
        signal=build_signal_mask(arrays, features, config),
    )
