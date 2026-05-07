from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
from xsignal.strategies.momentum_rotation_v1.prepare import PreparedArrays


@dataclass(frozen=True)
class SignalArrays:
    score: np.ndarray
    tradable_mask: np.ndarray


def _rolling_sum(values: np.ndarray, end_index: int, window: int) -> np.ndarray:
    start_index = end_index - window + 1
    if start_index < 0:
        return np.full(values.shape[1], np.nan, dtype=np.float64)
    return np.nansum(values[start_index : end_index + 1], axis=0)


def compute_momentum_signals(
    arrays: PreparedArrays,
    config: MomentumRotationConfig,
) -> SignalArrays:
    shape = arrays.close_1d.shape
    if arrays.close_1h.shape != shape or arrays.close_4h.shape != shape:
        raise ValueError("prepared close arrays must have matching shape")
    score = np.full(shape, np.nan, dtype=np.float64)
    tradable_mask = np.zeros(shape, dtype=bool)
    for index in range(shape[0]):
        if index < config.long_window_days:
            continue
        short_good = arrays.quality_1h_24h[index]
        medium_good = arrays.quality_4h_7d[index]
        long_good = arrays.quality_1d_30d[index]
        liquidity = _rolling_sum(arrays.quote_volume_1d, index, 7)
        liquid = liquidity >= config.min_rolling_7d_quote_volume
        current_positive = (
            (arrays.close_1h[index] > 0)
            & (arrays.close_4h[index] > 0)
            & (arrays.close_1d[index] > 0)
            & (arrays.close_1h[index - 1] > 0)
            & (arrays.close_4h[index - config.medium_window_days] > 0)
            & (arrays.close_1d[index - config.long_window_days] > 0)
        )
        eligible = short_good & medium_good & long_good & liquid & current_positive
        short_return = arrays.close_1h[index] / arrays.close_1h[index - 1] - 1.0
        medium_return = (
            arrays.close_4h[index] / arrays.close_4h[index - config.medium_window_days] - 1.0
        )
        long_return = (
            arrays.close_1d[index] / arrays.close_1d[index - config.long_window_days] - 1.0
        )
        combined = (
            config.short_return_weight * short_return
            + config.medium_return_weight * medium_return
            + config.long_return_weight * long_return
        )
        score[index, eligible] = combined[eligible]
        tradable_mask[index, eligible] = True
    return SignalArrays(score=score, tradable_mask=tradable_mask)
