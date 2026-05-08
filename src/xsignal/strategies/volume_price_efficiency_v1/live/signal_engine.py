from __future__ import annotations

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.features import compute_features
from xsignal.strategies.volume_price_efficiency_v1.live.config import (
    LiveTradingConfig,
    build_vpe_live_strategy_config,
)


def closed_bar_view(arrays: OhlcvArrays, *, closed_open_time: object) -> OhlcvArrays:
    matches = np.flatnonzero(arrays.open_times <= closed_open_time)
    if matches.size == 0:
        raise ValueError("closed_open_time is before available history")
    end = int(matches[-1]) + 1
    return OhlcvArrays(
        symbols=arrays.symbols,
        open_times=arrays.open_times[:end],
        open=arrays.open[:end],
        high=arrays.high[:end],
        low=arrays.low[:end],
        close=arrays.close[:end],
        quote_volume=arrays.quote_volume[:end],
        quality=arrays.quality[:end],
    )


def build_market_regime_mask(
    close: np.ndarray,
    *,
    lookback_bars: int,
    min_return: float,
) -> np.ndarray:
    output = np.zeros(close.shape, dtype=bool)
    for index in range(lookback_bars, close.shape[0]):
        start = close[index - lookback_bars]
        end = close[index]
        valid = np.isfinite(start) & np.isfinite(end) & (start > 0.0)
        returns = np.divide(
            end,
            start,
            out=np.full(close.shape[1], np.nan),
            where=valid,
        ) - 1.0
        finite = returns[np.isfinite(returns)]
        if finite.size and float(np.mean(finite)) >= min_return:
            output[index] = True
    return output


def build_live_signal_mask(arrays: OhlcvArrays, live_config: LiveTradingConfig) -> np.ndarray:
    strategy_config = build_vpe_live_strategy_config()
    features = compute_features(arrays, strategy_config)
    regime = build_market_regime_mask(
        arrays.close,
        lookback_bars=live_config.market_regime_lookback_bars,
        min_return=live_config.market_regime_min_return,
    )
    return features.signal & regime
