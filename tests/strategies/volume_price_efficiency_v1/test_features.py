from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.features import (
    FeatureArrays,
    build_signal_mask,
    compute_features,
)


def _arrays(
    *,
    open_values,
    high_values,
    low_values,
    close_values,
    quote_volume_values,
    quality_values=None,
) -> OhlcvArrays:
    row_count = len(open_values)
    quality_values = quality_values or [True] * row_count
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return OhlcvArrays(
        symbols=("BTCUSDT",),
        open_times=np.array([start + timedelta(hours=4 * i) for i in range(row_count)], dtype=object),
        open=np.array(open_values, dtype=np.float64).reshape(row_count, 1),
        high=np.array(high_values, dtype=np.float64).reshape(row_count, 1),
        low=np.array(low_values, dtype=np.float64).reshape(row_count, 1),
        close=np.array(close_values, dtype=np.float64).reshape(row_count, 1),
        quote_volume=np.array(quote_volume_values, dtype=np.float64).reshape(row_count, 1),
        quality=np.array(quality_values, dtype=bool).reshape(row_count, 1),
    )


def test_true_range_uses_prior_close_for_gaps():
    arrays = _arrays(
        open_values=[100, 120],
        high_values=[105, 122],
        low_values=[95, 119],
        close_values=[100, 121],
        quote_volume_values=[1000, 1000],
    )
    config = VolumePriceEfficiencyConfig(atr_window=1, volume_window=1, efficiency_lookback=1)

    features = compute_features(arrays, config)

    assert isinstance(features, FeatureArrays)
    assert features.true_range[:, 0].tolist() == [10.0, 22.0]
    assert features.atr[:, 0].tolist() == [10.0, 22.0]


def test_volume_baseline_excludes_current_bar():
    arrays = _arrays(
        open_values=[100, 100, 100, 100],
        high_values=[101, 101, 101, 101],
        low_values=[99, 99, 99, 99],
        close_values=[100.5, 100.5, 100.5, 100.5],
        quote_volume_values=[100, 200, 1000, 10_000],
    )
    config = VolumePriceEfficiencyConfig(atr_window=1, volume_window=3, efficiency_lookback=1)

    features = compute_features(arrays, config)

    assert np.isnan(features.volume_baseline[0, 0])
    assert features.volume_baseline[3, 0] == 200.0
    assert features.volume_unit[3, 0] == 50.0


def test_efficiency_threshold_excludes_current_bar():
    arrays = _arrays(
        open_values=[100, 100, 100, 100],
        high_values=[102, 104, 106, 120],
        low_values=[99, 99, 99, 99],
        close_values=[101, 103, 105, 119],
        quote_volume_values=[1000, 1000, 1000, 1000],
    )
    config = VolumePriceEfficiencyConfig(
        atr_window=1,
        volume_window=1,
        efficiency_lookback=2,
        efficiency_percentile=0.5,
        min_move_unit=0,
    )

    features = compute_features(arrays, config)

    assert np.isnan(features.efficiency_threshold[1, 0])
    assert features.efficiency_threshold[3, 0] == np.nanpercentile(
        features.efficiency[1:3, 0],
        50,
    )


def test_long_upper_wick_does_not_signal():
    arrays = _arrays(
        open_values=[100, 100, 100, 100],
        high_values=[101, 101, 101, 130],
        low_values=[99, 99, 99, 99],
        close_values=[100.5, 100.5, 100.5, 101],
        quote_volume_values=[1000, 1000, 1000, 1000],
    )
    config = VolumePriceEfficiencyConfig(
        atr_window=1,
        volume_window=1,
        efficiency_lookback=1,
        min_move_unit=0,
        min_volume_unit=0,
        min_close_position=0.7,
        min_body_ratio=0.4,
    )

    features = compute_features(arrays, config)

    assert features.close_position[3, 0] < 0.7
    assert features.body_ratio[3, 0] < 0.4
    assert not features.signal[3, 0]


def test_high_efficiency_bar_signals_when_all_filters_pass():
    arrays = _arrays(
        open_values=[100, 100, 100, 100],
        high_values=[101, 101, 101, 108],
        low_values=[99, 99, 99, 99],
        close_values=[100.5, 100.5, 100.5, 107],
        quote_volume_values=[1000, 1000, 1000, 1000],
    )
    config = VolumePriceEfficiencyConfig(
        atr_window=1,
        volume_window=1,
        efficiency_lookback=1,
        min_move_unit=0.5,
        min_volume_unit=0.3,
        min_close_position=0.7,
        min_body_ratio=0.4,
    )

    features = compute_features(arrays, config)

    assert features.signal.tolist() == [[False], [False], [False], [True]]


def test_bad_quality_row_never_signals():
    arrays = _arrays(
        open_values=[100, 100, 100, 100],
        high_values=[101, 101, 101, 108],
        low_values=[99, 99, 99, 99],
        close_values=[100.5, 100.5, 100.5, 107],
        quote_volume_values=[1000, 1000, 1000, 1000],
        quality_values=[True, True, True, False],
    )
    config = VolumePriceEfficiencyConfig(
        atr_window=1,
        volume_window=1,
        efficiency_lookback=1,
        min_move_unit=0,
        min_volume_unit=0,
    )

    features = compute_features(arrays, config)

    assert not features.signal[3, 0]


def test_build_signal_mask_reuses_features_with_different_filter_thresholds():
    arrays = _arrays(
        open_values=[100, 100, 100, 100],
        high_values=[101, 101, 101, 108],
        low_values=[99, 99, 99, 99],
        close_values=[100.5, 100.5, 100.5, 107],
        quote_volume_values=[1000, 1000, 1000, 1000],
    )
    loose_config = VolumePriceEfficiencyConfig(
        atr_window=1,
        volume_window=1,
        efficiency_lookback=1,
        min_move_unit=0.5,
        min_volume_unit=0.3,
        min_close_position=0.7,
        min_body_ratio=0.4,
    )
    strict_config = loose_config.model_copy(update={"min_move_unit": 10.0})

    features = compute_features(arrays, loose_config)
    strict_signal = build_signal_mask(arrays, features, strict_config)

    assert features.signal[3, 0]
    assert strict_signal.tolist() == [[False], [False], [False], [False]]
