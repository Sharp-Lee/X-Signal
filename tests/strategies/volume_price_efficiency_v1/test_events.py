from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.events import build_event_rows
from xsignal.strategies.volume_price_efficiency_v1.features import FeatureArrays


def _arrays(close_values=None, open_values=None) -> OhlcvArrays:
    close_values = close_values or [100, 110, 120, 130, 140]
    open_values = open_values or [100, 111, 121, 131, 141]
    row_count = len(close_values)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return OhlcvArrays(
        symbols=("BTCUSDT",),
        open_times=np.array([start + timedelta(days=i) for i in range(row_count)], dtype=object),
        open=np.array(open_values, dtype=np.float64).reshape(row_count, 1),
        high=np.array([value + 2 for value in close_values], dtype=np.float64).reshape(row_count, 1),
        low=np.array([value - 2 for value in close_values], dtype=np.float64).reshape(row_count, 1),
        close=np.array(close_values, dtype=np.float64).reshape(row_count, 1),
        quote_volume=np.full((row_count, 1), 1_000_000.0),
        quality=np.ones((row_count, 1), dtype=bool),
    )


def _features(signal_index=1, row_count=5) -> FeatureArrays:
    values = np.arange(row_count, dtype=np.float64).reshape(row_count, 1)
    signal = np.zeros((row_count, 1), dtype=bool)
    signal[signal_index, 0] = True
    return FeatureArrays(
        true_range=values + 10,
        atr=values + 20,
        move_unit=values + 30,
        volume_baseline=values + 40,
        volume_unit=values + 50,
        efficiency=values + 60,
        efficiency_threshold=values + 70,
        close_position=np.full((row_count, 1), 0.8),
        body_ratio=np.full((row_count, 1), 0.5),
        signal=signal,
    )


def test_signal_event_enters_next_open_and_uses_forward_close():
    arrays = _arrays()
    features = _features(signal_index=1)
    config = VolumePriceEfficiencyConfig(horizons=(1, 3), fee_bps=5, slippage_bps=5)

    events = build_event_rows(arrays, features, config)

    assert len(events) == 1
    event = events[0]
    assert event["symbol"] == "BTCUSDT"
    assert event["signal_open_time"] == "2026-01-02T00:00:00+00:00"
    assert event["decision_time"] == "2026-01-03T00:00:00+00:00"
    assert event["entry_open_time"] == "2026-01-03T00:00:00+00:00"
    assert event["entry_price"] == 121.0
    assert event["forward_return_1"] == 120.0 / 121.0 - 1.0
    assert event["forward_return_3"] == 140.0 / 121.0 - 1.0
    assert event["net_forward_return_1"] == event["forward_return_1"] - 0.002
    assert event["net_forward_return_3"] == event["forward_return_3"] - 0.002


def test_event_rows_include_signal_features():
    arrays = _arrays()
    features = _features(signal_index=1)

    event = build_event_rows(arrays, features, VolumePriceEfficiencyConfig(horizons=(1,)))[0]

    assert event["move_unit"] == 31.0
    assert event["volume_unit"] == 51.0
    assert event["efficiency"] == 61.0
    assert event["efficiency_threshold"] == 71.0
    assert event["close_position"] == 0.8
    assert event["body_ratio"] == 0.5
    assert event["quote_volume"] == 1_000_000.0
    assert event["volume_baseline"] == 41.0
    assert event["atr"] == 21.0


def test_unavailable_horizon_is_none():
    arrays = _arrays()
    features = _features(signal_index=3)

    event = build_event_rows(arrays, features, VolumePriceEfficiencyConfig(horizons=(1, 3)))[0]

    assert event["forward_return_1"] == 140.0 / 141.0 - 1.0
    assert event["forward_return_3"] is None
    assert event["net_forward_return_3"] is None


def test_missing_entry_open_skips_event():
    arrays = _arrays(open_values=[100, 111, np.nan, 131, 141])
    features = _features(signal_index=1)

    events = build_event_rows(arrays, features, VolumePriceEfficiencyConfig(horizons=(1,)))

    assert events == []


def test_missing_forward_close_makes_horizon_unavailable():
    arrays = _arrays(close_values=[100, 110, np.nan, 130, 140])
    features = _features(signal_index=1)

    event = build_event_rows(arrays, features, VolumePriceEfficiencyConfig(horizons=(1, 3)))[0]

    assert event["forward_return_1"] is None
    assert event["net_forward_return_1"] is None
    assert event["forward_return_3"] == 140.0 / 121.0 - 1.0
