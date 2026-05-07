from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.baseline import build_baseline_events
from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.features import FeatureArrays


def _arrays(row_count=10) -> OhlcvArrays:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    times = [start + timedelta(hours=4 * i) for i in range(row_count)]
    base = np.arange(row_count * 2, dtype=np.float64).reshape(row_count, 2) + 100.0
    return OhlcvArrays(
        symbols=("BTCUSDT", "ETHUSDT"),
        open_times=np.array(times, dtype=object),
        open=base.copy(),
        high=base + 2.0,
        low=base - 2.0,
        close=base + 1.0,
        quote_volume=np.full((row_count, 2), 1_000_000.0),
        quality=np.ones((row_count, 2), dtype=bool),
    )


def _features(row_count=10) -> FeatureArrays:
    values = np.arange(row_count * 2, dtype=np.float64).reshape(row_count, 2)
    signal = np.zeros((row_count, 2), dtype=bool)
    signal[1, 0] = True
    signal[3, 0] = True
    signal[2, 1] = True
    return FeatureArrays(
        true_range=values + 10,
        atr=values + 20,
        move_unit=values + 30,
        volume_baseline=values + 40,
        volume_unit=values + 50,
        efficiency=values + 60,
        efficiency_threshold=values + 70,
        close_position=np.full((row_count, 2), 0.8),
        body_ratio=np.full((row_count, 2), 0.5),
        signal=signal,
    )


def test_baseline_samples_non_signal_rows_by_symbol_month():
    arrays = _arrays()
    features = _features()

    rows = build_baseline_events(arrays, features, VolumePriceEfficiencyConfig(horizons=(1,)))

    assert len(rows) == 3
    assert {row["matched_signal_month"] for row in rows} == {"2026-01"}
    btc_rows = [row for row in rows if row["symbol"] == "BTCUSDT"]
    eth_rows = [row for row in rows if row["symbol"] == "ETHUSDT"]
    assert len(btc_rows) == 2
    assert len(eth_rows) == 1
    assert {row["signal_open_time"] for row in btc_rows}.isdisjoint(
        {"2026-01-01T04:00:00+00:00", "2026-01-01T12:00:00+00:00"}
    )
    assert all(row["matched_signal_count_for_symbol_month"] == 2 for row in btc_rows)
    assert all(row["matched_signal_count_for_symbol_month"] == 1 for row in eth_rows)


def test_baseline_sampling_is_deterministic_for_same_seed():
    arrays = _arrays()
    features = _features()
    config = VolumePriceEfficiencyConfig(horizons=(1,), baseline_seed=123)

    first = build_baseline_events(arrays, features, config)
    second = build_baseline_events(arrays, features, config)

    assert first == second


def test_baseline_skips_rows_without_entry_or_forward_return():
    arrays = _arrays(row_count=5)
    arrays.open[2, 0] = np.nan
    arrays.close[4, 0] = np.nan
    features = _features(row_count=5)
    features.signal[:, :] = False
    features.signal[1, 0] = True
    features.signal[2, 0] = True

    rows = build_baseline_events(arrays, features, VolumePriceEfficiencyConfig(horizons=(2,)))

    assert len(rows) < 2
    assert all(row["forward_return_2"] is not None for row in rows)


def test_baseline_respects_quality_mask():
    arrays = _arrays()
    arrays.quality[0, 0] = False
    arrays.quality[4, 0] = False
    features = _features()

    rows = build_baseline_events(arrays, features, VolumePriceEfficiencyConfig(horizons=(1,)))

    assert all(row["signal_open_time"] not in {"2026-01-01T00:00:00+00:00", "2026-01-01T16:00:00+00:00"} for row in rows)
