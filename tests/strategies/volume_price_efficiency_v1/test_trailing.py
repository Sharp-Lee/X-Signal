from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.features import FeatureArrays
from xsignal.strategies.volume_price_efficiency_v1.trailing import (
    simulate_trailing_stop,
)


def _arrays(
    *,
    symbols=("BTCUSDT",),
    open_values=None,
    high_values=None,
    low_values=None,
    close_values=None,
    quote_volume_values=None,
) -> OhlcvArrays:
    row_count = len(open_values)
    symbol_count = len(symbols)
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return OhlcvArrays(
        symbols=tuple(symbols),
        open_times=np.array([start + timedelta(hours=4 * i) for i in range(row_count)], dtype=object),
        open=np.array(open_values, dtype=np.float64).reshape(row_count, symbol_count),
        high=np.array(high_values, dtype=np.float64).reshape(row_count, symbol_count),
        low=np.array(low_values, dtype=np.float64).reshape(row_count, symbol_count),
        close=np.array(close_values, dtype=np.float64).reshape(row_count, symbol_count),
        quote_volume=np.array(quote_volume_values, dtype=np.float64).reshape(row_count, symbol_count),
        quality=np.ones((row_count, symbol_count), dtype=bool),
    )


def _features(
    signal_rows: set[int],
    row_count: int,
    *,
    atr_values: list[float] | None = None,
) -> FeatureArrays:
    shape = (row_count, 1)
    signal = np.zeros(shape, dtype=bool)
    for row in signal_rows:
        signal[row, 0] = True
    values = np.arange(row_count, dtype=np.float64).reshape(shape)
    atr = np.array(atr_values, dtype=np.float64).reshape(shape) if atr_values else values + 1.0
    return FeatureArrays(
        true_range=values + 10,
        atr=atr,
        move_unit=values + 2.0,
        volume_baseline=values + 3.0,
        volume_unit=values + 4.0,
        efficiency=values + 5.0,
        efficiency_threshold=values + 6.0,
        close_position=np.full((row_count, 1), 0.8),
        body_ratio=np.full((row_count, 1), 0.5),
        signal=signal,
    )


def test_trailing_simulation_exits_at_two_atr_below_highest_high():
    arrays = _arrays(
        open_values=[100, 100, 95],
        high_values=[105, 101, 96],
        low_values=[95, 89, 94],
        close_values=[104, 95, 95],
        quote_volume_values=[1000, 1000, 1000],
    )
    features = _features({0}, row_count=3, atr_values=[5, 5, 5])

    result = simulate_trailing_stop(
        arrays,
        features,
        VolumePriceEfficiencyConfig(fee_bps=0, slippage_bps=0),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade["entry_open_time"] == "2026-01-01T04:00:00+00:00"
    assert trade["exit_time"] == "2026-01-01T04:00:00+00:00"
    assert trade["entry_price"] == 100.0
    assert trade["exit_price"] == 90.0
    assert trade["stop_price_at_exit"] == 90.0
    assert trade["holding_bars"] == 1
    assert trade["realized_return"] == pytest.approx(-0.10)
    assert result.period_returns.tolist() == pytest.approx([-0.10, 0.0])
    assert result.equity.tolist() == pytest.approx([1.0, 0.90, 0.90])


def test_trailing_simulation_updates_stop_when_atr_changes():
    arrays = _arrays(
        open_values=[100, 100, 109],
        high_values=[105, 110, 111],
        low_values=[95, 95, 107],
        close_values=[104, 109, 108],
        quote_volume_values=[1000, 1000, 1000],
    )
    features = _features({0}, row_count=3, atr_values=[5, 1, 1])

    result = simulate_trailing_stop(
        arrays,
        features,
        VolumePriceEfficiencyConfig(fee_bps=0, slippage_bps=0),
    )

    assert len(result.trades) == 1
    trade = result.trades[0]
    assert trade["exit_time"] == "2026-01-01T08:00:00+00:00"
    assert trade["stop_price_at_exit"] == 108.0
    assert trade["exit_price"] == 108.0
    assert trade["highest_high"] == 110.0
    assert trade["atr_at_exit"] == 1.0
    assert trade["realized_return"] == pytest.approx(0.08)


def test_trailing_simulation_ignores_signals_while_position_is_open():
    arrays = _arrays(
        open_values=[100, 100, 101, 102, 103],
        high_values=[105, 101, 102, 103, 104],
        low_values=[95, 95, 95, 94, 92],
        close_values=[104, 101, 102, 103, 93],
        quote_volume_values=[1000, 1000, 1000, 1000, 1000],
    )
    features = _features({0, 2}, row_count=5, atr_values=[5, 5, 5, 5, 5])

    result = simulate_trailing_stop(
        arrays,
        features,
        VolumePriceEfficiencyConfig(fee_bps=0, slippage_bps=0),
    )

    assert len(result.trades) == 1
    assert result.trades[0]["signal_open_time"] == "2026-01-01T00:00:00+00:00"
    assert result.trades[0]["ignored_signal_count"] == 1


def test_trailing_simulation_allows_reentry_after_exit():
    arrays = _arrays(
        open_values=[100, 100, 96, 100, 99],
        high_values=[105, 101, 100, 101, 100],
        low_values=[95, 89, 95, 89, 95],
        close_values=[104, 95, 99, 95, 99],
        quote_volume_values=[1000, 1000, 1000, 1000, 1000],
    )
    features = _features({0, 2}, row_count=5, atr_values=[5, 5, 5, 5, 5])

    result = simulate_trailing_stop(
        arrays,
        features,
        VolumePriceEfficiencyConfig(fee_bps=0, slippage_bps=0),
    )

    assert len(result.trades) == 2
    assert [trade["signal_open_time"] for trade in result.trades] == [
        "2026-01-01T00:00:00+00:00",
        "2026-01-01T08:00:00+00:00",
    ]
    assert [trade["entry_open_time"] for trade in result.trades] == [
        "2026-01-01T04:00:00+00:00",
        "2026-01-01T12:00:00+00:00",
    ]


def test_trailing_simulation_locks_each_symbol_independently():
    arrays = _arrays(
        symbols=("BTCUSDT", "ETHUSDT"),
        open_values=[[100, 200], [100, 201], [101, 200], [102, 202], [103, 203]],
        high_values=[[105, 205], [101, 206], [102, 201], [103, 203], [104, 204]],
        low_values=[[95, 195], [95, 196], [95, 191], [94, 192], [89, 189]],
        close_values=[[104, 204], [101, 205], [102, 200], [103, 202], [93, 190]],
        quote_volume_values=[[1000, 1000]] * 5,
    )
    row_count = 5
    shape = (row_count, 2)
    values = np.arange(row_count * 2, dtype=np.float64).reshape(shape)
    signal = np.zeros(shape, dtype=bool)
    signal[0, 0] = True
    signal[1, 1] = True
    features = FeatureArrays(
        true_range=values + 10,
        atr=np.full(shape, 5.0),
        move_unit=values + 2.0,
        volume_baseline=values + 3.0,
        volume_unit=values + 4.0,
        efficiency=values + 5.0,
        efficiency_threshold=values + 6.0,
        close_position=np.full(shape, 0.8),
        body_ratio=np.full(shape, 0.5),
        signal=signal,
    )

    result = simulate_trailing_stop(
        arrays,
        features,
        VolumePriceEfficiencyConfig(fee_bps=0, slippage_bps=0),
    )

    assert [trade["symbol"] for trade in result.trades] == ["BTCUSDT", "ETHUSDT"]
    assert [trade["entry_open_time"] for trade in result.trades] == [
        "2026-01-01T04:00:00+00:00",
        "2026-01-01T08:00:00+00:00",
    ]
    assert result.positions[2].tolist() == [True, True]
