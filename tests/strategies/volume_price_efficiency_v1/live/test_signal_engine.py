from datetime import datetime, timezone

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.live.signal_engine import (
    build_market_regime_mask,
    closed_bar_view,
)


def _arrays() -> OhlcvArrays:
    times = np.array(
        [
            datetime(2026, 5, 7, tzinfo=timezone.utc),
            datetime(2026, 5, 8, tzinfo=timezone.utc),
            datetime(2026, 5, 9, tzinfo=timezone.utc),
        ],
        dtype=object,
    )
    values = np.array([[100.0], [110.0], [120.0]])
    return OhlcvArrays(
        symbols=("BTCUSDT",),
        open_times=times,
        open=values.copy(),
        high=values.copy() + 1,
        low=values.copy() - 1,
        close=values.copy(),
        quote_volume=np.full((3, 1), 1000.0),
        quality=np.ones((3, 1), dtype=bool),
    )


def test_closed_bar_view_excludes_forming_bar():
    arrays = _arrays()
    view = closed_bar_view(arrays, closed_open_time=arrays.open_times[1])
    assert view.open.shape == (2, 1)
    assert view.open_times[-1] == arrays.open_times[1]


def test_market_regime_mask_uses_closed_market_average():
    close = np.array(
        [
            [100.0, 100.0],
            [90.0, 90.0],
            [80.0, 80.0],
            [91.0, 99.0],
        ]
    )
    mask = build_market_regime_mask(close, lookback_bars=2, min_return=-0.10)
    assert not mask[2, 0]
    assert not mask[2, 1]
    assert mask[3, 0]
    assert mask[3, 1]
