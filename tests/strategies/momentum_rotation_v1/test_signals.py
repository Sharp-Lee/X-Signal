from __future__ import annotations

import numpy as np

from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
from xsignal.strategies.momentum_rotation_v1.prepare import PreparedArrays
from xsignal.strategies.momentum_rotation_v1.signals import compute_momentum_signals


def prepared(close_multiplier: float = 1.0) -> PreparedArrays:
    t = 40
    n = 2
    base = np.arange(1, t + 1, dtype=np.float64).reshape(t, 1)
    close_1d = close_multiplier * np.concatenate([base, base * 2], axis=1)
    close_1h = close_1d * 10
    close_4h = close_1d * 20
    return PreparedArrays(
        symbols=("BTCUSDT", "ETHUSDT"),
        rebalance_times=np.array(list(range(t)), dtype=object),
        close_1h=close_1h,
        close_4h=close_4h,
        close_1d=close_1d,
        quote_volume_1d=np.full((t, n), 1_000_000.0),
        complete_1h=np.ones((t, n), dtype=bool),
        complete_4h=np.ones((t, n), dtype=bool),
        complete_1d=np.ones((t, n), dtype=bool),
        quality_1h_24h=np.ones((t, n), dtype=bool),
        quality_4h_7d=np.ones((t, n), dtype=bool),
        quality_1d_30d=np.ones((t, n), dtype=bool),
    )


def test_compute_momentum_signals_uses_configured_windows_and_weights():
    config = MomentumRotationConfig()
    result = compute_momentum_signals(prepared(), config)

    expected_24h = 40 / 39 - 1
    expected_7d = 40 / 33 - 1
    expected_30d = 40 / 10 - 1
    expected = 0.4 * expected_24h + 0.4 * expected_7d + 0.2 * expected_30d

    assert np.isnan(result.score[:30]).all()
    assert result.tradable_mask[39, 0]
    assert result.score[39, 0] == expected


def test_compute_momentum_signals_requires_full_quality_window():
    arrays = prepared()
    arrays.quality_1h_24h[30, 0] = False

    result = compute_momentum_signals(arrays, MomentumRotationConfig())

    assert not result.tradable_mask[30, 0]
    assert np.isnan(result.score[30, 0])


def test_compute_momentum_signals_applies_liquidity_filter():
    arrays = prepared()
    arrays.quote_volume_1d[:, 1] = 1.0
    config = MomentumRotationConfig(min_rolling_7d_quote_volume=10_000.0)

    result = compute_momentum_signals(arrays, config)

    assert result.tradable_mask[39, 0]
    assert not result.tradable_mask[39, 1]
