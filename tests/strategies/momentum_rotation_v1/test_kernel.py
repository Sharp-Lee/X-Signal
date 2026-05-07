from __future__ import annotations

import numpy as np
import pytest

from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
from xsignal.strategies.momentum_rotation_v1.kernel import run_backtest
from xsignal.strategies.momentum_rotation_v1.prepare import PreparedArrays
from xsignal.strategies.momentum_rotation_v1.signals import SignalArrays


def arrays() -> PreparedArrays:
    close = np.array(
        [
            [100.0, 100.0, 100.0],
            [110.0, 90.0, 100.0],
            [121.0, 81.0, 100.0],
        ]
    )
    return PreparedArrays(
        symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        rebalance_times=np.array([0, 1, 2], dtype=object),
        close_1h=close,
        close_4h=close,
        close_1d=close,
        quote_volume_1d=np.ones_like(close),
        complete_1h=np.ones_like(close, dtype=bool),
        complete_4h=np.ones_like(close, dtype=bool),
        complete_1d=np.ones_like(close, dtype=bool),
        quality_1h_24h=np.ones_like(close, dtype=bool),
        quality_4h_7d=np.ones_like(close, dtype=bool),
        quality_1d_30d=np.ones_like(close, dtype=bool),
    )


def test_run_backtest_uses_weights_on_next_period_returns():
    signal = SignalArrays(
        score=np.array(
            [
                [3.0, 2.0, 1.0],
                [1.0, 3.0, 2.0],
                [1.0, 2.0, 3.0],
            ]
        ),
        tradable_mask=np.ones((3, 3), dtype=bool),
    )
    result = run_backtest(
        arrays(),
        signal,
        MomentumRotationConfig(top_n=1, fee_bps=0, slippage_bps=0),
    )

    assert result.weights.shape == (3, 3)
    assert result.weights[0].tolist() == [1.0, 0.0, 0.0]
    assert result.period_returns.tolist() == pytest.approx([0.1, -0.1])
    assert result.equity.tolist() == pytest.approx([1.0, 1.1, 0.99])


def test_run_backtest_applies_turnover_costs():
    signal = SignalArrays(
        score=np.array([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0], [1.0, 2.0, 3.0]]),
        tradable_mask=np.ones((3, 3), dtype=bool),
    )
    result = run_backtest(
        arrays(),
        signal,
        MomentumRotationConfig(top_n=1, fee_bps=10, slippage_bps=0),
    )

    assert result.turnover.tolist() == [1.0, 2.0, 2.0]
    assert result.costs[0] == 0.001


def test_run_backtest_fails_when_no_symbols_are_eligible():
    signal = SignalArrays(score=np.full((3, 3), np.nan), tradable_mask=np.zeros((3, 3), dtype=bool))

    with pytest.raises(ValueError, match="No eligible symbols"):
        run_backtest(arrays(), signal, MomentumRotationConfig())
