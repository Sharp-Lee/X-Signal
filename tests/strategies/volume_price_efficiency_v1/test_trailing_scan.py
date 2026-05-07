from __future__ import annotations

import numpy as np
import pytest

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing import (
    TrailingStopResult,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_scan import (
    build_trailing_scan_row,
    select_top_trailing_configs,
)


def _result(
    *,
    trade_count: int = 3,
    final_equity: float = 1.12,
    total_return: float = 0.12,
    max_drawdown: float = 0.03,
) -> TrailingStopResult:
    mid_equity = 1.0 - max_drawdown
    return TrailingStopResult(
        trades=[
            {
                "symbol": "BTCUSDT",
                "realized_return": 0.04,
                "net_realized_return": 0.038,
                "holding_bars": 2,
                "ignored_signal_count": 1,
            }
            for _ in range(trade_count)
        ],
        equity=np.array([1.0, mid_equity, final_equity], dtype=float),
        period_returns=np.array([mid_equity - 1.0, final_equity / mid_equity - 1.0], dtype=float),
        positions=np.zeros((3, 1), dtype=bool),
        stop_prices=np.full((3, 1), np.nan),
    )


def test_build_trailing_scan_row_scores_return_minus_drawdown():
    config = VolumePriceEfficiencyConfig(
        efficiency_percentile=0.9,
        min_move_unit=1.2,
        min_volume_unit=1.5,
        min_close_position=0.94,
        min_body_ratio=0.85,
    )

    row = build_trailing_scan_row(
        scan_id="trailscan",
        config=config,
        result=_result(total_return=0.12, max_drawdown=0.03),
        symbols=("BTCUSDT", "ETHUSDT"),
    )

    assert row["scan_id"] == "trailscan"
    assert row["config_hash"] == config.config_hash()
    assert row["efficiency_percentile"] == 0.9
    assert row["min_move_unit"] == 1.2
    assert row["symbol_count"] == 2
    assert row["trade_count"] == 3
    assert row["total_return"] == pytest.approx(0.12)
    assert row["max_drawdown"] == pytest.approx(0.03)
    assert row["score"] == pytest.approx(0.09)


def test_select_top_trailing_configs_filters_low_trade_counts_then_orders_by_score():
    rows = [
        {"config_hash": "low-trades", "trade_count": 5, "score": 1.0},
        {"config_hash": "b", "trade_count": 10, "score": 0.02},
        {"config_hash": "a", "trade_count": 12, "score": 0.04},
        {"config_hash": "c", "trade_count": 11, "score": 0.04},
    ]

    assert select_top_trailing_configs(rows, top_k=2, min_trades=10) == [
        {"config_hash": "a", "trade_count": 12, "score": 0.04},
        {"config_hash": "c", "trade_count": 11, "score": 0.04},
    ]
