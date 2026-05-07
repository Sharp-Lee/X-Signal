from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
from xsignal.strategies.momentum_rotation_v1.prepare import PreparedArrays
from xsignal.strategies.momentum_rotation_v1.signals import SignalArrays


@dataclass(frozen=True)
class BacktestResult:
    equity: np.ndarray
    period_returns: np.ndarray
    weights: np.ndarray
    turnover: np.ndarray
    costs: np.ndarray
    terminal_liquidation_cost: float = 0.0
    missing_weighted_return_count: int = 0
    missing_weighted_return_weight: float = 0.0


def _target_weights(score_row: np.ndarray, mask_row: np.ndarray, top_n: int) -> np.ndarray:
    eligible = np.flatnonzero(mask_row & np.isfinite(score_row))
    weights = np.zeros(score_row.shape[0], dtype=np.float64)
    if eligible.size == 0:
        return weights
    selected_count = min(top_n, eligible.size)
    eligible_scores = score_row[eligible]
    selected = eligible[np.argsort(eligible_scores)[-selected_count:]]
    weights[selected] = 1.0 / selected_count
    return weights


def _forward_fill_prices(values: np.ndarray) -> np.ndarray:
    filled = values.copy()
    for column_index in range(filled.shape[1]):
        last_price = np.nan
        for row_index in range(filled.shape[0]):
            value = filled[row_index, column_index]
            if np.isfinite(value):
                last_price = value
            elif np.isfinite(last_price):
                filled[row_index, column_index] = last_price
    return filled


def run_backtest(
    arrays: PreparedArrays,
    signals: SignalArrays,
    config: MomentumRotationConfig,
) -> BacktestResult:
    if arrays.close_1d.shape != signals.score.shape or signals.score.shape != signals.tradable_mask.shape:
        raise ValueError("array shapes do not match")
    t_count, n_count = arrays.close_1d.shape
    weights = np.zeros((t_count, n_count), dtype=np.float64)
    turnover = np.zeros(t_count, dtype=np.float64)
    costs = np.zeros(t_count, dtype=np.float64)
    previous = np.zeros(n_count, dtype=np.float64)
    cost_rate = (config.fee_bps + config.slippage_bps) / 10_000.0
    any_eligible = False
    for index in range(t_count):
        if index == 0 or index == t_count - 1:
            target = np.zeros(n_count, dtype=np.float64)
        else:
            target = _target_weights(
                signals.score[index - 1],
                signals.tradable_mask[index - 1],
                config.top_n,
            )
        if np.any(target):
            any_eligible = True
        weights[index] = target
        turnover[index] = np.sum(np.abs(target - previous))
        costs[index] = turnover[index] * cost_rate
        previous = target
    if not any_eligible:
        raise ValueError("No eligible symbols")
    raw_symbol_returns = arrays.close_1d[1:] / arrays.close_1d[:-1] - 1.0
    weighted_mask = weights[:-1] != 0.0
    missing_weighted_returns = (~np.isfinite(raw_symbol_returns)) & weighted_mask
    missing_weighted_return_count = int(np.count_nonzero(missing_weighted_returns))
    missing_weighted_return_weight = float(
        np.sum(np.abs(weights[:-1][missing_weighted_returns]))
    )
    pnl_close_1d = _forward_fill_prices(arrays.close_1d)
    symbol_returns = pnl_close_1d[1:] / pnl_close_1d[:-1] - 1.0
    if not np.all(np.isfinite(symbol_returns[weighted_mask])):
        raise ValueError("weighted portfolio returns contain NaN or infinite values")
    weighted_returns = np.zeros_like(symbol_returns)
    np.multiply(
        weights[:-1],
        symbol_returns,
        out=weighted_returns,
        where=weights[:-1] != 0.0,
    )
    period_returns = np.sum(weighted_returns, axis=1) - costs[:-1]
    if not np.all(np.isfinite(period_returns)):
        raise ValueError("portfolio returns contain NaN or infinite values")
    equity = np.empty(t_count, dtype=np.float64)
    equity[0] = config.initial_equity
    for index, period_return in enumerate(period_returns, start=1):
        equity[index] = equity[index - 1] * (1.0 + period_return)
    terminal_liquidation_cost = float(costs[-1])
    if terminal_liquidation_cost:
        equity[-1] *= 1.0 - terminal_liquidation_cost
    if not np.all(np.isfinite(equity)):
        raise ValueError("portfolio equity contains NaN or infinite values")
    return BacktestResult(
        equity=equity,
        period_returns=period_returns,
        weights=weights,
        turnover=turnover,
        costs=costs,
        terminal_liquidation_cost=terminal_liquidation_cost,
        missing_weighted_return_count=missing_weighted_return_count,
        missing_weighted_return_weight=missing_weighted_return_weight,
    )
