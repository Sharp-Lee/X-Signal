from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.features import FeatureArrays


@dataclass(frozen=True)
class TrailingStopResult:
    trades: list[dict[str, Any]]
    equity: np.ndarray
    period_returns: np.ndarray
    positions: np.ndarray
    stop_prices: np.ndarray


@dataclass
class _PyramidLot:
    entry_index: int
    entry_price: float
    atr_at_entry: float | None


@dataclass
class _OpenPosition:
    signal_index: int
    entry_index: int
    entry_price: float
    atr_at_entry: float | None
    highest_high: float
    stop_price: float
    lots: list[_PyramidLot]
    add_count: int = 0
    next_add_trigger: float | None = None
    ignored_signal_count: int = 0


def _json_time(value) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _as_float_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _timeframe_delta(arrays: OhlcvArrays) -> timedelta:
    if len(arrays.open_times) >= 2:
        return arrays.open_times[1] - arrays.open_times[0]
    return timedelta(days=1)


def _valid_entry_open(arrays: OhlcvArrays, entry_index: int, s_index: int) -> bool:
    if entry_index >= arrays.open.shape[0]:
        return False
    entry_price = arrays.open[entry_index, s_index]
    return bool(np.isfinite(entry_price) and entry_price > 0.0 and arrays.quality[entry_index, s_index])


def _stop_from(highest_high: float, atr: float, multiplier: float) -> float | None:
    if not np.isfinite(highest_high) or not np.isfinite(atr) or atr <= 0.0:
        return None
    return float(highest_high - multiplier * atr)


def _trade_row(
    *,
    arrays: OhlcvArrays,
    features: FeatureArrays,
    config: VolumePriceEfficiencyConfig,
    position: _OpenPosition,
    s_index: int,
    exit_index: int,
    exit_price: float,
    stop_price_at_exit: float,
) -> dict[str, Any]:
    signal_index = position.signal_index
    entry_index = position.entry_index
    gross_returns = [float(exit_price / lot.entry_price - 1.0) for lot in position.lots]
    net_returns = [gross_return - config.round_trip_cost for gross_return in gross_returns]
    gross_return = float(sum(gross_returns))
    net_return = float(sum(net_returns))
    lot_count = len(position.lots)
    lot_entry_prices = [float(lot.entry_price) for lot in position.lots]
    average_entry_price = float(sum(lot_entry_prices) / lot_count)
    signal_time = arrays.open_times[signal_index]
    return {
        "symbol": arrays.symbols[s_index],
        "signal_open_time": _json_time(signal_time),
        "decision_time": _json_time(signal_time + _timeframe_delta(arrays)),
        "entry_open_time": _json_time(arrays.open_times[entry_index]),
        "last_entry_open_time": _json_time(arrays.open_times[position.lots[-1].entry_index]),
        "exit_time": _json_time(arrays.open_times[exit_index]),
        "entry_price": float(position.entry_price),
        "average_entry_price": average_entry_price,
        "exit_price": float(exit_price),
        "stop_price_at_exit": float(stop_price_at_exit),
        "atr_at_entry": position.atr_at_entry,
        "atr_at_exit": _as_float_or_none(features.atr[exit_index, s_index]),
        "highest_high": float(position.highest_high),
        "realized_return": gross_return,
        "net_realized_return": net_return,
        "lot_count": lot_count,
        "add_count": int(position.add_count),
        "average_lot_net_realized_return": float(sum(net_returns) / lot_count),
        "best_lot_net_realized_return": float(max(net_returns)),
        "worst_lot_net_realized_return": float(min(net_returns)),
        "holding_bars": int(exit_index - entry_index + 1),
        "ignored_signal_count": int(position.ignored_signal_count),
        "move_unit": _as_float_or_none(features.move_unit[signal_index, s_index]),
        "volume_unit": _as_float_or_none(features.volume_unit[signal_index, s_index]),
        "efficiency": _as_float_or_none(features.efficiency[signal_index, s_index]),
        "efficiency_threshold": _as_float_or_none(
            features.efficiency_threshold[signal_index, s_index]
        ),
        "close_position": _as_float_or_none(features.close_position[signal_index, s_index]),
        "body_ratio": _as_float_or_none(features.body_ratio[signal_index, s_index]),
        "quote_volume": _as_float_or_none(arrays.quote_volume[signal_index, s_index]),
        "volume_baseline": _as_float_or_none(features.volume_baseline[signal_index, s_index]),
    }


def simulate_trailing_stop(
    arrays: OhlcvArrays,
    features: FeatureArrays,
    config: VolumePriceEfficiencyConfig,
    *,
    atr_multiplier: float = 2.0,
    pyramid_add_step_atr: float | None = None,
    pyramid_max_adds: int = 0,
) -> TrailingStopResult:
    if arrays.open.shape != features.signal.shape or features.signal.shape != features.atr.shape:
        raise ValueError("array and feature shapes do not match")
    if atr_multiplier <= 0.0:
        raise ValueError("atr_multiplier must be positive")
    if pyramid_max_adds < 0:
        raise ValueError("pyramid_max_adds must be non-negative")
    if pyramid_max_adds > 0 and (pyramid_add_step_atr is None or pyramid_add_step_atr <= 0.0):
        raise ValueError("pyramid_add_step_atr must be positive when pyramid_max_adds is enabled")

    t_count, s_count = arrays.open.shape
    positions = np.zeros((t_count, s_count), dtype=bool)
    stop_prices = np.full((t_count, s_count), np.nan, dtype=np.float64)
    trades: list[dict[str, Any]] = []
    completed_returns_by_bar: list[list[tuple[int, float]]] = [[] for _ in range(t_count)]

    for s_index in range(s_count):
        open_position: _OpenPosition | None = None
        scheduled_signal_index: int | None = None
        scheduled_add = False

        for t_index in range(t_count):
            opened_this_bar = False
            if open_position is None and scheduled_signal_index is not None:
                if t_index == scheduled_signal_index + 1 and _valid_entry_open(
                    arrays,
                    t_index,
                    s_index,
                ):
                    entry_price = float(arrays.open[t_index, s_index])
                    entry_atr = _as_float_or_none(features.atr[scheduled_signal_index, s_index])
                    initial_stop = _stop_from(
                        entry_price,
                        features.atr[scheduled_signal_index, s_index],
                        atr_multiplier,
                    )
                    if initial_stop is not None:
                        open_position = _OpenPosition(
                            signal_index=scheduled_signal_index,
                            entry_index=t_index,
                            entry_price=entry_price,
                            atr_at_entry=entry_atr,
                            highest_high=entry_price,
                            stop_price=initial_stop,
                            lots=[
                                _PyramidLot(
                                    entry_index=t_index,
                                    entry_price=entry_price,
                                    atr_at_entry=entry_atr,
                                )
                            ],
                            next_add_trigger=(
                                entry_price
                                + pyramid_add_step_atr * features.atr[scheduled_signal_index, s_index]
                                if pyramid_max_adds > 0
                                and pyramid_add_step_atr is not None
                                and np.isfinite(features.atr[scheduled_signal_index, s_index])
                                else None
                            ),
                        )
                        opened_this_bar = True
                if t_index >= scheduled_signal_index + 1:
                    scheduled_signal_index = None

            if open_position is not None:
                positions[t_index, s_index] = True
                stop_prices[t_index, s_index] = open_position.stop_price
                if scheduled_add:
                    add_confirmed = False
                    if _valid_entry_open(arrays, t_index, s_index):
                        entry_price = float(arrays.open[t_index, s_index])
                        add_confirmed = (
                            open_position.next_add_trigger is not None
                            and entry_price >= open_position.next_add_trigger
                        )
                    if add_confirmed:
                        open_position.lots.append(
                            _PyramidLot(
                                entry_index=t_index,
                                entry_price=entry_price,
                                atr_at_entry=_as_float_or_none(features.atr[t_index, s_index]),
                            )
                        )
                        open_position.add_count += 1
                        if (
                            open_position.add_count < pyramid_max_adds
                            and pyramid_add_step_atr is not None
                            and np.isfinite(features.atr[t_index, s_index])
                        ):
                            open_position.next_add_trigger = (
                                entry_price + pyramid_add_step_atr * features.atr[t_index, s_index]
                            )
                        else:
                            open_position.next_add_trigger = None
                    else:
                        open_position.next_add_trigger = None
                    scheduled_add = False
                if features.signal[t_index, s_index] and t_index != open_position.signal_index:
                    open_position.ignored_signal_count += 1
                if (
                    arrays.quality[t_index, s_index]
                    and np.isfinite(arrays.low[t_index, s_index])
                    and arrays.low[t_index, s_index] <= open_position.stop_price
                ):
                    trade = _trade_row(
                        arrays=arrays,
                        features=features,
                        config=config,
                        position=open_position,
                        s_index=s_index,
                        exit_index=t_index,
                        exit_price=open_position.stop_price,
                        stop_price_at_exit=open_position.stop_price,
                    )
                    trades.append(trade)
                    completed_returns_by_bar[t_index].append(
                        (s_index, float(trade["net_realized_return"]))
                    )
                    open_position = None
                    continue

                bar_high = arrays.high[t_index, s_index]
                if arrays.quality[t_index, s_index] and np.isfinite(bar_high):
                    open_position.highest_high = max(open_position.highest_high, float(bar_high))
                if (
                    open_position.next_add_trigger is not None
                    and open_position.add_count < pyramid_max_adds
                    and arrays.quality[t_index, s_index]
                    and np.isfinite(bar_high)
                    and bar_high >= open_position.next_add_trigger
                    and t_index + 1 < t_count
                ):
                    scheduled_add = True
                next_stop = _stop_from(
                    open_position.highest_high,
                    features.atr[t_index, s_index],
                    atr_multiplier,
                )
                if next_stop is not None:
                    open_position.stop_price = max(open_position.stop_price, next_stop)
                continue

            if not opened_this_bar and features.signal[t_index, s_index]:
                scheduled_signal_index = t_index if t_index + 1 < t_count else None

    equity = np.ones(t_count, dtype=np.float64)
    symbol_equity = np.full(s_count, 1.0 / s_count, dtype=np.float64) if s_count else np.array([])
    for index in range(1, t_count):
        for s_index, net_return in completed_returns_by_bar[index]:
            symbol_equity[s_index] *= 1.0 + net_return
        equity[index] = float(symbol_equity.sum()) if s_count else 1.0
    period_returns = equity[1:] / equity[:-1] - 1.0 if t_count > 1 else np.array([], dtype=np.float64)
    return TrailingStopResult(
        trades=trades,
        equity=equity,
        period_returns=period_returns,
        positions=positions,
        stop_prices=stop_prices,
    )
