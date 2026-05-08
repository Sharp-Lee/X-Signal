from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from xsignal.strategies.volume_price_efficiency_v1.live.broker import FakeBroker
from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig


@dataclass(frozen=True)
class LiveSymbolState:
    symbol: str
    position_state: str
    quantity: float
    entry_price: float | None
    highest_high: float | None
    stop_price: float | None
    add_count: int
    active_stop_order_id: str | None = None
    next_add_trigger: float | None = None
    closed_at: datetime | None = None

    @classmethod
    def flat(cls, symbol: str) -> "LiveSymbolState":
        return cls(symbol, "FLAT", 0.0, None, None, None, 0)

    @classmethod
    def open(
        cls,
        *,
        symbol: str,
        quantity: float,
        entry_price: float,
        highest_high: float,
        stop_price: float,
        add_count: int,
        next_add_trigger: float | None = None,
    ) -> "LiveSymbolState":
        return cls(
            symbol=symbol,
            position_state="OPEN",
            quantity=quantity,
            entry_price=entry_price,
            highest_high=highest_high,
            stop_price=stop_price,
            add_count=add_count,
            active_stop_order_id="active-stop",
            next_add_trigger=next_add_trigger,
        )


def on_signal(
    *,
    state: LiveSymbolState,
    broker: FakeBroker,
    config: LiveTradingConfig,
    entry_price: float,
    atr: float,
    quantity: float,
    now: datetime,
) -> LiveSymbolState:
    if state.position_state != "FLAT":
        return state
    broker.market_buy(symbol=state.symbol, quantity=quantity)
    stop_price = entry_price - config.atr_multiplier * atr
    broker.place_stop_market_close(symbol=state.symbol, stop_price=stop_price)
    return LiveSymbolState.open(
        symbol=state.symbol,
        quantity=quantity,
        entry_price=entry_price,
        highest_high=entry_price,
        stop_price=stop_price,
        add_count=0,
        next_add_trigger=entry_price + config.pyramid_add_step_atr * atr,
    )


def update_trailing_stop(
    *,
    state: LiveSymbolState,
    broker: FakeBroker,
    config: LiveTradingConfig,
    bar_high: float,
    atr: float,
    now: datetime,
) -> LiveSymbolState:
    if state.position_state != "OPEN" or state.highest_high is None or state.stop_price is None:
        return state
    highest_high = max(state.highest_high, bar_high)
    next_stop = highest_high - config.atr_multiplier * atr
    if next_stop <= state.stop_price:
        return replace(state, highest_high=highest_high)
    broker.place_stop_market_close(symbol=state.symbol, stop_price=next_stop)
    if state.active_stop_order_id is not None:
        broker.cancel_order(state.active_stop_order_id)
    return replace(
        state,
        highest_high=highest_high,
        stop_price=next_stop,
        active_stop_order_id="active-stop",
    )


def arm_pyramid_add(
    *,
    state: LiveSymbolState,
    config: LiveTradingConfig,
    bar_high: float,
) -> LiveSymbolState:
    if (
        state.position_state != "OPEN"
        or state.next_add_trigger is None
        or state.add_count >= config.pyramid_max_adds
        or bar_high < state.next_add_trigger
    ):
        return state
    return replace(state, position_state="ADD_ARMED")


def submit_pyramid_add(
    *,
    state: LiveSymbolState,
    broker: FakeBroker,
    config: LiveTradingConfig,
    execution_price: float,
    quantity: float,
    atr: float,
    now: datetime,
) -> LiveSymbolState:
    if state.position_state != "ADD_ARMED":
        return state
    if state.next_add_trigger is None or execution_price < state.next_add_trigger:
        return replace(state, position_state="OPEN", next_add_trigger=None)
    broker.market_buy(symbol=state.symbol, quantity=quantity)
    add_count = state.add_count + 1
    next_add_trigger = (
        execution_price + config.pyramid_add_step_atr * atr
        if add_count < config.pyramid_max_adds
        else None
    )
    return replace(
        state,
        position_state="OPEN",
        quantity=state.quantity + quantity,
        add_count=add_count,
        next_add_trigger=next_add_trigger,
    )


def on_stop_fill(*, state: LiveSymbolState, fill_price: float, now: datetime) -> LiveSymbolState:
    return replace(
        state,
        position_state="CLOSED",
        quantity=0.0,
        stop_price=None,
        active_stop_order_id=None,
        next_add_trigger=None,
        closed_at=now,
    )
