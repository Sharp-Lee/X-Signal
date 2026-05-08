from datetime import datetime, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.broker import FakeBroker
from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig
from xsignal.strategies.volume_price_efficiency_v1.live.state_machine import (
    LiveSymbolState,
    arm_pyramid_add,
    on_signal,
    on_stop_fill,
    submit_pyramid_add,
    update_trailing_stop,
)


NOW = datetime(2026, 5, 9, tzinfo=timezone.utc)


def test_signal_enters_long_and_places_protective_stop():
    broker = FakeBroker()
    state = LiveSymbolState.flat("BTCUSDT")
    next_state = on_signal(
        state=state,
        broker=broker,
        config=LiveTradingConfig(),
        entry_price=100.0,
        atr=5.0,
        quantity=0.2,
        now=NOW,
    )
    assert next_state.position_state == "OPEN"
    assert next_state.stop_price == 85.0
    assert next_state.next_add_trigger == 105.0
    assert broker.orders[-1].order_type == "STOP_MARKET"
    assert broker.orders[-1].side == "SELL"


def test_open_symbol_ignores_second_signal():
    broker = FakeBroker()
    state = LiveSymbolState.open(
        symbol="BTCUSDT",
        quantity=0.2,
        entry_price=100.0,
        highest_high=105.0,
        stop_price=90.0,
        add_count=0,
    )
    next_state = on_signal(
        state=state,
        broker=broker,
        config=LiveTradingConfig(),
        entry_price=106.0,
        atr=4.0,
        quantity=0.2,
        now=NOW,
    )
    assert next_state is state
    assert broker.orders == []


def test_trailing_stop_moves_up_protective_first():
    broker = FakeBroker()
    state = LiveSymbolState.open(
        symbol="BTCUSDT",
        quantity=0.2,
        entry_price=100.0,
        highest_high=110.0,
        stop_price=90.0,
        add_count=0,
    )
    next_state = update_trailing_stop(
        state=state,
        broker=broker,
        config=LiveTradingConfig(),
        bar_high=120.0,
        atr=5.0,
        now=NOW,
    )
    assert next_state.stop_price == 105.0
    assert [order.order_type for order in broker.orders] == ["STOP_MARKET"]
    assert broker.cancelled_order_ids == ["active-stop"]


def test_pyramid_add_arms_and_submits_only_if_trigger_holds():
    broker = FakeBroker()
    state = LiveSymbolState.open(
        symbol="BTCUSDT",
        quantity=0.2,
        entry_price=100.0,
        highest_high=104.0,
        stop_price=90.0,
        add_count=0,
        next_add_trigger=105.0,
    )
    armed = arm_pyramid_add(state=state, config=LiveTradingConfig(), bar_high=106.0)
    assert armed.position_state == "ADD_ARMED"
    added = submit_pyramid_add(
        state=armed,
        broker=broker,
        config=LiveTradingConfig(),
        execution_price=105.5,
        quantity=0.2,
        atr=5.0,
        now=NOW,
    )
    assert added.position_state == "OPEN"
    assert added.quantity == 0.4
    assert added.add_count == 1
    assert added.next_add_trigger is None
    assert broker.orders[-1].side == "BUY"


def test_pyramid_add_discards_when_trigger_is_lost():
    broker = FakeBroker()
    state = LiveSymbolState.open(
        symbol="BTCUSDT",
        quantity=0.2,
        entry_price=100.0,
        highest_high=104.0,
        stop_price=90.0,
        add_count=0,
        next_add_trigger=105.0,
    )
    armed = arm_pyramid_add(state=state, config=LiveTradingConfig(), bar_high=106.0)
    discarded = submit_pyramid_add(
        state=armed,
        broker=broker,
        config=LiveTradingConfig(),
        execution_price=104.5,
        quantity=0.2,
        atr=5.0,
        now=NOW,
    )
    assert discarded.position_state == "OPEN"
    assert discarded.quantity == 0.2
    assert discarded.add_count == 0
    assert discarded.next_add_trigger is None
    assert broker.orders == []


def test_stop_fill_closes_and_unlocks_symbol():
    state = LiveSymbolState.open(
        symbol="BTCUSDT",
        quantity=0.2,
        entry_price=100.0,
        highest_high=110.0,
        stop_price=90.0,
        add_count=0,
    )
    next_state = on_stop_fill(state=state, fill_price=90.0, now=NOW)
    assert next_state.position_state == "CLOSED"
    assert next_state.quantity == 0.0
