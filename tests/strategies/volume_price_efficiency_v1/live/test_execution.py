from datetime import datetime, timezone

import pytest

from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig
from xsignal.strategies.volume_price_efficiency_v1.live.execution import (
    enter_long_with_protection,
    replace_trailing_stop,
    submit_pyramid_add,
)
from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    OrderIntentStatus,
    OrderIntentType,
    PositionState,
)
from xsignal.strategies.volume_price_efficiency_v1.live.position_store import (
    LivePositionRecord,
    get_live_position,
    update_live_position,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore


NOW = datetime(2026, 5, 9, tzinfo=timezone.utc)


class PersistAssertingExecutionBroker:
    def __init__(self, store: LiveStore) -> None:
        self.store = store
        self.calls = []

    def market_buy(self, *, symbol, quantity, client_order_id):
        intent = self.store.get_order_intent_by_client_order_id(client_order_id)
        assert intent is not None
        assert intent.status == OrderIntentStatus.PENDING_SUBMIT
        self.calls.append(("market_buy", symbol, quantity, client_order_id))
        return {"symbol": symbol, "clientOrderId": client_order_id, "status": "FILLED"}

    def place_stop_market_close(self, *, symbol, stop_price, client_order_id):
        intent = self.store.get_order_intent_by_client_order_id(client_order_id)
        assert intent is not None
        assert intent.status == OrderIntentStatus.PENDING_SUBMIT
        self.calls.append(("place_stop_market_close", symbol, stop_price, client_order_id))
        return {"symbol": symbol, "clientAlgoId": client_order_id, "algoStatus": "NEW"}

    def cancel_order(self, *, symbol, client_order_id):
        self.calls.append(("cancel_order", symbol, client_order_id))
        return {}


def test_enter_long_persists_entry_and_stop_before_submit(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    broker = PersistAssertingExecutionBroker(store)

    record = enter_long_with_protection(
        store=store,
        broker=broker,
        config=LiveTradingConfig(),
        environment="testnet",
        symbol="BTCUSDT",
        quantity=0.2,
        entry_price=100.0,
        atr=5.0,
        now=NOW,
    )

    assert record.state == PositionState.OPEN
    assert record.stop_price == 85.0
    assert record.next_add_trigger == 105.0
    assert get_live_position(store, record.position_id) == record
    assert [store.get_order_intent_by_client_order_id(call[3]).intent_type for call in broker.calls[:2]] == [
        OrderIntentType.ENTRY,
        OrderIntentType.STOP_PLACE,
    ]


def test_replace_trailing_stop_moves_only_upward_and_cancels_old_before_new_stop(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    position_id = store.create_position(symbol="BTCUSDT", state=PositionState.OPEN)
    record = LivePositionRecord(
        position_id=position_id,
        symbol="BTCUSDT",
        state=PositionState.OPEN,
        entry_price=100.0,
        quantity=0.2,
        highest_high=110.0,
        stop_price=90.0,
        atr_at_entry=5.0,
        next_add_trigger=None,
        add_count=0,
        active_stop_client_order_id="old-stop",
        last_decision_open_time=None,
    )
    update_live_position(store, record)
    broker = PersistAssertingExecutionBroker(store)

    unchanged = replace_trailing_stop(
        store=store,
        broker=broker,
        environment="testnet",
        record=record,
        candidate_stop_price=89.0,
        now=NOW,
    )
    updated = replace_trailing_stop(
        store=store,
        broker=broker,
        environment="testnet",
        record=record,
        candidate_stop_price=95.0,
        now=NOW,
    )

    assert unchanged.stop_price == 90.0
    assert updated.stop_price == 95.0
    assert [call[0] for call in broker.calls] == ["cancel_order", "place_stop_market_close"]
    assert get_live_position(store, position_id).active_stop_client_order_id != "old-stop"


def test_replace_trailing_stop_locks_position_when_new_stop_is_rejected_after_cancel(tmp_path):
    class RejectingStopBroker(PersistAssertingExecutionBroker):
        def place_stop_market_close(self, *, symbol, stop_price, client_order_id):
            super().place_stop_market_close(
                symbol=symbol,
                stop_price=stop_price,
                client_order_id=client_order_id,
            )
            raise RuntimeError("Binance API error -4130")

    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    position_id = store.create_position(symbol="BTCUSDT", state=PositionState.OPEN)
    record = LivePositionRecord(
        position_id=position_id,
        symbol="BTCUSDT",
        state=PositionState.OPEN,
        entry_price=100.0,
        quantity=0.2,
        highest_high=110.0,
        stop_price=90.0,
        atr_at_entry=5.0,
        next_add_trigger=None,
        add_count=0,
        active_stop_client_order_id="old-stop",
        last_decision_open_time=None,
    )
    update_live_position(store, record)
    broker = RejectingStopBroker(store)

    with pytest.raises(RuntimeError, match="-4130"):
        replace_trailing_stop(
            store=store,
            broker=broker,
            environment="testnet",
            record=record,
            candidate_stop_price=95.0,
            now=NOW,
        )

    failed_intents = [
        store.get_order_intent_by_client_order_id(call[3])
        for call in broker.calls
        if call[0] == "place_stop_market_close"
    ]
    assert [call[0] for call in broker.calls] == ["cancel_order", "place_stop_market_close"]
    assert failed_intents[0].status == OrderIntentStatus.ERROR
    assert get_live_position(store, position_id).state == PositionState.ERROR_LOCKED


def test_submit_pyramid_add_persists_add_before_market_order(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    position_id = store.create_position(symbol="BTCUSDT", state=PositionState.OPEN)
    record = LivePositionRecord(
        position_id=position_id,
        symbol="BTCUSDT",
        state=PositionState.OPEN,
        entry_price=100.0,
        quantity=0.2,
        highest_high=110.0,
        stop_price=90.0,
        atr_at_entry=5.0,
        next_add_trigger=105.0,
        add_count=0,
        active_stop_client_order_id="stop",
        last_decision_open_time=None,
    )
    update_live_position(store, record)
    broker = PersistAssertingExecutionBroker(store)

    updated = submit_pyramid_add(
        store=store,
        broker=broker,
        environment="testnet",
        record=record,
        quantity=0.1,
        execution_price=106.0,
        now=NOW,
    )

    assert updated.quantity == 0.3
    assert updated.add_count == 1
    assert updated.next_add_trigger is None
    add_client_id = broker.calls[0][3]
    assert store.get_order_intent_by_client_order_id(add_client_id).intent_type == (
        OrderIntentType.PYRAMID_ADD
    )
