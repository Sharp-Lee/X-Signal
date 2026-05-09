from datetime import datetime, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.models import PositionState
from xsignal.strategies.volume_price_efficiency_v1.live.position_store import (
    LivePositionRecord,
    get_live_position,
    list_active_live_positions,
    update_live_position,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore


def test_live_position_fields_are_persisted_and_loaded(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    position_id = store.create_position(symbol="BTCUSDT", state=PositionState.OPEN)
    decision_time = datetime(2026, 5, 8, tzinfo=timezone.utc)

    update_live_position(
        store,
        LivePositionRecord(
            position_id=position_id,
            symbol="BTCUSDT",
            state=PositionState.OPEN,
            entry_price=100.0,
            quantity=0.2,
            highest_high=120.0,
            stop_price=90.0,
            atr_at_entry=5.0,
            next_add_trigger=105.0,
            add_count=0,
            active_stop_client_order_id="XV1TS...",
            last_decision_open_time=decision_time,
        ),
    )

    loaded = get_live_position(store, position_id)

    assert loaded == LivePositionRecord(
        position_id=position_id,
        symbol="BTCUSDT",
        state=PositionState.OPEN,
        entry_price=100.0,
        quantity=0.2,
        highest_high=120.0,
        stop_price=90.0,
        atr_at_entry=5.0,
        next_add_trigger=105.0,
        add_count=0,
        active_stop_client_order_id="XV1TS...",
        last_decision_open_time=decision_time,
    )


def test_list_active_live_positions_excludes_closed_and_flat(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    open_position = store.create_position(symbol="BTCUSDT", state=PositionState.OPEN)
    store.create_position(symbol="ETHUSDT", state=PositionState.CLOSED)
    store.create_position(symbol="SOLUSDT", state=PositionState.FLAT)
    update_live_position(
        store,
        LivePositionRecord(
            position_id=open_position,
            symbol="BTCUSDT",
            state=PositionState.OPEN,
            entry_price=100.0,
            quantity=0.2,
            highest_high=120.0,
            stop_price=90.0,
            atr_at_entry=5.0,
            next_add_trigger=None,
            add_count=1,
            active_stop_client_order_id="XV1TS...",
            last_decision_open_time=None,
        ),
    )

    active = list_active_live_positions(store)

    assert [item.position_id for item in active] == [open_position]
