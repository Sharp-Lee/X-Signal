from datetime import datetime, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    AccountSnapshot,
    OrderIntent,
    OrderIntentStatus,
    OrderIntentType,
    PositionState,
    SymbolMetadata,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore


def test_store_initializes_schema_and_persists_intent(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    position_id = store.create_position(symbol="BTCUSDT", state=PositionState.FLAT)
    intent = OrderIntent(
        intent_id="intent-1",
        position_id=position_id,
        symbol="BTCUSDT",
        intent_type=OrderIntentType.ENTRY,
        client_order_id="XV1TEBTC123",
        side="BUY",
        quantity=0.001,
        notional=20.0,
        price=None,
        stop_price=None,
        created_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
    )
    store.record_order_intent(intent)
    loaded = store.get_order_intent("intent-1")
    assert loaded == intent
    assert loaded.status == OrderIntentStatus.PENDING_SUBMIT


def test_store_updates_intent_status_and_lists_unresolved(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    position_id = store.create_position(symbol="BTCUSDT", state=PositionState.ENTRY_SUBMITTED)
    intent = OrderIntent(
        intent_id="intent-1",
        position_id=position_id,
        symbol="BTCUSDT",
        intent_type=OrderIntentType.ENTRY,
        client_order_id="XV1TEBTC123",
        side="BUY",
        quantity=0.001,
        notional=20.0,
        price=None,
        stop_price=None,
        created_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
    )

    store.record_order_intent(intent)
    store.update_order_intent_status(
        client_order_id="XV1TEBTC123",
        status=OrderIntentStatus.EXCHANGE_CONFIRMED,
        exchange_order_id="123456",
        exchange_status="FILLED",
        submitted_at=datetime(2026, 5, 9, 1, tzinfo=timezone.utc),
    )

    loaded = store.get_order_intent_by_client_order_id("XV1TEBTC123")
    unresolved = store.list_unresolved_order_intents()

    assert loaded.status == OrderIntentStatus.EXCHANGE_CONFIRMED
    assert loaded.exchange_order_id == "123456"
    assert loaded.exchange_status == "FILLED"
    assert loaded.submitted_at == datetime(2026, 5, 9, 1, tzinfo=timezone.utc)
    assert [item.client_order_id for item in unresolved] == ["XV1TEBTC123"]

    store.update_order_intent_status(
        client_order_id="XV1TEBTC123",
        status=OrderIntentStatus.RESOLVED,
        resolved_at=datetime(2026, 5, 9, 2, tzinfo=timezone.utc),
    )

    assert store.list_unresolved_order_intents() == []


def test_store_updates_and_lists_position_states(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    position_id = store.create_position(symbol="BTCUSDT", state=PositionState.OPEN)

    store.update_position_state(position_id, PositionState.ERROR_LOCKED)

    assert store.get_position_state(position_id) == PositionState.ERROR_LOCKED
    locked = store.list_positions_by_states([PositionState.ERROR_LOCKED])
    assert [(item["position_id"], item["symbol"], item["state"]) for item in locked] == [
        (position_id, "BTCUSDT", "ERROR_LOCKED")
    ]


def test_store_persists_metadata_and_account_snapshot(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    metadata = SymbolMetadata(
        symbol="BTCUSDT",
        status="TRADING",
        min_notional=5.0,
        quantity_step=0.001,
        price_tick=0.1,
        supports_stop_market=True,
        trigger_protect=0.05,
        updated_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
        min_quantity=0.001,
        max_quantity=1000.0,
        market_min_quantity=0.001,
        market_max_quantity=100.0,
        market_quantity_step=0.001,
    )
    snapshot = AccountSnapshot(
        mode="testnet",
        account_mode="one_way",
        asset_mode="single_asset_usdt",
        equity=1000.0,
        available_balance=900.0,
        open_notional=40.0,
        open_position_count=2,
        daily_realized_pnl=-3.0,
        captured_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
    )
    store.upsert_symbol_metadata(metadata)
    store.record_account_snapshot(snapshot)
    assert store.get_symbol_metadata("BTCUSDT") == metadata
    assert store.latest_account_snapshot() == snapshot


def test_store_persists_closed_market_bars_and_lists_them_in_time_order(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    first_open = datetime(2026, 5, 9, 8, 1, tzinfo=timezone.utc)
    second_open = datetime(2026, 5, 9, 8, tzinfo=timezone.utc)

    store.upsert_market_bar(
        {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "open_time": first_open,
            "open": 101.0,
            "high": 103.0,
            "low": 100.0,
            "close": 102.0,
            "quote_volume": 12.5,
            "is_complete": True,
        }
    )
    store.upsert_market_bar(
        {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "open_time": second_open,
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "quote_volume": 10.5,
            "is_complete": True,
        }
    )

    rows = store.list_market_bars(symbol="BTCUSDT", interval="1m")

    assert [row["open_time"] for row in rows] == [second_open, first_open]
    assert rows[0]["close"] == 101.0
    assert rows[1]["quote_volume"] == 12.5
    assert rows[0]["is_complete"] is True


def test_store_advances_market_cursor_without_regressing(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    first_open = datetime(2026, 5, 9, 8, tzinfo=timezone.utc)
    later_open = datetime(2026, 5, 9, 8, 5, tzinfo=timezone.utc)

    store.advance_market_cursor(symbol="BTCUSDT", interval="1m", open_time=later_open)
    store.advance_market_cursor(symbol="BTCUSDT", interval="1m", open_time=first_open)

    assert store.get_market_cursor(symbol="BTCUSDT", interval="1m") == later_open
