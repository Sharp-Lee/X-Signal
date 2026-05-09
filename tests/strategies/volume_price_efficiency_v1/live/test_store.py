from datetime import datetime, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    AccountSnapshot,
    OrderIntent,
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
