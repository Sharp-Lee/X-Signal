from datetime import datetime, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    OrderIntent,
    OrderIntentStatus,
    OrderIntentType,
    PositionState,
    SymbolMetadata,
)
from xsignal.strategies.volume_price_efficiency_v1.live.position_store import (
    LivePositionRecord,
    get_live_position,
    update_live_position,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore
from xsignal.strategies.volume_price_efficiency_v1.live.testnet_rehearsal import (
    close_rehearsal_position,
    open_protected_rehearsal_position,
)


NOW = datetime(2026, 5, 10, tzinfo=timezone.utc)


def _metadata(symbol: str) -> SymbolMetadata:
    return SymbolMetadata(
        symbol=symbol,
        status="TRADING",
        min_notional=5.0,
        quantity_step=0.01,
        price_tick=0.1,
        supports_stop_market=True,
        trigger_protect=0.05,
        updated_at=NOW,
        min_quantity=0.01,
        max_quantity=1000.0,
        market_min_quantity=0.01,
        market_max_quantity=1000.0,
        market_quantity_step=0.01,
    )


class OpenBroker:
    def __init__(self, store: LiveStore) -> None:
        self.store = store
        self.calls = []

    def get_symbol_metadata(self, symbol):
        return _metadata(symbol)

    def get_symbol_price(self, symbol):
        self.calls.append(("get_symbol_price", symbol))
        return 100.0

    def change_margin_type(self, symbol, margin_mode):
        self.calls.append(("change_margin_type", symbol, margin_mode))
        return {}

    def change_leverage(self, symbol, leverage):
        self.calls.append(("change_leverage", symbol, leverage))
        return {}

    def market_buy(self, *, symbol, quantity, client_order_id):
        intent = self.store.get_order_intent_by_client_order_id(client_order_id)
        assert intent is not None
        assert intent.status == OrderIntentStatus.PENDING_SUBMIT
        self.calls.append(("market_buy", symbol, quantity, client_order_id))
        return {"orderId": 11, "status": "FILLED"}

    def place_stop_market_close(self, *, symbol, stop_price, client_order_id):
        intent = self.store.get_order_intent_by_client_order_id(client_order_id)
        assert intent is not None
        assert intent.status == OrderIntentStatus.PENDING_SUBMIT
        self.calls.append(("place_stop_market_close", symbol, stop_price, client_order_id))
        return {"algoId": 22, "algoStatus": "NEW"}


def test_open_protected_rehearsal_position_uses_live_store_and_protects_before_return(
    tmp_path,
):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    broker = OpenBroker(store)

    result = open_protected_rehearsal_position(
        store=store,
        broker=broker,
        symbol="SOLUSDT",
        notional=8.0,
        stop_offset_pct=0.05,
        now=NOW,
    )

    record = get_live_position(store, result.position_id)
    assert result.symbol == "SOLUSDT"
    assert result.quantity == 0.08
    assert result.entry_price == 100.0
    assert result.stop_price == 95.0
    assert record is not None
    assert record.state == PositionState.OPEN
    assert record.active_stop_client_order_id == result.stop_client_order_id
    assert [call[0] for call in broker.calls] == [
        "get_symbol_price",
        "change_margin_type",
        "change_leverage",
        "market_buy",
        "place_stop_market_close",
    ]


class CloseBroker:
    def __init__(self) -> None:
        self.calls = []
        self.position_amount = "0.08"

    def cancel_order(self, *, symbol, client_order_id):
        self.calls.append(("cancel_order", symbol, client_order_id))
        return {"algoStatus": "CANCELED"}

    def market_sell_reduce_only(self, *, symbol, quantity, client_order_id):
        self.calls.append(("market_sell_reduce_only", symbol, quantity, client_order_id))
        self.position_amount = "0"
        return {"orderId": 33, "status": "FILLED"}

    def get_position_risk(self, *, symbol):
        self.calls.append(("get_position_risk", symbol))
        return [{"symbol": symbol, "positionSide": "BOTH", "positionAmt": self.position_amount}]


def test_close_rehearsal_position_cancels_strategy_stop_then_closes_reduce_only(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    position_id = store.create_position(symbol="SOLUSDT", state=PositionState.OPEN)
    record = LivePositionRecord(
        position_id=position_id,
        symbol="SOLUSDT",
        state=PositionState.OPEN,
        entry_price=100.0,
        quantity=0.08,
        highest_high=100.0,
        stop_price=95.0,
        atr_at_entry=5.0 / 3.0,
        next_add_trigger=101.0,
        add_count=0,
        active_stop_client_order_id="XV1TS...",
        last_decision_open_time=None,
        strategy_interval="rehearsal",
    )
    update_live_position(store, record)
    store.record_order_intent(
        OrderIntent(
            intent_id="XV1TS...",
            position_id=position_id,
            symbol="SOLUSDT",
            intent_type=OrderIntentType.STOP_PLACE,
            client_order_id="XV1TS...",
            side="SELL",
            quantity=0.0,
            notional=0.0,
            price=None,
            stop_price=95.0,
            created_at=NOW,
            status=OrderIntentStatus.EXCHANGE_CONFIRMED,
        )
    )
    broker = CloseBroker()

    result = close_rehearsal_position(
        store=store,
        broker=broker,
        symbol="SOLUSDT",
        position_id=position_id,
        now=NOW,
    )

    updated = get_live_position(store, position_id)
    close_intent = store.get_order_intent_by_client_order_id(result.close_client_order_id)
    stop_intent = store.get_order_intent_by_client_order_id("XV1TS...")
    assert result.final_position_amount == 0.0
    assert updated is not None
    assert updated.state == PositionState.CLOSED
    assert updated.active_stop_client_order_id is None
    assert stop_intent is not None
    assert stop_intent.status == OrderIntentStatus.RESOLVED
    assert close_intent is not None
    assert close_intent.intent_type == OrderIntentType.MANUAL_RECONCILE
    assert close_intent.status == OrderIntentStatus.RESOLVED
    assert [call[0] for call in broker.calls] == [
        "get_position_risk",
        "cancel_order",
        "market_sell_reduce_only",
        "get_position_risk",
    ]


def test_close_rehearsal_position_uses_exchange_position_amount_for_reduce_only(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    position_id = store.create_position(symbol="SOLUSDT", state=PositionState.OPEN)
    update_live_position(
        store,
        LivePositionRecord(
            position_id=position_id,
            symbol="SOLUSDT",
            state=PositionState.OPEN,
            entry_price=100.0,
            quantity=0.08,
            highest_high=100.0,
            stop_price=95.0,
            atr_at_entry=5.0 / 3.0,
            next_add_trigger=101.0,
            add_count=0,
            active_stop_client_order_id="XV1TS...",
            last_decision_open_time=None,
            strategy_interval="rehearsal",
        ),
    )

    class ExchangeQuantityBroker(CloseBroker):
        def __init__(self) -> None:
            super().__init__()
            self.position_amount = "0.10"

    broker = ExchangeQuantityBroker()

    result = close_rehearsal_position(
        store=store,
        broker=broker,
        symbol="SOLUSDT",
        position_id=position_id,
        now=NOW,
    )

    close_calls = [call for call in broker.calls if call[0] == "market_sell_reduce_only"]
    assert close_calls[0][2] == 0.10
    assert result.quantity == 0.10
