from datetime import datetime, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.ids import build_client_order_id
from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    OrderIntent,
    OrderIntentStatus,
    OrderIntentType,
    PositionState,
)
from xsignal.strategies.volume_price_efficiency_v1.live.reconcile import (
    ReconcileStatus,
    run_reconciliation_pass,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore


NOW = datetime(2026, 5, 9, tzinfo=timezone.utc)


class FakeReconcileBroker:
    def __init__(
        self,
        *,
        position_amount: str = "0.001",
        open_orders: list[dict] | None = None,
        queried_order: dict | None = None,
    ) -> None:
        self.position_amount = position_amount
        self.open_orders = open_orders if open_orders is not None else []
        self.queried_order = queried_order
        self.calls = []

    def get_position_risk(self, *, symbol):
        self.calls.append(("get_position_risk", symbol))
        return [
            {
                "symbol": symbol,
                "positionSide": "BOTH",
                "positionAmt": self.position_amount,
                "entryPrice": "100",
                "markPrice": "101",
            }
        ]

    def get_open_orders(self, *, symbol):
        self.calls.append(("get_open_orders", symbol))
        return list(self.open_orders)

    def get_order(self, *, symbol, client_order_id):
        self.calls.append(("get_order", symbol, client_order_id))
        if self.queried_order is None:
            raise RuntimeError("order not found")
        return self.queried_order

    def get_open_order(self, *, symbol, client_order_id):
        self.calls.append(("get_open_order", symbol, client_order_id))
        for order in self.open_orders:
            if order.get("clientAlgoId") == client_order_id:
                return order
        raise RuntimeError("algo order not found")

    def market_sell_reduce_only(self, *, symbol, quantity, client_order_id):
        self.calls.append(("market_sell_reduce_only", symbol, quantity, client_order_id))
        self.position_amount = "0"
        return {"symbol": symbol, "clientOrderId": client_order_id, "status": "FILLED"}


def _open_stop(client_id: str = "XV1TSBTCSTOP") -> dict:
    return {
        "symbol": "BTCUSDT",
        "clientAlgoId": client_id,
        "type": "STOP_MARKET",
        "side": "SELL",
        "closePosition": True,
        "algoStatus": "NEW",
    }


def _store_with_position(tmp_path, state: PositionState = PositionState.OPEN) -> tuple[LiveStore, str]:
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    position_id = store.create_position(symbol="BTCUSDT", state=state)
    return store, position_id


def _record_intent(
    store: LiveStore,
    *,
    position_id: str,
    intent_type: OrderIntentType,
    client_id: str,
    status: OrderIntentStatus = OrderIntentStatus.PENDING_SUBMIT,
) -> None:
    store.record_order_intent(
        OrderIntent(
            intent_id=client_id,
            position_id=position_id,
            symbol="BTCUSDT",
            intent_type=intent_type,
            client_order_id=client_id,
            side="SELL" if intent_type in {OrderIntentType.STOP_PLACE, OrderIntentType.STOP_REPLACE} else "BUY",
            quantity=0.001,
            notional=20.0,
            price=None,
            stop_price=95.0 if intent_type in {OrderIntentType.STOP_PLACE, OrderIntentType.STOP_REPLACE} else None,
            created_at=NOW,
            status=status,
        )
    )


def test_reconcile_marks_local_open_with_exchange_position_and_stop_as_protected(tmp_path):
    store, position_id = _store_with_position(tmp_path, PositionState.OPEN)
    _record_intent(
        store,
        position_id=position_id,
        intent_type=OrderIntentType.STOP_PLACE,
        client_id="XV1TSBTCSTOP",
    )
    broker = FakeReconcileBroker(open_orders=[_open_stop()])

    summary = run_reconciliation_pass(
        store=store,
        broker=broker,
        symbols=["BTCUSDT"],
        environment="testnet",
        allow_repair=False,
        now=NOW,
    )

    assert summary.findings[0].status == ReconcileStatus.PROTECTED
    assert store.get_position_state(position_id) == PositionState.OPEN
    assert store.get_order_intent_by_client_order_id("XV1TSBTCSTOP").status == (
        OrderIntentStatus.EXCHANGE_CONFIRMED
    )


def test_reconcile_locks_unprotected_local_open_position_in_read_only_mode(tmp_path):
    store, position_id = _store_with_position(tmp_path, PositionState.OPEN)
    broker = FakeReconcileBroker(open_orders=[])

    summary = run_reconciliation_pass(
        store=store,
        broker=broker,
        symbols=["BTCUSDT"],
        environment="testnet",
        allow_repair=False,
        now=NOW,
    )

    assert summary.findings[0].status == ReconcileStatus.ERROR_LOCKED
    assert "unprotected" in summary.findings[0].reason
    assert store.get_position_state(position_id) == PositionState.ERROR_LOCKED
    assert not any(call[0] == "market_sell_reduce_only" for call in broker.calls)


def test_reconcile_repairs_unprotected_owned_long_by_persisting_close_before_submit(tmp_path):
    store, position_id = _store_with_position(tmp_path, PositionState.OPEN)
    broker = FakeReconcileBroker(open_orders=[])

    summary = run_reconciliation_pass(
        store=store,
        broker=broker,
        symbols=["BTCUSDT"],
        environment="testnet",
        allow_repair=True,
        now=NOW,
    )

    close_calls = [call for call in broker.calls if call[0] == "market_sell_reduce_only"]
    close_client_id = close_calls[0][3]
    close_intent = store.get_order_intent_by_client_order_id(close_client_id)
    assert summary.findings[0].status == ReconcileStatus.CLOSED
    assert close_intent.intent_type == OrderIntentType.MANUAL_RECONCILE
    assert close_intent.status == OrderIntentStatus.RESOLVED
    assert store.get_position_state(position_id) == PositionState.CLOSED


def test_reconcile_does_not_close_unknown_exchange_position_when_local_flat(tmp_path):
    store, position_id = _store_with_position(tmp_path, PositionState.FLAT)
    broker = FakeReconcileBroker(open_orders=[])

    summary = run_reconciliation_pass(
        store=store,
        broker=broker,
        symbols=["BTCUSDT"],
        environment="testnet",
        allow_repair=True,
        now=NOW,
    )

    assert summary.findings[0].status == ReconcileStatus.ERROR_LOCKED
    assert "local state is FLAT" in summary.findings[0].reason
    assert store.get_position_state(position_id) == PositionState.ERROR_LOCKED
    assert not any(call[0] == "market_sell_reduce_only" for call in broker.calls)


def test_reconcile_queries_pending_entry_client_id_after_restart(tmp_path):
    store, position_id = _store_with_position(tmp_path, PositionState.ENTRY_SUBMITTED)
    entry_id = build_client_order_id(
        env="testnet",
        intent="ENTRY",
        symbol="BTCUSDT",
        position_id=position_id,
        sequence=1,
    )
    stop_id = build_client_order_id(
        env="testnet",
        intent="STOP_PLACE",
        symbol="BTCUSDT",
        position_id=position_id,
        sequence=2,
    )
    _record_intent(
        store,
        position_id=position_id,
        intent_type=OrderIntentType.ENTRY,
        client_id=entry_id,
        status=OrderIntentStatus.PENDING_SUBMIT,
    )
    _record_intent(
        store,
        position_id=position_id,
        intent_type=OrderIntentType.STOP_PLACE,
        client_id=stop_id,
        status=OrderIntentStatus.PENDING_SUBMIT,
    )
    broker = FakeReconcileBroker(
        open_orders=[_open_stop(stop_id)],
        queried_order={"symbol": "BTCUSDT", "clientOrderId": entry_id, "orderId": 7, "status": "FILLED"},
    )

    summary = run_reconciliation_pass(
        store=store,
        broker=broker,
        symbols=["BTCUSDT"],
        environment="testnet",
        allow_repair=False,
        now=NOW,
    )

    assert summary.findings[0].status == ReconcileStatus.PROTECTED
    assert store.get_order_intent_by_client_order_id(entry_id).exchange_status == "FILLED"
    assert ("get_order", "BTCUSDT", entry_id) in broker.calls
