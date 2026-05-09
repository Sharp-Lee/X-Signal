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
    run_testnet_deploy_verify,
    run_testnet_rehearsal,
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


class FullRehearsalBroker(OpenBroker):
    def __init__(self, store: LiveStore) -> None:
        super().__init__(store)
        self.position_amount = "0"
        self.position_price = "0"
        self.open_stop_client_order_id = None

    def market_buy(self, *, symbol, quantity, client_order_id):
        order = super().market_buy(
            symbol=symbol,
            quantity=quantity,
            client_order_id=client_order_id,
        )
        self.position_amount = str(quantity)
        self.position_price = "100.0"
        return order

    def place_stop_market_close(self, *, symbol, stop_price, client_order_id):
        order = super().place_stop_market_close(
            symbol=symbol,
            stop_price=stop_price,
            client_order_id=client_order_id,
        )
        self.open_stop_client_order_id = client_order_id
        return order

    def get_all_position_risk(self):
        return [
            {
                "symbol": "SOLUSDT",
                "positionSide": "BOTH",
                "positionAmt": self.position_amount,
            }
        ]

    def get_position_risk(self, *, symbol):
        self.calls.append(("get_position_risk", symbol))
        return [
            {
                "symbol": symbol,
                "positionSide": "BOTH",
                "positionAmt": self.position_amount,
                "entryPrice": self.position_price,
            }
        ]

    def get_order(self, *, symbol, client_order_id):
        return {"symbol": symbol, "clientOrderId": client_order_id, "status": "FILLED"}

    def get_open_order(self, *, symbol, client_order_id):
        return {
            "symbol": symbol,
            "clientAlgoId": client_order_id,
            "algoStatus": "NEW",
            "type": "STOP_MARKET",
            "side": "SELL",
            "closePosition": True,
        }

    def get_open_orders(self, *, symbol):
        if self.open_stop_client_order_id is None:
            return []
        return [
            {
                "symbol": symbol,
                "clientAlgoId": self.open_stop_client_order_id,
                "algoStatus": "NEW",
                "type": "STOP_MARKET",
                "side": "SELL",
                "closePosition": True,
            }
        ]

    def cancel_order(self, *, symbol, client_order_id):
        self.calls.append(("cancel_order", symbol, client_order_id))
        self.open_stop_client_order_id = None
        return {"algoStatus": "CANCELED"}

    def market_sell_reduce_only(self, *, symbol, quantity, client_order_id):
        self.calls.append(("market_sell_reduce_only", symbol, quantity, client_order_id))
        self.position_amount = "0"
        return {"orderId": 44, "status": "FILLED"}


def test_run_testnet_rehearsal_opens_restarts_closes_and_reports_clean(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    broker = FullRehearsalBroker(store)
    restarts = []

    report = run_testnet_rehearsal(
        store=store,
        broker=broker,
        symbol="SOLUSDT",
        notional=8.0,
        stop_offset_pct=0.05,
        service_name="xsignal-vpe-testnet-stream-daemon.service",
        restart_runner=restarts.append,
        sleep_after_restart_seconds=0,
        now=NOW,
    )

    assert report.status == "OK"
    assert report.open.position_id == "SOLUSDT-1"
    assert report.protected_reconcile.error_count == 0
    assert report.protected_reconcile.findings[0].status == "PROTECTED"
    assert report.restart == {
        "enabled": True,
        "service": "xsignal-vpe-testnet-stream-daemon.service",
        "status": "RESTARTED",
    }
    assert restarts == ["xsignal-vpe-testnet-stream-daemon.service"]
    assert report.close.final_position_amount == 0.0
    assert report.final_reconcile.error_count == 0
    assert report.final_reconcile.findings[0].status == "CLEAN"


def test_run_testnet_rehearsal_closes_position_when_restart_fails(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    broker = FullRehearsalBroker(store)

    def fail_restart(service_name):
        raise RuntimeError(f"restart failed for {service_name}")

    report = run_testnet_rehearsal(
        store=store,
        broker=broker,
        symbol="SOLUSDT",
        notional=8.0,
        stop_offset_pct=0.05,
        service_name="xsignal-vpe-testnet-stream-daemon.service",
        restart_runner=fail_restart,
        sleep_after_restart_seconds=0,
        now=NOW,
    )

    assert report.status == "ERROR"
    assert report.restart["status"] == "ERROR"
    assert "restart failed" in report.restart["error"]
    assert report.close.final_position_amount == 0.0
    assert report.final_reconcile.error_count == 0
    assert get_live_position(store, report.open.position_id).state == PositionState.CLOSED


def _ok_status() -> dict:
    return {
        "overall": "OK",
        "warnings": [],
        "service_active": True,
        "live_service_active": False,
        "live_guard_present": False,
        "journal": {
            "rest_429_since_clean": 0,
            "stream_errors_since_clean": 0,
            "reconcile_error_since_clean": 0,
        },
    }


def test_run_testnet_deploy_verify_checks_status_rehearsal_and_final_status(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    broker = FullRehearsalBroker(store)
    status_calls = []

    def status_collector():
        status_calls.append(len(status_calls) + 1)
        return _ok_status()

    report = run_testnet_deploy_verify(
        store=store,
        db_path=tmp_path / "live.sqlite",
        broker=broker,
        symbol="SOLUSDT",
        notional=8.0,
        stop_offset_pct=0.05,
        service_name="xsignal-vpe-testnet-stream-daemon.service",
        status_collector=status_collector,
        restart_runner=lambda service_name: None,
        sleep_after_restart_seconds=0,
        now=NOW,
    )

    assert report.status == "OK"
    assert status_calls == [1, 2]
    assert report.pre_status["overall"] == "OK"
    assert report.rehearsal is not None
    assert report.rehearsal.status == "OK"
    assert report.post_status["overall"] == "OK"
    assert report.checks == {
        "pre_status_ok": True,
        "rehearsal_ok": True,
        "post_status_ok": True,
        "post_journal_clean": True,
        "live_guard_absent": True,
        "live_service_inactive": True,
    }


def test_run_testnet_deploy_verify_skips_rehearsal_when_pre_status_warn(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()

    def warn_status():
        status = _ok_status()
        status["overall"] = "WARN"
        status["warnings"] = ["socket_queue_nonzero"]
        return status

    report = run_testnet_deploy_verify(
        store=store,
        db_path=tmp_path / "live.sqlite",
        broker=object(),
        symbol="SOLUSDT",
        notional=8.0,
        stop_offset_pct=0.05,
        service_name="xsignal-vpe-testnet-stream-daemon.service",
        status_collector=warn_status,
        restart_runner=lambda service_name: None,
        sleep_after_restart_seconds=0,
        now=NOW,
    )

    assert report.status == "ERROR"
    assert report.rehearsal is None
    assert report.post_status is None
    assert "pre_status_not_ok" in report.errors
