from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
from decimal import Decimal
import subprocess
import time

from xsignal.strategies.volume_price_efficiency_v1.live.binance_rest import BinanceApiError
from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig
from xsignal.strategies.volume_price_efficiency_v1.live.execution import enter_long_with_protection
from xsignal.strategies.volume_price_efficiency_v1.live.ids import build_client_order_id
from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    OrderIntent,
    OrderIntentStatus,
    OrderIntentType,
    PositionState,
)
from xsignal.strategies.volume_price_efficiency_v1.live.order_normalizer import SymbolRules
from xsignal.strategies.volume_price_efficiency_v1.live.position_store import (
    LivePositionRecord,
    get_live_position,
    list_active_live_positions,
    update_live_position,
)
from xsignal.strategies.volume_price_efficiency_v1.live.reconcile import (
    ReconcileSummary,
    run_reconciliation_pass,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore


@dataclass(frozen=True)
class ProtectedOpenResult:
    symbol: str
    position_id: str
    quantity: float
    requested_notional: float
    effective_notional: float
    entry_price: float
    stop_offset_pct: float
    stop_price: float
    entry_client_order_id: str
    stop_client_order_id: str


@dataclass(frozen=True)
class ProtectedCloseResult:
    symbol: str
    position_id: str
    quantity: float
    canceled_stop_client_order_id: str | None
    close_client_order_id: str
    final_position_amount: float


@dataclass(frozen=True)
class TestnetRehearsalReport:
    status: str
    open: ProtectedOpenResult
    protected_reconcile: ReconcileSummary
    restart: dict[str, object]
    post_restart_reconcile: ReconcileSummary
    close: ProtectedCloseResult
    final_reconcile: ReconcileSummary

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "open": vars(self.open),
            "protected_reconcile": self.protected_reconcile.to_dict(),
            "restart": self.restart,
            "post_restart_reconcile": self.post_restart_reconcile.to_dict(),
            "close": vars(self.close),
            "final_reconcile": self.final_reconcile.to_dict(),
        }


def run_testnet_rehearsal(
    *,
    store: LiveStore,
    broker,
    symbol: str,
    notional: float,
    stop_offset_pct: float,
    service_name: str,
    restart_service: bool = True,
    restart_runner=None,
    sleep_after_restart_seconds: float = 4.0,
    sleeper=time.sleep,
    now: datetime | None = None,
) -> TestnetRehearsalReport:
    now = now or datetime.now(timezone.utc)
    restart_runner = restart_runner or _restart_systemd_service
    opened = open_protected_rehearsal_position(
        store=store,
        broker=broker,
        symbol=symbol,
        notional=notional,
        stop_offset_pct=stop_offset_pct,
        now=now,
    )
    protected_reconcile = run_reconciliation_pass(
        store=store,
        broker=broker,
        symbols=[symbol],
        environment="testnet",
        allow_repair=False,
        now=now,
    )
    if restart_service and protected_reconcile.error_count == 0:
        try:
            restart_runner(service_name)
        except Exception as exc:
            restart = {
                "enabled": True,
                "service": service_name,
                "status": "ERROR",
                "error": str(exc),
            }
        else:
            if sleep_after_restart_seconds > 0:
                sleeper(sleep_after_restart_seconds)
            restart = {
                "enabled": True,
                "service": service_name,
                "status": "RESTARTED",
            }
    elif restart_service:
        restart = {
            "enabled": True,
            "service": service_name,
            "status": "SKIPPED_RECONCILE_ERROR",
        }
    else:
        restart = {
            "enabled": False,
            "service": service_name,
            "status": "SKIPPED",
        }
    post_restart_reconcile = run_reconciliation_pass(
        store=store,
        broker=broker,
        symbols=[symbol],
        environment="testnet",
        allow_repair=False,
        now=datetime.now(timezone.utc),
    )
    closed = close_rehearsal_position(
        store=store,
        broker=broker,
        symbol=symbol,
        position_id=opened.position_id,
        now=datetime.now(timezone.utc),
    )
    final_reconcile = run_reconciliation_pass(
        store=store,
        broker=broker,
        symbols=[symbol],
        environment="testnet",
        allow_repair=False,
        now=datetime.now(timezone.utc),
    )
    status = (
        "OK"
        if protected_reconcile.error_count == 0
        and post_restart_reconcile.error_count == 0
        and final_reconcile.error_count == 0
        and closed.final_position_amount == 0.0
        and restart["status"] in {"RESTARTED", "SKIPPED"}
        else "ERROR"
    )
    return TestnetRehearsalReport(
        status=status,
        open=opened,
        protected_reconcile=protected_reconcile,
        restart=restart,
        post_restart_reconcile=post_restart_reconcile,
        close=closed,
        final_reconcile=final_reconcile,
    )


def open_protected_rehearsal_position(
    *,
    store: LiveStore,
    broker,
    symbol: str,
    notional: float,
    stop_offset_pct: float,
    now: datetime | None = None,
    environment: str = "testnet",
) -> ProtectedOpenResult:
    now = now or datetime.now(timezone.utc)
    if notional <= 0:
        raise ValueError("notional must be positive")
    if stop_offset_pct <= 0 or stop_offset_pct >= 1:
        raise ValueError("stop_offset_pct must be in (0, 1)")

    metadata = broker.get_symbol_metadata(symbol)
    if metadata.status != "TRADING":
        raise ValueError(f"{symbol} is not TRADING")
    rules = SymbolRules.from_metadata(metadata)
    entry_price = Decimal(str(broker.get_symbol_price(symbol)))
    quantity = rules.market_quantity_from_notional(notional=notional, price=entry_price)
    stop_price = rules.normalize_price(entry_price * (Decimal("1") - Decimal(str(stop_offset_pct))))
    atr = (entry_price - stop_price) / Decimal(str(LiveTradingConfig().atr_multiplier))
    if atr <= 0:
        raise ValueError("computed rehearsal ATR must be positive")

    _set_isolated_margin(broker, symbol)
    broker.change_leverage(symbol, 1)
    store.upsert_symbol_metadata(metadata)
    record = enter_long_with_protection(
        store=store,
        broker=broker,
        config=LiveTradingConfig(),
        environment=environment,
        symbol=symbol,
        quantity=float(quantity),
        entry_price=float(entry_price),
        atr=float(atr),
        now=now,
        strategy_interval="rehearsal",
    )
    entry_intent, stop_intent = _entry_and_stop_client_ids(
        store=store,
        position_id=record.position_id,
    )
    return ProtectedOpenResult(
        symbol=symbol,
        position_id=record.position_id,
        quantity=record.quantity,
        requested_notional=float(notional),
        effective_notional=float(quantity * entry_price),
        entry_price=float(entry_price),
        stop_offset_pct=stop_offset_pct,
        stop_price=float(record.stop_price or 0.0),
        entry_client_order_id=entry_intent,
        stop_client_order_id=stop_intent,
    )


def close_rehearsal_position(
    *,
    store: LiveStore,
    broker,
    symbol: str,
    position_id: str | None = None,
    now: datetime | None = None,
    environment: str = "testnet",
) -> ProtectedCloseResult:
    now = now or datetime.now(timezone.utc)
    record = _select_active_position(store=store, symbol=symbol, position_id=position_id)
    close_client_order_id = build_client_order_id(
        env=environment,
        intent=OrderIntentType.MANUAL_RECONCILE.value,
        symbol=symbol,
        position_id=record.position_id,
        sequence=_next_sequence(store=store, position_id=record.position_id),
    )
    exchange_position = _find_position(broker.get_position_risk(symbol=symbol), symbol)
    close_quantity = _position_amount(exchange_position)
    canceled_stop_client_order_id = record.active_stop_client_order_id
    try:
        if canceled_stop_client_order_id:
            broker.cancel_order(symbol=symbol, client_order_id=canceled_stop_client_order_id)
            store.update_order_intent_status(
                client_order_id=canceled_stop_client_order_id,
                status=OrderIntentStatus.RESOLVED,
                exchange_status="CANCELED_BY_REHEARSAL_CLOSE",
                resolved_at=now,
            )
    except Exception:
        store.update_position_state(record.position_id, PositionState.ERROR_LOCKED)
        raise

    if close_quantity <= Decimal("0"):
        update_live_position(
            store,
            replace(
                record,
                state=PositionState.CLOSED,
                quantity=0.0,
                active_stop_client_order_id=None,
                next_add_trigger=None,
            ),
        )
        store.update_position_state(record.position_id, PositionState.CLOSED)
        return ProtectedCloseResult(
            symbol=symbol,
            position_id=record.position_id,
            quantity=0.0,
            canceled_stop_client_order_id=canceled_stop_client_order_id,
            close_client_order_id=close_client_order_id,
            final_position_amount=0.0,
        )

    _record_close_intent(
        store=store,
        record=record,
        quantity=close_quantity,
        close_client_order_id=close_client_order_id,
        now=now,
    )
    try:
        order = broker.market_sell_reduce_only(
            symbol=symbol,
            quantity=float(close_quantity),
            client_order_id=close_client_order_id,
        )
    except Exception as exc:
        store.update_order_intent_status(
            client_order_id=close_client_order_id,
            status=OrderIntentStatus.ERROR,
            resolved_at=now,
            last_error=str(exc),
        )
        store.update_position_state(record.position_id, PositionState.ERROR_LOCKED)
        raise

    final_position = _find_position(broker.get_position_risk(symbol=symbol), symbol)
    final_amount = _position_amount(final_position)
    if final_amount > Decimal("0"):
        store.update_order_intent_status(
            client_order_id=close_client_order_id,
            status=OrderIntentStatus.EXCHANGE_CONFIRMED,
            exchange_order_id=str(order.get("orderId")) if order.get("orderId") is not None else None,
            exchange_status=str(order.get("status", "UNKNOWN")),
            submitted_at=now,
        )
        store.update_position_state(record.position_id, PositionState.ERROR_LOCKED)
        return ProtectedCloseResult(
            symbol=symbol,
            position_id=record.position_id,
            quantity=float(close_quantity),
            canceled_stop_client_order_id=canceled_stop_client_order_id,
            close_client_order_id=close_client_order_id,
            final_position_amount=float(final_amount),
        )

    store.update_order_intent_status(
        client_order_id=close_client_order_id,
        status=OrderIntentStatus.RESOLVED,
        exchange_order_id=str(order.get("orderId")) if order.get("orderId") is not None else None,
        exchange_status=str(order.get("status", "FILLED")),
        submitted_at=now,
        resolved_at=now,
    )
    update_live_position(
        store,
        replace(
            record,
            state=PositionState.CLOSED,
            quantity=float(close_quantity),
            active_stop_client_order_id=None,
            next_add_trigger=None,
        ),
    )
    store.update_position_state(record.position_id, PositionState.CLOSED)
    return ProtectedCloseResult(
        symbol=symbol,
        position_id=record.position_id,
        quantity=float(close_quantity),
        canceled_stop_client_order_id=canceled_stop_client_order_id,
        close_client_order_id=close_client_order_id,
        final_position_amount=0.0,
    )


def _set_isolated_margin(broker, symbol: str) -> None:
    try:
        broker.change_margin_type(symbol, "isolated")
    except BinanceApiError as exc:
        if exc.code == -4046:
            return
        raise


def _restart_systemd_service(service_name: str) -> None:
    subprocess.run(["systemctl", "restart", service_name], check=True)


def _entry_and_stop_client_ids(*, store: LiveStore, position_id: str) -> tuple[str, str]:
    rows = store.connection.execute(
        """
        select intent_type, client_order_id
        from order_intents
        where position_id = ?
        order by created_at, intent_id
        """,
        (position_id,),
    ).fetchall()
    by_type = {row["intent_type"]: row["client_order_id"] for row in rows}
    return by_type[OrderIntentType.ENTRY.value], by_type[OrderIntentType.STOP_PLACE.value]


def _select_active_position(
    *,
    store: LiveStore,
    symbol: str,
    position_id: str | None,
) -> LivePositionRecord:
    if position_id is not None:
        record = get_live_position(store, position_id)
        if record is None:
            raise ValueError(f"missing local position {position_id}")
        if record.symbol != symbol:
            raise ValueError(f"position {position_id} belongs to {record.symbol}, not {symbol}")
        if record.state not in _ACTIVE_STATES:
            raise ValueError(f"position {position_id} is not active")
        return record

    matches = [record for record in list_active_live_positions(store) if record.symbol == symbol]
    if not matches:
        raise ValueError(f"no active local position for {symbol}")
    if len(matches) > 1:
        raise ValueError(f"multiple active local positions for {symbol}; pass --position-id")
    return matches[0]


_ACTIVE_STATES = {
    PositionState.OPEN,
    PositionState.ADD_ARMED,
    PositionState.ADD_SUBMITTED,
    PositionState.STOP_REPLACING,
    PositionState.EXITING,
}


def _record_close_intent(
    *,
    store: LiveStore,
    record: LivePositionRecord,
    quantity: Decimal,
    close_client_order_id: str,
    now: datetime,
) -> None:
    store.record_order_intent(
        OrderIntent(
            intent_id=close_client_order_id,
            position_id=record.position_id,
            symbol=record.symbol,
            intent_type=OrderIntentType.MANUAL_RECONCILE,
            client_order_id=close_client_order_id,
            side="SELL",
            quantity=float(quantity),
            notional=0.0,
            price=None,
            stop_price=None,
            created_at=now,
        )
    )


def _next_sequence(*, store: LiveStore, position_id: str) -> int:
    row = store.connection.execute(
        "select count(*) from order_intents where position_id = ?",
        (position_id,),
    ).fetchone()
    return int(row[0]) + 1


def _find_position(payload, symbol: str) -> dict:
    positions = payload if isinstance(payload, list) else [payload]
    for item in positions:
        if item.get("symbol") == symbol and item.get("positionSide", "BOTH") in {"BOTH", "LONG"}:
            return item
    return {"symbol": symbol, "positionSide": "BOTH", "positionAmt": "0"}


def _position_amount(position: dict) -> Decimal:
    return Decimal(str(position.get("positionAmt", "0"))).copy_abs()
