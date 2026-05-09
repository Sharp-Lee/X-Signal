from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
from enum import StrEnum

from xsignal.strategies.volume_price_efficiency_v1.live.ids import build_client_order_id
from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    OrderIntent,
    OrderIntentStatus,
    OrderIntentType,
    PositionState,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore


class ReconcileStatus(StrEnum):
    CLEAN = "CLEAN"
    PROTECTED = "PROTECTED"
    CLOSED = "CLOSED"
    ERROR_LOCKED = "ERROR_LOCKED"


@dataclass(frozen=True)
class ReconcileFinding:
    symbol: str
    position_id: str | None
    status: ReconcileStatus
    reason: str
    actions: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "position_id": self.position_id,
            "status": self.status.value,
            "reason": self.reason,
            "actions": list(self.actions),
        }


@dataclass(frozen=True)
class ReconcileSummary:
    environment: str
    allow_repair: bool
    findings: tuple[ReconcileFinding, ...]

    @property
    def error_count(self) -> int:
        return sum(1 for finding in self.findings if finding.status == ReconcileStatus.ERROR_LOCKED)

    @property
    def repaired_count(self) -> int:
        return sum(1 for finding in self.findings if finding.actions)

    def to_dict(self) -> dict[str, object]:
        return {
            "environment": self.environment,
            "allow_repair": self.allow_repair,
            "error_count": self.error_count,
            "repaired_count": self.repaired_count,
            "findings": [finding.to_dict() for finding in self.findings],
        }


_LOCAL_RECONCILE_STATES = [
    PositionState.FLAT,
    PositionState.ENTRY_SUBMITTED,
    PositionState.OPEN,
    PositionState.ADD_ARMED,
    PositionState.ADD_SUBMITTED,
    PositionState.STOP_REPLACING,
    PositionState.EXITING,
    PositionState.ERROR_LOCKED,
]

_REGULAR_ORDER_INTENTS = {
    OrderIntentType.ENTRY,
    OrderIntentType.PYRAMID_ADD,
    OrderIntentType.MANUAL_RECONCILE,
}

_STOP_ORDER_INTENTS = {
    OrderIntentType.STOP_PLACE,
    OrderIntentType.STOP_REPLACE,
}


def run_reconciliation_pass(
    *,
    store: LiveStore,
    broker,
    symbols: list[str],
    environment: str,
    allow_repair: bool,
    now: datetime | None = None,
) -> ReconcileSummary:
    now = now or datetime.now(timezone.utc)
    _refresh_unresolved_intents(store=store, broker=broker, now=now)
    local_positions = store.list_positions_by_states(_LOCAL_RECONCILE_STATES)
    exchange_positions = _positions_by_symbol(broker.get_all_position_risk())
    findings = []
    for symbol in symbols:
        local_position = _find_latest_position_for_symbol(local_positions, symbol)
        findings.append(
            _reconcile_symbol(
                store=store,
                broker=broker,
                symbol=symbol,
                exchange_position=exchange_positions.get(symbol),
                local_position=local_position,
                environment=environment,
                allow_repair=allow_repair,
                now=now,
            )
        )
    return ReconcileSummary(
        environment=environment,
        allow_repair=allow_repair,
        findings=tuple(findings),
    )


def _refresh_unresolved_intents(*, store: LiveStore, broker, now: datetime) -> None:
    for intent in store.list_unresolved_order_intents():
        if intent.intent_type in _REGULAR_ORDER_INTENTS:
            _refresh_regular_order_intent(store=store, broker=broker, intent=intent, now=now)
        elif intent.intent_type in _STOP_ORDER_INTENTS:
            _refresh_stop_order_intent(store=store, broker=broker, intent=intent, now=now)


def _refresh_regular_order_intent(
    *,
    store: LiveStore,
    broker,
    intent: OrderIntent,
    now: datetime,
) -> None:
    try:
        order = broker.get_order(symbol=intent.symbol, client_order_id=intent.client_order_id)
    except Exception as exc:
        store.update_order_intent_status(
            client_order_id=intent.client_order_id,
            status=intent.status,
            last_error=str(exc),
        )
        return
    status = str(order.get("status", "UNKNOWN"))
    if status in {"CANCELED", "EXPIRED", "REJECTED"}:
        store.update_order_intent_status(
            client_order_id=intent.client_order_id,
            status=OrderIntentStatus.ERROR,
            exchange_order_id=str(order.get("orderId")) if order.get("orderId") is not None else None,
            exchange_status=status,
            resolved_at=now,
        )
        return
    store.update_order_intent_status(
        client_order_id=intent.client_order_id,
        status=OrderIntentStatus.EXCHANGE_CONFIRMED,
        exchange_order_id=str(order.get("orderId")) if order.get("orderId") is not None else None,
        exchange_status=status,
        submitted_at=now,
    )


def _refresh_stop_order_intent(
    *,
    store: LiveStore,
    broker,
    intent: OrderIntent,
    now: datetime,
) -> None:
    try:
        order = broker.get_open_order(symbol=intent.symbol, client_order_id=intent.client_order_id)
    except Exception as exc:
        store.update_order_intent_status(
            client_order_id=intent.client_order_id,
            status=intent.status,
            last_error=str(exc),
        )
        return
    status = str(order.get("algoStatus") or order.get("status") or "UNKNOWN")
    if _is_open_strategy_stop(order):
        store.update_order_intent_status(
            client_order_id=intent.client_order_id,
            status=OrderIntentStatus.EXCHANGE_CONFIRMED,
            exchange_order_id=(
                str(order.get("algoId")) if order.get("algoId") is not None else None
            ),
            exchange_status=status,
            submitted_at=now,
        )


def _reconcile_symbol(
    *,
    store: LiveStore,
    broker,
    symbol: str,
    exchange_position,
    local_position,
    environment: str,
    allow_repair: bool,
    now: datetime,
) -> ReconcileFinding:
    position_amount = _position_amount(
        exchange_position or {"symbol": symbol, "positionSide": "BOTH", "positionAmt": "0"}
    )

    if position_amount <= 0:
        if local_position is not None and local_position["state"] not in {
            PositionState.FLAT.value,
            PositionState.CLOSED.value,
        }:
            store.update_position_state(local_position["position_id"], PositionState.CLOSED)
            return ReconcileFinding(
                symbol=symbol,
                position_id=local_position["position_id"],
                status=ReconcileStatus.CLOSED,
                reason="Binance position is flat",
            )
        return ReconcileFinding(
            symbol=symbol,
            position_id=local_position["position_id"] if local_position is not None else None,
            status=ReconcileStatus.CLEAN,
            reason="Binance position is flat",
        )

    open_orders = broker.get_open_orders(symbol=symbol)
    strategy_stops = [order for order in open_orders if _is_open_strategy_stop(order)]

    if local_position is None:
        return ReconcileFinding(
            symbol=symbol,
            position_id=None,
            status=ReconcileStatus.ERROR_LOCKED,
            reason="Binance has a long position but no local strategy position exists",
        )

    position_id = local_position["position_id"]
    local_state = PositionState(local_position["state"])
    if local_state == PositionState.FLAT:
        store.update_position_state(position_id, PositionState.ERROR_LOCKED)
        return ReconcileFinding(
            symbol=symbol,
            position_id=position_id,
            status=ReconcileStatus.ERROR_LOCKED,
            reason="Binance has a long position while local state is FLAT",
        )

    if strategy_stops:
        store.update_position_state(position_id, PositionState.OPEN)
        return ReconcileFinding(
            symbol=symbol,
            position_id=position_id,
            status=ReconcileStatus.PROTECTED,
            reason="Binance long position has an active strategy stop",
        )

    if not allow_repair:
        store.update_position_state(position_id, PositionState.ERROR_LOCKED)
        return ReconcileFinding(
            symbol=symbol,
            position_id=position_id,
            status=ReconcileStatus.ERROR_LOCKED,
            reason="Binance long position is unprotected and repair mode is disabled",
        )

    return _close_unprotected_owned_position(
        store=store,
        broker=broker,
        symbol=symbol,
        position_id=position_id,
        position_amount=position_amount,
        environment=environment,
        now=now,
    )


def _close_unprotected_owned_position(
    *,
    store: LiveStore,
    broker,
    symbol: str,
    position_id: str,
    position_amount: Decimal,
    environment: str,
    now: datetime,
) -> ReconcileFinding:
    sequence = _next_intent_sequence(store=store, position_id=position_id)
    client_order_id = build_client_order_id(
        env=environment,
        intent=OrderIntentType.MANUAL_RECONCILE.value,
        symbol=symbol,
        position_id=position_id,
        sequence=sequence,
    )
    store.record_order_intent(
        OrderIntent(
            intent_id=client_order_id,
            position_id=position_id,
            symbol=symbol,
            intent_type=OrderIntentType.MANUAL_RECONCILE,
            client_order_id=client_order_id,
            side="SELL",
            quantity=float(position_amount),
            notional=0.0,
            price=None,
            stop_price=None,
            created_at=now,
        )
    )
    try:
        order = broker.market_sell_reduce_only(
            symbol=symbol,
            quantity=position_amount,
            client_order_id=client_order_id,
        )
    except Exception as exc:
        store.update_order_intent_status(
            client_order_id=client_order_id,
            status=OrderIntentStatus.ERROR,
            resolved_at=now,
            last_error=str(exc),
        )
        store.update_position_state(position_id, PositionState.ERROR_LOCKED)
        return ReconcileFinding(
            symbol=symbol,
            position_id=position_id,
            status=ReconcileStatus.ERROR_LOCKED,
            reason=f"unprotected emergency close failed: {exc}",
            actions=("persist_close_intent",),
        )

    final_position = _find_position(broker.get_position_risk(symbol=symbol), symbol)
    if _position_amount(final_position) > 0:
        store.update_order_intent_status(
            client_order_id=client_order_id,
            status=OrderIntentStatus.EXCHANGE_CONFIRMED,
            exchange_order_id=str(order.get("orderId")) if order.get("orderId") is not None else None,
            exchange_status=str(order.get("status", "UNKNOWN")),
            submitted_at=now,
        )
        store.update_position_state(position_id, PositionState.ERROR_LOCKED)
        return ReconcileFinding(
            symbol=symbol,
            position_id=position_id,
            status=ReconcileStatus.ERROR_LOCKED,
            reason="unprotected emergency close submitted but position is still open",
            actions=("persist_close_intent", "submit_reduce_only_close"),
        )

    store.update_order_intent_status(
        client_order_id=client_order_id,
        status=OrderIntentStatus.RESOLVED,
        exchange_order_id=str(order.get("orderId")) if order.get("orderId") is not None else None,
        exchange_status=str(order.get("status", "FILLED")),
        submitted_at=now,
        resolved_at=now,
    )
    store.update_position_state(position_id, PositionState.CLOSED)
    return ReconcileFinding(
        symbol=symbol,
        position_id=position_id,
        status=ReconcileStatus.CLOSED,
        reason="unprotected strategy long closed by repair mode",
        actions=("persist_close_intent", "submit_reduce_only_close", "verify_flat"),
    )


def _next_intent_sequence(*, store: LiveStore, position_id: str) -> int:
    row = store.connection.execute(
        "select count(*) from order_intents where position_id = ?",
        (position_id,),
    ).fetchone()
    return int(row[0]) + 1


def _find_latest_position_for_symbol(local_positions, symbol: str):
    matches = [row for row in local_positions if row["symbol"] == symbol]
    return matches[-1] if matches else None


def _positions_by_symbol(payload) -> dict[str, dict]:
    result = {}
    for item in payload:
        if item.get("positionSide", "BOTH") in {"BOTH", "LONG"}:
            result[str(item.get("symbol"))] = item
    return result


def _find_position(payload, symbol: str) -> dict:
    positions = payload if isinstance(payload, list) else [payload]
    for item in positions:
        if item.get("symbol") == symbol and item.get("positionSide", "BOTH") in {"BOTH", "LONG"}:
            return item
    return {"symbol": symbol, "positionSide": "BOTH", "positionAmt": "0"}


def _position_amount(position: dict) -> Decimal:
    return Decimal(str(position.get("positionAmt", "0"))).copy_abs()


def _is_open_strategy_stop(order: dict) -> bool:
    client_id = str(order.get("clientAlgoId") or order.get("clientOrderId") or "")
    status = str(order.get("algoStatus") or order.get("status") or "")
    order_type = str(order.get("type") or order.get("orderType") or "")
    return (
        client_id.startswith("XV1")
        and status in {"NEW", "PARTIALLY_FILLED"}
        and order_type == "STOP_MARKET"
        and str(order.get("side")) == "SELL"
        and _truthy(order.get("closePosition"))
    )


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"
