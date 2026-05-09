from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from decimal import Decimal

from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig
from xsignal.strategies.volume_price_efficiency_v1.live.ids import build_client_order_id
from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    OrderIntent,
    OrderIntentStatus,
    OrderIntentType,
    PositionState,
)
from xsignal.strategies.volume_price_efficiency_v1.live.position_store import (
    LivePositionRecord,
    update_live_position,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore


def enter_long_with_protection(
    *,
    store: LiveStore,
    broker,
    config: LiveTradingConfig,
    environment: str,
    symbol: str,
    quantity: float,
    entry_price: float,
    atr: float,
    now: datetime,
    strategy_interval: str | None = None,
) -> LivePositionRecord:
    position_id = store.create_position(symbol=symbol, state=PositionState.ENTRY_SUBMITTED)
    entry_client_id = _client_id(
        environment=environment,
        intent=OrderIntentType.ENTRY,
        symbol=symbol,
        position_id=position_id,
        sequence=1,
    )
    _record_intent(
        store=store,
        intent_type=OrderIntentType.ENTRY,
        position_id=position_id,
        symbol=symbol,
        client_order_id=entry_client_id,
        side="BUY",
        quantity=quantity,
        notional=quantity * entry_price,
        price=None,
        stop_price=None,
        now=now,
    )
    order = broker.market_buy(symbol=symbol, quantity=quantity, client_order_id=entry_client_id)
    store.update_order_intent_status(
        client_order_id=entry_client_id,
        status=OrderIntentStatus.EXCHANGE_CONFIRMED,
        exchange_order_id=_order_id(order),
        exchange_status=str(order.get("status", "UNKNOWN")),
        submitted_at=now,
    )

    stop_price = entry_price - config.atr_multiplier * atr
    stop_client_id = _client_id(
        environment=environment,
        intent=OrderIntentType.STOP_PLACE,
        symbol=symbol,
        position_id=position_id,
        sequence=2,
    )
    _record_intent(
        store=store,
        intent_type=OrderIntentType.STOP_PLACE,
        position_id=position_id,
        symbol=symbol,
        client_order_id=stop_client_id,
        side="SELL",
        quantity=0.0,
        notional=0.0,
        price=None,
        stop_price=stop_price,
        now=now,
    )
    stop_order = broker.place_stop_market_close(
        symbol=symbol,
        stop_price=stop_price,
        client_order_id=stop_client_id,
    )
    store.update_order_intent_status(
        client_order_id=stop_client_id,
        status=OrderIntentStatus.EXCHANGE_CONFIRMED,
        exchange_order_id=_algo_id(stop_order),
        exchange_status=str(stop_order.get("algoStatus") or stop_order.get("status") or "UNKNOWN"),
        submitted_at=now,
    )

    record = LivePositionRecord(
        position_id=position_id,
        symbol=symbol,
        state=PositionState.OPEN,
        entry_price=entry_price,
        quantity=quantity,
        highest_high=entry_price,
        stop_price=stop_price,
        atr_at_entry=atr,
        next_add_trigger=entry_price + config.pyramid_add_step_atr * atr,
        add_count=0,
        active_stop_client_order_id=stop_client_id,
        last_decision_open_time=None,
        strategy_interval=strategy_interval,
    )
    update_live_position(store, record)
    return record


def replace_trailing_stop(
    *,
    store: LiveStore,
    broker,
    environment: str,
    record: LivePositionRecord,
    candidate_stop_price: float,
    now: datetime,
) -> LivePositionRecord:
    if record.stop_price is not None and candidate_stop_price <= record.stop_price:
        return record
    sequence = _next_sequence(store=store, position_id=record.position_id)
    client_order_id = _client_id(
        environment=environment,
        intent=OrderIntentType.STOP_REPLACE,
        symbol=record.symbol,
        position_id=record.position_id,
        sequence=sequence,
    )
    _record_intent(
        store=store,
        intent_type=OrderIntentType.STOP_REPLACE,
        position_id=record.position_id,
        symbol=record.symbol,
        client_order_id=client_order_id,
        side="SELL",
        quantity=0.0,
        notional=0.0,
        price=None,
        stop_price=candidate_stop_price,
        now=now,
    )
    if record.active_stop_client_order_id:
        try:
            broker.cancel_order(symbol=record.symbol, client_order_id=record.active_stop_client_order_id)
        except Exception as exc:
            store.update_order_intent_status(
                client_order_id=client_order_id,
                status=OrderIntentStatus.ERROR,
                resolved_at=now,
                last_error=str(exc),
            )
            raise
    try:
        stop_order = broker.place_stop_market_close(
            symbol=record.symbol,
            stop_price=candidate_stop_price,
            client_order_id=client_order_id,
        )
    except Exception as exc:
        store.update_order_intent_status(
            client_order_id=client_order_id,
            status=OrderIntentStatus.ERROR,
            resolved_at=now,
            last_error=str(exc),
        )
        update_live_position(store, replace(record, state=PositionState.ERROR_LOCKED))
        raise
    store.update_order_intent_status(
        client_order_id=client_order_id,
        status=OrderIntentStatus.EXCHANGE_CONFIRMED,
        exchange_order_id=_algo_id(stop_order),
        exchange_status=str(stop_order.get("algoStatus") or stop_order.get("status") or "UNKNOWN"),
        submitted_at=now,
    )
    updated = replace(
        record,
        state=PositionState.OPEN,
        stop_price=candidate_stop_price,
        active_stop_client_order_id=client_order_id,
    )
    update_live_position(store, updated)
    return updated


def submit_pyramid_add(
    *,
    store: LiveStore,
    broker,
    environment: str,
    record: LivePositionRecord,
    quantity: float,
    execution_price: float,
    now: datetime,
) -> LivePositionRecord:
    sequence = _next_sequence(store=store, position_id=record.position_id)
    client_order_id = _client_id(
        environment=environment,
        intent=OrderIntentType.PYRAMID_ADD,
        symbol=record.symbol,
        position_id=record.position_id,
        sequence=sequence,
    )
    _record_intent(
        store=store,
        intent_type=OrderIntentType.PYRAMID_ADD,
        position_id=record.position_id,
        symbol=record.symbol,
        client_order_id=client_order_id,
        side="BUY",
        quantity=quantity,
        notional=quantity * execution_price,
        price=None,
        stop_price=None,
        now=now,
    )
    order = broker.market_buy(symbol=record.symbol, quantity=quantity, client_order_id=client_order_id)
    store.update_order_intent_status(
        client_order_id=client_order_id,
        status=OrderIntentStatus.EXCHANGE_CONFIRMED,
        exchange_order_id=_order_id(order),
        exchange_status=str(order.get("status", "UNKNOWN")),
        submitted_at=now,
    )
    updated = replace(
        record,
        state=PositionState.OPEN,
        quantity=float(Decimal(str(record.quantity)) + Decimal(str(quantity))),
        add_count=record.add_count + 1,
        next_add_trigger=None,
    )
    update_live_position(store, updated)
    return updated


def _record_intent(
    *,
    store: LiveStore,
    intent_type: OrderIntentType,
    position_id: str,
    symbol: str,
    client_order_id: str,
    side: str,
    quantity: float,
    notional: float,
    price: float | None,
    stop_price: float | None,
    now: datetime,
) -> None:
    store.record_order_intent(
        OrderIntent(
            intent_id=client_order_id,
            position_id=position_id,
            symbol=symbol,
            intent_type=intent_type,
            client_order_id=client_order_id,
            side=side,
            quantity=quantity,
            notional=notional,
            price=price,
            stop_price=stop_price,
            created_at=now,
        )
    )


def _client_id(
    *,
    environment: str,
    intent: OrderIntentType,
    symbol: str,
    position_id: str,
    sequence: int,
) -> str:
    return build_client_order_id(
        env=environment,
        intent=intent.value,
        symbol=symbol,
        position_id=position_id,
        sequence=sequence,
    )


def _next_sequence(*, store: LiveStore, position_id: str) -> int:
    row = store.connection.execute(
        "select count(*) from order_intents where position_id = ?",
        (position_id,),
    ).fetchone()
    return int(row[0]) + 1


def _order_id(order: dict) -> str | None:
    return str(order.get("orderId")) if order.get("orderId") is not None else None


def _algo_id(order: dict) -> str | None:
    return str(order.get("algoId")) if order.get("algoId") is not None else None
