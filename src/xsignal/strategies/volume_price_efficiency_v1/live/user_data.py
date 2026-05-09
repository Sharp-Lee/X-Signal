from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from enum import StrEnum
import json
from urllib.parse import quote

from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    OrderIntent,
    OrderIntentStatus,
    OrderIntentType,
    PositionState,
)
from xsignal.strategies.volume_price_efficiency_v1.live.position_store import (
    get_live_position,
    list_active_live_positions,
    update_live_position,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore


TESTNET_USER_DATA_WS_BASE_URL = "wss://stream.binancefuture.com/ws"
LIVE_USER_DATA_WS_BASE_URL = "wss://fstream.binance.com/private/ws"
STRATEGY_CLIENT_ORDER_PREFIX = "XV1"


class UserDataEventType(StrEnum):
    ORDER_TRADE_UPDATE = "ORDER_TRADE_UPDATE"
    ACCOUNT_UPDATE = "ACCOUNT_UPDATE"
    LISTEN_KEY_EXPIRED = "listenKeyExpired"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class UserOrderTradeUpdate:
    event_type: UserDataEventType
    event_time: datetime
    transaction_time: datetime
    symbol: str
    client_order_id: str
    side: str
    order_type: str
    execution_type: str
    order_status: str
    order_id: str | None
    cumulative_filled_quantity: float
    average_price: float | None
    stop_price: float | None
    reduce_only: bool
    close_position: bool


@dataclass(frozen=True)
class UserPositionUpdate:
    symbol: str
    position_amount: float
    entry_price: float | None
    position_side: str


@dataclass(frozen=True)
class UserAccountUpdate:
    event_type: UserDataEventType
    event_time: datetime
    transaction_time: datetime
    reason: str
    positions: tuple[UserPositionUpdate, ...]


@dataclass(frozen=True)
class UnknownUserDataEvent:
    event_type: UserDataEventType
    event_time: datetime | None = None
    raw_event_type: str | None = None


UserDataEvent = UserOrderTradeUpdate | UserAccountUpdate | UnknownUserDataEvent


@dataclass(frozen=True)
class UserDataApplyResult:
    intent_updated: bool = False
    position_closed: bool = False
    positions_synced: int = 0
    ignored_reason: str | None = None


def user_data_ws_url_for_mode(mode: str, listen_key: str) -> str:
    if mode == "testnet":
        return f"{TESTNET_USER_DATA_WS_BASE_URL.rstrip('/')}/{quote(listen_key, safe='')}"
    if mode == "live":
        return f"{LIVE_USER_DATA_WS_BASE_URL.rstrip('/')}?listenKey={quote(listen_key, safe='')}"
    raise ValueError(f"unsupported mode: {mode}")


def parse_user_data_event(payload: dict) -> UserDataEvent:
    event_type = str(payload.get("e") or "")
    event_time = _ms_to_dt(payload.get("E"))
    transaction_time = _ms_to_dt(payload.get("T")) or event_time
    if event_type == UserDataEventType.ORDER_TRADE_UPDATE.value:
        order = payload.get("o") or {}
        return UserOrderTradeUpdate(
            event_type=UserDataEventType.ORDER_TRADE_UPDATE,
            event_time=event_time or datetime.now(timezone.utc),
            transaction_time=transaction_time or event_time or datetime.now(timezone.utc),
            symbol=str(order.get("s") or payload.get("s") or ""),
            client_order_id=str(order.get("c") or ""),
            side=str(order.get("S") or ""),
            order_type=str(order.get("o") or ""),
            execution_type=str(order.get("x") or ""),
            order_status=str(order.get("X") or ""),
            order_id=str(order["i"]) if order.get("i") is not None else None,
            cumulative_filled_quantity=_float(order.get("z"), default=0.0),
            average_price=_optional_float(order.get("ap")),
            stop_price=_optional_float(order.get("sp")),
            reduce_only=_truthy(order.get("R")),
            close_position=_truthy(order.get("cp")),
        )
    if event_type == UserDataEventType.ACCOUNT_UPDATE.value:
        account = payload.get("a") or {}
        positions = []
        for item in account.get("P") or []:
            positions.append(
                UserPositionUpdate(
                    symbol=str(item.get("s") or ""),
                    position_amount=_float(item.get("pa"), default=0.0),
                    entry_price=_optional_float(item.get("ep")),
                    position_side=str(item.get("ps") or "BOTH"),
                )
            )
        return UserAccountUpdate(
            event_type=UserDataEventType.ACCOUNT_UPDATE,
            event_time=event_time or datetime.now(timezone.utc),
            transaction_time=transaction_time or event_time or datetime.now(timezone.utc),
            reason=str(account.get("m") or ""),
            positions=tuple(positions),
        )
    if event_type == UserDataEventType.LISTEN_KEY_EXPIRED.value:
        return UnknownUserDataEvent(
            event_type=UserDataEventType.LISTEN_KEY_EXPIRED,
            event_time=event_time,
            raw_event_type=event_type,
        )
    return UnknownUserDataEvent(
        event_type=UserDataEventType.UNKNOWN,
        event_time=event_time,
        raw_event_type=event_type or None,
    )


def apply_user_data_event(store: LiveStore, event: UserDataEvent) -> UserDataApplyResult:
    if isinstance(event, UserOrderTradeUpdate):
        return _apply_order_trade_update(store, event)
    if isinstance(event, UserAccountUpdate):
        return _apply_account_update(store, event)
    return UserDataApplyResult(ignored_reason="unsupported_event")


async def run_user_data_stream(
    *,
    store: LiveStore,
    broker,
    mode: str,
    stop_event: asyncio.Event,
    service=None,
    keepalive_interval_seconds: float = 1800.0,
    reconnect_backoff_seconds: float = 5.0,
    sleep_func=asyncio.sleep,
) -> None:
    while not stop_event.is_set():
        try:
            listen_key = await asyncio.to_thread(broker.start_user_data_stream)
        except Exception as exc:  # noqa: BLE001
            print(
                json.dumps(
                    {"event": "user_data_stream_error", "error": str(exc)},
                    sort_keys=True,
                ),
                flush=True,
            )
            await sleep_func(reconnect_backoff_seconds)
            continue
        keepalive_task = asyncio.create_task(
            _keepalive_listen_key_loop(
                broker=broker,
                listen_key=listen_key,
                stop_event=stop_event,
                interval_seconds=keepalive_interval_seconds,
            )
        )
        try:
            import websockets

            async with websockets.connect(
                user_data_ws_url_for_mode(mode, listen_key),
                ping_interval=180,
                ping_timeout=600,
            ) as websocket:
                print(
                    json.dumps({"event": "user_data_stream_connected"}, sort_keys=True),
                    flush=True,
                )
                async for message in websocket:
                    if stop_event.is_set():
                        return
                    payload = json.loads(message.decode("utf-8") if isinstance(message, bytes) else message)
                    event = parse_user_data_event(payload)
                    if event.event_type == UserDataEventType.LISTEN_KEY_EXPIRED:
                        break
                    result = apply_user_data_event(store, event)
                    if service is not None and (
                        result.position_closed or result.positions_synced > 0
                    ):
                        service.refresh_active_symbols()
        except Exception as exc:  # noqa: BLE001
            if stop_event.is_set():
                return
            print(
                json.dumps({"event": "user_data_stream_error", "error": str(exc)}, sort_keys=True),
                flush=True,
            )
            await sleep_func(reconnect_backoff_seconds)
        finally:
            keepalive_task.cancel()
            await _await_cancelled(keepalive_task)
            await asyncio.to_thread(_close_listen_key_safely, broker, listen_key)


def _apply_order_trade_update(
    store: LiveStore,
    event: UserOrderTradeUpdate,
) -> UserDataApplyResult:
    if not event.client_order_id.startswith(STRATEGY_CLIENT_ORDER_PREFIX):
        return UserDataApplyResult(ignored_reason="external_order")
    intent = store.get_order_intent_by_client_order_id(event.client_order_id)
    if intent is None:
        return UserDataApplyResult(ignored_reason="unknown_strategy_order")

    status = _intent_status_from_order_event(intent=intent, event=event)
    store.update_order_intent_status(
        client_order_id=event.client_order_id,
        status=status,
        exchange_order_id=event.order_id,
        exchange_status=event.order_status,
        submitted_at=event.transaction_time
        if status == OrderIntentStatus.EXCHANGE_CONFIRMED
        else None,
        resolved_at=event.transaction_time
        if status in {OrderIntentStatus.RESOLVED, OrderIntentStatus.ERROR}
        else None,
    )
    position_closed = False
    if (
        event.order_status == "FILLED"
        and intent.intent_type in _CLOSE_POSITION_INTENT_TYPES
        and event.side == "SELL"
    ):
        _close_position_for_intent(store=store, intent=intent)
        position_closed = True
    elif (
        event.order_status in {"CANCELED", "EXPIRED"}
        and intent.intent_type in {OrderIntentType.STOP_PLACE, OrderIntentType.STOP_REPLACE}
    ):
        _error_lock_if_current_stop_was_cancelled(
            store=store,
            intent=intent,
            client_order_id=event.client_order_id,
        )
    return UserDataApplyResult(intent_updated=True, position_closed=position_closed)


def _apply_account_update(store: LiveStore, event: UserAccountUpdate) -> UserDataApplyResult:
    synced = 0
    closed = False
    active_positions = list_active_live_positions(store)
    for update in event.positions:
        if update.position_side not in {"BOTH", "LONG"}:
            continue
        records = [record for record in active_positions if record.symbol == update.symbol]
        for record in records:
            amount = abs(update.position_amount)
            if amount <= 0:
                _close_position_record(store=store, record=record)
                closed = True
                synced += 1
                continue
            update_live_position(
                store,
                replace(
                    record,
                    state=PositionState.OPEN,
                    quantity=amount,
                    entry_price=update.entry_price or record.entry_price,
                ),
            )
            synced += 1
    return UserDataApplyResult(position_closed=closed, positions_synced=synced)


_CLOSE_POSITION_INTENT_TYPES = {
    OrderIntentType.STOP_PLACE,
    OrderIntentType.STOP_REPLACE,
    OrderIntentType.MANUAL_RECONCILE,
    OrderIntentType.STOP_EXIT_OBSERVED,
}


def _intent_status_from_order_event(
    *,
    intent: OrderIntent,
    event: UserOrderTradeUpdate,
) -> OrderIntentStatus:
    if event.order_status in {"NEW", "PARTIALLY_FILLED"}:
        return OrderIntentStatus.EXCHANGE_CONFIRMED
    if event.order_status == "FILLED":
        return OrderIntentStatus.RESOLVED
    if event.order_status in {"CANCELED", "EXPIRED"}:
        if intent.intent_type in {OrderIntentType.STOP_PLACE, OrderIntentType.STOP_REPLACE}:
            return OrderIntentStatus.RESOLVED
        return OrderIntentStatus.ERROR
    if event.order_status in {"REJECTED"}:
        return OrderIntentStatus.ERROR
    return intent.status


def _close_position_for_intent(*, store: LiveStore, intent: OrderIntent) -> None:
    record = get_live_position(store, intent.position_id)
    if record is None:
        store.update_position_state(intent.position_id, PositionState.CLOSED)
        return
    _close_position_record(store=store, record=record)


def _error_lock_if_current_stop_was_cancelled(
    *,
    store: LiveStore,
    intent: OrderIntent,
    client_order_id: str,
) -> None:
    record = get_live_position(store, intent.position_id)
    if record is None or record.active_stop_client_order_id != client_order_id:
        return
    update_live_position(store, replace(record, state=PositionState.ERROR_LOCKED))


def _close_position_record(*, store: LiveStore, record) -> None:
    update_live_position(
        store,
        replace(
            record,
            state=PositionState.CLOSED,
            quantity=0.0,
            stop_price=None,
            next_add_trigger=None,
            active_stop_client_order_id=None,
        ),
    )


async def _keepalive_listen_key_loop(
    *,
    broker,
    listen_key: str,
    stop_event: asyncio.Event,
    interval_seconds: float,
) -> None:
    while not stop_event.is_set():
        await asyncio.sleep(interval_seconds)
        if stop_event.is_set():
            return
        await asyncio.to_thread(broker.keepalive_user_data_stream, listen_key)


async def _await_cancelled(task: asyncio.Task) -> None:
    try:
        await task
    except asyncio.CancelledError:
        return


def _close_listen_key_safely(broker, listen_key: str) -> None:
    try:
        broker.close_user_data_stream(listen_key)
    except Exception:  # noqa: BLE001
        return


def _ms_to_dt(value) -> datetime | None:
    if value is None:
        return None
    return datetime.fromtimestamp(int(value) / 1000, tz=timezone.utc)


def _optional_float(value) -> float | None:
    if value in {None, ""}:
        return None
    return float(value)


def _float(value, *, default: float) -> float:
    if value in {None, ""}:
        return default
    return float(value)


def _truthy(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() == "true"
