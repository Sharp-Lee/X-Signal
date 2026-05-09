from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal
import time
import uuid

from xsignal.strategies.volume_price_efficiency_v1.live.binance_rest import BinanceApiError
from xsignal.strategies.volume_price_efficiency_v1.live.ids import build_client_order_id
from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    OrderIntent,
    OrderIntentStatus,
    OrderIntentType,
    PositionState,
)
from xsignal.strategies.volume_price_efficiency_v1.live.order_normalizer import (
    SymbolRules,
    floor_to_step,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore


@dataclass(frozen=True)
class TestnetLifecycleResult:
    symbol: str
    quantity: float
    stop_offset_pct: float
    stop_price: float
    entry_client_order_id: str
    stop_client_order_id: str
    close_client_order_id: str
    opened_position_amount: float
    final_position_amount: float


class UnknownStopSubmitStatus(RuntimeError):
    pass


def run_testnet_lifecycle(
    *,
    broker,
    symbol: str,
    quantity: float,
    stop_offset_pct: float,
    store: LiveStore | None = None,
    environment: str = "testnet",
    position_id: str | None = None,
    price_tick: float | None = None,
    symbol_rules: SymbolRules | None = None,
    poll_attempts: int = 5,
    poll_sleep_seconds: float = 0.25,
) -> TestnetLifecycleResult:
    if quantity <= 0:
        raise ValueError("quantity must be positive")
    if stop_offset_pct <= 0 or stop_offset_pct >= 1:
        raise ValueError("stop_offset_pct must be in (0, 1)")
    if price_tick is not None and price_tick <= 0:
        raise ValueError("price_tick must be positive")
    if poll_attempts <= 0:
        raise ValueError("poll_attempts must be positive")
    normalized_quantity = (
        symbol_rules.normalize_market_quantity(quantity) if symbol_rules is not None else quantity
    )

    position_id = _prepare_lifecycle_position(
        store=store,
        symbol=symbol,
        position_id=position_id,
    )
    entry_client_order_id = build_client_order_id(
        env=environment,
        intent="ENTRY",
        symbol=symbol,
        position_id=position_id,
        sequence=1,
    )
    stop_client_order_id = build_client_order_id(
        env=environment,
        intent="STOP_PLACE",
        symbol=symbol,
        position_id=position_id,
        sequence=2,
    )
    close_client_order_id = build_client_order_id(
        env=environment,
        intent="MANUAL_RECONCILE",
        symbol=symbol,
        position_id=position_id,
        sequence=3,
    )

    stop_placed = False
    opened_position_amount = 0.0
    stop_price = 0.0
    try:
        _set_isolated_margin(broker, symbol)
        broker.change_leverage(symbol, 1)
        _record_lifecycle_intent(
            store=store,
            intent_id=entry_client_order_id,
            position_id=position_id,
            symbol=symbol,
            intent_type=OrderIntentType.ENTRY,
            side="BUY",
            quantity=normalized_quantity,
            notional=0.0,
            price=None,
            stop_price=None,
        )
        entry_submit_error = _submit_market_buy_with_reconcile(
            broker=broker,
            symbol=symbol,
            quantity=normalized_quantity,
            client_order_id=entry_client_order_id,
        )
        _update_lifecycle_intent(
            store=store,
            client_order_id=entry_client_order_id,
            status=OrderIntentStatus.EXCHANGE_CONFIRMED,
            exchange_status="UNKNOWN" if entry_submit_error is None else "UNKNOWN_AFTER_ERROR",
            submitted_at=_utc_now(),
        )

        try:
            opened_position = _wait_for_long_position(
                broker=broker,
                symbol=symbol,
                attempts=poll_attempts,
                sleep_seconds=poll_sleep_seconds,
            )
        except RuntimeError as exc:
            if entry_submit_error is not None:
                raise RuntimeError("entry submit status is unknown and no long position is visible") from exc
            raise
        _update_position_state(store=store, position_id=position_id, state=PositionState.OPEN)
        opened_position_amount = _position_amount(opened_position)
        reference_price = _position_reference_price(opened_position)
        stop_price = reference_price * (1 - stop_offset_pct)
        stop_price_for_order = stop_price
        if symbol_rules is not None:
            stop_price_for_order = symbol_rules.normalize_price(Decimal(str(stop_price)))
            stop_price = float(stop_price_for_order)
        elif price_tick is not None:
            stop_price = _round_down_to_tick(stop_price, price_tick)
            stop_price_for_order = stop_price
        if stop_price <= 0:
            raise RuntimeError("computed stop price is not positive")

        _record_lifecycle_intent(
            store=store,
            intent_id=stop_client_order_id,
            position_id=position_id,
            symbol=symbol,
            intent_type=OrderIntentType.STOP_PLACE,
            side="SELL",
            quantity=0.0,
            notional=0.0,
            price=None,
            stop_price=stop_price,
        )
        try:
            stop_placed = _submit_stop_with_reconcile(
                broker=broker,
                symbol=symbol,
                stop_price=stop_price_for_order,
                client_order_id=stop_client_order_id,
            )
        except UnknownStopSubmitStatus:
            stop_placed = True
            raise
        _update_lifecycle_intent(
            store=store,
            client_order_id=stop_client_order_id,
            status=OrderIntentStatus.EXCHANGE_CONFIRMED,
            exchange_status="NEW",
            submitted_at=_utc_now(),
        )

        protected_position = _wait_for_long_position(
            broker=broker,
            symbol=symbol,
            attempts=poll_attempts,
            sleep_seconds=poll_sleep_seconds,
        )
        protected_position_amount = _position_amount(protected_position)
        opened_position_amount = protected_position_amount

        open_stop = broker.get_open_order(
            symbol=symbol,
            client_order_id=stop_client_order_id,
        )
        if not _is_open_order(open_stop):
            raise RuntimeError("protective stop is not open")

        broker.cancel_order(symbol=symbol, client_order_id=stop_client_order_id)
        stop_placed = False
        _update_lifecycle_intent(
            store=store,
            client_order_id=stop_client_order_id,
            status=OrderIntentStatus.RESOLVED,
            exchange_status="CANCELED_BY_LIFECYCLE",
            resolved_at=_utc_now(),
        )
        _record_lifecycle_intent(
            store=store,
            intent_id=close_client_order_id,
            position_id=position_id,
            symbol=symbol,
            intent_type=OrderIntentType.MANUAL_RECONCILE,
            side="SELL",
            quantity=protected_position_amount,
            notional=0.0,
            price=None,
            stop_price=None,
        )
        _submit_close_with_reconcile(
            broker=broker,
            symbol=symbol,
            quantity=protected_position_amount,
            client_order_id=close_client_order_id,
        )

        final_position = _wait_for_flat_position(
            broker=broker,
            symbol=symbol,
            attempts=poll_attempts,
            sleep_seconds=poll_sleep_seconds,
        )
        final_position_amount = _position_amount(final_position)
        _update_lifecycle_intent(
            store=store,
            client_order_id=close_client_order_id,
            status=OrderIntentStatus.RESOLVED,
            exchange_status="FILLED_OR_FLAT",
            submitted_at=_utc_now(),
            resolved_at=_utc_now(),
        )
        _update_position_state(store=store, position_id=position_id, state=PositionState.CLOSED)

        return TestnetLifecycleResult(
            symbol=symbol,
            quantity=float(normalized_quantity),
            stop_offset_pct=stop_offset_pct,
            stop_price=stop_price,
            entry_client_order_id=entry_client_order_id,
            stop_client_order_id=stop_client_order_id,
            close_client_order_id=close_client_order_id,
            opened_position_amount=opened_position_amount,
            final_position_amount=final_position_amount,
        )
    except Exception:
        _cleanup_testnet_lifecycle(
            broker=broker,
            symbol=symbol,
            stop_placed=stop_placed,
            stop_client_order_id=stop_client_order_id,
            opened_position_amount=opened_position_amount,
            close_client_order_id=close_client_order_id,
        )
        raise


def _prepare_lifecycle_position(
    *,
    store: LiveStore | None,
    symbol: str,
    position_id: str | None,
) -> str:
    if store is None:
        return position_id or uuid.uuid4().hex[:16]
    if position_id is None:
        return store.create_position(symbol=symbol, state=PositionState.ENTRY_SUBMITTED)
    _ensure_position_row(store=store, position_id=position_id, symbol=symbol)
    store.update_position_state(position_id, PositionState.ENTRY_SUBMITTED)
    return position_id


def _ensure_position_row(*, store: LiveStore, position_id: str, symbol: str) -> None:
    if store.get_position_state(position_id) is not None:
        return
    store.connection.execute(
        """
        insert into positions(position_id, symbol, state, updated_at)
        values (?, ?, ?, ?)
        """,
        (position_id, symbol, PositionState.ENTRY_SUBMITTED.value, _utc_now().isoformat()),
    )
    store.connection.commit()


def _record_lifecycle_intent(
    *,
    store: LiveStore | None,
    intent_id: str,
    position_id: str,
    symbol: str,
    intent_type: OrderIntentType,
    side: str,
    quantity,
    notional: float,
    price: float | None,
    stop_price: float | None,
) -> None:
    if store is None:
        return
    store.record_order_intent(
        OrderIntent(
            intent_id=intent_id,
            position_id=position_id,
            symbol=symbol,
            intent_type=intent_type,
            client_order_id=intent_id,
            side=side,
            quantity=float(quantity),
            notional=notional,
            price=price,
            stop_price=stop_price,
            created_at=_utc_now(),
        )
    )


def _update_lifecycle_intent(
    *,
    store: LiveStore | None,
    client_order_id: str,
    status: OrderIntentStatus,
    exchange_status: str,
    submitted_at: datetime | None = None,
    resolved_at: datetime | None = None,
) -> None:
    if store is None:
        return
    store.update_order_intent_status(
        client_order_id=client_order_id,
        status=status,
        exchange_status=exchange_status,
        submitted_at=submitted_at,
        resolved_at=resolved_at,
    )


def _update_position_state(
    *,
    store: LiveStore | None,
    position_id: str,
    state: PositionState,
) -> None:
    if store is not None:
        store.update_position_state(position_id, state)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _set_isolated_margin(broker, symbol: str) -> None:
    try:
        broker.change_margin_type(symbol, "isolated")
    except BinanceApiError as exc:
        if exc.code == -4046:
            return
        raise


def _submit_market_buy_with_reconcile(
    *,
    broker,
    symbol: str,
    quantity,
    client_order_id: str,
) -> Exception | None:
    try:
        broker.market_buy(
            symbol=symbol,
            quantity=quantity,
            client_order_id=client_order_id,
        )
        return None
    except Exception as exc:
        if not _is_unknown_exchange_error(exc):
            raise
        _query_regular_order_after_unknown_submit(
            broker=broker,
            symbol=symbol,
            client_order_id=client_order_id,
        )
        return exc


def _submit_stop_with_reconcile(
    *,
    broker,
    symbol: str,
    stop_price,
    client_order_id: str,
) -> bool:
    try:
        broker.place_stop_market_close(
            symbol=symbol,
            stop_price=stop_price,
            client_order_id=client_order_id,
        )
        return True
    except Exception as exc:
        if not _is_unknown_exchange_error(exc):
            raise
        try:
            open_stop = broker.get_open_order(
                symbol=symbol,
                client_order_id=client_order_id,
            )
        except Exception as query_exc:
            raise UnknownStopSubmitStatus("protective stop submit status is unknown") from query_exc
        if not _is_open_order(open_stop):
            raise RuntimeError("protective stop submit did not leave an open stop")
        return True


def _submit_close_with_reconcile(
    *,
    broker,
    symbol: str,
    quantity: float,
    client_order_id: str,
) -> Exception | None:
    try:
        broker.market_sell_reduce_only(
            symbol=symbol,
            quantity=quantity,
            client_order_id=client_order_id,
        )
        return None
    except Exception as exc:
        if not _is_unknown_exchange_error(exc):
            raise
        _query_regular_order_after_unknown_submit(
            broker=broker,
            symbol=symbol,
            client_order_id=client_order_id,
        )
        return exc


def _query_regular_order_after_unknown_submit(
    *,
    broker,
    symbol: str,
    client_order_id: str,
) -> dict | None:
    try:
        order = broker.get_order(symbol=symbol, client_order_id=client_order_id)
    except AttributeError:
        return None
    except Exception:
        return None
    status = order.get("status")
    if status in {"CANCELED", "EXPIRED", "REJECTED"}:
        raise RuntimeError(f"order {client_order_id} resolved to terminal status {status}")
    return order


def _is_unknown_exchange_error(exc: Exception) -> bool:
    return not isinstance(exc, BinanceApiError) or exc.status >= 500


def _cleanup_testnet_lifecycle(
    *,
    broker,
    symbol: str,
    stop_placed: bool,
    stop_client_order_id: str,
    opened_position_amount: float,
    close_client_order_id: str,
) -> None:
    if stop_placed:
        try:
            broker.cancel_order(symbol=symbol, client_order_id=stop_client_order_id)
        except Exception:
            pass
    current_position_amount = opened_position_amount
    try:
        current_position = _find_position(broker.get_position_risk(symbol=symbol), symbol)
        current_position_amount = _position_amount(current_position)
    except RuntimeError:
        current_position_amount = 0.0
    except Exception:
        pass
    if current_position_amount > 0:
        try:
            broker.market_sell_reduce_only(
                symbol=symbol,
                quantity=current_position_amount,
                client_order_id=close_client_order_id,
            )
        except Exception:
            pass


def _wait_for_long_position(
    *,
    broker,
    symbol: str,
    attempts: int,
    sleep_seconds: float,
) -> dict:
    last_position = None
    for attempt in range(attempts):
        try:
            last_position = _find_position(broker.get_position_risk(symbol=symbol), symbol)
        except RuntimeError:
            last_position = _zero_position(symbol)
        if _position_amount(last_position) > 0:
            return last_position
        _sleep_before_next_poll(attempt=attempt, attempts=attempts, sleep_seconds=sleep_seconds)
    raise RuntimeError("testnet lifecycle did not open a long position")


def _wait_for_flat_position(
    *,
    broker,
    symbol: str,
    attempts: int,
    sleep_seconds: float,
) -> dict:
    last_position = None
    for attempt in range(attempts):
        try:
            last_position = _find_position(broker.get_position_risk(symbol=symbol), symbol)
        except RuntimeError:
            return _zero_position(symbol)
        if abs(_position_amount(last_position)) <= 1e-12:
            return last_position
        _sleep_before_next_poll(attempt=attempt, attempts=attempts, sleep_seconds=sleep_seconds)
    raise RuntimeError("testnet lifecycle did not return to flat")


def _sleep_before_next_poll(*, attempt: int, attempts: int, sleep_seconds: float) -> None:
    if attempt + 1 < attempts and sleep_seconds > 0:
        time.sleep(sleep_seconds)


def _find_position(payload, symbol: str) -> dict:
    positions = payload if isinstance(payload, list) else [payload]
    for item in positions:
        if item.get("symbol") != symbol:
            continue
        if item.get("positionSide", "BOTH") in {"BOTH", "LONG"}:
            return item
    raise RuntimeError(f"missing position risk for {symbol}")


def _zero_position(symbol: str) -> dict:
    return {"symbol": symbol, "positionSide": "BOTH", "positionAmt": "0"}


def _position_amount(position: dict) -> float:
    return float(position.get("positionAmt", 0.0))


def _position_reference_price(position: dict) -> float:
    for field in ("entryPrice", "markPrice"):
        value = float(position.get(field, 0.0))
        if value > 0:
            return value
    raise RuntimeError("position has no usable entry or mark price")


def _is_open_order(order: dict) -> bool:
    status = order.get("algoStatus") or order.get("status")
    return status in {"NEW", "PARTIALLY_FILLED"}


def _round_down_to_tick(price: float, price_tick: float) -> float:
    return float(floor_to_step(Decimal(str(price)), Decimal(str(price_tick))))
