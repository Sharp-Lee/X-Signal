from __future__ import annotations

from dataclasses import dataclass
import time
import uuid

from xsignal.strategies.volume_price_efficiency_v1.live.binance_rest import BinanceApiError
from xsignal.strategies.volume_price_efficiency_v1.live.ids import build_client_order_id


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


def run_testnet_lifecycle(
    *,
    broker,
    symbol: str,
    quantity: float,
    stop_offset_pct: float,
    position_id: str | None = None,
    price_tick: float | None = None,
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

    position_id = position_id or uuid.uuid4().hex[:16]
    entry_client_order_id = build_client_order_id(
        env="testnet",
        intent="ENTRY",
        symbol=symbol,
        position_id=position_id,
        sequence=1,
    )
    stop_client_order_id = build_client_order_id(
        env="testnet",
        intent="STOP_PLACE",
        symbol=symbol,
        position_id=position_id,
        sequence=2,
    )
    close_client_order_id = build_client_order_id(
        env="testnet",
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
        broker.market_buy(
            symbol=symbol,
            quantity=quantity,
            client_order_id=entry_client_order_id,
        )

        opened_position = _wait_for_long_position(
            broker=broker,
            symbol=symbol,
            attempts=poll_attempts,
            sleep_seconds=poll_sleep_seconds,
        )
        opened_position_amount = _position_amount(opened_position)
        reference_price = _position_reference_price(opened_position)
        stop_price = reference_price * (1 - stop_offset_pct)
        if price_tick is not None:
            stop_price = _round_down_to_tick(stop_price, price_tick)
        if stop_price <= 0:
            raise RuntimeError("computed stop price is not positive")

        broker.place_stop_market_close(
            symbol=symbol,
            stop_price=stop_price,
            client_order_id=stop_client_order_id,
        )
        stop_placed = True

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
        broker.market_sell_reduce_only(
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

        return TestnetLifecycleResult(
            symbol=symbol,
            quantity=quantity,
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


def _set_isolated_margin(broker, symbol: str) -> None:
    try:
        broker.change_margin_type(symbol, "isolated")
    except BinanceApiError as exc:
        if exc.code == -4046:
            return
        raise


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
    ticks = int(price / price_tick)
    return ticks * price_tick
