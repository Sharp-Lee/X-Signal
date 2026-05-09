from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class PositionState(StrEnum):
    FLAT = "FLAT"
    ENTRY_SUBMITTED = "ENTRY_SUBMITTED"
    OPEN = "OPEN"
    ADD_ARMED = "ADD_ARMED"
    ADD_SUBMITTED = "ADD_SUBMITTED"
    STOP_REPLACING = "STOP_REPLACING"
    EXITING = "EXITING"
    CLOSED = "CLOSED"
    ERROR_LOCKED = "ERROR_LOCKED"


class OrderIntentType(StrEnum):
    ENTRY = "ENTRY"
    PYRAMID_ADD = "PYRAMID_ADD"
    STOP_PLACE = "STOP_PLACE"
    STOP_REPLACE = "STOP_REPLACE"
    STOP_EXIT_OBSERVED = "STOP_EXIT_OBSERVED"
    MANUAL_RECONCILE = "MANUAL_RECONCILE"


class OrderIntentStatus(StrEnum):
    PENDING_SUBMIT = "PENDING_SUBMIT"
    SUBMITTED = "SUBMITTED"
    EXCHANGE_CONFIRMED = "EXCHANGE_CONFIRMED"
    RESOLVED = "RESOLVED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class SymbolMetadata:
    symbol: str
    status: str
    min_notional: float
    quantity_step: float
    price_tick: float
    supports_stop_market: bool
    trigger_protect: float
    updated_at: datetime
    min_quantity: float = 0.0
    max_quantity: float | None = None
    market_min_quantity: float = 0.0
    market_max_quantity: float | None = None
    market_quantity_step: float = 0.0


@dataclass(frozen=True)
class AccountSnapshot:
    mode: str
    account_mode: str
    asset_mode: str
    equity: float
    available_balance: float
    open_notional: float
    open_position_count: int
    daily_realized_pnl: float
    captured_at: datetime


@dataclass(frozen=True)
class OrderIntent:
    intent_id: str
    position_id: str
    symbol: str
    intent_type: OrderIntentType
    client_order_id: str
    side: str
    quantity: float
    notional: float
    price: float | None
    stop_price: float | None
    created_at: datetime
    status: OrderIntentStatus = OrderIntentStatus.PENDING_SUBMIT
    exchange_order_id: str | None = None
    exchange_status: str | None = None
    submitted_at: datetime | None = None
    resolved_at: datetime | None = None
    last_error: str | None = None


@dataclass(frozen=True)
class RiskResult:
    accepted: bool
    reason: str
