from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from xsignal.strategies.volume_price_efficiency_v1.live.models import PositionState
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore


@dataclass(frozen=True)
class LivePositionRecord:
    position_id: str
    symbol: str
    state: PositionState
    entry_price: float | None
    quantity: float
    highest_high: float | None
    stop_price: float | None
    atr_at_entry: float | None
    next_add_trigger: float | None
    add_count: int
    active_stop_client_order_id: str | None
    last_decision_open_time: datetime | None
    strategy_interval: str | None = None
    last_stop_replace_at: datetime | None = None


def update_live_position(store: LiveStore, record: LivePositionRecord) -> None:
    store.connection.execute(
        """
        update positions set
          state = ?,
          entry_price = ?,
          quantity = ?,
          highest_high = ?,
          stop_price = ?,
          atr_at_entry = ?,
          next_add_trigger = ?,
          add_count = ?,
          active_stop_client_order_id = ?,
          last_decision_open_time = ?,
          strategy_interval = ?,
          last_stop_replace_at = ?
        where position_id = ?
        """,
        (
            record.state.value,
            record.entry_price,
            record.quantity,
            record.highest_high,
            record.stop_price,
            record.atr_at_entry,
            record.next_add_trigger,
            record.add_count,
            record.active_stop_client_order_id,
            _dt(record.last_decision_open_time),
            record.strategy_interval,
            _dt(record.last_stop_replace_at),
            record.position_id,
        ),
    )
    store.connection.commit()


def get_live_position(store: LiveStore, position_id: str) -> LivePositionRecord | None:
    row = store.connection.execute(
        "select * from positions where position_id = ?",
        (position_id,),
    ).fetchone()
    if row is None:
        return None
    return _from_row(row)


def list_active_live_positions(store: LiveStore) -> list[LivePositionRecord]:
    rows = store.connection.execute(
        """
        select *
        from positions
        where state in (?, ?, ?, ?, ?)
        order by created_at, position_id
        """,
        (
            PositionState.OPEN.value,
            PositionState.ADD_ARMED.value,
            PositionState.ADD_SUBMITTED.value,
            PositionState.STOP_REPLACING.value,
            PositionState.EXITING.value,
        ),
    ).fetchall()
    return [_from_row(row) for row in rows]


def _from_row(row) -> LivePositionRecord:
    return LivePositionRecord(
        position_id=row["position_id"],
        symbol=row["symbol"],
        state=PositionState(row["state"]),
        entry_price=row["entry_price"],
        quantity=float(row["quantity"] or 0.0),
        highest_high=row["highest_high"],
        stop_price=row["stop_price"],
        atr_at_entry=row["atr_at_entry"],
        next_add_trigger=row["next_add_trigger"],
        add_count=int(row["add_count"] or 0),
        active_stop_client_order_id=row["active_stop_client_order_id"],
        last_decision_open_time=_parse_dt(row["last_decision_open_time"]),
        strategy_interval=row["strategy_interval"],
        last_stop_replace_at=_parse_dt(row["last_stop_replace_at"]),
    )


def _dt(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _parse_dt(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value is not None else None
