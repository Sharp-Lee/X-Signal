from __future__ import annotations

from dataclasses import astuple
from datetime import datetime
from pathlib import Path
import sqlite3

from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    AccountSnapshot,
    OrderIntent,
    OrderIntentStatus,
    OrderIntentType,
    PositionState,
    SymbolMetadata,
)


def _dt(value: datetime) -> str:
    return value.isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _optional_dt(value: datetime | None) -> str | None:
    return _dt(value) if value is not None else None


def _parse_optional_dt(value: str | None) -> datetime | None:
    return _parse_dt(value) if value is not None else None


class LiveStore:
    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self.connection.row_factory = sqlite3.Row

    @classmethod
    def open(cls, path: Path) -> "LiveStore":
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(sqlite3.connect(path))

    def initialize(self) -> None:
        self.connection.executescript(
            """
            create table if not exists positions (
              position_id text primary key,
              symbol text not null,
              state text not null,
              created_at text not null default CURRENT_TIMESTAMP,
              updated_at text
            );
            create table if not exists order_intents (
              intent_id text primary key,
              position_id text not null,
              symbol text not null,
              intent_type text not null,
              client_order_id text not null unique,
              side text not null,
              quantity real not null,
              notional real not null,
              price real,
              stop_price real,
              created_at text not null,
              status text not null default 'PENDING_SUBMIT',
              exchange_order_id text,
              exchange_status text,
              submitted_at text,
              resolved_at text,
              last_error text
            );
            create table if not exists symbol_metadata (
              symbol text primary key,
              status text not null,
              min_notional real not null,
              quantity_step real not null,
              price_tick real not null,
              supports_stop_market integer not null,
              trigger_protect real not null,
              updated_at text not null,
              min_quantity real not null default 0,
              max_quantity real,
              market_min_quantity real not null default 0,
              market_max_quantity real,
              market_quantity_step real not null default 0
            );
            create table if not exists account_snapshots (
              id integer primary key autoincrement,
              mode text not null,
              account_mode text not null,
              asset_mode text not null,
              equity real not null,
              available_balance real not null,
              open_notional real not null,
              open_position_count integer not null,
              daily_realized_pnl real not null,
              captured_at text not null
            );
            """
        )
        self._ensure_position_columns()
        self._ensure_order_intent_columns()
        self._ensure_symbol_metadata_columns()
        self.connection.commit()

    def _ensure_position_columns(self) -> None:
        columns = {
            row["name"] for row in self.connection.execute("pragma table_info(positions)").fetchall()
        }
        if "updated_at" not in columns:
            self.connection.execute("alter table positions add column updated_at text")

    def _ensure_order_intent_columns(self) -> None:
        columns = {
            row["name"]
            for row in self.connection.execute("pragma table_info(order_intents)").fetchall()
        }
        for name, definition in {
            "status": "text not null default 'PENDING_SUBMIT'",
            "exchange_order_id": "text",
            "exchange_status": "text",
            "submitted_at": "text",
            "resolved_at": "text",
            "last_error": "text",
        }.items():
            if name not in columns:
                self.connection.execute(f"alter table order_intents add column {name} {definition}")

    def _ensure_symbol_metadata_columns(self) -> None:
        columns = {
            row["name"]
            for row in self.connection.execute("pragma table_info(symbol_metadata)").fetchall()
        }
        for name, definition in {
            "min_quantity": "real not null default 0",
            "max_quantity": "real",
            "market_min_quantity": "real not null default 0",
            "market_max_quantity": "real",
            "market_quantity_step": "real not null default 0",
        }.items():
            if name not in columns:
                self.connection.execute(
                    f"alter table symbol_metadata add column {name} {definition}"
                )

    def create_position(self, *, symbol: str, state: PositionState) -> str:
        row = self.connection.execute("select count(*) from positions").fetchone()
        position_id = f"{symbol}-{row[0] + 1}"
        self.connection.execute(
            "insert into positions(position_id, symbol, state, updated_at) values (?, ?, ?, ?)",
            (position_id, symbol, state.value, _dt(datetime.now().astimezone())),
        )
        self.connection.commit()
        return position_id

    def get_position_state(self, position_id: str) -> PositionState | None:
        row = self.connection.execute(
            "select state from positions where position_id = ?",
            (position_id,),
        ).fetchone()
        if row is None:
            return None
        return PositionState(row["state"])

    def update_position_state(self, position_id: str, state: PositionState) -> None:
        self.connection.execute(
            "update positions set state = ?, updated_at = ? where position_id = ?",
            (state.value, _dt(datetime.now().astimezone()), position_id),
        )
        self.connection.commit()

    def list_positions_by_states(self, states: list[PositionState]) -> list[sqlite3.Row]:
        if not states:
            return []
        placeholders = ",".join("?" for _ in states)
        return list(
            self.connection.execute(
                f"""
                select position_id, symbol, state, created_at, updated_at
                from positions
                where state in ({placeholders})
                order by created_at, position_id
                """,
                tuple(state.value for state in states),
            ).fetchall()
        )

    def record_order_intent(self, intent: OrderIntent) -> None:
        self.connection.execute(
            """
            insert into order_intents(
              intent_id, position_id, symbol, intent_type, client_order_id,
              side, quantity, notional, price, stop_price, created_at, status,
              exchange_order_id, exchange_status, submitted_at, resolved_at, last_error
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(intent_id) do update set
              position_id = excluded.position_id,
              symbol = excluded.symbol,
              intent_type = excluded.intent_type,
              client_order_id = excluded.client_order_id,
              side = excluded.side,
              quantity = excluded.quantity,
              notional = excluded.notional,
              price = excluded.price,
              stop_price = excluded.stop_price
            """,
            (
                intent.intent_id,
                intent.position_id,
                intent.symbol,
                intent.intent_type.value,
                intent.client_order_id,
                intent.side,
                intent.quantity,
                intent.notional,
                intent.price,
                intent.stop_price,
                _dt(intent.created_at),
                intent.status.value,
                intent.exchange_order_id,
                intent.exchange_status,
                _optional_dt(intent.submitted_at),
                _optional_dt(intent.resolved_at),
                intent.last_error,
            ),
        )
        self.connection.commit()

    def get_order_intent(self, intent_id: str) -> OrderIntent | None:
        row = self.connection.execute(
            "select * from order_intents where intent_id = ?",
            (intent_id,),
        ).fetchone()
        if row is None:
            return None
        return self._order_intent_from_row(row)

    def get_order_intent_by_client_order_id(self, client_order_id: str) -> OrderIntent | None:
        row = self.connection.execute(
            "select * from order_intents where client_order_id = ?",
            (client_order_id,),
        ).fetchone()
        if row is None:
            return None
        return self._order_intent_from_row(row)

    def list_unresolved_order_intents(self) -> list[OrderIntent]:
        rows = self.connection.execute(
            """
            select *
            from order_intents
            where status not in (?, ?)
            order by created_at, intent_id
            """,
            (OrderIntentStatus.RESOLVED.value, OrderIntentStatus.ERROR.value),
        ).fetchall()
        return [self._order_intent_from_row(row) for row in rows]

    def update_order_intent_status(
        self,
        *,
        client_order_id: str,
        status: OrderIntentStatus,
        exchange_order_id: str | None = None,
        exchange_status: str | None = None,
        submitted_at: datetime | None = None,
        resolved_at: datetime | None = None,
        last_error: str | None = None,
    ) -> None:
        assignments = ["status = ?"]
        values: list[object] = [status.value]
        optional_values = {
            "exchange_order_id": exchange_order_id,
            "exchange_status": exchange_status,
            "submitted_at": _optional_dt(submitted_at),
            "resolved_at": _optional_dt(resolved_at),
            "last_error": last_error,
        }
        for column, value in optional_values.items():
            if value is not None:
                assignments.append(f"{column} = ?")
                values.append(value)
        values.append(client_order_id)
        self.connection.execute(
            f"update order_intents set {', '.join(assignments)} where client_order_id = ?",
            tuple(values),
        )
        self.connection.commit()

    def _order_intent_from_row(self, row: sqlite3.Row) -> OrderIntent:
        return OrderIntent(
            intent_id=row["intent_id"],
            position_id=row["position_id"],
            symbol=row["symbol"],
            intent_type=OrderIntentType(row["intent_type"]),
            client_order_id=row["client_order_id"],
            side=row["side"],
            quantity=row["quantity"],
            notional=row["notional"],
            price=row["price"],
            stop_price=row["stop_price"],
            created_at=_parse_dt(row["created_at"]),
            status=OrderIntentStatus(row["status"]),
            exchange_order_id=row["exchange_order_id"],
            exchange_status=row["exchange_status"],
            submitted_at=_parse_optional_dt(row["submitted_at"]),
            resolved_at=_parse_optional_dt(row["resolved_at"]),
            last_error=row["last_error"],
        )

    def upsert_symbol_metadata(self, metadata: SymbolMetadata) -> None:
        self.connection.execute(
            """
            insert into symbol_metadata(
              symbol, status, min_notional, quantity_step, price_tick,
              supports_stop_market, trigger_protect, updated_at,
              min_quantity, max_quantity, market_min_quantity, market_max_quantity,
              market_quantity_step
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(symbol) do update set
              status = excluded.status,
              min_notional = excluded.min_notional,
              quantity_step = excluded.quantity_step,
              price_tick = excluded.price_tick,
              supports_stop_market = excluded.supports_stop_market,
              trigger_protect = excluded.trigger_protect,
              updated_at = excluded.updated_at,
              min_quantity = excluded.min_quantity,
              max_quantity = excluded.max_quantity,
              market_min_quantity = excluded.market_min_quantity,
              market_max_quantity = excluded.market_max_quantity,
              market_quantity_step = excluded.market_quantity_step
            """,
            (
                metadata.symbol,
                metadata.status,
                metadata.min_notional,
                metadata.quantity_step,
                metadata.price_tick,
                int(metadata.supports_stop_market),
                metadata.trigger_protect,
                _dt(metadata.updated_at),
                metadata.min_quantity,
                metadata.max_quantity,
                metadata.market_min_quantity,
                metadata.market_max_quantity,
                metadata.market_quantity_step,
            ),
        )
        self.connection.commit()

    def get_symbol_metadata(self, symbol: str) -> SymbolMetadata | None:
        row = self.connection.execute(
            "select * from symbol_metadata where symbol = ?",
            (symbol,),
        ).fetchone()
        if row is None:
            return None
        return SymbolMetadata(
            symbol=row["symbol"],
            status=row["status"],
            min_notional=row["min_notional"],
            quantity_step=row["quantity_step"],
            price_tick=row["price_tick"],
            supports_stop_market=bool(row["supports_stop_market"]),
            trigger_protect=row["trigger_protect"],
            updated_at=_parse_dt(row["updated_at"]),
            min_quantity=row["min_quantity"],
            max_quantity=row["max_quantity"],
            market_min_quantity=row["market_min_quantity"],
            market_max_quantity=row["market_max_quantity"],
            market_quantity_step=row["market_quantity_step"],
        )

    def record_account_snapshot(self, snapshot: AccountSnapshot) -> None:
        self.connection.execute(
            """
            insert into account_snapshots(
              mode, account_mode, asset_mode, equity, available_balance,
              open_notional, open_position_count, daily_realized_pnl, captured_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (*astuple(snapshot)[:-1], _dt(snapshot.captured_at)),
        )
        self.connection.commit()

    def latest_account_snapshot(self) -> AccountSnapshot | None:
        row = self.connection.execute(
            "select * from account_snapshots order by id desc limit 1",
        ).fetchone()
        if row is None:
            return None
        return AccountSnapshot(
            mode=row["mode"],
            account_mode=row["account_mode"],
            asset_mode=row["asset_mode"],
            equity=row["equity"],
            available_balance=row["available_balance"],
            open_notional=row["open_notional"],
            open_position_count=row["open_position_count"],
            daily_realized_pnl=row["daily_realized_pnl"],
            captured_at=_parse_dt(row["captured_at"]),
        )
