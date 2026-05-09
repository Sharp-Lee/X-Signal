from __future__ import annotations

from dataclasses import astuple
from datetime import datetime
from pathlib import Path
import sqlite3

from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    AccountSnapshot,
    OrderIntent,
    OrderIntentType,
    PositionState,
    SymbolMetadata,
)


def _dt(value: datetime) -> str:
    return value.isoformat()


def _parse_dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


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
              created_at text not null default CURRENT_TIMESTAMP
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
              created_at text not null
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
        self._ensure_symbol_metadata_columns()
        self.connection.commit()

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
            "insert into positions(position_id, symbol, state) values (?, ?, ?)",
            (position_id, symbol, state.value),
        )
        self.connection.commit()
        return position_id

    def record_order_intent(self, intent: OrderIntent) -> None:
        self.connection.execute(
            """
            insert into order_intents(
              intent_id, position_id, symbol, intent_type, client_order_id,
              side, quantity, notional, price, stop_price, created_at
            ) values (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
