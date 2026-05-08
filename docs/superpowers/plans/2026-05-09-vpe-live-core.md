# VPE Live Core Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the offline live-trading core for the daily VPE strategy: preset config, shared-equity sizing, risk gate, SQLite state store, fake broker, state machine, closed-bar signal wrapper, and replay/status CLI.

**Architecture:** This plan deliberately stops before real Binance order submission. The first deliverable is a deterministic live core that can replay historical daily bars through the same state machine that will later drive testnet/live trading. The strategy layer remains pure signal computation; order decisions flow through capital allocation, risk checks, broker intents, and persisted state.

**Tech Stack:** Python 3.12, stdlib `sqlite3`, `dataclasses`, `enum`, `pydantic`, `numpy`, existing VPE research modules, `pytest`, `ruff`.

---

## Scope Boundary

This plan implements Phase 2 from the design spec: offline live-simulation harness. It does not add the real Binance SDK/REST/WebSocket adapter yet. The next plan will add Binance testnet connectivity after this core passes unit tests and replay smoke tests.

Design spec: `docs/superpowers/specs/2026-05-09-vpe-live-trading-design.md`

## File Structure

- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/__init__.py`
  - Package marker for live-core modules.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/config.py`
  - Fixed live preset, account assumptions, risk defaults, and mode validation.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/ids.py`
  - Compact deterministic Binance client order ids with a 36-character limit.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/models.py`
  - Enums and dataclasses for positions, intents, orders, fills, symbol metadata, account snapshots, and risk results.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/store.py`
  - SQLite schema and repository methods for config, signals, positions, orders, fills, risk events, and heartbeats.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/capital.py`
  - Shared-equity position sizing.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/risk.py`
  - Intent validation before broker submission.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/broker.py`
  - Broker protocol and fake broker used by tests/replay.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/state_machine.py`
  - Per-symbol lifecycle: entry, open, add, stop replacement, close, error lock.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/signal_engine.py`
  - Closed-bar wrapper around existing VPE features and fixed regime rule.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/replay.py`
  - Offline replay harness that feeds daily bars into the live core.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/cli.py`
  - `replay`, `status`, and `reconcile` commands for offline/fake-broker mode.
- Modify `pyproject.toml`
  - Add `xsignal-vpe-live = "xsignal.strategies.volume_price_efficiency_v1.live.cli:main"`.
- Create tests under `tests/strategies/volume_price_efficiency_v1/live/`.

---

### Task 1: Live Preset And Compact Client Order IDs

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/__init__.py`
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/config.py`
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/ids.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_config.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_ids.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/strategies/volume_price_efficiency_v1/live/test_config.py`:

```python
import pytest

from xsignal.strategies.volume_price_efficiency_v1.live.config import (
    LiveTradingConfig,
    build_vpe_live_strategy_config,
)


def test_live_defaults_match_design_spec():
    config = LiveTradingConfig()
    assert config.mode == "testnet"
    assert config.account_mode == "one_way"
    assert config.margin_mode == "isolated"
    assert config.asset_mode == "single_asset_usdt"
    assert config.direction == "long_only"
    assert config.leverage == 1
    assert config.base_position_fraction == 0.05
    assert config.per_symbol_notional_cap == 20.0
    assert config.total_open_notional_cap == 100.0
    assert config.max_open_positions == 5
    assert config.max_daily_realized_loss == 50.0


def test_live_mode_requires_acknowledgement():
    with pytest.raises(ValueError, match="live acknowledgement"):
        LiveTradingConfig(mode="live")
    assert LiveTradingConfig(mode="live", live_acknowledgement=True).mode == "live"


def test_strategy_preset_matches_final_holdout_values():
    strategy_config = build_vpe_live_strategy_config()
    assert strategy_config.timeframe == "1d"
    assert strategy_config.fill_policy == "raw"
    assert strategy_config.signal_mode == "seed_efficiency"
    assert strategy_config.min_move_unit == 0.7
    assert strategy_config.min_volume_unit == 0.3
    assert strategy_config.min_close_position == 0.7
    assert strategy_config.min_body_ratio == 0.4
    assert strategy_config.seed_efficiency_lookback == 4
    assert strategy_config.seed_min_efficiency_ratio_to_max == 2.0
    assert strategy_config.seed_min_efficiency_ratio_to_mean == 5.0
    assert strategy_config.seed_max_volume_unit == 0.8
    assert strategy_config.seed_bottom_lookback == 60
    assert strategy_config.seed_max_close_position_in_range == 0.6
```

- [ ] **Step 2: Write failing client order id tests**

Create `tests/strategies/volume_price_efficiency_v1/live/test_ids.py`:

```python
from xsignal.strategies.volume_price_efficiency_v1.live.ids import build_client_order_id


def test_client_order_id_is_deterministic_and_short():
    first = build_client_order_id(
        env="testnet",
        intent="PYRAMID_ADD",
        symbol="1000RATSUSDT",
        position_id="position-with-a-very-long-id",
        sequence=12,
    )
    second = build_client_order_id(
        env="testnet",
        intent="PYRAMID_ADD",
        symbol="1000RATSUSDT",
        position_id="position-with-a-very-long-id",
        sequence=12,
    )
    assert first == second
    assert len(first) <= 36
    assert first.startswith("XV1T")


def test_client_order_id_changes_for_different_sequence():
    first = build_client_order_id(
        env="testnet",
        intent="ENTRY",
        symbol="BTCUSDT",
        position_id="abc",
        sequence=1,
    )
    second = build_client_order_id(
        env="testnet",
        intent="ENTRY",
        symbol="BTCUSDT",
        position_id="abc",
        sequence=2,
    )
    assert first != second
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest \
  tests/strategies/volume_price_efficiency_v1/live/test_config.py \
  tests/strategies/volume_price_efficiency_v1/live/test_ids.py -q
```

Expected: import errors for the new live modules.

- [ ] **Step 4: Add minimal implementation**

Create `src/xsignal/strategies/volume_price_efficiency_v1/live/__init__.py` as an empty file.

Create `src/xsignal/strategies/volume_price_efficiency_v1/live/config.py`:

```python
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)


LiveMode = Literal["testnet", "live", "reconcile-only"]


class LiveTradingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: LiveMode = "testnet"
    live_acknowledgement: bool = False
    account_mode: str = "one_way"
    margin_mode: str = "isolated"
    asset_mode: str = "single_asset_usdt"
    direction: str = "long_only"
    leverage: int = 1
    base_position_fraction: float = 0.05
    per_symbol_notional_cap: float = 20.0
    total_open_notional_cap: float = 100.0
    max_open_positions: int = 5
    max_daily_realized_loss: float = 50.0
    min_quote_notional: float = 5.0
    atr_multiplier: float = 3.0
    pyramid_add_step_atr: float = 1.0
    pyramid_max_adds: int = 1
    market_regime_lookback_bars: int = 30
    market_regime_min_return: float = -0.10

    @model_validator(mode="after")
    def _validate(self) -> "LiveTradingConfig":
        if self.mode == "live" and not self.live_acknowledgement:
            raise ValueError("live acknowledgement is required for live mode")
        if self.account_mode != "one_way":
            raise ValueError("account_mode must be one_way")
        if self.margin_mode != "isolated":
            raise ValueError("margin_mode must be isolated")
        if self.asset_mode != "single_asset_usdt":
            raise ValueError("asset_mode must be single_asset_usdt")
        if self.direction != "long_only":
            raise ValueError("direction must be long_only")
        if self.leverage != 1:
            raise ValueError("leverage must be 1")
        if not 0.0 < self.base_position_fraction <= 1.0:
            raise ValueError("base_position_fraction must be in (0, 1]")
        if self.per_symbol_notional_cap <= 0.0:
            raise ValueError("per_symbol_notional_cap must be positive")
        if self.total_open_notional_cap <= 0.0:
            raise ValueError("total_open_notional_cap must be positive")
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be positive")
        if self.max_daily_realized_loss <= 0.0:
            raise ValueError("max_daily_realized_loss must be positive")
        if self.min_quote_notional <= 0.0:
            raise ValueError("min_quote_notional must be positive")
        if self.atr_multiplier <= 0.0:
            raise ValueError("atr_multiplier must be positive")
        if self.pyramid_add_step_atr <= 0.0:
            raise ValueError("pyramid_add_step_atr must be positive")
        if self.pyramid_max_adds != 1:
            raise ValueError("pyramid_max_adds must be 1 for the first live preset")
        return self


def build_vpe_live_strategy_config() -> VolumePriceEfficiencyConfig:
    return VolumePriceEfficiencyConfig(
        signal_mode="seed_efficiency",
        min_move_unit=0.7,
        min_volume_unit=0.3,
        min_close_position=0.7,
        min_body_ratio=0.4,
        seed_efficiency_lookback=4,
        seed_min_efficiency_ratio_to_max=2.0,
        seed_min_efficiency_ratio_to_mean=5.0,
        seed_max_volume_unit=0.8,
        seed_bottom_lookback=60,
        seed_max_close_position_in_range=0.6,
    )
```

Create `src/xsignal/strategies/volume_price_efficiency_v1/live/ids.py`:

```python
from __future__ import annotations

import hashlib


_INTENT_CODES = {
    "ENTRY": "E",
    "PYRAMID_ADD": "A",
    "STOP_PLACE": "S",
    "STOP_REPLACE": "R",
    "STOP_EXIT_OBSERVED": "X",
    "MANUAL_RECONCILE": "M",
}


def _env_code(env: str) -> str:
    if env == "testnet":
        return "T"
    if env == "live":
        return "L"
    if env == "reconcile-only":
        return "R"
    raise ValueError(f"unsupported env: {env}")


def build_client_order_id(
    *,
    env: str,
    intent: str,
    symbol: str,
    position_id: str,
    sequence: int,
) -> str:
    if sequence < 0:
        raise ValueError("sequence must be non-negative")
    intent_code = _INTENT_CODES.get(intent)
    if intent_code is None:
        raise ValueError(f"unsupported intent: {intent}")
    digest = hashlib.sha1(f"{symbol}|{position_id}|{sequence}".encode()).hexdigest()[:14]
    symbol_code = "".join(ch for ch in symbol.upper() if ch.isalnum())[:10]
    client_id = f"XV1{_env_code(env)}{intent_code}{symbol_code}{digest}{sequence:02d}"
    if len(client_id) > 36:
        client_id = f"XV1{_env_code(env)}{intent_code}{digest}{sequence:02d}"
    return client_id
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest \
  tests/strategies/volume_price_efficiency_v1/live/test_config.py \
  tests/strategies/volume_price_efficiency_v1/live/test_ids.py -q
```

Expected: all tests pass.

Commit:

```bash
git add \
  src/xsignal/strategies/volume_price_efficiency_v1/live \
  tests/strategies/volume_price_efficiency_v1/live
git commit -m "feat: add vpe live preset and order ids"
```

---

### Task 2: Live Models And SQLite Store

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/models.py`
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/store.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_store.py`

- [ ] **Step 1: Write failing store tests**

Create `tests/strategies/volume_price_efficiency_v1/live/test_store.py`:

```python
from datetime import datetime, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    AccountSnapshot,
    OrderIntent,
    OrderIntentType,
    PositionState,
    SymbolMetadata,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore


def test_store_initializes_schema_and_persists_intent(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    position_id = store.create_position(symbol="BTCUSDT", state=PositionState.FLAT)
    intent = OrderIntent(
        intent_id="intent-1",
        position_id=position_id,
        symbol="BTCUSDT",
        intent_type=OrderIntentType.ENTRY,
        client_order_id="XV1TEBTC123",
        side="BUY",
        quantity=0.001,
        notional=20.0,
        price=None,
        stop_price=None,
        created_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
    )
    store.record_order_intent(intent)
    loaded = store.get_order_intent("intent-1")
    assert loaded == intent


def test_store_persists_metadata_and_account_snapshot(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    metadata = SymbolMetadata(
        symbol="BTCUSDT",
        status="TRADING",
        min_notional=5.0,
        quantity_step=0.001,
        price_tick=0.1,
        supports_stop_market=True,
        trigger_protect=0.05,
        updated_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
    )
    snapshot = AccountSnapshot(
        mode="testnet",
        account_mode="one_way",
        asset_mode="single_asset_usdt",
        equity=1000.0,
        available_balance=900.0,
        open_notional=40.0,
        open_position_count=2,
        daily_realized_pnl=-3.0,
        captured_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
    )
    store.upsert_symbol_metadata(metadata)
    store.record_account_snapshot(snapshot)
    assert store.get_symbol_metadata("BTCUSDT") == metadata
    assert store.latest_account_snapshot() == snapshot
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/live/test_store.py -q
```

Expected: import errors for `models` and `store`.

- [ ] **Step 3: Implement live models**

Create `src/xsignal/strategies/volume_price_efficiency_v1/live/models.py` with frozen dataclasses and enums:

```python
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
```

- [ ] **Step 4: Implement SQLite store**

Create `src/xsignal/strategies/volume_price_efficiency_v1/live/store.py` with explicit schema, `sqlite3.Row`, and JSON-free primitive columns. Implement the methods used by the tests:

```python
from __future__ import annotations

from dataclasses import astuple
from datetime import datetime
import sqlite3
from pathlib import Path

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
              updated_at text not null
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
        self.connection.commit()

    def create_position(self, *, symbol: str, state: PositionState) -> str:
        position_id = f"{symbol}-{self.connection.execute('select count(*) from positions').fetchone()[0] + 1}"
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
            insert into symbol_metadata values (?, ?, ?, ?, ?, ?, ?, ?)
            on conflict(symbol) do update set
              status = excluded.status,
              min_notional = excluded.min_notional,
              quantity_step = excluded.quantity_step,
              price_tick = excluded.price_tick,
              supports_stop_market = excluded.supports_stop_market,
              trigger_protect = excluded.trigger_protect,
              updated_at = excluded.updated_at
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
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/live/test_store.py -q
```

Expected: all tests pass.

Commit:

```bash
git add \
  src/xsignal/strategies/volume_price_efficiency_v1/live/models.py \
  src/xsignal/strategies/volume_price_efficiency_v1/live/store.py \
  tests/strategies/volume_price_efficiency_v1/live/test_store.py
git commit -m "feat: add vpe live state store"
```

---

### Task 3: Shared Capital Allocator And Risk Gate

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/capital.py`
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/risk.py`
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/models.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_capital.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_risk.py`

- [ ] **Step 1: Write failing capital tests**

Create `tests/strategies/volume_price_efficiency_v1/live/test_capital.py`:

```python
from xsignal.strategies.volume_price_efficiency_v1.live.capital import size_entry_notional
from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig
from xsignal.strategies.volume_price_efficiency_v1.live.models import AccountSnapshot


def _snapshot(equity: float, available: float, open_notional: float = 0.0) -> AccountSnapshot:
    from datetime import datetime, timezone

    return AccountSnapshot(
        mode="testnet",
        account_mode="one_way",
        asset_mode="single_asset_usdt",
        equity=equity,
        available_balance=available,
        open_notional=open_notional,
        open_position_count=0,
        daily_realized_pnl=0.0,
        captured_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
    )


def test_entry_notional_uses_shared_equity_and_cap():
    config = LiveTradingConfig()
    assert size_entry_notional(config, _snapshot(1000.0, 1000.0)) == 20.0
    assert size_entry_notional(config, _snapshot(200.0, 200.0)) == 10.0


def test_entry_notional_respects_available_balance_and_total_cap():
    config = LiveTradingConfig()
    assert size_entry_notional(config, _snapshot(1000.0, 8.0)) == 8.0
    assert size_entry_notional(config, _snapshot(1000.0, 1000.0, open_notional=95.0)) == 5.0
```

- [ ] **Step 2: Write failing risk tests**

Create `tests/strategies/volume_price_efficiency_v1/live/test_risk.py`:

```python
from datetime import datetime, timedelta, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig
from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    AccountSnapshot,
    OrderIntent,
    OrderIntentType,
    PositionState,
    SymbolMetadata,
)
from xsignal.strategies.volume_price_efficiency_v1.live.risk import evaluate_intent


NOW = datetime(2026, 5, 9, tzinfo=timezone.utc)


def _metadata(**overrides) -> SymbolMetadata:
    data = dict(
        symbol="BTCUSDT",
        status="TRADING",
        min_notional=5.0,
        quantity_step=0.001,
        price_tick=0.1,
        supports_stop_market=True,
        trigger_protect=0.05,
        updated_at=NOW,
    )
    data.update(overrides)
    return SymbolMetadata(**data)


def _snapshot(**overrides) -> AccountSnapshot:
    data = dict(
        mode="testnet",
        account_mode="one_way",
        asset_mode="single_asset_usdt",
        equity=1000.0,
        available_balance=1000.0,
        open_notional=0.0,
        open_position_count=0,
        daily_realized_pnl=0.0,
        captured_at=NOW,
    )
    data.update(overrides)
    return AccountSnapshot(**data)


def _intent(**overrides) -> OrderIntent:
    data = dict(
        intent_id="intent-1",
        position_id="BTCUSDT-1",
        symbol="BTCUSDT",
        intent_type=OrderIntentType.ENTRY,
        client_order_id="XV1TEBTC123",
        side="BUY",
        quantity=0.001,
        notional=20.0,
        price=None,
        stop_price=None,
        created_at=NOW,
    )
    data.update(overrides)
    return OrderIntent(**data)


def test_risk_accepts_valid_entry():
    result = evaluate_intent(
        config=LiveTradingConfig(),
        intent=_intent(),
        metadata=_metadata(),
        account=_snapshot(),
        position_state=PositionState.FLAT,
        now=NOW,
    )
    assert result.accepted
    assert result.reason == "accepted"


def test_risk_rejects_wrong_account_mode():
    result = evaluate_intent(
        config=LiveTradingConfig(),
        intent=_intent(),
        metadata=_metadata(),
        account=_snapshot(account_mode="hedge"),
        position_state=PositionState.FLAT,
        now=NOW,
    )
    assert not result.accepted
    assert result.reason == "account_mode_mismatch"


def test_risk_rejects_stale_metadata_and_low_notional():
    result = evaluate_intent(
        config=LiveTradingConfig(),
        intent=_intent(notional=2.0),
        metadata=_metadata(updated_at=NOW - timedelta(hours=2)),
        account=_snapshot(),
        position_state=PositionState.FLAT,
        now=NOW,
    )
    assert not result.accepted
    assert result.reason == "metadata_stale"
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest \
  tests/strategies/volume_price_efficiency_v1/live/test_capital.py \
  tests/strategies/volume_price_efficiency_v1/live/test_risk.py -q
```

Expected: import errors for `capital` and `risk`.

- [ ] **Step 4: Implement capital allocator**

Create `src/xsignal/strategies/volume_price_efficiency_v1/live/capital.py`:

```python
from __future__ import annotations

from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig
from xsignal.strategies.volume_price_efficiency_v1.live.models import AccountSnapshot


def size_entry_notional(config: LiveTradingConfig, account: AccountSnapshot) -> float:
    desired = account.equity * config.base_position_fraction
    capped = min(desired, config.per_symbol_notional_cap)
    remaining_total = max(config.total_open_notional_cap - account.open_notional, 0.0)
    return max(min(capped, account.available_balance, remaining_total), 0.0)
```

- [ ] **Step 5: Add `RiskResult` and risk gate**

Append this dataclass to `models.py`:

```python
@dataclass(frozen=True)
class RiskResult:
    accepted: bool
    reason: str
```

Create `src/xsignal/strategies/volume_price_efficiency_v1/live/risk.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta

from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig
from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    AccountSnapshot,
    OrderIntent,
    PositionState,
    RiskResult,
    SymbolMetadata,
)


def evaluate_intent(
    *,
    config: LiveTradingConfig,
    intent: OrderIntent,
    metadata: SymbolMetadata,
    account: AccountSnapshot,
    position_state: PositionState,
    now: datetime,
) -> RiskResult:
    if account.mode != config.mode:
        return RiskResult(False, "mode_mismatch")
    if account.account_mode != config.account_mode:
        return RiskResult(False, "account_mode_mismatch")
    if account.asset_mode != config.asset_mode:
        return RiskResult(False, "asset_mode_mismatch")
    if metadata.updated_at < now - timedelta(minutes=30):
        return RiskResult(False, "metadata_stale")
    if metadata.status != "TRADING":
        return RiskResult(False, "symbol_not_trading")
    if not metadata.supports_stop_market:
        return RiskResult(False, "stop_market_not_supported")
    if intent.notional < metadata.min_notional:
        return RiskResult(False, "below_min_notional")
    if intent.notional > config.per_symbol_notional_cap:
        return RiskResult(False, "per_symbol_cap_exceeded")
    if account.open_notional + intent.notional > config.total_open_notional_cap:
        return RiskResult(False, "total_cap_exceeded")
    if account.open_position_count >= config.max_open_positions and position_state == PositionState.FLAT:
        return RiskResult(False, "max_open_positions_exceeded")
    if account.daily_realized_pnl <= -config.max_daily_realized_loss:
        return RiskResult(False, "daily_loss_limit_exceeded")
    return RiskResult(True, "accepted")
```

- [ ] **Step 6: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest \
  tests/strategies/volume_price_efficiency_v1/live/test_capital.py \
  tests/strategies/volume_price_efficiency_v1/live/test_risk.py -q
```

Expected: all tests pass.

Commit:

```bash
git add \
  src/xsignal/strategies/volume_price_efficiency_v1/live/capital.py \
  src/xsignal/strategies/volume_price_efficiency_v1/live/risk.py \
  src/xsignal/strategies/volume_price_efficiency_v1/live/models.py \
  tests/strategies/volume_price_efficiency_v1/live/test_capital.py \
  tests/strategies/volume_price_efficiency_v1/live/test_risk.py
git commit -m "feat: add vpe live capital and risk checks"
```

---

### Task 4: Fake Broker And Position State Machine

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/broker.py`
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/state_machine.py`
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/models.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_state_machine.py`

- [ ] **Step 1: Write failing state machine tests**

Create `tests/strategies/volume_price_efficiency_v1/live/test_state_machine.py`:

```python
from datetime import datetime, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.broker import FakeBroker
from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig
from xsignal.strategies.volume_price_efficiency_v1.live.state_machine import (
    LiveSymbolState,
    on_signal,
    on_stop_fill,
    update_trailing_stop,
)


NOW = datetime(2026, 5, 9, tzinfo=timezone.utc)


def test_signal_enters_long_and_places_protective_stop():
    broker = FakeBroker()
    state = LiveSymbolState.flat("BTCUSDT")
    next_state = on_signal(
        state=state,
        broker=broker,
        config=LiveTradingConfig(),
        entry_price=100.0,
        atr=5.0,
        quantity=0.2,
        now=NOW,
    )
    assert next_state.position_state == "OPEN"
    assert next_state.stop_price == 85.0
    assert broker.orders[-1].order_type == "STOP_MARKET"
    assert broker.orders[-1].side == "SELL"


def test_open_symbol_ignores_second_signal():
    broker = FakeBroker()
    state = LiveSymbolState.open(
        symbol="BTCUSDT",
        quantity=0.2,
        entry_price=100.0,
        highest_high=105.0,
        stop_price=90.0,
        add_count=0,
    )
    next_state = on_signal(
        state=state,
        broker=broker,
        config=LiveTradingConfig(),
        entry_price=106.0,
        atr=4.0,
        quantity=0.2,
        now=NOW,
    )
    assert next_state is state
    assert broker.orders == []


def test_trailing_stop_moves_up_protective_first():
    broker = FakeBroker()
    state = LiveSymbolState.open(
        symbol="BTCUSDT",
        quantity=0.2,
        entry_price=100.0,
        highest_high=110.0,
        stop_price=90.0,
        add_count=0,
    )
    next_state = update_trailing_stop(
        state=state,
        broker=broker,
        config=LiveTradingConfig(),
        bar_high=120.0,
        atr=5.0,
        now=NOW,
    )
    assert next_state.stop_price == 105.0
    assert [order.order_type for order in broker.orders] == ["STOP_MARKET"]
    assert broker.cancelled_order_ids == ["active-stop"]


def test_stop_fill_closes_and_unlocks_symbol():
    state = LiveSymbolState.open(
        symbol="BTCUSDT",
        quantity=0.2,
        entry_price=100.0,
        highest_high=110.0,
        stop_price=90.0,
        add_count=0,
    )
    next_state = on_stop_fill(state=state, fill_price=90.0, now=NOW)
    assert next_state.position_state == "CLOSED"
    assert next_state.quantity == 0.0
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/live/test_state_machine.py -q
```

Expected: import errors for `broker` and `state_machine`.

- [ ] **Step 3: Implement fake broker**

Create `src/xsignal/strategies/volume_price_efficiency_v1/live/broker.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BrokerOrder:
    symbol: str
    side: str
    order_type: str
    quantity: float | None
    stop_price: float | None
    close_position: bool


class FakeBroker:
    def __init__(self) -> None:
        self.orders: list[BrokerOrder] = []
        self.cancelled_order_ids: list[str] = []

    def market_buy(self, *, symbol: str, quantity: float) -> BrokerOrder:
        order = BrokerOrder(symbol, "BUY", "MARKET", quantity, None, False)
        self.orders.append(order)
        return order

    def place_stop_market_close(self, *, symbol: str, stop_price: float) -> BrokerOrder:
        order = BrokerOrder(symbol, "SELL", "STOP_MARKET", None, stop_price, True)
        self.orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> None:
        self.cancelled_order_ids.append(order_id)
```

- [ ] **Step 4: Implement state machine**

Create `src/xsignal/strategies/volume_price_efficiency_v1/live/state_machine.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

from xsignal.strategies.volume_price_efficiency_v1.live.broker import FakeBroker
from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig


@dataclass(frozen=True)
class LiveSymbolState:
    symbol: str
    position_state: str
    quantity: float
    entry_price: float | None
    highest_high: float | None
    stop_price: float | None
    add_count: int
    active_stop_order_id: str | None = None
    closed_at: datetime | None = None

    @classmethod
    def flat(cls, symbol: str) -> "LiveSymbolState":
        return cls(symbol, "FLAT", 0.0, None, None, None, 0)

    @classmethod
    def open(
        cls,
        *,
        symbol: str,
        quantity: float,
        entry_price: float,
        highest_high: float,
        stop_price: float,
        add_count: int,
    ) -> "LiveSymbolState":
        return cls(
            symbol=symbol,
            position_state="OPEN",
            quantity=quantity,
            entry_price=entry_price,
            highest_high=highest_high,
            stop_price=stop_price,
            add_count=add_count,
            active_stop_order_id="active-stop",
        )


def on_signal(
    *,
    state: LiveSymbolState,
    broker: FakeBroker,
    config: LiveTradingConfig,
    entry_price: float,
    atr: float,
    quantity: float,
    now: datetime,
) -> LiveSymbolState:
    if state.position_state != "FLAT":
        return state
    broker.market_buy(symbol=state.symbol, quantity=quantity)
    stop_price = entry_price - config.atr_multiplier * atr
    broker.place_stop_market_close(symbol=state.symbol, stop_price=stop_price)
    return LiveSymbolState.open(
        symbol=state.symbol,
        quantity=quantity,
        entry_price=entry_price,
        highest_high=entry_price,
        stop_price=stop_price,
        add_count=0,
    )


def update_trailing_stop(
    *,
    state: LiveSymbolState,
    broker: FakeBroker,
    config: LiveTradingConfig,
    bar_high: float,
    atr: float,
    now: datetime,
) -> LiveSymbolState:
    if state.position_state != "OPEN" or state.highest_high is None or state.stop_price is None:
        return state
    highest_high = max(state.highest_high, bar_high)
    next_stop = highest_high - config.atr_multiplier * atr
    if next_stop <= state.stop_price:
        return replace(state, highest_high=highest_high)
    broker.place_stop_market_close(symbol=state.symbol, stop_price=next_stop)
    if state.active_stop_order_id is not None:
        broker.cancel_order(state.active_stop_order_id)
    return replace(
        state,
        highest_high=highest_high,
        stop_price=next_stop,
        active_stop_order_id="active-stop",
    )


def on_stop_fill(*, state: LiveSymbolState, fill_price: float, now: datetime) -> LiveSymbolState:
    return replace(
        state,
        position_state="CLOSED",
        quantity=0.0,
        stop_price=None,
        active_stop_order_id=None,
        closed_at=now,
    )
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/live/test_state_machine.py -q
```

Expected: all tests pass.

Commit:

```bash
git add \
  src/xsignal/strategies/volume_price_efficiency_v1/live/broker.py \
  src/xsignal/strategies/volume_price_efficiency_v1/live/state_machine.py \
  tests/strategies/volume_price_efficiency_v1/live/test_state_machine.py
git commit -m "feat: add vpe live fake broker state machine"
```

---

### Task 5: Closed-Bar Signal Engine With Fixed Regime

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/signal_engine.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_signal_engine.py`

- [ ] **Step 1: Write failing signal engine tests**

Create `tests/strategies/volume_price_efficiency_v1/live/test_signal_engine.py`:

```python
from datetime import datetime, timezone

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.live.signal_engine import (
    build_market_regime_mask,
    closed_bar_view,
)


def _arrays() -> OhlcvArrays:
    times = np.array(
        [
            datetime(2026, 5, 7, tzinfo=timezone.utc),
            datetime(2026, 5, 8, tzinfo=timezone.utc),
            datetime(2026, 5, 9, tzinfo=timezone.utc),
        ],
        dtype=object,
    )
    values = np.array([[100.0], [110.0], [120.0]])
    return OhlcvArrays(
        symbols=("BTCUSDT",),
        open_times=times,
        open=values.copy(),
        high=values.copy() + 1,
        low=values.copy() - 1,
        close=values.copy(),
        quote_volume=np.full((3, 1), 1000.0),
        quality=np.ones((3, 1), dtype=bool),
    )


def test_closed_bar_view_excludes_forming_bar():
    arrays = _arrays()
    view = closed_bar_view(arrays, closed_open_time=arrays.open_times[1])
    assert view.open.shape == (2, 1)
    assert view.open_times[-1] == arrays.open_times[1]


def test_market_regime_mask_uses_closed_history_only():
    close = np.array(
        [
            [100.0, 100.0],
            [90.0, 90.0],
            [89.0, 95.0],
            [91.0, 99.0],
        ]
    )
    mask = build_market_regime_mask(close, lookback_bars=2, min_return=-0.10)
    assert not mask[2, 0]
    assert mask[3, 0]
    assert mask[3, 1]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/live/test_signal_engine.py -q
```

Expected: import error for `signal_engine`.

- [ ] **Step 3: Implement signal engine helpers**

Create `src/xsignal/strategies/volume_price_efficiency_v1/live/signal_engine.py`:

```python
from __future__ import annotations

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.features import compute_features
from xsignal.strategies.volume_price_efficiency_v1.live.config import (
    LiveTradingConfig,
    build_vpe_live_strategy_config,
)


def closed_bar_view(arrays: OhlcvArrays, *, closed_open_time: object) -> OhlcvArrays:
    matches = np.flatnonzero(arrays.open_times <= closed_open_time)
    if matches.size == 0:
        raise ValueError("closed_open_time is before available history")
    end = int(matches[-1]) + 1
    return OhlcvArrays(
        symbols=arrays.symbols,
        open_times=arrays.open_times[:end],
        open=arrays.open[:end],
        high=arrays.high[:end],
        low=arrays.low[:end],
        close=arrays.close[:end],
        quote_volume=arrays.quote_volume[:end],
        quality=arrays.quality[:end],
    )


def build_market_regime_mask(
    close: np.ndarray,
    *,
    lookback_bars: int,
    min_return: float,
) -> np.ndarray:
    output = np.zeros(close.shape, dtype=bool)
    for index in range(lookback_bars, close.shape[0]):
        start = close[index - lookback_bars]
        end = close[index]
        valid = np.isfinite(start) & np.isfinite(end) & (start > 0.0)
        returns = np.divide(end, start, out=np.full(close.shape[1], np.nan), where=valid) - 1.0
        finite = returns[np.isfinite(returns)]
        if finite.size and float(np.mean(finite)) >= min_return:
            output[index] = True
    return output


def build_live_signal_mask(arrays: OhlcvArrays, live_config: LiveTradingConfig) -> np.ndarray:
    strategy_config = build_vpe_live_strategy_config()
    features = compute_features(arrays, strategy_config)
    regime = build_market_regime_mask(
        arrays.close,
        lookback_bars=live_config.market_regime_lookback_bars,
        min_return=live_config.market_regime_min_return,
    )
    return features.signal & regime
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/live/test_signal_engine.py -q
```

Expected: all tests pass.

Commit:

```bash
git add \
  src/xsignal/strategies/volume_price_efficiency_v1/live/signal_engine.py \
  tests/strategies/volume_price_efficiency_v1/live/test_signal_engine.py
git commit -m "feat: add vpe live closed bar signal engine"
```

---

### Task 6: Offline Replay And CLI

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/replay.py`
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/cli.py`
- Modify: `pyproject.toml`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_replay.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_cli.py`

- [ ] **Step 1: Write failing replay test**

Create `tests/strategies/volume_price_efficiency_v1/live/test_replay.py`:

```python
from xsignal.strategies.volume_price_efficiency_v1.live.replay import ReplaySummary


def test_replay_summary_counts_are_plain_dataclass():
    summary = ReplaySummary(processed_bars=10, accepted_signals=2, submitted_orders=4)
    assert summary.processed_bars == 10
    assert summary.accepted_signals == 2
    assert summary.submitted_orders == 4
```

- [ ] **Step 2: Write failing CLI registration test**

Create `tests/strategies/volume_price_efficiency_v1/live/test_cli.py`:

```python
import pytest

from xsignal.strategies.volume_price_efficiency_v1.live.cli import build_parser


def test_cli_has_replay_status_and_reconcile_commands():
    parser = build_parser()
    subcommands = parser._subparsers._group_actions[0].choices
    assert {"replay", "status", "reconcile"} <= set(subcommands)


def test_status_requires_database_path():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["status"])
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
.venv/bin/python -m pytest \
  tests/strategies/volume_price_efficiency_v1/live/test_replay.py \
  tests/strategies/volume_price_efficiency_v1/live/test_cli.py -q
```

Expected: import errors for `replay` and `cli`.

- [ ] **Step 4: Implement replay summary and CLI parser**

Create `src/xsignal/strategies/volume_price_efficiency_v1/live/replay.py`:

```python
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReplaySummary:
    processed_bars: int
    accepted_signals: int
    submitted_orders: int
```

Create `src/xsignal/strategies/volume_price_efficiency_v1/live/cli.py`:

```python
from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xsignal-vpe-live")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay")
    replay.add_argument("--root", type=Path, default=Path("data"))
    replay.add_argument("--db", type=Path, required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--db", type=Path, required=True)

    reconcile = subparsers.add_parser("reconcile")
    reconcile.add_argument("--db", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    return 0
```

Modify `pyproject.toml`:

```toml
xsignal-vpe-live = "xsignal.strategies.volume_price_efficiency_v1.live.cli:main"
```

- [ ] **Step 5: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest \
  tests/strategies/volume_price_efficiency_v1/live/test_replay.py \
  tests/strategies/volume_price_efficiency_v1/live/test_cli.py -q
```

Expected: all tests pass.

Commit:

```bash
git add \
  src/xsignal/strategies/volume_price_efficiency_v1/live/replay.py \
  src/xsignal/strategies/volume_price_efficiency_v1/live/cli.py \
  tests/strategies/volume_price_efficiency_v1/live/test_replay.py \
  tests/strategies/volume_price_efficiency_v1/live/test_cli.py \
  pyproject.toml
git commit -m "feat: add vpe live offline cli"
```

---

### Task 7: Full Verification And Handoff To Binance Testnet Plan

**Files:**
- Modify: `README.md`
- No code files unless verification exposes a defect.

- [ ] **Step 1: Add README section for live core**

Append this section to `README.md`:

````markdown
## Volume Price Efficiency Live Core

The first live-trading implementation phase is an offline core. It builds the
same state machine, shared-equity sizing, risk gate, SQLite state store, and
fake broker that will later drive Binance testnet/live trading.

Run the offline CLI:

```bash
xsignal-vpe-live replay --root data --db data/live/vpe-live.sqlite
xsignal-vpe-live status --db data/live/vpe-live.sqlite
xsignal-vpe-live reconcile --db data/live/vpe-live.sqlite
```

Production order submission is not part of the offline core. The Binance
testnet adapter is the next implementation plan after this core passes tests.
````

- [ ] **Step 2: Run live test slice**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/live -q
```

Expected: all live tests pass.

- [ ] **Step 3: Run full project tests**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 4: Run lint**

Run:

```bash
.venv/bin/ruff check .
```

Expected: no lint errors.

- [ ] **Step 5: Commit README and any verification fixes**

Run:

```bash
git add README.md
git commit -m "docs: document vpe live core workflow"
```

If verification required code fixes, include those exact files in the same final commit and mention the failing command that drove the fix in the commit body.

---

## Plan Self-Review Checklist

- Spec coverage in this plan:
  - Fixed live preset: Task 1.
  - No CLI production order path: Task 4 uses a fake broker only; real Binance is outside this plan.
  - Shared capital and compounding sizing: Task 3.
  - Compact idempotent client order ids: Task 1.
  - SQLite persistence foundation: Task 2.
  - Risk gate foundation: Task 3.
  - Per-symbol state machine and protective-first stop replacement: Task 4.
  - Closed-bar signal wrapper and fixed market regime: Task 5.
  - Offline operator CLI: Task 6.
  - Project verification and README handoff: Task 7.
- Scope gap intentionally deferred:
  - Real Binance USD-M adapter, testnet order smoke, user data stream handling, production `systemd`, and live guarded trading need a second plan after this offline core is merged.
- Blank-item scan:
  - No empty fill-in markers or vague catch-all steps are used.
- Type consistency:
  - `LiveTradingConfig`, `OrderIntent`, `OrderIntentType`, `PositionState`, `SymbolMetadata`, `AccountSnapshot`, `RiskResult`, `FakeBroker`, and `LiveSymbolState` are introduced before use.
