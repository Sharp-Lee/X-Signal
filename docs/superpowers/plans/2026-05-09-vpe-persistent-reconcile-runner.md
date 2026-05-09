# VPE Persistent Reconcile Runner Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Binance USD-M testnet lifecycle restart-safe by persisting every order intent before submission and adding a testnet reconciliation runner that can classify, repair, or lock unsafe local/Binance mismatches.

**Architecture:** Extend the SQLite store as the local audit source for order intent status and position state. Add a reconciliation module that queries Binance by deterministic client ids plus current position/open stop state, then applies conservative repair actions only when explicitly allowed. Wire this into guarded testnet CLI commands while keeping production trading disabled.

**Tech Stack:** Python 3.12, SQLite, pytest, Binance USD-M Futures REST adapter, existing VPE live models.

---

### Task 1: Persist Order Intent Status

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/models.py`
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/store.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_store.py`

- [ ] **Step 1: Write failing tests**

Add tests that create an `OrderIntent`, verify the default status is `PENDING_SUBMIT`, update it to `EXCHANGE_CONFIRMED`, list unresolved intents, and update a position state to `ERROR_LOCKED`.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
/Users/wukong/mylife/X-Signal/.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/live/test_store.py -q
```

Expected: fail because status columns and helper methods do not exist.

- [ ] **Step 3: Implement store/model support**

Add:

```python
class OrderIntentStatus(StrEnum):
    PENDING_SUBMIT = "PENDING_SUBMIT"
    SUBMITTED = "SUBMITTED"
    EXCHANGE_CONFIRMED = "EXCHANGE_CONFIRMED"
    RESOLVED = "RESOLVED"
    ERROR = "ERROR"
```

Add default status/audit fields to `OrderIntent`, migrate missing columns in `LiveStore.initialize()`, and implement:

```python
get_order_intent_by_client_order_id(...)
list_unresolved_order_intents(...)
update_order_intent_status(...)
get_position_state(...)
update_position_state(...)
list_positions_by_states(...)
```

- [ ] **Step 4: Run tests to verify GREEN**

Run the same store test file and keep existing tests passing.

### Task 2: Add Reconciliation Runner

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/reconcile.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_reconcile.py`

- [ ] **Step 1: Write failing tests**

Cover these restart cases:

- local `OPEN` + Binance long + active strategy stop => `PROTECTED`
- local `OPEN` + Binance long + no active strategy stop + read-only => `ERROR_LOCKED`
- local `OPEN` + Binance long + no active strategy stop + repair => persist close intent before reduce-only close, then mark closed after flat
- local `FLAT` + Binance long => `ERROR_LOCKED`, no auto-close
- pending entry intent after timeout => query regular order by client id and reconcile to protected if position and stop exist

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
/Users/wukong/mylife/X-Signal/.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/live/test_reconcile.py -q
```

Expected: fail because `live.reconcile` is missing.

- [ ] **Step 3: Implement conservative reconciliation**

Create `ReconcileResult` and `run_reconciliation_pass(...)`. The runner must:

- query unresolved regular/algo intents by client id
- read local non-closed positions
- query Binance position risk and open algo orders per symbol
- treat only `XV1` client ids as strategy-owned orders
- mark protected matches as `PROTECTED`
- mark unknown Binance positions as `ERROR_LOCKED`
- in repair mode, close an owned unprotected long with a persisted `MANUAL_RECONCILE` intent before submitting reduce-only market sell

- [ ] **Step 4: Run tests to verify GREEN**

Run the new reconcile tests and the store tests.

### Task 3: Persist Testnet Lifecycle Submissions

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/testnet_lifecycle.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_testnet_lifecycle.py`

- [ ] **Step 1: Write failing tests**

Add a store-backed lifecycle test proving the entry intent is recorded before `market_buy`, stop intent before `place_stop_market_close`, and close intent before `market_sell_reduce_only`.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
/Users/wukong/mylife/X-Signal/.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/live/test_testnet_lifecycle.py -q
```

Expected: fail because lifecycle has no store hook.

- [ ] **Step 3: Implement persistence hook**

Add optional `store` and `environment` parameters to `run_testnet_lifecycle`. When provided, create or reuse a position id, record each intent before the corresponding broker POST, and update statuses after confirmed order/position outcomes.

- [ ] **Step 4: Run tests to verify GREEN**

Run lifecycle, store, and reconcile tests.

### Task 4: Add Guarded Testnet Reconcile CLI

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/cli.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_binance_cli.py`

- [ ] **Step 1: Write failing tests**

Add parser and command tests for:

```bash
xsignal-vpe-live testnet-reconcile --db data/live/vpe-testnet.sqlite --symbol BTCUSDT
xsignal-vpe-live testnet-reconcile --db data/live/vpe-testnet.sqlite --symbol BTCUSDT --repair --i-understand-testnet-order
```

The command must refuse repair without the acknowledgement flag and must not print secrets.

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
/Users/wukong/mylife/X-Signal/.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/live/test_binance_cli.py -q
```

- [ ] **Step 3: Implement CLI wiring**

Open and initialize `LiveStore`, build the testnet broker, call `run_reconciliation_pass`, print JSON summary, and return non-zero for `ERROR_LOCKED`.

- [ ] **Step 4: Run tests to verify GREEN**

Run CLI, reconcile, lifecycle, and store tests.

### Task 5: Document Operator Flow

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update docs**

Document that `--db` enables persistent lifecycle audit, `testnet-reconcile` is the restart recovery command, read-only mode only reports, and repair mode can submit testnet close orders.

- [ ] **Step 2: Verify docs formatting**

Run:

```bash
git diff --check
```

### Task 6: Final Verification

**Files:**
- No code files unless fixes are required.

- [ ] **Step 1: Run targeted tests**

```bash
/Users/wukong/mylife/X-Signal/.venv/bin/python -m pytest \
  tests/strategies/volume_price_efficiency_v1/live/test_store.py \
  tests/strategies/volume_price_efficiency_v1/live/test_reconcile.py \
  tests/strategies/volume_price_efficiency_v1/live/test_testnet_lifecycle.py \
  tests/strategies/volume_price_efficiency_v1/live/test_binance_cli.py -q
```

- [ ] **Step 2: Run full suite and lint**

```bash
/Users/wukong/mylife/X-Signal/.venv/bin/python -m pytest -q
/Users/wukong/mylife/X-Signal/.venv/bin/ruff check .
git diff --check
```

- [ ] **Step 3: Run real testnet lifecycle**

```bash
/Users/wukong/mylife/X-Signal/.venv/bin/python -c 'from xsignal.strategies.volume_price_efficiency_v1.live.cli import main; raise SystemExit(main(["testnet-lifecycle","--symbol","BTCUSDT","--quantity","0.001","--stop-offset-pct","0.05","--db","data/live/vpe-testnet.sqlite","--i-understand-testnet-order"]))'
```

- [ ] **Step 4: Run read-only testnet reconcile**

```bash
/Users/wukong/mylife/X-Signal/.venv/bin/python -c 'from xsignal.strategies.volume_price_efficiency_v1.live.cli import main; raise SystemExit(main(["testnet-reconcile","--db","data/live/vpe-testnet.sqlite","--symbol","BTCUSDT"]))'
```

Expected: no open BTCUSDT position and no active strategy stop after lifecycle cleanup.
