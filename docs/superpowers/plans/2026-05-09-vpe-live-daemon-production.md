# VPE Live Daemon Production Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy the first production-capable VPE live trading daemon: automated daily signal scan, shared-capital entry, exchange-side protective stop, one ATR pyramid add, trailing-stop maintenance, restart reconciliation, and guarded live-mode wiring.

**Architecture:** Keep Binance I/O inside broker/data adapters, keep strategy math pure, and drive live trading through a one-shot `run-cycle` command suitable for systemd timers. Testnet and live use the same code path, but live mode requires production credentials plus an explicit acknowledgement guard before any order submission.

**Tech Stack:** Python 3.12, SQLite, Binance USD-M Futures REST, pytest, ruff, systemd timer on `alpha`.

---

## Files

- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/market_data.py`: fetch USD-M perpetual symbols and closed daily klines, then build `OhlcvArrays`.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/position_store.py`: focused helpers for live strategy position fields stored in SQLite.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/execution.py`: persist-and-submit entry, stop, stop replacement, and pyramid add intents.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/runner.py`: one-shot automated cycle for reconcile, scan, entries, stop maintenance, and adds.
- Modify `src/xsignal/strategies/volume_price_efficiency_v1/live/binance_adapter.py`: add production base URL, generic broker class, account snapshot, price, and all-symbol metadata helpers.
- Modify `src/xsignal/strategies/volume_price_efficiency_v1/live/cli.py`: add `run-cycle`, `live-smoke`, and guarded live/prod config parsing.
- Modify `src/xsignal/strategies/volume_price_efficiency_v1/live/store.py`: migrate position lifecycle fields and add audit helpers.
- Modify `README.md`: document alpha deployment, testnet automatic runner, live guard, and operator commands.
- Add tests under `tests/strategies/volume_price_efficiency_v1/live/`.

## Task 1: Broker And Market Data Foundation

- [ ] Write failing tests for:
  - live production base URL is `https://fapi.binance.com`
  - broker can be constructed for `testnet` and `live`
  - `fetch_closed_daily_klines` excludes the currently forming daily bar
  - daily kline payloads convert into quality `OhlcvArrays`
- [ ] Run targeted tests and confirm RED.
- [ ] Implement `market_data.py` and generic broker environment selection.
- [ ] Run targeted tests and confirm GREEN.

## Task 2: Live Position Persistence

- [ ] Write failing tests for SQLite migration of:
  - `entry_price`
  - `quantity`
  - `highest_high`
  - `stop_price`
  - `atr_at_entry`
  - `next_add_trigger`
  - `add_count`
  - `active_stop_client_order_id`
  - `last_decision_open_time`
- [ ] Run targeted tests and confirm RED.
- [ ] Implement migration and helper methods in `position_store.py` backed by `LiveStore`.
- [ ] Run targeted tests and confirm GREEN.

## Task 3: Persisted Execution Helpers

- [ ] Write failing tests proving every external order action persists intent before broker call:
  - initial market entry
  - initial protective stop
  - stop replacement
  - pyramid add
- [ ] Run targeted tests and confirm RED.
- [ ] Implement `execution.py` with idempotent client ids, `SymbolRules` normalization, and safe stop replacement fallback.
- [ ] Run targeted tests and confirm GREEN.

## Task 4: Automated VPE Cycle

- [ ] Write failing tests for `run_live_cycle`:
  - reconciles first
  - blocks live mode without acknowledgement
  - scans symbols deterministically
  - ignores symbols already open locally
  - opens on latest closed-bar signal
  - maintains trailing stop upward only
  - arms and submits one pyramid add only when execution price confirms trigger
- [ ] Run targeted tests and confirm RED.
- [ ] Implement `runner.py` one-shot cycle.
- [ ] Run targeted tests and confirm GREEN.

## Task 5: CLI And Deployment Surfaces

- [ ] Write failing tests for:
  - `xsignal-vpe-live run-cycle --mode testnet --db ...`
  - `xsignal-vpe-live run-cycle --mode live` refusing without `--i-understand-live-order`
  - `xsignal-vpe-live live-smoke` read-only account/metadata check
- [ ] Run targeted tests and confirm RED.
- [ ] Implement CLI commands without printing secrets.
- [ ] Update README with alpha/systemd commands.
- [ ] Run targeted tests and confirm GREEN.

## Task 6: Verification And Alpha Deployment

- [ ] Run full local verification:

```bash
/Users/wukong/mylife/X-Signal/.venv/bin/python -m pytest -q
/Users/wukong/mylife/X-Signal/.venv/bin/ruff check .
git diff --check
```

- [ ] Run remote testnet `run-cycle` in `--max-symbols 3` smoke mode.
- [ ] Install `xsignal-vpe-testnet-auto-cycle.timer` on alpha.
- [ ] Install but do not start `xsignal-vpe-live-auto-cycle.timer` until `/etc/xsignal/binance-live.env` and `/etc/xsignal/enable-live-trading` exist.
- [ ] Verify:
  - `https://alpha.tradingviewbots.xyz/healthz` returns `ok`
  - testnet timer is active
  - read-only reconcile is `CLEAN`
  - live timer is inactive/guarded unless explicitly enabled
