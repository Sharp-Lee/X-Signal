# Volume Price Efficiency Trailing Stop Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Simulate `volume_price_efficiency_v1` on the reserved holdout window with per-symbol locking and a 2 ATR dynamic trailing stop.

**Architecture:** Keep the simulator inside the strategy module. Reuse the existing OHLCV loader and signal detector, but add a new stateful trade simulator, trade artifact writer, and CLI entrypoint dedicated to the holdout test. The simulator processes symbols independently so one symbol's open trade blocks only that symbol, not the whole universe.

**Tech Stack:** Python 3.12, NumPy, PyArrow, Parquet, pydantic, pytest, ruff.

---

### Task 1: Trailing Stop Simulator Core

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/trailing.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_trailing.py`

- [ ] Write tests for immediate stop breach, ATR update moving the stop upward, signal lockout while in position, and re-entry after exit.
- [ ] Implement a per-symbol long-only state machine that uses the signal from `features.signal`, enters on next open, and exits at `highest_high - 2 * current_atr`.
- [ ] Run `pytest tests/strategies/volume_price_efficiency_v1/test_trailing.py -q`.
- [ ] Commit with `feat: add volume price efficiency trailing simulator core`.

### Task 2: Trailing Stop Artifacts

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/trailing_artifacts.py`
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/paths.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_trailing_artifacts.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_paths.py`

- [ ] Write tests for trade table columns, summary fields, and run directory validation.
- [ ] Add a `trailing_runs` path and a `trailing_run_dir(run_id)` helper.
- [ ] Add `write_trailing_run_artifacts(...)` with `manifest.json`, `summary.json`, `trades.parquet`, `equity_curve.parquet`, and `daily_positions.parquet`.
- [ ] Run the new artifact and path tests.
- [ ] Commit with `feat: write volume price efficiency trailing artifacts`.

### Task 3: Holdout Test CLI

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/trailing_cli.py`
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/cli.py`
- Modify: `README.md`
- Test: `tests/strategies/volume_price_efficiency_v1/test_trailing_cli.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_cli.py`

- [ ] Write tests that the trailing CLI loads the reserved holdout window, uses the existing signal detector, and writes trailing artifacts.
- [ ] Add a `trail` subcommand with `--holdout-days`, `--run-id`, `--atr-multiplier` fixed at `2.0`, and `--offline`.
- [ ] Update README with the new holdout test command and note that it is separate from the scan phase.
- [ ] Run strategy tests and ruff.
- [ ] Commit with `feat: add volume price efficiency trailing stop cli`.

### Task 4: Real Holdout Smoke

**Files:**
- No code files expected.

- [ ] Run the holdout trailing test on the reserved window.
- [ ] Inspect trade count, symbol-level lockout, stop fills, and summary metrics.
- [ ] Run full `pytest -q`, `ruff check .`, and `git diff --check`.
- [ ] Push the branch, create a PR, and merge once clean.
