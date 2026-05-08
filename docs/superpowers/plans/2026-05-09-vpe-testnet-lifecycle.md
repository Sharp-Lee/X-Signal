# VPE Testnet Lifecycle Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a guarded Binance USD-M Futures testnet lifecycle command that opens a tiny testnet long, places a protective stop, verifies position/order state, cancels protection, closes the position, and confirms the symbol returns to flat.

**Architecture:** Keep all exchange I/O inside `BinanceUsdFuturesTestnetBroker`. The CLI command orchestrates only testnet lifecycle smoke execution and requires explicit `--i-understand-testnet-order` acknowledgement. It never supports production mode.

**Tech Stack:** Python 3.12 stdlib, existing Binance REST adapter, existing live CLI, `pytest`, `ruff`.

---

### Task 1: Broker Query And Close Methods

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/binance_adapter.py`
- Modify: `tests/strategies/volume_price_efficiency_v1/live/test_binance_adapter.py`

- [ ] Add failing tests for `get_position_risk`, `get_open_order`, `get_open_orders`, and `market_sell_reduce_only`.
- [ ] Implement methods using official endpoints:
  - `GET /fapi/v3/positionRisk`
  - `GET /fapi/v1/openOrder`
  - `GET /fapi/v1/openOrders`
  - `POST /fapi/v1/order` with `SELL`, `MARKET`, `reduceOnly=true`
- [ ] Run adapter tests and commit.

### Task 2: Lifecycle Orchestrator

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/testnet_lifecycle.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_testnet_lifecycle.py`

- [ ] Add failing tests for success path using a fake broker.
- [ ] Add failing tests that cleanup attempts to cancel stop and close position when a later step fails.
- [ ] Implement `run_testnet_lifecycle()` with:
  - set isolated margin and 1x leverage
  - market buy
  - place `STOP_MARKET closePosition=true`
  - verify open position amount is positive
  - verify stop order is open
  - cancel stop
  - market sell reduce-only
  - verify position amount returns to zero
- [ ] Run lifecycle tests and commit.

### Task 3: Guarded CLI And Real Testnet Smoke

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/cli.py`
- Modify: `README.md`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_binance_cli.py`

- [ ] Add `testnet-lifecycle` parser tests.
- [ ] Require `--i-understand-testnet-order`.
- [ ] Load `.secrets/binance-testnet.env` when present, without printing secrets.
- [ ] Run unit tests, full tests, ruff.
- [ ] Run real command with the local testnet key:
  - `xsignal-vpe-live testnet-lifecycle --symbol BTCUSDT --quantity 0.001 --stop-offset-pct 0.05 --i-understand-testnet-order`
- [ ] Confirm command exits zero and reports final position amount `0`.
- [ ] Commit and merge to `main`.

## Self-Review

- This plan uses real testnet matching-engine orders only after explicit CLI acknowledgement.
- It never enables production trading.
- Secrets remain in `.secrets/`, ignored by git and not printed.
