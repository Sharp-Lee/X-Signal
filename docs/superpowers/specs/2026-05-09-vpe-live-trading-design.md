# VPE Live Trading Design

## Goal

Turn the selected `volume_price_efficiency_v1` daily strategy into a fully
automated Binance USD-M Futures trading service.

The live system must:

- monitor all eligible USD-M perpetual symbols
- generate signals from closed daily candles only
- enter long automatically when a signal is confirmed
- keep each symbol locked while it has an open strategy position
- maintain exchange-side protective stop orders
- trail stops with `highest_high - 3 * current_atr`
- pyramid once when price advances by `1 * current_atr`
- size positions from shared account equity so all positions share capital and
  profits/losses compound into later trades
- persist every decision, order, fill, and state transition
- recover safely after restart by reconciling local state with Binance state

This is not a generic live trading platform. It is a strategy-specific execution
service for the current daily VPE strategy.

## Fixed Production Assumptions

Account and market:

- Venue: Binance USD-M Futures.
- Product: perpetual futures only.
- Account mode: One-way Mode.
- Margin mode: Isolated Margin.
- Asset mode: Single-Asset USDT.
- Direction: long-only.
- Initial leverage: `1x`.
- Capital model: all strategy positions share the same USDT equity pool.
- First rollout: testnet, then live with the same code path.

Strategy parameters:

- `timeframe = 1d`
- `fill_policy = raw`
- `signal_mode = seed_efficiency`
- `min_move_unit = 0.7`
- `min_volume_unit = 0.3`
- `min_close_position = 0.7`
- `min_body_ratio = 0.4`
- `seed_efficiency_lookback = 4`
- `seed_min_efficiency_ratio_to_max = 2.0`
- `seed_min_efficiency_ratio_to_mean = 5.0`
- `seed_max_volume_unit = 0.8`
- `seed_bottom_lookback = 60`
- `seed_max_close_position_in_range = 0.6`
- `regime = market_lookback_return >= -0.10`
- `market_lookback_bars = 30`
- `atr_window = 14`
- `atr_multiplier = 3.0`
- `pyramid_add_step_atr = 1.0`
- `pyramid_max_adds = 1`

The live implementation must make these values explicit in a named preset. It
must not silently inherit research defaults that differ from the final holdout
configuration.

## Non-Goals

The first live version will not include:

- short selling
- cross margin
- hedge mode
- multi-asset margin
- discretionary manual entries
- generic strategy plugins
- automatic parameter search
- Web3 alpha feeds as direct trade triggers
- CLI-based order placement in the production path

`binance-cli` may be used only as an operator smoke-test or emergency inspection
tool. The service itself must call Binance through an internal broker adapter
using the official SDK or direct REST/WebSocket clients.

## Approach Options

### Option A: Signal alert only

Generate signals and notify the operator. The operator places trades manually.

This is safest operationally but does not meet the goal because entries,
pyramid adds, and stop replacement are not automated.

### Option B: Semi-automatic entry with manual risk management

The service opens entries automatically, then the operator maintains stops and
adds manually.

This is worse than Option A for this strategy because the highest risk part is
intraday stop maintenance. A missing or stale stop would create avoidable live
risk.

### Option C: Full live service with mandatory guards

The service owns the complete lifecycle: data ingestion, signal generation,
entry, pyramid add, exchange-side stop replacement, reconciliation, and kill
switches.

This is the recommended design because it matches the user's goal. The guardrail
is not to keep humans in the order loop; the guardrail is to make the service
stateful, auditable, idempotent, testnet-first, and easy to kill.

## Architecture

```text
Binance Market Data / Existing Canonical Data
    -> Daily Bar Finalizer
    -> VPE Signal Engine
    -> Position State Machine
    -> Capital Allocator
    -> Risk Gate
    -> Binance Broker Adapter
    -> SQLite State Store / Audit Log
    -> Operator Status CLI
```

Strict boundaries:

- The strategy layer computes features and signals only.
- The strategy layer never calls Binance and never creates orders.
- The state machine decides the next intended position transition.
- The capital allocator sizes approved intents from one shared strategy equity
  pool.
- The risk gate has the final veto before any order intent reaches the broker.
- The broker adapter is the only module that signs Binance requests.
- The state store is the source of local truth, but startup must reconcile it
  with Binance open positions and open orders before trading.

## Components

### Live Runner

Long-running process responsible for scheduling, dependency wiring, heartbeats,
graceful shutdown, and top-level error handling.

It exposes three operating modes:

- `testnet`: real Binance testnet orders.
- `live`: real Binance production orders.
- `reconcile-only`: no new orders, only local/Binance state reconciliation and
  status reporting.

The service requires an explicit live acknowledgement flag or environment
variable before production trading is allowed.

### Daily Bar Finalizer

Responsible for creating the exact closed daily bar view used by the signal
engine.

Inputs:

- Existing canonical Parquet daily bars for historical warmup.
- Binance USD-M daily kline endpoint or stream for the newest closed bar.

Rules:

- Only closed candles are eligible for signals.
- A daily bar is not final until Binance marks it closed or REST returns it as a
  completed candle after the daily boundary.
- The signal for bar `D` can only be evaluated after `D` is closed.
- Live entry for a signal on `D` happens at the first tradable moment after that
  close, not before the close.
- If the latest bar is missing, duplicated, stale, or not strictly aligned to the
  expected daily open time, the symbol is blocked for new entries.

### Signal Engine

Wraps the existing VPE feature and signal logic for incremental live use.

Responsibilities:

- load enough daily history per symbol to compute ATR, volume baseline,
  efficiency comparisons, bottom-position features, and market regime
- reuse the same formulas as research/backtest code
- output deterministic signal candidates with all feature values needed for
  audit
- emit one signal per symbol per closed daily bar at most

It must avoid lookahead by using only bars whose open time is less than or equal
to the just-finalized decision bar.

### Position State Machine

The state machine is the center of the live system.

Symbol states:

- `FLAT`
- `ENTRY_SUBMITTED`
- `OPEN`
- `ADD_ARMED`
- `ADD_SUBMITTED`
- `STOP_REPLACING`
- `EXITING`
- `CLOSED`
- `ERROR_LOCKED`

Order intents:

- `ENTRY`
- `PYRAMID_ADD`
- `STOP_PLACE`
- `STOP_REPLACE`
- `STOP_EXIT_OBSERVED`
- `MANUAL_RECONCILE`

Main lifecycle:

1. When a closed daily bar produces a signal and the symbol is `FLAT`, create an
   `ENTRY` intent for the next live execution point.
2. After entry fill, transition to `OPEN`, initialize one lot, highest high, ATR,
   and protective stop.
3. Maintain an exchange-side stop-market close order.
4. On each new finalized daily bar, recompute current ATR and move the strategy
   stop upward only if `highest_high - 3 * current_atr` is higher than the
   existing stop.
5. If high reaches the pyramid trigger, create one `PYRAMID_ADD` intent. The
   add executes only if the next execution point is still at or above the
   trigger, matching the research rule.
6. While a symbol is not `FLAT`, ignore new VPE entry signals for that symbol.
7. When Binance reports the stop filled, transition through `EXITING` to
   `CLOSED`, persist realized fills, and unlock the symbol.

### Risk Gate

The risk gate validates every intent immediately before broker submission.

Minimum required checks:

- global kill switch is not active
- environment is explicit: `testnet` or `live`
- API keys are present and scoped
- Binance server time is reachable
- symbol metadata is fresh
- symbol is trading and supports required order types
- account is One-way Mode
- account is Single-Asset Mode
- symbol margin type is isolated
- symbol leverage is `1x`
- no local/Binance position mismatch
- no unexpected open orders for the symbol
- requested quantity passes min notional and lot-size filters
- requested stop price passes tick-size and trigger-protect filters
- per-symbol notional is within configured limit
- total open notional is within configured limit
- open position count is within configured limit
- daily realized loss is within configured limit
- order retry count has not exceeded the configured limit
- local state DB is writable

If any check fails, the symbol moves to `ERROR_LOCKED` or the intent is rejected
with an auditable `risk_events` row.

### Capital Allocator

The live strategy uses shared account capital, not isolated per-symbol research
equity curves.

Rules:

- Fetch available USDT balance and account equity during each sizing pass.
- Compute order notional from current strategy equity, so realized profits and
  losses compound into later entries.
- Reserve capital already committed to open strategy positions before sizing a
  new entry.
- Pyramid adds use the same shared pool and must pass the same risk checks as
  initial entries.
- If several symbols signal on the same daily close, allocate from one shared
  capital snapshot in deterministic order.
- Never assume a fixed `1 / symbol_count` allocation in live trading.

First guarded sizing default:

- base entry notional = `min(20 USDT, 5% of strategy equity)`
- pyramid add notional = same notional as the initial lot for that position,
  capped again by available shared capital and risk limits
- if remaining available capital cannot satisfy Binance min-notional filters,
  skip the order and persist a risk rejection

### Binance Broker Adapter

The broker adapter owns all exchange I/O.

Required capabilities:

- fetch exchange information and filters
- fetch account configuration
- fetch position mode and multi-assets mode
- set isolated margin type when needed
- set leverage to `1x` when needed
- place market entry orders
- place market pyramid add orders
- place stop-market close orders
- cancel stale stop orders
- query open orders
- query positions
- receive user data stream order updates
- receive account updates
- map Binance errors to stable internal error categories

Production order submission must be direct SDK/REST/WebSocket code. Shelling out
to `binance-cli` is forbidden in this adapter.

### State Store

Use SQLite for the first version. It is local, simple, transactional, and enough
for one strategy service running under a single service lock.

Tables:

- `service_config`
- `symbol_metadata`
- `bars`
- `signals`
- `positions`
- `position_lots`
- `order_intents`
- `orders`
- `fills`
- `stop_history`
- `risk_events`
- `reconciliation_runs`
- `engine_heartbeats`
- `audit_log`

Every mutable transition must be persisted before or immediately after the
external effect it represents. Order submissions must use deterministic client
order ids so retries are idempotent.

Client order id shape:

```text
XSIG-VPE1-<env>-<symbol>-<intent>-<position_id>-<seq>
```

## Stop Logic

The backtest exits at the stop price when daily low breaches the stop. Live
trading does not wait until the next daily bar to exit. It maintains a
Binance exchange-side protective stop.

Live rule:

- Initial stop after entry: `entry_price - 3 * atr_at_signal`.
- Daily trailing candidate: `highest_high_since_entry - 3 * current_atr`.
- Stop can move upward only.
- Protective stop order: USD-M Futures algo `STOP_MARKET`.
- Stop side: `SELL`.
- One-way position side: `BOTH`.
- Stop trigger source: `CONTRACT_PRICE`, matching the trade-price candles used
  in research.
- Stop close behavior: `closePosition=true` so the stop covers the full current
  long after entry and after the pyramid add.
- `reduceOnly` and explicit `quantity` are not sent with the close-position
  stop.
- Stop replacement must be done with cancel-and-replace semantics that never
  leaves the position unprotected longer than necessary.
- If cancel succeeds but new stop placement fails, retry up to three times over
  30 seconds.
- If no valid active stop exists after those retries, submit a market emergency
  exit for the current long quantity, mark the symbol `ERROR_LOCKED`, and alert.

These parameters must still be verified on Binance testnet before production
trading, but they are fixed for the first implementation plan.

## Pyramid Logic

The selected live preset allows one add:

- After the initial fill, set `next_add_trigger = entry_price + 1 * atr_at_entry`.
- If a finalized daily bar reaches the trigger, arm an add for the next
  execution point.
- The add executes only if that execution price is still at or above the stored
  trigger.
- If price falls below the trigger by the execution point, discard the add
  opportunity.
- After the add fills, update lots, average entry, highest high, notional
  exposure, and protective stop quantity.

The stop protects the full current position size after an add.

## Startup Reconciliation

Startup must be conservative.

Sequence:

1. Open DB and acquire a single-service lock.
2. Load local config and verify it matches the running preset.
3. Fetch Binance account mode, positions, open orders, and symbol metadata.
4. Compare local open positions with Binance positions.
5. Compare local active orders with Binance open orders.
6. Rebuild derived state such as active stop order id and filled lots from
   persisted fills plus Binance data.
7. If state matches, enter normal trading.
8. If state does not match, enter `reconcile-only` or `ERROR_LOCKED` until the
   mismatch is resolved.

The service must never assume a local `FLAT` state is safe if Binance shows a
non-zero position.

## Scheduling

The daily strategy has two timing loops:

- Daily finalization loop around Binance UTC daily close.
- Intraday safety loop for stop/order/account reconciliation.

Recommended first version:

- Evaluate new signals shortly after the Binance daily candle closes.
- Reconcile positions and open orders every 30-60 seconds.
- Keep user data stream connected for faster order/fill updates.
- Refresh exchange information periodically and on startup.

Stops are exchange-side, so intraday stop execution does not depend on the
Python process being alive at the exact stop moment.

## Observability

Operator surfaces:

- `xsignal-vpe-live status`: concise health, mode, open positions, stops, last
  signal time, last heartbeat, and risk locks.
- `xsignal-vpe-live reconcile`: run one reconciliation pass without placing new
  entries.
- `xsignal-vpe-live disable-trading`: activate local kill switch.
- `xsignal-vpe-live enable-trading`: clear local kill switch after explicit
  confirmation.

Logs must include:

- signal candidate accepted/rejected
- risk gate rejection reasons
- order submission request and Binance response ids
- fill updates
- stop replacement attempts
- reconciliation mismatch details
- account mode mismatch
- emergency exits or locks

## Configuration And Secrets

Use environment variables for secrets:

- `BINANCE_API_KEY`
- `BINANCE_SECRET_KEY`

Use config files for non-secret risk limits and mode:

- environment: `testnet` or `live`
- base position fraction
- per-symbol notional cap
- total open notional cap
- max open positions
- max daily realized loss
- emergency policy
- database path

First guarded defaults:

- base position fraction: `5%` of current strategy equity
- per-symbol notional cap: `20 USDT`
- total open notional cap: `100 USDT`
- max open positions: `5`
- max daily realized loss: `50 USDT`
- entry order type: `MARKET`
- entry slippage guard: reject if the latest best ask is more than `1%` above
  the latest contract price snapshot used by the broker adapter
- emergency policy: market-exit the current long quantity if a valid protective
  stop cannot be restored after three retries
- production service manager: `systemd` on the cloud server
- alert destination in the first version: structured logs plus status CLI;
  webhook notification is a follow-up after the core service is stable

API key requirements:

- Futures trading permission only.
- IP whitelist enabled where possible.
- No withdrawal permission.
- Separate testnet and live keys.

## Data Leakage And Lookahead Controls

Live controls:

- No signal uses the currently forming daily bar.
- No trade uses a signal before the signal bar has closed.
- Feature windows include only historical closed bars.
- Market regime is computed using only closed bars up to the decision bar.
- Entry after a signal occurs only after the close of that signal bar.
- Pyramid adds are armed by closed-bar evidence and confirmed by the next
  execution price.

These controls intentionally mirror the backtest assumptions while adapting
stops to safer exchange-side execution.

## Testing Strategy

Unit tests:

- signal engine uses closed bars only
- state transitions for entry, open, add armed, add submitted, stop replacing,
  exit, and error lock
- idempotent client order ids
- risk gate rejects stale metadata, wrong account mode, bad symbol filters, and
  position mismatches
- stop replacement never lowers a stop
- pyramid add is discarded if the next execution price loses the trigger

Integration tests with fake broker:

- complete entry -> stop -> close lifecycle
- entry -> add -> stop lifecycle
- restart recovery with matching Binance state
- restart recovery with mismatched Binance state
- duplicate order retry with same client order id

Testnet smoke:

- connect to Binance testnet
- fetch exchange information
- verify account mode
- set leverage and isolated margin for a small test symbol
- place a tiny entry order
- place protective stop
- observe fill/order updates through user data stream
- cancel/close and reconcile to flat

No production trading is enabled before the testnet smoke passes.

## Rollout Plan

Phase 1: Design and plan.

- Commit this spec.
- Write an implementation plan with review checkpoints.

Phase 2: Offline live-simulation harness.

- Build state machine, risk gate, DB schema, and fake broker.
- Replay historical daily bars through the live state machine.

Phase 3: Testnet service.

- Add Binance adapter.
- Run real testnet orders with tiny notional.
- Verify stop placement, stop replacement, fill handling, and restart recovery.

Phase 4: Live dry-run with real market data.

- Connect production market/account read-only endpoints.
- Do not submit orders.
- Verify signal timing, symbol metadata, account mode, and reconciliation.

Phase 5: Live guarded trading.

- Enable production order submission with small shared-equity sizing and tight
  notional caps.
- Keep kill switch active in operator workflow.
- Review every order/fill/state transition after the first signals.

## Implementation Plan Inputs

The implementation plan should use the fixed defaults in this design. Any
change to account mode, leverage, margin mode, stop trigger source, live notional
limits, base position fraction, or emergency policy requires an explicit spec
update before production trading.
