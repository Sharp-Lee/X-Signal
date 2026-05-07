# Volume Price Efficiency Trailing Stop Design

## Goal

Add a test-set simulator for `volume_price_efficiency_v1` that trades the
signal with per-symbol locking and ATR trailing stops.

## Decision

The simulator uses:

- per-symbol independent locking
- entry on the next bar open after a signal
- exit on `highest_high - 2 * current_atr`
- ATR recomputed on every bar from the data available up to that bar
- stop fills at the stop price
- any new signal for a symbol already in position is ignored

This is not a search phase. It is a production-test simulation phase on the
reserved holdout window.

## Scope

In scope:

- per-symbol state machine for flat / long / stopped
- trailing stop updates with dynamic ATR
- per-symbol lockout while a position is open
- output rows for entries, exits, stop price, ATR, and realized return
- summary artifacts for win rate, mean return, and drawdown style diagnostics

Out of scope:

- parameter scanning
- portfolio allocation across symbols
- short selling
- pyramiding
- partial fills

## State Machine

For each symbol:

1. When flat and a signal appears on bar `t`, schedule entry at `open[t+1]`.
2. After entry, track:
   - entry price
   - highest high seen so far
   - current ATR recomputed each bar
   - stop price = `highest_high - 2 * current_atr`
3. If the next bar's low breaches the stop, exit at the stop price.
4. If the position is open, ignore any further signals for that symbol.
5. After exit, the symbol becomes eligible for a new entry again.

## Data Windows

The simulator runs only on the final reserved holdout window.
The holdout window is the same one already reserved by the scan phase.

## Outputs

Write a new strategy-owned run directory with:

- `manifest.json`
- `summary.json`
- `trades.parquet`
- `equity_curve.parquet`
- `daily_positions.parquet`

`trades.parquet` should contain one row per completed trade with:

- symbol
- signal_open_time
- entry_open_time
- exit_time
- entry_price
- exit_price
- stop_price_at_exit
- atr_at_entry
- atr_at_exit
- highest_high
- realized_return
- holding_bars

## Validation

The first implementation should be validated with unit tests on small synthetic
arrays covering:

- immediate stop breach
- stop moved upward as ATR updates
- second signal ignored while already long
- re-entry after exit on the same symbol
