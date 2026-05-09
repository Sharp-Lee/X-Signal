# X-Signal

X-Signal is a strategy research project optimized for custom high-performance backtests.

The shared data foundation exposes canonical deduplicated bars through:

```bash
xsignal-export ensure --timeframe 1h
```

Strategies should call the canonical export layer first, then build their own arrays and backtest kernels.

## Canonical Bar Export

Export or reuse canonical bars for the full available ClickHouse history:

```bash
xsignal-export ensure --timeframe 1h
```

Use raw exchange-derived aggregation by default:

```bash
xsignal-export ensure --timeframe 1h --fill-policy raw
```

Use flat previous-close zero-volume filling when a strategy needs continuous bars:

```bash
xsignal-export ensure --timeframe 1h --fill-policy prev_close_zero_volume
```

Export or reuse canonical bars for one explicit partition:

```bash
xsignal-export ensure --timeframe 1h --year 2026 --month 5 --root data
```

The export layer writes:

- `data/canonical_bars/timeframe=1h/fill_policy=raw/year=2026/month=05/bars.<run-id>.parquet`
- `data/canonical_bars/timeframe=1h/fill_policy=raw/year=2026/month=05/manifest.json`
- `data/canonical_bars/_catalog/timeframe=1h/fill_policy=raw.json`

Strategies should call `ensure_canonical_bars` before reading canonical Parquet.
When a requested partition is already complete, the export layer returns the
local Parquet dataset without opening a ClickHouse connection.
The manifest is the atomic pointer to the immutable Parquet file for that
partition.
Raw and filled datasets are separate canonical identities and never satisfy
each other's cache entries.

Validate a raw canonical partition against Binance USD-M Futures:

```bash
python scripts/validate_binance_klines.py \
  --manifest data/canonical_bars/timeframe=1h/fill_policy=raw/year=2020/month=02/manifest.json \
  --symbol BTCUSDT \
  --timeframe 1h \
  --limit 10
```

## Momentum Rotation V1

Run the first strategy-specific backtest:

```bash
xsignal-momentum-v1 run --root data --top-n 10 --fee-bps 5 --slippage-bps 5
```

The strategy consumes canonical `1h`, `4h`, and `1d` raw bars, prepares dense arrays, computes a multi-timeframe momentum score, and writes outputs under:

```text
data/strategies/momentum_rotation_v1/runs/<run_id>/
```

Expected outputs:

- `manifest.json`
- `summary.json`
- `equity_curve.parquet`
- `daily_positions.parquet`

Run only a reserved production-test window after choosing parameters:

```bash
xsignal-momentum-v1 run \
  --root data \
  --offline \
  --run-id final-production-test \
  --top-n 10 \
  --fee-bps 5 \
  --slippage-bps 5 \
  --start-date 2025-11-09
```

Scan a lightweight parameter grid without writing per-combination position files:

```bash
xsignal-momentum-v1 scan \
  --root data \
  --offline \
  --top-n 5,10,20 \
  --fee-bps 5 \
  --slippage-bps 2.5,5 \
  --min-rolling-7d-quote-volume 0,10000000 \
  --holdout-days 180
```

The scan prepares canonical arrays once, reuses the prepared cache by default,
excludes the most recent 180 days from parameter selection by default, and
writes:

```text
data/strategies/momentum_rotation_v1/scans/<scan_id>/
```

Expected scan outputs:

- `manifest.json`
- `summary.json`
- `summary.csv`

Select parameters from a scan using return minus risk penalties:

```bash
xsignal-momentum-v1 select \
  --root data \
  --scan-id <scan_id> \
  --selection-id selected-v1 \
  --max-drawdown-lte 0.35 \
  --missing-weight-lte 0.1 \
  --min-periods 1000 \
  --drawdown-penalty 1 \
  --missing-return-penalty 1
```

The selector writes:

```text
data/strategies/momentum_rotation_v1/scans/<scan_id>/selections/<selection_id>.json
```

The selection file includes the chosen parameters and a ready-to-run holdout
command for the reserved production-test window. Hard filters are applied before
scoring, so unacceptable drawdown, missing-price exposure, or too-short research
windows are excluded instead of merely penalized.

## Volume Price Efficiency V1

Run the daily volume-price-efficiency event study in fully offline mode:

```bash
xsignal-vpe-v1 run --root data --offline --run-id smoke-vpe-v1
```

The event study consumes canonical `1d` raw bars, detects unusually efficient
upward candles, samples matched non-signal baseline candles, and writes:

```text
data/strategies/volume_price_efficiency_v1/runs/<run_id>/
```

Expected outputs:

- `manifest.json`
- `summary.json`
- `events.parquet`
- `baseline_events.parquet`

Scan a compact research-only parameter grid and reserve the latest 180 days as
holdout:

```bash
xsignal-vpe-v1 scan --root data --offline --scan-id smoke-vpe-scan
```

The scan excludes holdout rows from all parameter rankings and writes:

```text
data/strategies/volume_price_efficiency_v1/scans/<scan_id>/
```

Expected scan outputs:

- `manifest.json`
- `summary.json`
- `summary.csv`
- `top_configs.json`
- `bucket_summary.parquet`

Scan the same research window with the actual trailing-stop simulator:

```bash
xsignal-vpe-v1 trail-scan \
  --root data \
  --offline \
  --scan-id vpe-trailing-scan \
  --atr-multiplier 1.5,2,2.5,3,4,5,6
```

Use this before the final holdout trail when the production exit logic is a
trailing stop. It ranks configs by research `total_return - max_drawdown`, keeps
the latest 180 days reserved, searches ATR trailing-stop width on research data,
and does not trade holdout rows.

Expected trailing scan outputs:

- `manifest.json`
- `summary.json`
- `summary.csv`
- `top_configs.json`

Diagnose a fixed trailing-stop config across research and holdout without using
the diagnostic output for parameter selection:

```bash
xsignal-vpe-v1 trail-diagnose \
  --root data \
  --offline \
  --diagnostic-id vpe-trailing-diagnostic \
  --efficiency-percentile 0.9 \
  --min-move-unit 1.2 \
  --min-volume-unit 1.0 \
  --min-close-position 0.94 \
  --min-body-ratio 0.85
```

Expected diagnostic outputs:

- `manifest.json`
- `time_summary.parquet`
- `bucket_summary.parquet`

Run the trailing-stop production test on the reserved holdout window:

```bash
xsignal-vpe-v1 trail \
  --root data \
  --offline \
  --run-id vpe-trailing-holdout \
  --atr-multiplier 3
```

This is separate from the scan phase: it uses the chosen signal settings,
reserves the latest 180 days by default, trades only that holdout window, enters
on the next daily open after a signal, locks each symbol independently while a
position is open, and exits at the research-selected ATR trailing stop.

Expected trailing outputs:

- `manifest.json`
- `summary.json`
- `trades.parquet`
- `equity_curve.parquet`
- `daily_positions.parquet`

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

## Binance USD-M Testnet Smoke

The live broker adapter talks to Binance USD-M Futures testnet through signed
REST requests. It does not use `binance-cli` for production or testnet order
paths.

Set testnet credentials:

```bash
export BINANCE_API_KEY=...
export BINANCE_SECRET_KEY=...
```

Or keep them in the local ignored file `.secrets/binance-testnet.env`:

```bash
BINANCE_API_KEY=...
BINANCE_SECRET_KEY=...
```

Run a read-only testnet smoke check:

```bash
xsignal-vpe-live testnet-smoke --symbol BTCUSDT
```

Validate order parameters without entering the matching engine:

```bash
xsignal-vpe-live testnet-smoke \
  --symbol BTCUSDT \
  --quantity 0.001 \
  --submit-test-order
```

`--submit-test-order` uses Binance `/fapi/v1/order/test`; it validates signed
order payloads but does not create a live position. Real testnet lifecycle
trading requires a separate acknowledgement because it enters the Binance
testnet matching engine:

```bash
xsignal-vpe-live testnet-lifecycle \
  --symbol BTCUSDT \
  --quantity 0.001 \
  --stop-offset-pct 0.05 \
  --i-understand-testnet-order
```

`--stop-offset-pct 0.05` means a 5% protective stop offset for this smoke run.
The command opens a tiny testnet long, places a `STOP_MARKET closePosition=true`
protection order, verifies the position and stop, cancels the stop, closes with
a reduce-only market sell, and verifies the symbol is flat again. Production
trading remains disabled.

Before submitting lifecycle orders, the command loads Binance `exchangeInfo`
for the symbol and builds local order rules:

- market quantities are floored to `MARKET_LOT_SIZE.stepSize`
- quantities must satisfy `MARKET_LOT_SIZE.minQty/maxQty`
- stop prices are floored to `PRICE_FILTER.tickSize`
- notional sizing helpers reject orders below `MIN_NOTIONAL`
- client order ids use ASCII-only compact digests, so Chinese-name symbols do
  not leak into Binance `newClientOrderId` / `clientAlgoId`

The first live preset still forces isolated margin and `1x` leverage. Different
symbol maximum-leverage brackets therefore do not affect order acceptance yet;
if leverage is raised later, `/fapi/v1/leverageBracket` must become part of the
preflight gate.

Signed order submission can return an HTTP timeout after Binance has already
accepted the request. The testnet lifecycle treats these as unknown states, not
as simple failures:

- entry market-buy timeout: query `/fapi/v1/order` by `newClientOrderId`, then
  verify the position before placing protection
- protective stop timeout: query `/fapi/v1/algoOrder` by `clientAlgoId`; if the
  stop cannot be confirmed, attempt to cancel that `clientAlgoId` and close the
  position
- reduce-only close timeout: query `/fapi/v1/order` and verify the position is
  flat before declaring success

This is still a smoke lifecycle, not a full production recovery daemon. The
production runner should persist every client id before submission and resume
the same reconciliation flow after process restart.
