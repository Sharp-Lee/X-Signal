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
xsignal-vpe-live status --db data/live/vpe-live.sqlite --no-system
xsignal-vpe-status --db data/live/vpe-live.sqlite --no-system
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

On servers, pass the managed testnet environment file explicitly:

```bash
--env-file /etc/xsignal/binance-testnet.env
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
  --db data/live/vpe-testnet.sqlite \
  --i-understand-testnet-order
```

`--stop-offset-pct 0.05` means a 5% protective stop offset for this smoke run.
The command opens a tiny testnet long, places a `STOP_MARKET closePosition=true`
protection order, verifies the position and stop, cancels the stop, closes with
a reduce-only market sell, and verifies the symbol is flat again. Production
trading remains disabled.

When `--db` is provided, the lifecycle opens a local SQLite store, initializes
the schema, and persists each deterministic client id before the corresponding
Binance order submission. The entry, protective stop, and reduce-only close
intents remain available for restart reconciliation and audit.

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

Restart reconciliation for the testnet path is now available:

```bash
xsignal-vpe-live testnet-reconcile \
  --db data/live/vpe-testnet.sqlite \
  --env-file /etc/xsignal/binance-testnet.env \
  --symbol BTCUSDT
```

The default command is read-only. It loads unresolved local order intents,
queries Binance by `newClientOrderId` or `clientAlgoId`, compares local position
state with Binance position risk and open algo orders, and prints a JSON
summary. It returns non-zero when a symbol is locked because local and Binance
state do not match safely.

Repair mode is intentionally explicit because it may submit a testnet
reduce-only close order for a strategy-owned long position that has no active
strategy stop:

```bash
xsignal-vpe-live testnet-reconcile \
  --db data/live/vpe-testnet.sqlite \
  --env-file /etc/xsignal/binance-testnet.env \
  --symbol BTCUSDT \
  --repair \
  --i-understand-testnet-order
```

Repair mode does not close unknown positions when local state is `FLAT`; those
are marked `ERROR_LOCKED` for manual inspection. Production trading remains
disabled until the same reconciliation guarantees are wired into the long-running
service loop.

For repeatable testnet rehearsals, use the protected-position commands instead
of ad hoc scripts. Opening a protected rehearsal position persists the local
position and deterministic order intents before submitting the entry and
protective stop:

```bash
xsignal-vpe-live testnet-open-protected \
  --db data/live/vpe-testnet.sqlite \
  --env-file /etc/xsignal/binance-testnet.env \
  --symbol SOLUSDT \
  --notional 8 \
  --stop-offset-pct 0.05 \
  --i-understand-testnet-order
```

Closing a protected rehearsal position cancels the strategy stop, persists a
reduce-only close intent, submits the reduce-only market close, verifies the
symbol is flat, and marks the local position closed:

```bash
xsignal-vpe-live testnet-close-protected \
  --db data/live/vpe-testnet.sqlite \
  --env-file /etc/xsignal/binance-testnet.env \
  --symbol SOLUSDT \
  --position-id SOLUSDT-1 \
  --i-understand-testnet-order
```

Both commands are testnet-only and require the explicit acknowledgement flag
because they change Binance testnet account state.

Run the full deployment rehearsal in one command:

```bash
xsignal-vpe-live testnet-rehearsal \
  --db data/live/vpe-testnet.sqlite \
  --env-file /etc/xsignal/binance-testnet.env \
  --symbol ADAUSDT \
  --notional 8 \
  --stop-offset-pct 0.05 \
  --i-understand-testnet-order
```

This command opens a protected testnet position, runs read-only reconciliation,
restarts the testnet stream daemon, reconciles again, closes the rehearsal
position through the audited reduce-only path, and runs a final read-only
reconciliation. It returns non-zero if any reconciliation step reports an error
or the close verification is not flat.

For deployment gating, run the verify wrapper:

```bash
xsignal-vpe-live testnet-deploy-verify \
  --db data/live/vpe-testnet.sqlite \
  --env-file /etc/xsignal/binance-testnet.env \
  --symbol ADAUSDT \
  --notional 8 \
  --stop-offset-pct 0.05 \
  --i-understand-testnet-order
```

It checks pre-deploy status first and skips testnet order submission when the
daemon is already unhealthy. When pre-status is clean, it runs
`testnet-rehearsal`, collects final status, checks recent journal health, and
returns non-zero unless the whole deployment gate is clean.

## VPE Automatic Live Cycle

The preferred automatic runner is the realtime WebSocket daemon. In steady
state it uses full-universe `1m` kline WebSocket streams for every selected
`TRADING` USDT perpetual symbol, then locally aggregates the configured signal
intervals. REST kline calls are recovery-only: startup and reconnect gap
recovery read persisted `1m` cursors, fetch missing closed `1m` bars, and replay
them through the same local pipeline. Signals are screened only when a locally
aggregated bar is complete.

Unclosed `1m` updates are memory-only. They are not persisted as market bars;
only symbols with active strategy positions consume them for trailing-stop and
pyramid-add maintenance. Closed `1m` bars are batch-written to SQLite and then
aggregated into the configured signal intervals. This keeps the WebSocket read
path separate from the exchange order path and prevents full-market market data
from competing with entry, stop-replacement, and reconciliation REST calls.

Run the testnet daemon locally:

```bash
xsignal-vpe-live stream-daemon \
  --mode testnet \
  --db data/live/vpe-testnet.sqlite \
  --interval 1h \
  --interval 4h \
  --interval 1d \
  --lookback-bars 120
```

If `--interval` is omitted, the daemon defaults to `1h`, `4h`, and `1d`.
Any Binance USD-M kline interval is accepted, including `1m`, `3m`, `5m`,
`15m`, `30m`, `1h`, `2h`, `4h`, `6h`, `8h`, `12h`, `1d`, `3d`, `1w`, and
`1M`.

WebSocket streams are chunked by `--max-streams`; the default is `200`, so the
current full USDT perpetual universe runs in a few combined-stream connections
instead of one large socket or dozens of tiny sockets. Each connection is
proactively rotated before Binance's 24-hour hard disconnect: by default the
daemon reconnects after 23 hours with up to 30 minutes of deterministic jitter
per stream chunk. A `stream_rotation_due` log line is therefore expected daily.

On startup and before every WebSocket reconnect, gap recovery is limited to
symbols with active strategy positions. For those symbols, the daemon reads the
persisted `1m` cursor, fetches any missing closed `1m` bars through REST, stores
them locally, and replays them through the local aggregator so trailing-stop and
pyramid state can be safely maintained. Symbols without active positions are
treated like a fresh start: the daemon does not chase missed historical signals
or backfill their downtime gap before opening the full-universe WebSocket. On
the very first run for a symbol, no historical `1m` gap is fetched; the REST-seeded
closed `1h`/`4h`/`1d` buffers provide initial signal context, and local `1m`
retention begins from that startup point. Recovered historical bars update
buffers and protective position state, but they do not open delayed entries or
submit delayed pyramid adds; the daemon waits for fresh realtime triggers after
recovery.

The daemon also keeps an entry health gate. New entries are blocked until the
startup reconciliation pass is clean. Later WebSocket reconnect failures, REST
rate-limit errors, or reconciliation errors close the gate again; the next clean
reconciliation pass reopens it. This gate only blocks new entries. It does not
stop realtime high/last-price maintenance for active positions, so trailing
stops and pyramid-add checks continue to run.

Operator status is available as either the `xsignal-vpe-live status` subcommand
or the shorter standalone command:

```bash
xsignal-vpe-status --db /var/lib/xsignal/live/vpe-testnet.sqlite
xsignal-vpe-status --db /var/lib/xsignal/live/vpe-testnet.sqlite --json
```

The status command summarizes the deployed revision, systemd service state,
live guard state, WebSocket socket queues, recent daemon journal signals,
persisted bar/cursor lag, active positions, and unresolved order intents. It
returns non-zero when warnings are present. Use `--no-system` for local SQLite
inspection without systemd/journal/socket checks.

The one-shot cycle remains available as a fallback/manual check:

Run one automatic strategy cycle on Binance USD-M Futures testnet:

```bash
xsignal-vpe-live run-cycle \
  --mode testnet \
  --db data/live/vpe-testnet.sqlite \
  --lookback-bars 120
```

The cycle performs startup reconciliation first, loads recent closed daily
klines, computes the fixed VPE live preset, maintains existing stops/adds, and
opens new long positions only when the latest closed daily bar has a signal.
If no signal exists, it exits without submitting orders.

Production mode uses the same code path but is guarded by two controls:

```bash
export XSIGNAL_ENABLE_LIVE_TRADING=1
xsignal-vpe-live run-cycle \
  --mode live \
  --db data/live/vpe-live.sqlite \
  --lookback-bars 120 \
  --i-understand-live-order
```

On servers, the equivalent guard file is:

```text
/etc/xsignal/enable-live-trading
```

Production credentials should be stored separately from testnet credentials:

```text
/etc/xsignal/binance-live.env
```

Read-only live account smoke check:

```bash
xsignal-vpe-live live-smoke --symbol BTCUSDT --env-file /etc/xsignal/binance-live.env
```

Systemd unit templates live in:

```text
deploy/systemd/xsignal-vpe-testnet-stream-daemon.service
deploy/systemd/xsignal-vpe-live-stream-daemon.service
deploy/systemd/xsignal-vpe-testnet-auto-cycle.service
deploy/systemd/xsignal-vpe-testnet-auto-cycle.timer
deploy/systemd/xsignal-vpe-live-auto-cycle.service
deploy/systemd/xsignal-vpe-live-auto-cycle.timer
```

The realtime testnet service is the primary automatic trading service. The
daily timer is intended only as a fallback/manual one-shot runner. The live
service includes `ConditionPathExists=/etc/xsignal/enable-live-trading`, so
installing the live unit files does not enable live order submission unless the
operator deliberately creates that file and provides production API keys.

For alpha operations, rehearsals, and incident checks, use the runbook:

```text
docs/operations/vpe-live-runbook.md
```
