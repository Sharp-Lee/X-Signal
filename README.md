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

Run the 4h volume-price-efficiency event study in fully offline mode:

```bash
xsignal-vpe-v1 run --root data --offline --run-id smoke-vpe-v1
```

The event study consumes canonical `4h` raw bars, detects unusually efficient
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

Run the trailing-stop production test on the reserved holdout window:

```bash
xsignal-vpe-v1 trail --root data --offline --run-id vpe-trailing-holdout
```

This is separate from the scan phase: it uses the chosen signal settings,
reserves the latest 180 days by default, trades only that holdout window, enters
on the next 4h open after a signal, locks each symbol independently while a
position is open, and exits at a 2 ATR trailing stop.

Expected trailing outputs:

- `manifest.json`
- `summary.json`
- `trades.parquet`
- `equity_curve.parquet`
- `daily_positions.parquet`
