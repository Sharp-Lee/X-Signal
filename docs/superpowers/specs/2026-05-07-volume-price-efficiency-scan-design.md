# Volume Price Efficiency Scan Design

## Goal

Add a strategy-specific scan and stratified diagnostics path for
`volume_price_efficiency_v1`. The scan answers whether the 4h volume-price
efficiency signal has stable edge in narrower parameter regions before adding
trailing stops, rolling exits, or portfolio state.

## Decision: Reserve Holdout

The scan must reserve a holdout set. The current event study shows a weak gross
edge at longer horizons but negative mean returns after round-trip friction.
That is exactly the situation where parameter mining is dangerous.

Default holdout:

- Reserve the most recent `180` days.
- Exclude holdout bars from all scan summaries, rankings, and top-config output.
- Record holdout boundaries in scan manifests.
- Do not run selected parameters on holdout in this phase.

The holdout is the final production-test window. It should be touched only after
we choose parameters from research/validation evidence.

## Scope

In scope:

- Scan a compact grid of signal parameters:
  - `efficiency_percentile`
  - `min_move_unit`
  - `min_volume_unit`
  - `min_close_position`
  - `min_body_ratio`
- Reuse canonical offline 4h raw Parquet.
- Compute event-vs-baseline metrics for each config.
- Write scan artifacts:
  - `manifest.json`
  - `summary.json`
  - `summary.csv`
  - `top_configs.json`
  - `bucket_summary.parquet`
- Add stratified diagnostics for each config using research-only events.

Out of scope:

- Trailing stops.
- Rolling positions.
- Bottom filters.
- Portfolio construction.
- Selecting final production parameters from holdout.

## Data Split

Split by 4h `open_time` before feature computation and event building.

For `holdout_days = 180`:

- `holdout_start = last_open_time - 180 days`
- research rows are strictly before `holdout_start`
- holdout rows are at or after `holdout_start`

The metadata records:

- `holdout_days`
- `research_start`
- `research_end`
- `holdout_start`
- `holdout_end`

If `holdout_days = 0`, the full dataset is research and holdout fields are
`null`. Negative values are invalid. A window that consumes all rows is invalid.

## Metrics

Each scan row should include:

- config fields and `config_hash`
- event count and baseline event count
- symbol count
- per-horizon event mean, net mean, baseline mean, and event-minus-baseline mean
- a primary score for ranking

Primary score:

```text
score = event_minus_baseline_mean_return at ranking_horizon
```

Default ranking horizon is `30` 4h bars. This keeps the first scan aligned with
the only horizon that currently has the best gross event-vs-baseline spread.

## Stratified Diagnostics

For each config, bucket signal events by feature quantiles:

- `efficiency`
- `move_unit`
- `volume_unit`
- `close_position`
- `body_ratio`

Default bucket count is `5`. Each bucket row includes the config hash, feature
name, bucket index, bucket bounds, event count, and per-horizon mean/net mean.

These buckets are not used to pick a winner automatically. They are for reading
where the edge lives.

## CLI

Add:

```bash
xsignal-vpe-v1 scan --root data --offline --scan-id <scan_id>
```

Defaults:

- `--holdout-days 180`
- `--ranking-horizon 30`
- compact parameter grid
- `--top-k 20`

The existing `run` command remains unchanged.

## Review Notes

This design intentionally keeps the scan as event-study analysis. If the scan
finds stable positive research evidence in narrow buckets, the next phase should
implement a separate trailing-stop and rolling-position simulator for the chosen
signal family.
