# Volume Price Efficiency V1 Design

Date: 2026-05-07

## Purpose

`volume_price_efficiency_v1` is a strategy-specific 4h event study. Its first
job is not to produce a tradable portfolio. Its job is to answer one narrow
question:

```text
Does a 4h candle with unusually high upward price movement per unit of relative
volume have positive forward return edge?
```

The strategy should stay separate from `momentum_rotation_v1`. Momentum rotation
is a daily cross-sectional allocation strategy. This idea is an event-triggered
bar pattern that may later become a fixed-hold or trailing-roll strategy.

## First Version Scope

The first version only detects 4h signal bars and measures forward returns after
the signal.

It does not include:

- Bottom regime filters.
- Market-wide risk-off filters.
- Portfolio construction.
- Position overlap handling.
- Trailing stops.
- Rolling position continuation.
- Funding, leverage, margin, or liquidation modeling.

Those rules should be added only after the event itself shows evidence of edge.

## Data Dependencies

The strategy uses canonical Parquet only. It should not read ClickHouse from the
strategy hot path.

Required canonical request:

```text
timeframe = 4h
fill_policy = raw
```

Required columns:

- `symbol`
- `open_time`
- `open`
- `high`
- `low`
- `close`
- `quote_volume`
- `bar_count`
- `expected_1m_count`
- `is_complete`
- `has_synthetic`
- `fill_policy`

All signal calculations must use only bars whose close time is less than or
equal to the signal decision time. For a 4h bar with `open_time = t`, its close
time is `t + 4h`. A signal on that bar may only enter at the next 4h bar open.

## Signal Intuition

The intended bar is not merely a high-return bar. It is a bar where upward price
displacement is unusually efficient relative to the symbol's own recent volume
baseline.

The signal should favor bars that:

- Close up.
- Have meaningful real-body movement relative to recent volatility.
- Use non-trivial but not necessarily extreme volume.
- Close near the high of the bar.
- Have a body large enough that the move was not only a wick.
- Are unusually efficient compared with the same symbol's recent efficiency
  distribution.

## Normalized Bar Features

All features are computed per symbol on 4h bars.

### True Range And ATR

Use 4h true range:

```text
true_range_t = max(
  high_t - low_t,
  abs(high_t - close_{t-1}),
  abs(low_t - close_{t-1})
)
```

Use a trailing ATR:

```text
atr_t = mean(true_range_{t-atr_window+1 ... t})
```

Default:

```text
atr_window = 14
```

The current bar's range is allowed in `atr_t` for feature normalization because
the signal is evaluated only after the current 4h bar has closed.

### Upward Movement Unit

The first version should measure body-based upward displacement:

```text
up_move_t = max(close_t - open_t, 0)
move_unit_t = up_move_t / atr_t
```

This avoids treating a long upper wick as a clean upward push.

### Relative Volume Unit

Use a trailing median of quote volume:

```text
volume_baseline_t = median(quote_volume_{t-volume_window ... t-1})
volume_unit_t = quote_volume_t / volume_baseline_t
```

Default:

```text
volume_window = 60
```

The baseline excludes the current bar. This is important. Including the signal
bar in its own volume baseline would dampen exactly the abnormal volume behavior
we are trying to measure.

### Efficiency

Define:

```text
efficiency_t = move_unit_t / max(volume_unit_t, volume_floor)
```

Default:

```text
volume_floor = 0.2
```

The floor prevents tiny-volume bars from creating artificially infinite
efficiency.

### Close Position And Body Ratio

Use:

```text
range_t = high_t - low_t
close_position_t = (close_t - low_t) / range_t
body_ratio_t = abs(close_t - open_t) / range_t
```

Rows with non-positive range are invalid for signal generation.

## Default Signal Rule

The first version should use this default rule:

```text
signal_t =
  quality_good_t
  and efficiency_t > efficiency_{t-1}
  and efficiency_t > rolling_percentile(efficiency_{t-lookback ... t-1}, 0.90)
  and move_unit_t >= 0.5
  and volume_unit_t >= 0.3
  and close_position_t >= 0.7
  and body_ratio_t >= 0.4
```

Default:

```text
efficiency_lookback = 120
efficiency_percentile = 0.90
min_move_unit = 0.5
min_volume_unit = 0.3
min_close_position = 0.7
min_body_ratio = 0.4
```

The rolling percentile excludes the current bar. This prevents the current
signal candidate from raising its own threshold.

## Data Quality Rules

A bar can participate in feature calculation only when:

```text
is_complete == true
has_synthetic == false
bar_count == expected_1m_count
open > 0
high > 0
low > 0
close > 0
high >= max(open, close)
low <= min(open, close)
quote_volume > 0
```

If a required lookback value is missing or invalid, the signal for that symbol
and timestamp is false.

The raw first version should not forward-fill OHLCV inputs for signal
calculation.

## Event Timing And Leakage Boundaries

For a signal bar `t`:

```text
bar interval      = [open_time_t, open_time_t + 4h)
decision time     = open_time_t + 4h
entry price       = open_{t+1}
forward close H   = close_{t+H}
```

The event study should not use `open_{t+1}` or any later value to decide whether
`signal_t` is true.

Forward returns should be measured from next bar open:

```text
forward_return_H = close_{t+H} / open_{t+1} - 1
```

Default horizons:

```text
H = 1, 3, 6, 12, 30
```

These correspond to roughly:

```text
4h, 12h, 1d, 2d, 5d
```

If `open_{t+1}` or `close_{t+H}` is missing or invalid, that horizon should be
counted as unavailable for the event, not silently filled.

## Event Study Outputs

Each run should write artifacts under:

```text
data/strategies/volume_price_efficiency_v1/runs/<run_id>/
```

Required artifacts:

```text
manifest.json
events.parquet
baseline_events.parquet
summary.json
```

`events.parquet` should include one row per signal event:

- `symbol`
- `signal_open_time`
- `decision_time`
- `entry_open_time`
- `entry_price`
- signal features:
  - `move_unit`
  - `volume_unit`
  - `efficiency`
  - `efficiency_threshold`
  - `close_position`
  - `body_ratio`
  - `quote_volume`
  - `volume_baseline`
  - `atr`
- forward returns:
  - `forward_return_1`
  - `forward_return_3`
  - `forward_return_6`
  - `forward_return_12`
  - `forward_return_30`
  - `net_forward_return_1`
  - `net_forward_return_3`
  - `net_forward_return_6`
  - `net_forward_return_12`
  - `net_forward_return_30`

Net forward returns should subtract a configurable round-trip cost:

```text
net_forward_return_H = forward_return_H - 2 * (fee_bps + slippage_bps) / 10_000
```

The event study is not a portfolio simulation, so this is only a simple
round-trip friction estimate. It is still useful because a weak intraday edge can
disappear after realistic costs.

`baseline_events.parquet` should use the same schema as `events.parquet` and add:

- `matched_signal_month`
- `matched_signal_count_for_symbol_month`

`summary.json` should include:

- `event_count`
- `baseline_event_count`
- `symbol_count`
- `first_signal_time`
- `last_signal_time`
- per-horizon:
  - available event count
  - mean return
  - median return
  - win rate
  - p10, p25, p75, p90
  - net mean return
  - net median return
  - net win rate
  - baseline mean return
  - baseline median return
  - baseline win rate
  - event minus baseline mean return
  - event minus baseline median return
- top contributing symbols by event count
- top contributing symbols by average forward return

## Baselines

The event study needs at least one baseline to avoid fooling ourselves.

First version baseline:

```text
For each symbol and month, sample non-signal 4h bars with the same count as
signal bars where possible. Measure the same forward returns.
```

The baseline should use the same data-quality and forward-return availability
rules. It does not need to be random in the first version; deterministic sampling
with a fixed seed is enough for repeatability.

The summary should report event minus baseline return deltas for each horizon.

## Performance Shape

The first implementation can prepare dense arrays per symbol:

```text
open[T, N]
high[T, N]
low[T, N]
close[T, N]
quote_volume[T, N]
quality[T, N]
```

`T` is 4h time and `N` is symbol count. Feature calculations should be vectorized
over symbols where practical. A small Python loop over time is acceptable for the
first version because 4h history is modest. If parameter scans become slow, move
the feature kernel to Numba later.

## Parameter Scan Shape

Do not start with a huge grid. First scan only the parameters that define the
event shape:

```text
efficiency_percentile = 0.80, 0.90, 0.95
min_move_unit = 0.3, 0.5, 0.8
min_volume_unit = 0.2, 0.3, 0.5
min_close_position = 0.6, 0.7, 0.8
min_body_ratio = 0.3, 0.4, 0.5
```

The first acceptance gate is not total return. It is whether signal events
outperform matched non-signal baseline bars with enough event count to matter.

## Success Criteria

The first version is useful if it can answer:

- How often does the signal occur?
- Which symbols produce most signals?
- Do forward returns beat matched baseline bars?
- Which horizon has the best median and win-rate improvement?
- Are results robust across symbols rather than concentrated in one or two names?

If no horizon beats the matched baseline after costs or if events are too rare,
the idea should be revised before adding bottom filters or rolling exits.

## Later Phases

Phase B: fixed-hold backtest.

- Enter next 4h open after signal.
- Hold a fixed number of bars.
- Add costs and slippage.
- Define overlap handling and max concurrent positions.

Phase C: trailing-roll strategy.

- Continue holding while efficiency or trend confirmation persists.
- Exit on trailing stop, failed structure, or efficiency decay.
- Add portfolio-level exposure and risk-off controls.

Bottom filters should be considered after Phase A proves the event has standalone
edge. The bottom filter should improve selectivity, not rescue a weak event.
