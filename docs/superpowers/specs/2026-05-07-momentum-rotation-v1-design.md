# Momentum Rotation V1 Design

Date: 2026-05-07

## Purpose

`momentum_rotation_v1` is the first strategy-specific high-performance backtest for X-Signal. Its purpose is to validate the canonical Parquet data foundation under a realistic research workload:

- Hundreds of Binance USD-M symbols.
- Multiple canonical timeframes.
- Full historical data.
- Cross-sectional ranking.
- Repeated portfolio rebalancing.

This strategy is intentionally narrow. It should not become a general backtest engine. It should build only the preparation code, arrays, and kernel needed for this multi-timeframe long-only momentum rotation idea.

## Strategy Summary

The strategy ranks all eligible symbols by multi-timeframe momentum and holds the strongest names with equal weights.

Initial behavior:

- Universe: all canonical symbols, filtered by data quality and liquidity.
- Data: canonical Parquet only, never direct ClickHouse reads from the strategy.
- Timeframes: `1h`, `4h`, and `1d`.
- Fill policy: `raw`.
- Portfolio: long-only.
- Rebalance cadence: daily.
- Selection: Top N by momentum score.
- Weighting: equal weight among selected symbols.
- Leverage: none in the first version.
- Fees: fixed basis-point assumption.
- Slippage: fixed basis-point assumption in the first version.

The first version should optimize for speed, observability, and easy debugging over strategy sophistication.

## Non-Goals

This design does not introduce:

- A universal strategy interface.
- A reusable order simulator.
- Short portfolios.
- Funding-rate modeling.
- Margin and liquidation simulation.
- Intrabar order fill modeling.
- A generic factor library.
- A generic parameter optimization framework.

Those can be added later only when a specific strategy requires them.

## Data Dependencies

The strategy must call the canonical export layer before reading data:

```text
ensure_canonical_bars(timeframe="1h", fill_policy="raw")
ensure_canonical_bars(timeframe="4h", fill_policy="raw")
ensure_canonical_bars(timeframe="1d", fill_policy="raw")
```

It should then read manifest-selected Parquet files from:

```text
data/canonical_bars/timeframe=<tf>/fill_policy=raw/...
```

Required columns:

- `symbol`
- `open_time`
- `close`
- `quote_volume`
- `bar_count`
- `expected_1m_count`
- `is_complete`
- `has_synthetic`
- `fill_policy`

The strategy should treat `is_complete` and `has_synthetic` as eligibility inputs even though the first version uses `raw`, where `has_synthetic` should normally be false.

The first version should require every source bar inside the score windows to satisfy:

```text
is_complete == true
has_synthetic == false
```

The rule applies to every source bar inside each lookback window, not only the two endpoint bars used for return calculation. This conservative rule keeps the first result easy to audit. Later versions can relax this per strategy if the data-quality trade-off is intentional.

## Prepared Array Layout

Preparation should create a strategy-owned cache under:

```text
data/strategies/momentum_rotation_v1/cache/
```

Recommended arrays:

```text
symbols.json
times_1d.npy
close_1h.npy
close_4h.npy
close_1d.npy
quote_volume_1d.npy
complete_1h.npy
complete_4h.npy
complete_1d.npy
tradable_mask_1d.npy
score_1d.npy
```

The hot backtest path should use dense time-major arrays:

```text
close[T, N]
score[T, N]
tradable_mask[T, N]
```

`T` is daily rebalance time. `N` is the symbol count after canonical symbol alignment. Dense arrays are appropriate because the strategy compares all symbols at each rebalance point.

Array dtypes:

- Prices and scores: `float64` for the first version.
- Masks: `bool`.
- Symbol indices: integer arrays when needed.

Downcasting to `float32` can be evaluated later after result parity checks.

## Signal Definition

At each daily rebalance timestamp, compute a momentum score for every symbol that passes eligibility filters.

Initial return windows:

- Short: 24 hours from `1h` close data.
- Medium: 7 days from `4h` close data.
- Long: 30 days from `1d` close data.

Initial score:

```text
score = 0.4 * return_24h + 0.4 * return_7d + 0.2 * return_30d
```

Return definition:

```text
return_window = close_now / close_window_ago - 1
```

Time alignment:

- All timestamps are UTC.
- Daily rebalance timestamps are derived from completed `1d` bars.
- A rebalance timestamp `t` means the daily bar ending at `t` has just closed.
- The `1h` and `4h` close values used at `t` must be the latest completed bars with close time `<= t`.
- If the latest required `1h`, `4h`, or `1d` bar is missing at `t`, the symbol is ineligible for that rebalance.
- Returns and masks must be shifted so a decision at `t` cannot use any bar closing after `t`.

Eligibility rules:

- All required windows must exist.
- All source bars inside the lookback windows must pass the data-quality rule.
- The symbol must have positive close prices.
- The symbol must meet the liquidity filter.

Initial liquidity filter:

```text
rolling_7d_quote_volume >= configured minimum
```

The first implementation may set the default minimum to zero for maximum coverage, but the config must expose it because liquidity filtering is central to real-world interpretation.

## Portfolio Rules

At each daily rebalance:

1. Rank eligible symbols by score descending.
2. Select the top `N`.
3. Allocate equal target weight to selected symbols.
4. Set all other symbols to zero target weight.
5. Apply transaction costs based on absolute weight turnover.

Default parameters:

```text
top_n = 10
rebalance_frequency = 1d
fee_bps = 5
slippage_bps = 5
initial_equity = 1.0
```

Return accounting:

- Use close-to-close daily returns for selected symbols.
- All timestamps are UTC.
- A daily bar with `open_time = d` represents the interval `[d, d + 1d)`.
- The close of that bar becomes available at `d + 1d`.
- Rebalance decisions made after that close are applied to the next daily return.
- Portfolio return is the weighted sum of next-period symbol returns minus turnover cost.
- PnL is realized over the following daily period.

This avoids lookahead by shifting weights forward one rebalance interval before applying returns.

## Backtest Kernel

The first kernel can be a focused NumPy implementation with small explicit loops over daily time. This is acceptable because daily `T` is modest and cross-sectional work can stay vectorized.

Hot-loop responsibilities:

- Apply eligibility mask.
- Rank scores.
- Build target weights.
- Compute turnover.
- Apply fees and slippage.
- Compute portfolio returns.
- Write equity curve and selected holdings.

If parameter sweeps or intraday rebalancing make this too slow, the next step is a Numba kernel with the same input arrays and output contract.

## Outputs

Each run should write outputs under:

```text
data/strategies/momentum_rotation_v1/runs/<run_id>/
```

Required artifacts:

```text
manifest.json
equity_curve.parquet
daily_positions.parquet
summary.json
```

Optional first-version artifact:

```text
daily_scores.parquet
```

The run manifest should include:

- Strategy name and version.
- Git commit.
- Config hash.
- Canonical dataset manifest references.
- Timeframes and fill policy.
- Symbol count.
- Date range.
- Score parameters.
- Eligibility parameters.
- Fee and slippage assumptions.
- Runtime duration.
- Output paths.

## Error Handling

Preparation should fail fast when:

- Required canonical partitions are missing and export fails.
- A required Parquet manifest does not match `fill_policy=raw`.
- Required columns are missing.
- Timeframes cannot be aligned to the daily rebalance index.
- No symbols pass eligibility filters.
- Any score window would use future data.

Backtest should fail fast when:

- Array shapes do not match.
- Weights contain NaN or infinite values.
- Portfolio equity becomes NaN or infinite.
- Config values are invalid, such as `top_n <= 0` or negative costs.

## Validation

Unit tests should cover:

- Score calculation for a small deterministic fixture.
- Lookahead prevention through weight shifting.
- Top N selection with missing or ineligible symbols.
- Turnover cost calculation.
- Empty-eligibility failure.
- Manifest/config hash creation.

Integration smoke should cover:

- Calling canonical export for `1h`, `4h`, and `1d`.
- Preparing arrays for a small date range.
- Running a short backtest over a small symbol subset.
- Writing all expected artifacts.

Performance smoke should report:

- Canonical data load time.
- Array preparation time.
- Kernel runtime.
- Symbols processed.
- Daily bars processed.

The first useful benchmark target is not an absolute latency number. It is to make runtime visible and stable enough that later strategy changes can be compared honestly.

## Open Parameters

The implementation should expose these as config values:

- `top_n`
- `fee_bps`
- `slippage_bps`
- `min_rolling_7d_quote_volume`
- `short_return_weight`
- `medium_return_weight`
- `long_return_weight`
- `start_date`
- `end_date`

The implementation should keep the first config small and avoid a broad parameter grid until the single-run path is correct and fast.

## Recommendation

Build `momentum_rotation_v1` as a strategy-owned package that consumes canonical Parquet, prepares dense arrays, and runs a narrow long-only Top N daily rebalance kernel.

This gives the project a realistic first backtest without introducing a general engine. It also creates a concrete performance baseline for the canonical data foundation.
