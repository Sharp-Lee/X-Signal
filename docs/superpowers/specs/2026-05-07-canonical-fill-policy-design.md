# Canonical Fill Policy Design

Date: 2026-05-07

## Purpose

Canonical bars should support two distinct research needs:

- Preserve the exchange-derived raw kline semantics exactly.
- Provide an optional continuous-bar view for strategies that expect every symbol and timestamp to have a bar.

The project should not hide synthetic data inside the default canonical export. Filling missing klines changes the interpretation of price, volume, and tradability, so it must be explicit in the request, manifest, catalog, and output columns.

## Decision

Add a `fill_policy` dimension to canonical bar requests.

Initial policies:

- `raw`: aggregate only the 1m rows that exist in ClickHouse.
- `prev_close_zero_volume`: fill missing 1m rows with a flat synthetic bar using the previous real close and zero activity, then aggregate to the requested timeframe.

The default remains `raw`.

Example caller intent:

```python
ensure_canonical_bars(timeframe="1h", fill_policy="raw")
ensure_canonical_bars(timeframe="1h", fill_policy="prev_close_zero_volume")
```

## Semantics

### Raw

The `raw` policy keeps the current behavior:

- `open`: first present 1m open in the target interval.
- `high`: maximum present 1m high.
- `low`: minimum present 1m low.
- `close`: last present 1m close.
- volume fields: sum of present 1m rows.
- `trade_count`: sum of present 1m rows.
- `bar_count`: number of present 1m rows.
- `is_complete`: whether `bar_count` equals the expected 1m count for the interval.

This policy is the best source for data quality analysis and exchange-faithful research.

### Prev Close Zero Volume

The `prev_close_zero_volume` policy creates missing 1m rows before aggregation:

- `open = previous_real_close`
- `high = previous_real_close`
- `low = previous_real_close`
- `close = previous_real_close`
- `volume = 0`
- `quote_volume = 0`
- `trade_count = 0`
- `taker_buy_volume = 0`
- `taker_buy_quote_volume = 0`
- `is_synthetic_1m = true`

Then the normal OHLCV aggregation runs over real and synthetic 1m rows.

The output target-timeframe bars should include:

- `bar_count`: real 1m rows present before filling.
- `synthetic_1m_count`: synthetic 1m rows inserted.
- `expected_1m_count`: expected number of 1m rows for this target interval.
- `is_complete`: whether all expected timestamps are represented after filling.
- `has_synthetic`: whether `synthetic_1m_count > 0`.
- `fill_policy`: the string policy used to create the dataset.

This policy is useful for continuous matrix construction, but strategies must treat `has_synthetic` as a tradability/data-quality mask input.

## Fill Boundaries

Filling must not create fake history outside a symbol's observed trading range.

For each symbol:

- Do not fill before the first real 1m row.
- Do not fill after the last real 1m row for historical sealed partitions.
- For the current mutable partition, fill only up to the last globally safe timestamp selected by the export process.
- Do not fill across a known delisting boundary once lifecycle metadata exists.

At project start, symbol lifecycle metadata may be incomplete. The initial implementation should derive symbol bounds from observed ClickHouse 1m rows and record this limitation in the manifest.

## Timeframe Scope

The fill policy should work for any fixed-length timeframe supported by the canonical exporter.

Recommended first implementation scope:

- `1m`
- `3m`
- `5m`
- `15m`
- `30m`
- `1h`
- `2h`
- `4h`
- `6h`
- `8h`
- `12h`
- `1d`

Defer `3d`, `1w`, and `1M` until their Binance boundary semantics are separately validated. `3d` buckets need explicit cross-year alignment rules, monthly expected counts are dynamic, and weekly alignment must match Binance exactly before they become canonical.

## Storage Layout

`fill_policy` must be part of the canonical identity so raw and filled datasets cannot overwrite or satisfy each other.

Recommended layout:

```text
data/canonical_bars/
  _catalog/
    timeframe=1h/fill_policy=raw.json
    timeframe=1h/fill_policy=prev_close_zero_volume.json
  timeframe=1h/
    fill_policy=raw/
      year=2026/
        month=05/
          manifest.json
          bars.<run-id>.parquet
    fill_policy=prev_close_zero_volume/
      year=2026/
        month=05/
          manifest.json
          bars.<run-id>.parquet
```

The manifest should include:

- `fill_policy`
- `aggregation_semantics_version`
- `synthetic_generation_version`
- `source_table`
- `deduplication_mode`
- `query_hash`
- `row_count`
- `synthetic_1m_count_total`
- `incomplete_raw_bar_count`
- `symbol_bound_policy`

## Catalog Rules

A partition is complete only when all identity fields match:

- timeframe
- fill policy
- dataset version
- source table
- deduplication mode
- aggregation semantics version
- synthetic generation version
- partition key
- query hash
- Parquet metadata and row count

The catalog must treat a raw partition as missing for a filled request, and a filled partition as missing for a raw request.

## Validation

Validation should include three layers.

1. Unit fixtures:
   - Missing middle 1m rows become flat zero-volume synthetic rows.
   - Leading missing rows before first real bar are not filled.
   - Target OHLCV aggregation over synthetic rows is correct.
   - `synthetic_1m_count`, `has_synthetic`, and `is_complete` are correct.

2. Binance comparison:
   - For raw bars, sampled `1h`, `4h`, and `1d` outputs should match Binance USD-M Futures `/fapi/v1/klines` within decimal display tolerance.
   - Filled bars are not expected to match Binance unless the exchange itself emits the same synthetic semantics for the tested interval.

3. Strategy-readiness:
   - Strategy preparation must expose `has_synthetic` and data-quality masks to the backtest kernel.
   - Backtest run manifests must record the fill policy.

## Non-Goals

This design does not introduce:

- A general missing-data imputation framework.
- Forward-filled volume or trade counts.
- Synthetic prices before a symbol's first real observation.
- Symbol lifecycle metadata ingestion.
- A universal strategy policy for whether synthetic bars are tradable.

Those choices belong either to a later data-quality layer or to individual strategy preparation code.

## Recommendation

Implement `raw` and `prev_close_zero_volume` as separate canonical dataset identities.

Keep `raw` as the default and make filled bars opt-in. This preserves the exchange-faithful dataset while allowing high-performance strategies to request a continuous matrix when that is the desired modeling assumption.
