# Strategy Backtest Architecture Design

Date: 2026-05-06

## Purpose

X-Signal is designed for fast strategy research, not for a general-purpose backtesting platform. Each new strategy idea should be allowed to build a custom high-performance backtest path while sharing only the data and experiment infrastructure that improves speed, correctness, and reproducibility.

The guiding rule is:

> Keep raw data authoritative in ClickHouse, precompute only high-reuse standard bars, then let each strategy build its own optimal in-memory layout and backtest kernel.

## Current Data Context

The current production data source is ClickHouse on the cloud server.

- Database: `xgate`
- Main table: `xgate.klines_1m`
- Engine: `ReplacingMergeTree(version)`
- Partition key: `toYYYYMM(open_time)`
- Sort key: `(symbol, open_time)`
- Rows observed: about 606 million
- Compressed size observed: about 33 GiB
- Symbols observed: about 533
- Time range observed: `2020-01-01 00:00:00` to `2026-05-06 11:36:00 UTC`

The table contains recent duplicate `(symbol, open_time)` keys because `ReplacingMergeTree` does not guarantee query-time deduplication unless `FINAL` or an explicit version selection is used. Any canonical export or strategy dataset must therefore define a deterministic deduplication rule.

## Non-Goals

This project will not start by building:

- A universal backtest engine
- A strategy plugin framework
- A generic event simulator
- A generic order management abstraction
- A large feature store for every possible indicator or timeframe

These abstractions can be introduced later only when repeated strategy work proves they remove real duplication without slowing research.

## Architecture

The system has four layers:

```text
ClickHouse raw 1m data
  -> canonical standard bars
  -> strategy-specific prepared arrays
  -> custom high-performance backtest kernels
```

### 1. Raw Data Layer

ClickHouse remains the authoritative source for raw 1m kline data. It should be optimized for ingestion, repair, deduplication, and coarse extraction, not for running every backtest loop directly.

Responsibilities:

- Store raw 1m klines.
- Preserve source and version metadata.
- Support deterministic extraction for historical research.
- Provide standard aggregation queries for canonical bars.

Deduplication rule:

- For correctness, canonical exports should use `FINAL` or equivalent `argMax(..., version)` logic.
- Exports should record the exact query mode in a manifest.
- Recent mutable partitions should be refreshable.
- Older sealed partitions can be treated as stable once ingestion is complete.

### 2. Canonical Bars Layer

The project should maintain global reusable Parquet datasets for high-reuse timeframes:

- `1h`
- `4h`
- `1d`

These are worth precomputing because many strategies will test across hundreds of symbols and these timeframes. Precomputing avoids repeated database scans and standardizes bar semantics across strategies.

Initial layout:

```text
data/canonical_bars/
  timeframe=1h/
    year=2025/
      month=01/
        bars.parquet
  timeframe=4h/
    year=2025/
      month=01/
        bars.parquet
  timeframe=1d/
    year=2025/
      bars.parquet
```

Required columns:

- `symbol`
- `open_time`
- `open`
- `high`
- `low`
- `close`
- `volume`
- `quote_volume`
- `trade_count`
- `taker_buy_volume`
- `taker_buy_quote_volume`
- optional data quality fields such as `bar_count` and `is_complete`

Numeric representation:

- ClickHouse can store prices and volumes as `Decimal`.
- Canonical Parquet should store research bars as `Float64` unless a strategy requires exact decimal arithmetic.
- Backtest arrays may downcast to `Float32` for speed and memory after strategy-level validation.

Aggregation semantics:

- `open`: first 1m open in the target interval
- `high`: maximum high in the target interval
- `low`: minimum low in the target interval
- `close`: last 1m close in the target interval
- volume fields: sum over the interval
- `trade_count`: sum over the interval
- `bar_count`: number of 1m bars present
- `is_complete`: whether the interval has the expected number of 1m bars

The canonical layer is not a full feature store. It should not contain strategy-specific indicators unless several strategies repeatedly need exactly the same feature and semantics.

### 3. Strategy Preparation Layer

Each strategy owns its own preparation code. This layer reads canonical bars and writes the fastest format for that strategy's access pattern.

Example:

```text
strategies/
  cross_section_momentum/
    prepare.py
    backtest.py
    config.yaml
    cache/
      close_1h.npy
      volume_1h.npy
      tradable_mask_1h.npy
      symbols.json
      times.npy
```

Strategy preparation may create:

- Dense arrays such as `close[T, N]`, `volume[T, N]`, and `mask[T, N]`
- Symbol-major arrays such as `close[N, T]`
- Sparse event tables
- Memory-mapped arrays for large parameter sweeps
- Precomputed ranks, rolling windows, signals, or eligibility masks

The strategy decides the layout. A cross-sectional strategy over 300 symbols likely wants time-major arrays. A single-symbol trend strategy may prefer symbol-major or one file per symbol.

### 4. Backtest Kernel Layer

Each strategy writes a custom backtest kernel optimized for its logic. The project should encourage simple, explicit kernels over a shared general engine.

Recommended execution options:

- NumPy vectorization for simple matrix operations
- Numba for Python-native loops over time and symbols
- Rust or C++ extensions for stateful, branch-heavy, or very large parameter sweeps
- Polars or DuckDB only for preparation and analysis, not hot loops

The only shared contract is the output shape.

Required output artifacts:

- Equity curve
- Position history or compressed position events
- Trade log if the strategy has discrete trades
- Summary metrics
- Run manifest

## Experiment Manifest

Every run should write a manifest so custom backtests remain reproducible.

Suggested fields:

- run id
- strategy name
- strategy git commit
- strategy config hash
- canonical dataset version
- ClickHouse source table
- extraction query or query hash
- deduplication mode
- timeframe
- symbol universe
- start time and end time
- fee and slippage assumptions
- preparation cache paths
- output paths
- runtime environment

This manifest is the main shared discipline that lets the project stay flexible without becoming messy.

## Refresh Model

Canonical Parquet should not be treated as a one-time export forever. It should be versioned and refreshable.

Recommended policy:

- Historical sealed months: export once, verify, then keep stable.
- Current month: refresh regularly because late or duplicate records may arrive.
- When ingestion bugs are fixed: rebuild affected partitions.
- When aggregation semantics change: create a new dataset version rather than silently replacing old data.

## Performance Strategy

ClickHouse is fast enough for extraction and standard aggregation. It should do heavy columnar scans and group-by work. It should not be in the inner loop of strategy parameter searches.

Parquet is the reusable storage layer. It avoids repeated ClickHouse queries and makes strategy development portable.

Numpy, memory maps, Arrow arrays, Numba, Rust, or C++ are the hot-loop layer. This is where ultra-high performance should be achieved.

Expected workflow for a new strategy:

```text
1. Select canonical timeframe and universe.
2. Generate strategy-specific arrays.
3. Run parameter sweeps on arrays.
4. Save manifest and reports.
5. Promote only truly reusable logic back into shared utilities.
```

## Error Handling And Data Quality

The system should fail loudly when data quality affects a backtest.

Preparation should check:

- Missing bars
- Duplicate bars before deduplication
- Incomplete 1h, 4h, or 1d intervals
- Unexpected symbol gaps
- Non-positive prices
- Suspicious zero-volume periods
- Timezone assumptions
- Universe membership and delisting assumptions

Each strategy can choose whether to exclude, forward-fill, mask, or penalize missing data, but that choice must be recorded in the run manifest.

## Testing Approach

Testing should focus on correctness of shared data semantics and strategy-specific kernels.

Shared tests:

- Deduplication returns one row per `(symbol, open_time)`.
- Aggregated OHLCV bars match hand-calculated fixtures.
- `bar_count` and `is_complete` are correct.
- Canonical export partition paths are deterministic.
- Manifests include required fields.

Strategy tests:

- Prepared array shapes match expected time and symbol axes.
- Masks correctly represent missing or untradable bars.
- The backtest kernel matches a small hand-calculated scenario.
- Parameter sweeps are deterministic.

## Recommended Initial Build

The first implementation should create only the minimum shared substrate:

```text
data/
  canonical_bars/

strategies/
  template_strategy/
    prepare.py
    backtest.py
    config.yaml

src/
  xsignal/
    data/
      clickhouse.py
      canonical_bars.py
    runs/
      manifest.py
    metrics/
      summary.py
```

After that, each real strategy should be implemented as its own folder with its own preparation and backtest path.

## Design Decision

The project should precompute global `1h`, `4h`, and `1d` canonical Parquet bars from ClickHouse 1m data because these timeframes are expected to be reused across many strategies and hundreds of symbols.

The project should not build a universal backtest engine. It should standardize data extraction, canonical bars, manifests, and metrics, while leaving every strategy free to implement its own high-performance prepared arrays and backtest kernel.
