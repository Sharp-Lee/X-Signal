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

Export or reuse canonical bars for one explicit partition:

```bash
xsignal-export ensure --timeframe 1h --year 2026 --month 5 --root data
```

The export layer writes:

- `data/canonical_bars/timeframe=1h/year=2026/month=05/bars.<run-id>.parquet`
- `data/canonical_bars/timeframe=1h/year=2026/month=05/manifest.json`
- `data/canonical_bars/_catalog/timeframe=1h.json`

Strategies should call `ensure_canonical_bars` before reading canonical Parquet.
When a requested partition is already complete, the export layer returns the
local Parquet dataset without opening a ClickHouse connection.
The manifest is the atomic pointer to the immutable Parquet file for that
partition.
