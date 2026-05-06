# Canonical Timeframes And Fill Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend canonical exports to the boundary-safe fixed-length Binance USD-M kline intervals in this phase and add explicit raw versus filled dataset semantics.

**Architecture:** Keep ClickHouse as the authoritative 1m source and keep canonical Parquet as the reusable strategy data layer. Add `fill_policy` to the canonical dataset identity so raw bars and synthetic-filled bars are stored, validated, cataloged, and reused independently. Generate fixed-length timeframe SQL from metadata instead of hard-coded `1h`, `4h`, and `1d` branches.

**Tech Stack:** Python 3.12, `clickhouse-connect`, `pyarrow`, `pydantic`, `filelock`, `pytest`, `ruff`, Binance USD-M Futures `/fapi/v1/klines` for manual validation.

---

## Scope

Implement boundary-safe fixed-length Binance intervals:

```text
1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d
```

Do not implement `3d`, `1w`, or `1M` in this plan. `3d` buckets need explicit cross-year alignment rules, and weekly/monthly Binance boundary semantics require separate validation before becoming canonical.

Implement fill policies:

```text
raw
prev_close_zero_volume
```

The default policy is `raw`.

## File Structure

Modify existing files:

```text
src/xsignal/data/canonical_bars.py
src/xsignal/data/paths.py
src/xsignal/data/query_templates.py
src/xsignal/data/catalog.py
src/xsignal/data/canonical_export.py
src/xsignal/runs/manifest.py
tests/data/test_canonical_bars.py
tests/data/test_query_templates.py
tests/data/test_catalog.py
tests/data/test_canonical_export.py
README.md
```

Create validation helper:

```text
scripts/validate_binance_klines.py
```

Responsibilities:

- `canonical_bars.py`: timeframe metadata, fill-policy model, canonical request identity, partitioning rules, expected counts.
- `paths.py`: include `fill_policy` in canonical partition paths, locks, manifests, and catalogs.
- `query_templates.py`: build raw and filled aggregation SQL from timeframe metadata.
- `manifest.py`: persist fill-policy and synthetic-data metadata.
- `catalog.py`: validate fill-policy identity and the correct output columns.
- `canonical_export.py`: pass fill policy through CLI, query construction, manifest creation, and cache checks.
- `scripts/validate_binance_klines.py`: compare sampled raw canonical bars with Binance USD-M Futures.

## Task 1: Timeframe Metadata And Fill Policy Model

**Files:**
- Modify: `src/xsignal/data/canonical_bars.py`
- Modify: `tests/data/test_canonical_bars.py`

- [ ] **Step 1: Write failing tests for supported fixed intervals**

Append or update `tests/data/test_canonical_bars.py`:

```python
from xsignal.data.canonical_bars import (
    FILL_POLICIES,
    FIXED_TIMEFRAME_SPECS,
    FillPolicy,
    TimeframeSpec,
    expected_1m_count,
    timeframe_spec,
    validate_fill_policy,
)


def test_supported_timeframes_are_binance_fixed_length_intervals():
    assert set(FIXED_TIMEFRAME_SPECS) == {
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
    }


def test_timeframe_specs_define_clickhouse_interval_and_partition_grain():
    assert timeframe_spec("15m") == TimeframeSpec(
        name="15m",
        minutes=15,
        clickhouse_interval="INTERVAL 15 minute",
        partition_grain="month",
    )
    assert timeframe_spec("1d") == TimeframeSpec(
        name="1d",
        minutes=1440,
        clickhouse_interval="INTERVAL 1 day",
        partition_grain="year",
    )


def test_expected_bar_counts_cover_all_fixed_intervals():
    assert expected_1m_count("1m") == 1
    assert expected_1m_count("30m") == 30
    assert expected_1m_count("12h") == 720
    assert expected_1m_count("1d") == 1440


def test_fill_policy_defaults_and_validation():
    assert FILL_POLICIES == {"raw", "prev_close_zero_volume"}
    assert validate_fill_policy("raw") == "raw"
    assert validate_fill_policy("prev_close_zero_volume") == "prev_close_zero_volume"


def test_fill_policy_rejects_unknown_policy():
    with pytest.raises(ValueError, match="Unsupported fill_policy"):
        validate_fill_policy("forward_volume")


def test_three_day_timeframe_is_deferred_until_bucket_boundaries_are_validated():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        timeframe_spec("3d")
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest tests/data/test_canonical_bars.py -q
```

Expected: FAIL because `TimeframeSpec`, `FIXED_TIMEFRAME_SPECS`, `FILL_POLICIES`, and `validate_fill_policy` do not exist.

- [ ] **Step 3: Implement timeframe and fill-policy metadata**

Update `src/xsignal/data/canonical_bars.py` with these definitions:

```python
@dataclass(frozen=True)
class TimeframeSpec:
    name: str
    minutes: int
    clickhouse_interval: str
    partition_grain: str


FIXED_TIMEFRAME_SPECS = {
    "1m": TimeframeSpec("1m", 1, "INTERVAL 1 minute", "month"),
    "3m": TimeframeSpec("3m", 3, "INTERVAL 3 minute", "month"),
    "5m": TimeframeSpec("5m", 5, "INTERVAL 5 minute", "month"),
    "15m": TimeframeSpec("15m", 15, "INTERVAL 15 minute", "month"),
    "30m": TimeframeSpec("30m", 30, "INTERVAL 30 minute", "month"),
    "1h": TimeframeSpec("1h", 60, "INTERVAL 1 hour", "month"),
    "2h": TimeframeSpec("2h", 120, "INTERVAL 2 hour", "month"),
    "4h": TimeframeSpec("4h", 240, "INTERVAL 4 hour", "month"),
    "6h": TimeframeSpec("6h", 360, "INTERVAL 6 hour", "month"),
    "8h": TimeframeSpec("8h", 480, "INTERVAL 8 hour", "month"),
    "12h": TimeframeSpec("12h", 720, "INTERVAL 12 hour", "month"),
    "1d": TimeframeSpec("1d", 1440, "INTERVAL 1 day", "year"),
}

SUPPORTED_TIMEFRAMES = set(FIXED_TIMEFRAME_SPECS)
EXPECTED_1M_COUNTS = {name: spec.minutes for name, spec in FIXED_TIMEFRAME_SPECS.items()}
FILL_POLICIES = {"raw", "prev_close_zero_volume"}
FillPolicy = str


def timeframe_spec(timeframe: str) -> TimeframeSpec:
    validate_timeframe(timeframe)
    return FIXED_TIMEFRAME_SPECS[timeframe]


def validate_fill_policy(fill_policy: str) -> str:
    if fill_policy not in FILL_POLICIES:
        supported = ", ".join(sorted(FILL_POLICIES))
        raise ValueError(f"Unsupported fill_policy {fill_policy!r}; supported: {supported}")
    return fill_policy
```

- [ ] **Step 4: Update request and partition rules**

Change `CanonicalRequest`:

```python
@dataclass(frozen=True)
class CanonicalRequest:
    timeframe: str
    universe: str = "all"
    range_name: str = "full_history"
    dataset_version: str = "v1"
    fill_policy: str = "raw"

    def __post_init__(self) -> None:
        validate_timeframe(self.timeframe)
        validate_fill_policy(self.fill_policy)
        if self.universe != "all":
            raise ValueError("Only universe='all' is supported for canonical exports at project start")
        if self.range_name != "full_history":
            raise ValueError("Only range_name='full_history' is supported for canonical exports at project start")
```

Change `Partition.__post_init__` so yearly partitions are driven by metadata:

```python
spec = timeframe_spec(self.timeframe)
if spec.partition_grain == "year":
    if self.month is not None:
        raise ValueError("Yearly partitions must not include a month")
    return
if self.month is None:
    raise ValueError("Monthly partitions require a month")
```

Change `Partition.from_datetime`:

```python
if timeframe_spec(timeframe).partition_grain == "year":
    return cls(timeframe=timeframe, year=value.year)
return cls(timeframe=timeframe, year=value.year, month=value.month)
```

- [ ] **Step 5: Run model tests**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest tests/data/test_canonical_bars.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/xsignal/data/canonical_bars.py tests/data/test_canonical_bars.py
git commit -m "feat: add fixed timeframe and fill policy metadata"
```

## Task 2: Fill Policy In Paths And Manifest Identity

**Files:**
- Modify: `src/xsignal/data/paths.py`
- Modify: `src/xsignal/runs/manifest.py`
- Modify: `tests/data/test_canonical_bars.py`
- Modify: `tests/data/test_catalog.py`

- [ ] **Step 1: Write failing path tests**

Add tests in `tests/data/test_canonical_bars.py`:

```python
def test_canonical_paths_include_fill_policy(tmp_path):
    partition = Partition(timeframe="1h", year=2026, month=5)
    paths = CanonicalPaths(root=tmp_path, fill_policy="prev_close_zero_volume")

    assert paths.partition_dir(partition) == (
        tmp_path
        / "canonical_bars"
        / "timeframe=1h"
        / "fill_policy=prev_close_zero_volume"
        / "year=2026"
        / "month=05"
    )
    assert paths.catalog_path("1h") == (
        tmp_path
        / "canonical_bars"
        / "_catalog"
        / "timeframe=1h"
        / "fill_policy=prev_close_zero_volume.json"
    )
    assert paths.lock_path(partition).name == (
        "timeframe=1h__fill_policy=prev_close_zero_volume__year=2026__month=05.lock"
    )
```

- [ ] **Step 2: Write failing manifest tests**

Update the `make_manifest` helper in `tests/data/test_catalog.py` to include:

```python
"fill_policy": "raw",
"synthetic_generation_version": "none",
"synthetic_1m_count_total": 0,
"incomplete_raw_bar_count": 0,
"symbol_bound_policy": "observed_1m_bounds",
```

Add:

```python
def test_export_manifest_requires_fill_policy_identity(tmp_path):
    partition = Partition(timeframe="1h", year=2026, month=5)
    parquet_path = CanonicalPaths(root=tmp_path).published_parquet_path(partition, "abc123")

    with pytest.raises(ValueError):
        make_manifest(partition, parquet_path, fill_policy=" ")
```

- [ ] **Step 3: Run tests and verify they fail**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest tests/data/test_canonical_bars.py tests/data/test_catalog.py -q
```

Expected: FAIL because paths do not accept `fill_policy` and manifest fields do not exist.

- [ ] **Step 4: Implement path identity**

Change `CanonicalPaths` in `src/xsignal/data/paths.py`:

```python
@dataclass(frozen=True)
class CanonicalPaths:
    root: Path
    fill_policy: str = "raw"

    def __post_init__(self) -> None:
        validate_fill_policy(self.fill_policy)

    @property
    def base(self) -> Path:
        return self.root / "canonical_bars"

    def partition_dir(self, partition: Partition) -> Path:
        path = (
            self.base
            / f"timeframe={partition.timeframe}"
            / f"fill_policy={self.fill_policy}"
            / f"year={partition.year:04d}"
        )
        if partition.month is not None:
            path = path / f"month={partition.month:02d}"
        return path

    def lock_path(self, partition: Partition) -> Path:
        lock_name = (
            partition.key.replace("/", "__")
            + f"__fill_policy={self.fill_policy}.lock"
        )
        return self.base / "_locks" / lock_name

    def catalog_path(self, timeframe: str) -> Path:
        validate_timeframe(timeframe)
        return self.base / "_catalog" / f"timeframe={timeframe}" / f"fill_policy={self.fill_policy}.json"
```

Place `fill_policy` before year in the lock file name if needed to match the test exactly.

- [ ] **Step 5: Implement manifest fields**

Change `ExportManifest` in `src/xsignal/runs/manifest.py`:

```python
class ExportManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_version: str
    source_table: str
    timeframe: str
    fill_policy: str
    partition_key: str
    deduplication_mode: str
    aggregation_semantics_version: str
    synthetic_generation_version: str
    query_hash: str
    row_count: int
    parquet_path: str
    exported_at: str
    synthetic_1m_count_total: int = 0
    incomplete_raw_bar_count: int = 0
    symbol_bound_policy: str = "observed_1m_bounds"
```

Add the new string fields to `_reject_empty_strings`.

- [ ] **Step 6: Run path and manifest tests**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest tests/data/test_canonical_bars.py tests/data/test_catalog.py -q
```

Expected: PASS after updating expected paths that previously pointed directly under `timeframe=.../year=...`.

- [ ] **Step 7: Commit**

```bash
git add src/xsignal/data/paths.py src/xsignal/runs/manifest.py tests/data/test_canonical_bars.py tests/data/test_catalog.py
git commit -m "feat: include fill policy in canonical identity"
```

## Task 3: Raw Query Generalization

**Files:**
- Modify: `src/xsignal/data/query_templates.py`
- Modify: `tests/data/test_query_templates.py`

- [ ] **Step 1: Write failing tests for interval generation**

Add to `tests/data/test_query_templates.py`:

```python
@pytest.mark.parametrize(
    ("timeframe", "interval_sql", "expected_count"),
    [
        ("1m", "INTERVAL 1 minute", 1),
        ("3m", "INTERVAL 3 minute", 3),
        ("15m", "INTERVAL 15 minute", 15),
        ("2h", "INTERVAL 2 hour", 120),
        ("8h", "INTERVAL 8 hour", 480),
        ("12h", "INTERVAL 12 hour", 720),
    ],
)
def test_build_raw_aggregate_query_supports_fixed_binance_intervals(
    timeframe,
    interval_sql,
    expected_count,
):
    sql = build_aggregate_query(
        timeframe=timeframe,
        fill_policy="raw",
        start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    assert f"toStartOfInterval(k.open_time, {interval_sql}, 'UTC')" in sql
    assert f"bar_count = {expected_count} AS is_complete" in sql
    assert "0 AS synthetic_1m_count" in sql
    assert "0 AS has_synthetic" in sql
    assert "'raw' AS fill_policy" in sql
```

- [ ] **Step 2: Run query tests and verify they fail**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest tests/data/test_query_templates.py -q
```

Expected: FAIL because `build_aggregate_query` does not accept `fill_policy` and interval generation is hard-coded.

- [ ] **Step 3: Generalize raw query generation**

In `src/xsignal/data/query_templates.py`, replace `_interval_sql` with:

```python
def _interval_sql(timeframe: str) -> str:
    return timeframe_spec(timeframe).clickhouse_interval
```

Change the function signature:

```python
def build_aggregate_query(
    timeframe: str,
    start: datetime,
    end: datetime,
    fill_policy: str = "raw",
) -> str:
```

Route raw policy:

```python
validate_fill_policy(fill_policy)
if fill_policy == "prev_close_zero_volume":
    return build_filled_aggregate_query(timeframe, start, end)
return build_raw_aggregate_query(timeframe, start, end)
```

Create `build_raw_aggregate_query` using the existing SQL body and add output fields:

```sql
0 AS synthetic_1m_count,
bar_count AS expected_1m_count,
0 AS has_synthetic,
'raw' AS fill_policy
```

Use `expected_count AS expected_1m_count` rather than `bar_count AS expected_1m_count` in the final implementation:

```sql
toUInt16({expected_count}) AS expected_1m_count,
0 AS synthetic_1m_count,
0 AS has_synthetic,
'raw' AS fill_policy
```

- [ ] **Step 4: Run query tests**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest tests/data/test_query_templates.py -q
```

Expected: PASS after updating exact SQL snapshots for the new columns.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/data/query_templates.py tests/data/test_query_templates.py
git commit -m "feat: generalize raw aggregate query intervals"
```

## Task 4: Filled Query Template

**Files:**
- Modify: `src/xsignal/data/query_templates.py`
- Modify: `tests/data/test_query_templates.py`

- [ ] **Step 1: Write failing SQL-shape tests**

Add:

```python
def test_build_filled_aggregate_query_marks_synthetic_counts():
    sql = build_aggregate_query(
        timeframe="1h",
        fill_policy="prev_close_zero_volume",
        start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end=datetime(2026, 5, 2, tzinfo=timezone.utc),
    )

    assert "WITH" in sql
    assert "minute_grid" in sql
    assert "previous_real_close" in sql
    assert "synthetic_1m_count" in sql
    assert "has_synthetic" in sql
    assert "'prev_close_zero_volume' AS fill_policy" in sql
    assert "FROM xgate.klines_1m AS k FINAL" in sql
```

Add:

```python
def test_build_filled_aggregate_query_rejects_unsupported_policy():
    with pytest.raises(ValueError, match="Unsupported fill_policy"):
        build_aggregate_query(
            timeframe="1h",
            fill_policy="forward_volume",
            start=datetime(2026, 5, 1, tzinfo=timezone.utc),
            end=datetime(2026, 5, 2, tzinfo=timezone.utc),
        )
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest tests/data/test_query_templates.py -q
```

Expected: FAIL because filled query generation does not exist.

- [ ] **Step 3: Implement `build_filled_aggregate_query`**

Add a first implementation with explicit semantics:

```python
def build_filled_aggregate_query(timeframe: str, start: datetime, end: datetime) -> str:
    start_utc = _normalize_utc_datetime(start)
    end_utc = _normalize_utc_datetime(end)
    if end_utc <= start_utc:
        raise ValueError("end must be after start")

    interval = _interval_sql(timeframe)
    expected_count = expected_1m_count(timeframe)
    start_sql = _format_clickhouse_datetime(start_utc)
    end_sql = _format_clickhouse_datetime(end_utc)
    return f"""
WITH
    toDateTime('{start_sql}', 'UTC') AS start_time,
    toDateTime('{end_sql}', 'UTC') AS end_time,
    {expected_count} AS expected_count
SELECT
    symbol,
    bucket_open_time AS open_time,
    toFloat64(argMin(open, minute_open_time)) AS open,
    toFloat64(max(high)) AS high,
    toFloat64(min(low)) AS low,
    toFloat64(argMax(close, minute_open_time)) AS close,
    toFloat64(sum(volume)) AS volume,
    toFloat64(sum(quote_volume)) AS quote_volume,
    toUInt64(sum(trade_count)) AS trade_count,
    toFloat64(sum(taker_buy_volume)) AS taker_buy_volume,
    toFloat64(sum(taker_buy_quote_volume)) AS taker_buy_quote_volume,
    toUInt16(sum(is_real_1m)) AS bar_count,
    toUInt16(sum(is_synthetic_1m)) AS synthetic_1m_count,
    toUInt16(expected_count) AS expected_1m_count,
    bar_count + synthetic_1m_count = expected_count AS is_complete,
    synthetic_1m_count > 0 AS has_synthetic,
    'prev_close_zero_volume' AS fill_policy
FROM
(
    SELECT
        symbol,
        minute_open_time,
        toStartOfInterval(minute_open_time, {interval}, 'UTC') AS bucket_open_time,
        if(is_real_1m, open, previous_real_close) AS open,
        if(is_real_1m, high, previous_real_close) AS high,
        if(is_real_1m, low, previous_real_close) AS low,
        if(is_real_1m, close, previous_real_close) AS close,
        if(is_real_1m, volume, 0) AS volume,
        if(is_real_1m, quote_volume, 0) AS quote_volume,
        if(is_real_1m, trade_count, 0) AS trade_count,
        if(is_real_1m, taker_buy_volume, 0) AS taker_buy_volume,
        if(is_real_1m, taker_buy_quote_volume, 0) AS taker_buy_quote_volume,
        is_real_1m,
        NOT is_real_1m AS is_synthetic_1m
    FROM
    (
        SELECT
            g.symbol,
            g.minute_open_time,
            r.open,
            r.high,
            r.low,
            r.close,
            r.volume,
            r.quote_volume,
            r.trade_count,
            r.taker_buy_volume,
            r.taker_buy_quote_volume,
            r.close IS NOT NULL AS is_real_1m,
            anyLast(r.close) OVER (
                PARTITION BY g.symbol
                ORDER BY g.minute_open_time
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS previous_real_close
        FROM minute_grid AS g
        LEFT JOIN raw_1m AS r
            ON g.symbol = r.symbol
           AND g.minute_open_time = r.open_time
    )
    WHERE previous_real_close IS NOT NULL
)
GROUP BY
    symbol,
    bucket_open_time
ORDER BY
    bucket_open_time,
    symbol
""".strip()
```

Then add the CTE definitions before the final `SELECT`:

```sql
raw_1m AS
(
    SELECT *
    FROM xgate.klines_1m AS k FINAL
    WHERE k.open_time >= start_time
      AND k.open_time < end_time
),
symbols AS
(
    SELECT DISTINCT symbol
    FROM raw_1m
),
minute_grid AS
(
    SELECT
        symbol,
        addMinutes(start_time, minute_offset) AS minute_open_time
    FROM symbols
    ARRAY JOIN range(0, dateDiff('minute', start_time, end_time)) AS minute_offset
)
```

This first implementation fills gaps inside the exported partition for symbols with at least one real row in that partition. It does not fill a symbol that has no real row in the partition.

- [ ] **Step 4: Run query tests**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest tests/data/test_query_templates.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/data/query_templates.py tests/data/test_query_templates.py
git commit -m "feat: add filled aggregate query template"
```

## Task 5: Catalog Validation For Raw And Filled Columns

**Files:**
- Modify: `src/xsignal/data/canonical_bars.py`
- Modify: `src/xsignal/data/catalog.py`
- Modify: `tests/data/test_catalog.py`

- [ ] **Step 1: Write failing catalog tests for fill policy mismatch**

Add:

```python
def test_catalog_treats_fill_policy_mismatch_as_stale(tmp_path):
    paths = CanonicalPaths(root=tmp_path, fill_policy="prev_close_zero_volume")
    catalog = Catalog(paths=paths)
    partition = Partition(timeframe="1h", year=2026, month=5)
    parquet_path = paths.published_parquet_path(partition, "abc123")
    write_canonical_parquet(parquet_path, fill_policy="prev_close_zero_volume")
    manifest = make_manifest(
        partition,
        parquet_path,
        fill_policy="raw",
        synthetic_generation_version="none",
    )
    paths.manifest_path(partition).write_text(manifest.model_dump_json(indent=2))

    assert catalog.status(partition, dataset_version="v1") == PartitionStatus.STALE
```

Add:

```python
def test_catalog_accepts_filled_output_columns(tmp_path):
    paths = CanonicalPaths(root=tmp_path, fill_policy="prev_close_zero_volume")
    catalog = Catalog(paths=paths)
    partition = Partition(timeframe="1h", year=2026, month=5)
    parquet_path = paths.published_parquet_path(partition, "abc123")
    write_canonical_parquet(parquet_path, fill_policy="prev_close_zero_volume")
    manifest = make_manifest(
        partition,
        parquet_path,
        fill_policy="prev_close_zero_volume",
        synthetic_generation_version="prev-close-zero-volume-v1",
        synthetic_1m_count_total=2,
        incomplete_raw_bar_count=1,
    )
    paths.manifest_path(partition).write_text(manifest.model_dump_json(indent=2))

    assert catalog.status(partition, dataset_version="v1") == PartitionStatus.COMPLETE
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest tests/data/test_catalog.py -q
```

Expected: FAIL because catalog does not validate fill policy or filled columns.

- [ ] **Step 3: Define canonical columns by fill policy**

In `src/xsignal/data/canonical_bars.py`, replace `CANONICAL_BAR_COLUMNS` with:

```python
BASE_CANONICAL_BAR_COLUMNS = (
    "symbol",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "bar_count",
    "synthetic_1m_count",
    "expected_1m_count",
    "is_complete",
    "has_synthetic",
    "fill_policy",
)

CANONICAL_BAR_COLUMNS_BY_FILL_POLICY = {
    "raw": BASE_CANONICAL_BAR_COLUMNS,
    "prev_close_zero_volume": BASE_CANONICAL_BAR_COLUMNS,
}


def canonical_bar_columns(fill_policy: str) -> tuple[str, ...]:
    validate_fill_policy(fill_policy)
    return CANONICAL_BAR_COLUMNS_BY_FILL_POLICY[fill_policy]
```

- [ ] **Step 4: Validate manifest fill fields**

Update `Catalog.status_for_manifest`:

```python
if manifest.fill_policy != self.paths.fill_policy:
    return PartitionStatus.STALE
if manifest.fill_policy == "raw" and manifest.synthetic_generation_version != "none":
    return PartitionStatus.STALE
if manifest.fill_policy == "prev_close_zero_volume" and (
    manifest.synthetic_generation_version != "prev-close-zero-volume-v1"
):
    return PartitionStatus.STALE
if manifest.synthetic_1m_count_total < 0:
    return PartitionStatus.STALE
if manifest.incomplete_raw_bar_count < 0:
    return PartitionStatus.STALE
```

Change metadata column validation:

```python
if tuple(parquet_metadata.schema.names) != canonical_bar_columns(manifest.fill_policy):
    return PartitionStatus.STALE
```

- [ ] **Step 5: Update catalog entries**

Include fill metadata in `Catalog.mark_complete` entry:

```python
entry = {
    "dataset_version": manifest.dataset_version,
    "fill_policy": manifest.fill_policy,
    "row_count": manifest.row_count,
    "query_hash": manifest.query_hash,
    "parquet_path": manifest.parquet_path,
    "exported_at": manifest.exported_at,
    "synthetic_generation_version": manifest.synthetic_generation_version,
    "synthetic_1m_count_total": manifest.synthetic_1m_count_total,
    "incomplete_raw_bar_count": manifest.incomplete_raw_bar_count,
}
```

- [ ] **Step 6: Run catalog tests**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest tests/data/test_catalog.py -q
```

Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add src/xsignal/data/canonical_bars.py src/xsignal/data/catalog.py tests/data/test_catalog.py
git commit -m "feat: validate canonical fill policy catalogs"
```

## Task 6: Export Orchestration And CLI Fill Policy

**Files:**
- Modify: `src/xsignal/data/canonical_export.py`
- Modify: `tests/data/test_canonical_export.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing orchestration tests**

Add:

```python
def test_ensure_passes_fill_policy_to_query_and_paths(tmp_path):
    exporter = FakeExporter()
    paths = CanonicalPaths(root=tmp_path, fill_policy="prev_close_zero_volume")
    partition = Partition(timeframe="1h", year=2026, month=5)

    result = ensure_canonical_bars(
        request=CanonicalRequest(timeframe="1h", fill_policy="prev_close_zero_volume"),
        paths=paths,
        partitions=[partition],
        exporter=exporter,
        now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
    )

    assert result.request.fill_policy == "prev_close_zero_volume"
    assert "prev_close_zero_volume" in str(paths.manifest_path(partition))
    assert "'prev_close_zero_volume' AS fill_policy" in exporter.calls[0][0]
```

Add CLI test:

```python
def test_cli_accepts_fill_policy_for_single_partition(tmp_path, monkeypatch):
    exporter = FakeExporter()

    class FakeClickHouseClient:
        def __init__(self, _config):
            pass

        def write_parquet(self, sql, path):
            return exporter.export(sql, path)

    monkeypatch.setattr("xsignal.data.canonical_export.ClickHouseClient", FakeClickHouseClient)

    exit_code = main(
        [
            "ensure",
            "--timeframe",
            "1h",
            "--fill-policy",
            "prev_close_zero_volume",
            "--year",
            "2026",
            "--month",
            "5",
            "--root",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert (
        tmp_path
        / "canonical_bars"
        / "timeframe=1h"
        / "fill_policy=prev_close_zero_volume"
        / "year=2026"
        / "month=05"
        / "manifest.json"
    ).is_file()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest tests/data/test_canonical_export.py -q
```

Expected: FAIL because CLI and orchestration ignore fill policy.

- [ ] **Step 3: Wire request, paths, query, and manifest**

Update `ensure_canonical_bars`:

```python
if paths.fill_policy != request.fill_policy:
    raise ValueError("CanonicalPaths fill_policy must match request fill_policy")
sql = build_aggregate_query(
    request.timeframe,
    start,
    end,
    fill_policy=request.fill_policy,
)
```

Set manifest fields:

```python
synthetic_generation_version = (
    "none"
    if request.fill_policy == "raw"
    else "prev-close-zero-volume-v1"
)
manifest = ExportManifest(
    dataset_version=request.dataset_version,
    source_table=CLICKHOUSE_SOURCE_TABLE,
    timeframe=request.timeframe,
    fill_policy=request.fill_policy,
    partition_key=partition.key,
    deduplication_mode="FINAL",
    aggregation_semantics_version="ohlcv-v2",
    synthetic_generation_version=synthetic_generation_version,
    query_hash=query_hash(sql),
    row_count=row_count,
    parquet_path=str(target_parquet),
    exported_at=clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
    synthetic_1m_count_total=0,
    incomplete_raw_bar_count=0,
    symbol_bound_policy="observed_1m_bounds",
)
```

In the same task, read Parquet metadata after export and compute synthetic totals:

```python
def _manifest_quality_counts(parquet_path: Path) -> tuple[int, int]:
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    table = pq.read_table(
        parquet_path,
        columns=["synthetic_1m_count", "bar_count", "expected_1m_count"],
    )
    synthetic_total = pc.sum(table["synthetic_1m_count"]).as_py() or 0
    incomplete_total = pc.sum(
        pc.cast(pc.less(table["bar_count"], table["expected_1m_count"]), "int64")
    ).as_py() or 0
    return int(synthetic_total), int(incomplete_total)
```

Use these values before creating the manifest.

- [ ] **Step 4: Add CLI option**

Update `_ensure_command`:

```python
request = CanonicalRequest(timeframe=args.timeframe, fill_policy=args.fill_policy)
paths = CanonicalPaths(root=Path(args.root), fill_policy=args.fill_policy)
```

Update parser:

```python
ensure_parser.add_argument("--fill-policy", default="raw")
```

- [ ] **Step 5: Run canonical export tests**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest tests/data/test_canonical_export.py -q
```

Expected: PASS.

- [ ] **Step 6: Update README**

Add:

```markdown
Use raw exchange-derived aggregation by default:

```bash
xsignal-export ensure --timeframe 1h --fill-policy raw
```

Use flat previous-close zero-volume filling when a strategy needs continuous bars:

```bash
xsignal-export ensure --timeframe 1h --fill-policy prev_close_zero_volume
```
```

- [ ] **Step 7: Commit**

```bash
git add src/xsignal/data/canonical_export.py tests/data/test_canonical_export.py README.md
git commit -m "feat: wire fill policy through canonical export"
```

## Task 7: Full-History Partition Discovery For New Timeframes

**Files:**
- Modify: `src/xsignal/data/canonical_export.py`
- Modify: `tests/data/test_canonical_export.py`

- [ ] **Step 1: Write failing partition discovery tests**

Add:

```python
def test_partitions_for_full_history_uses_months_for_12h_timeframe():
    partitions = partitions_for_full_history(
        timeframe="12h",
        start=datetime(2026, 4, 10, tzinfo=timezone.utc),
        end=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    assert partitions == [
        Partition(timeframe="12h", year=2026, month=4),
        Partition(timeframe="12h", year=2026, month=5),
    ]


def test_partitions_for_full_history_uses_years_for_1d_timeframe():
    partitions = partitions_for_full_history(
        timeframe="1d",
        start=datetime(2025, 12, 31, tzinfo=timezone.utc),
        end=datetime(2026, 5, 6, tzinfo=timezone.utc),
    )

    assert partitions == [
        Partition(timeframe="1d", year=2025),
        Partition(timeframe="1d", year=2026),
    ]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest tests/data/test_canonical_export.py -q
```

Expected: FAIL if partition discovery is still hard-coded instead of using timeframe metadata.

- [ ] **Step 3: Use timeframe metadata for partition grain**

Update:

```python
if timeframe_spec(timeframe).partition_grain == "year":
    ...
```

Keep the exact-year-boundary exclusion logic from the current `1d` implementation. `3d`, `1w`, and `1M` stay unsupported until bucket-boundary semantics are validated.

- [ ] **Step 4: Run partition tests**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest tests/data/test_canonical_export.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/data/canonical_export.py tests/data/test_canonical_export.py
git commit -m "feat: partition full history by timeframe metadata"
```

## Task 8: Binance Raw Kline Validation Script

**Files:**
- Create: `scripts/validate_binance_klines.py`
- Modify: `README.md`

- [ ] **Step 1: Write the script**

Create `scripts/validate_binance_klines.py`:

```python
from __future__ import annotations

import argparse
import json
import urllib.parse
import urllib.request
from decimal import Decimal
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq


FIELD_MAP = {
    "open": 1,
    "high": 2,
    "low": 3,
    "close": 4,
    "volume": 5,
    "quote_volume": 7,
    "trade_count": 8,
    "taker_buy_volume": 9,
    "taker_buy_quote_volume": 10,
}


def fetch_binance(symbol: str, interval: str, start_ms: int, limit: int) -> list[list[object]]:
    query = urllib.parse.urlencode(
        {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_ms,
            "limit": limit,
        }
    )
    url = f"https://fapi.binance.com/fapi/v1/klines?{query}"
    with urllib.request.urlopen(url, timeout=20) as response:
        return json.loads(response.read().decode("utf-8"))


def compare_rows(local_rows: list[dict[str, object]], remote_rows: list[list[object]], tolerance: Decimal):
    mismatches = []
    for local, remote in zip(local_rows, remote_rows):
        if int(local["open_time"]) != int(remote[0]) // 1000:
            mismatches.append(
                {
                    "open_time": local["open_time"],
                    "field": "open_time",
                    "local": local["open_time"],
                    "remote": int(remote[0]) // 1000,
                }
            )
            continue
        for field, index in FIELD_MAP.items():
            if field == "trade_count":
                if int(local[field]) != int(remote[index]):
                    mismatches.append(
                        {
                            "open_time": local["open_time"],
                            "field": field,
                            "local": local[field],
                            "remote": remote[index],
                        }
                    )
                continue
            diff = abs(Decimal(str(local[field])) - Decimal(str(remote[index])))
            if diff > tolerance:
                mismatches.append(
                    {
                        "open_time": local["open_time"],
                        "field": field,
                        "local": local[field],
                        "remote": remote[index],
                        "diff": str(diff),
                    }
                )
    return mismatches


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate raw canonical bars against Binance USD-M klines")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--timeframe", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--tolerance", default="0.000001")
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text())
    if manifest["fill_policy"] != "raw":
        raise SystemExit("Binance raw comparison requires fill_policy=raw")

    table = pq.read_table(manifest["parquet_path"])
    filtered = table.filter(
        pc.and_(
            pc.equal(table["symbol"], args.symbol),
            pc.equal(table["is_complete"], 1),
        )
    ).slice(0, args.limit)
    local_rows = filtered.to_pylist()
    if not local_rows:
        raise SystemExit("No complete local rows found for requested symbol")

    remote_rows = fetch_binance(
        args.symbol,
        args.timeframe,
        int(local_rows[0]["open_time"]) * 1000,
        len(local_rows),
    )
    mismatches = compare_rows(local_rows, remote_rows, Decimal(args.tolerance))
    result = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "checked": len(local_rows),
        "mismatches": len(mismatches),
        "first_mismatches": mismatches[:5],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 1 if mismatches else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run ruff on the script**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m ruff check scripts/validate_binance_klines.py
```

Expected: PASS.

- [ ] **Step 3: Document manual use**

Add to README:

```markdown
Validate a raw canonical partition against Binance USD-M Futures:

```bash
python scripts/validate_binance_klines.py \
  --manifest data/canonical_bars/timeframe=1h/fill_policy=raw/year=2020/month=02/manifest.json \
  --symbol BTCUSDT \
  --timeframe 1h \
  --limit 10
```
```

- [ ] **Step 4: Commit**

```bash
git add scripts/validate_binance_klines.py README.md
git commit -m "chore: add binance raw kline validator"
```

## Task 9: Full Local Verification

**Files:**
- No code changes unless verification reveals a bug.

- [ ] **Step 1: Run ruff**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m ruff check .
```

Expected:

```text
All checks passed!
```

- [ ] **Step 2: Run pytest**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m pytest -q
```

Expected: all tests pass.

- [ ] **Step 3: Inspect status**

Run:

```bash
git status --short
```

Expected: clean working tree.

## Task 10: Server Smoke Validation

**Files:**
- No code changes unless validation reveals a bug.

- [ ] **Step 1: Ensure ClickHouse tunnel**

Run:

```bash
ssh -f -N -L 8123:127.0.0.1:8123 X || true
lsof -nP -iTCP:8123 -sTCP:LISTEN || true
ssh X 'docker ps --format "{{.Names}} {{.Status}}" | grep xgate-clickhouse'
```

Expected: local `8123` listener exists and `xgate-clickhouse` is healthy.

- [ ] **Step 2: Export a raw new interval partition**

Run:

```bash
rm -rf data/canonical_bars/timeframe=15m/fill_policy=raw/year=2020/month=02
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m xsignal.data.canonical_export ensure \
  --timeframe 15m \
  --fill-policy raw \
  --year 2020 \
  --month 2 \
  --root data
```

Expected: `CanonicalDataset(...)`.

- [ ] **Step 3: Export a filled partition**

Run:

```bash
rm -rf data/canonical_bars/timeframe=1h/fill_policy=prev_close_zero_volume/year=2020/month=02
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 -m xsignal.data.canonical_export ensure \
  --timeframe 1h \
  --fill-policy prev_close_zero_volume \
  --year 2020 \
  --month 2 \
  --root data
```

Expected: `CanonicalDataset(...)`.

- [ ] **Step 4: Inspect manifests**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 - <<'PY'
from pathlib import Path
import json
for path in [
    Path("data/canonical_bars/timeframe=15m/fill_policy=raw/year=2020/month=02/manifest.json"),
    Path("data/canonical_bars/timeframe=1h/fill_policy=prev_close_zero_volume/year=2020/month=02/manifest.json"),
]:
    manifest = json.loads(path.read_text())
    print(path)
    print({
        "fill_policy": manifest["fill_policy"],
        "row_count": manifest["row_count"],
        "synthetic_1m_count_total": manifest["synthetic_1m_count_total"],
        "incomplete_raw_bar_count": manifest["incomplete_raw_bar_count"],
        "parquet_path": manifest["parquet_path"],
    })
PY
```

Expected: raw manifest has `fill_policy=raw`; filled manifest has `fill_policy=prev_close_zero_volume`.

- [ ] **Step 5: Validate raw partition against Binance**

Run:

```bash
/Users/wukong/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 scripts/validate_binance_klines.py \
  --manifest data/canonical_bars/timeframe=15m/fill_policy=raw/year=2020/month=02/manifest.json \
  --symbol BTCUSDT \
  --timeframe 15m \
  --limit 10
```

Expected:

```json
{
  "checked": 10,
  "mismatches": 0
}
```

- [ ] **Step 6: Confirm cache reuse is idempotent**

Run the raw export command from Step 2 twice and compare `stat` and `shasum` for:

```text
manifest.json
bars.<run-id>.parquet
_catalog/timeframe=15m/fill_policy=raw.json
```

Expected: repeated `ensure` does not rewrite existing complete files.

## Self-Review

Spec coverage:

- Fixed Binance intervals: Tasks 1, 3, 7, 10.
- Raw default behavior: Tasks 1, 3, 6.
- `prev_close_zero_volume` identity: Tasks 1, 2, 4, 5, 6.
- Separate raw and filled storage/catalog identity: Tasks 2, 5, 6.
- Manifest fill metadata: Tasks 2, 5, 6, 10.
- Binance raw validation: Task 8 and Task 10.
- Strategy mask columns: Tasks 3, 4, 5.

Completeness scan:

- No incomplete marker text remains in the plan.

Type consistency:

- `fill_policy` is consistently the request, path, manifest, catalog, and CLI field.
- `prev_close_zero_volume` is the public policy string.
- `prev-close-zero-volume-v1` is the manifest synthetic generation version.
