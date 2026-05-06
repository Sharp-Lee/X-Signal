# Canonical Export Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the shared `ensure_canonical_bars` data foundation that checks, exports, validates, catalogs, and returns deduplicated canonical Parquet bars for `1h`, `4h`, and `1d`.

**Architecture:** ClickHouse remains the raw 1m source. The export layer owns deduplication, aggregation, partition manifests, catalog updates, locks, and atomic publication. Strategies call this layer before building their own high-performance arrays.

**Tech Stack:** Python 3.12, `clickhouse-connect`, `pyarrow`, `pydantic`, `filelock`, `pytest`, `ruff`.

---

## File Structure

Create the project as a small Python package. Keep the shared data substrate focused and independent from strategy code.

```text
pyproject.toml
README.md
.gitignore
src/
  xsignal/
    __init__.py
    data/
      __init__.py
      canonical_bars.py
      canonical_export.py
      catalog.py
      clickhouse.py
      locks.py
      paths.py
      query_templates.py
    runs/
      __init__.py
      manifest.py
tests/
  data/
    test_canonical_bars.py
    test_catalog.py
    test_locks.py
    test_query_templates.py
    test_canonical_export.py
```

Responsibilities:

- `canonical_bars.py`: timeframe definitions, aggregation expectations, partition model.
- `paths.py`: deterministic filesystem paths for canonical data, manifests, temp files, and locks.
- `catalog.py`: read/write catalog JSON and decide whether a partition is complete.
- `locks.py`: filesystem lock wrapper for partition exports.
- `query_templates.py`: ClickHouse SQL builder for deduplicated aggregation.
- `clickhouse.py`: ClickHouse connection and query execution helpers.
- `canonical_export.py`: public `ensure_canonical_bars` orchestration.
- `runs/manifest.py`: shared manifest models used by export and strategy runs.

## Task 1: Project Scaffold

**Files:**
- Create: `pyproject.toml`
- Create: `.gitignore`
- Create: `README.md`
- Create: `src/xsignal/__init__.py`
- Create: `src/xsignal/data/__init__.py`
- Create: `src/xsignal/runs/__init__.py`

- [ ] **Step 1: Create package metadata**

Write `pyproject.toml`:

```toml
[project]
name = "xsignal"
version = "0.1.0"
description = "High-performance custom strategy research foundation"
requires-python = ">=3.12"
dependencies = [
  "clickhouse-connect>=0.8.0",
  "filelock>=3.15.0",
  "pyarrow>=16.0.0",
  "pydantic>=2.7.0",
]

[project.optional-dependencies]
dev = [
  "pytest>=8.2.0",
  "ruff>=0.5.0",
]

[project.scripts]
xsignal-export = "xsignal.data.canonical_export:main"

[build-system]
requires = ["setuptools>=69.0"]
build-backend = "setuptools.build_meta"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[tool.ruff]
line-length = 100
target-version = "py312"
```

- [ ] **Step 2: Add ignore rules**

Write `.gitignore`:

```gitignore
.venv/
__pycache__/
.pytest_cache/
.ruff_cache/
*.pyc
data/canonical_bars/
data/tmp/
runs/
```

- [ ] **Step 3: Add a short README**

Write `README.md`:

```markdown
# X-Signal

X-Signal is a strategy research project optimized for custom high-performance backtests.

The shared data foundation exposes canonical deduplicated bars through:

```bash
xsignal-export ensure --timeframe 1h
```

Strategies should call the canonical export layer first, then build their own arrays and backtest kernels.
```

- [ ] **Step 4: Add empty package markers**

Create empty files:

```text
src/xsignal/__init__.py
src/xsignal/data/__init__.py
src/xsignal/runs/__init__.py
```

- [ ] **Step 5: Verify package scaffold**

Run:

```bash
python3 -m pip install -e ".[dev]"
pytest -q
```

Expected:

```text
no tests ran
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml .gitignore README.md src/xsignal tests
git commit -m "chore: scaffold xsignal package"
```

## Task 2: Timeframe And Partition Models

**Files:**
- Create: `src/xsignal/data/canonical_bars.py`
- Test: `tests/data/test_canonical_bars.py`

- [ ] **Step 1: Write failing tests**

Write `tests/data/test_canonical_bars.py`:

```python
from datetime import datetime, timezone

import pytest

from xsignal.data.canonical_bars import (
    SUPPORTED_TIMEFRAMES,
    CanonicalRequest,
    Partition,
    expected_1m_count,
)


def test_supported_timeframes_are_explicit():
    assert SUPPORTED_TIMEFRAMES == {"1h", "4h", "1d"}


def test_canonical_request_defaults_to_all_symbols_full_history():
    request = CanonicalRequest(timeframe="1h")

    assert request.timeframe == "1h"
    assert request.universe == "all"
    assert request.range_name == "full_history"


def test_rejects_unsupported_timeframe():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        CanonicalRequest(timeframe="15m")


def test_expected_bar_counts():
    assert expected_1m_count("1h") == 60
    assert expected_1m_count("4h") == 240
    assert expected_1m_count("1d") == 1440


def test_partition_from_datetime():
    partition = Partition.from_datetime(
        timeframe="1h",
        value=datetime(2026, 5, 6, 11, 36, tzinfo=timezone.utc),
    )

    assert partition.timeframe == "1h"
    assert partition.year == 2026
    assert partition.month == 5
    assert partition.key == "timeframe=1h/year=2026/month=05"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/data/test_canonical_bars.py -q
```

Expected: FAIL because `xsignal.data.canonical_bars` does not exist.

- [ ] **Step 3: Implement timeframe and partition models**

Write `src/xsignal/data/canonical_bars.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


SUPPORTED_TIMEFRAMES = {"1h", "4h", "1d"}
EXPECTED_1M_COUNTS = {
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


def validate_timeframe(timeframe: str) -> str:
    if timeframe not in SUPPORTED_TIMEFRAMES:
        supported = ", ".join(sorted(SUPPORTED_TIMEFRAMES))
        raise ValueError(f"Unsupported timeframe {timeframe!r}; supported: {supported}")
    return timeframe


def expected_1m_count(timeframe: str) -> int:
    return EXPECTED_1M_COUNTS[validate_timeframe(timeframe)]


@dataclass(frozen=True)
class CanonicalRequest:
    timeframe: str
    universe: str = "all"
    range_name: str = "full_history"
    dataset_version: str = "v1"

    def __post_init__(self) -> None:
        validate_timeframe(self.timeframe)
        if self.universe != "all":
            raise ValueError("Only universe='all' is supported for canonical exports at project start")
        if self.range_name != "full_history":
            raise ValueError("Only range_name='full_history' is supported for canonical exports at project start")


@dataclass(frozen=True)
class Partition:
    timeframe: str
    year: int
    month: int | None = None

    @classmethod
    def from_datetime(cls, timeframe: str, value: datetime) -> "Partition":
        validate_timeframe(timeframe)
        if timeframe == "1d":
            return cls(timeframe=timeframe, year=value.year)
        return cls(timeframe=timeframe, year=value.year, month=value.month)

    @property
    def key(self) -> str:
        if self.month is None:
            return f"timeframe={self.timeframe}/year={self.year:04d}"
        return f"timeframe={self.timeframe}/year={self.year:04d}/month={self.month:02d}"
```

- [ ] **Step 4: Verify tests pass**

Run:

```bash
pytest tests/data/test_canonical_bars.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/data/canonical_bars.py tests/data/test_canonical_bars.py
git commit -m "feat: add canonical bar models"
```

## Task 3: Canonical Paths

**Files:**
- Create: `src/xsignal/data/paths.py`
- Test: `tests/data/test_canonical_bars.py`

- [ ] **Step 1: Extend tests**

Append to `tests/data/test_canonical_bars.py`:

```python
from pathlib import Path

from xsignal.data.paths import CanonicalPaths


def test_canonical_paths_are_deterministic(tmp_path):
    partition = Partition(timeframe="1h", year=2026, month=5)
    paths = CanonicalPaths(root=tmp_path)

    assert paths.parquet_path(partition) == (
        tmp_path / "canonical_bars" / "timeframe=1h" / "year=2026" / "month=05" / "bars.parquet"
    )
    assert paths.manifest_path(partition) == (
        tmp_path / "canonical_bars" / "timeframe=1h" / "year=2026" / "month=05" / "manifest.json"
    )
    assert paths.lock_path(partition) == (
        tmp_path / "canonical_bars" / "_locks" / "timeframe=1h__year=2026__month=05.lock"
    )
    assert paths.catalog_path("1h") == tmp_path / "canonical_bars" / "_catalog" / "timeframe=1h.json"
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/data/test_canonical_bars.py::test_canonical_paths_are_deterministic -q
```

Expected: FAIL because `xsignal.data.paths` does not exist.

- [ ] **Step 3: Implement paths**

Write `src/xsignal/data/paths.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from xsignal.data.canonical_bars import Partition, validate_timeframe


@dataclass(frozen=True)
class CanonicalPaths:
    root: Path

    @property
    def base(self) -> Path:
        return self.root / "canonical_bars"

    def partition_dir(self, partition: Partition) -> Path:
        path = self.base / f"timeframe={partition.timeframe}" / f"year={partition.year:04d}"
        if partition.month is not None:
            path = path / f"month={partition.month:02d}"
        return path

    def parquet_path(self, partition: Partition) -> Path:
        return self.partition_dir(partition) / "bars.parquet"

    def temp_parquet_path(self, partition: Partition, run_id: str) -> Path:
        return self.partition_dir(partition) / f".bars.{run_id}.tmp.parquet"

    def manifest_path(self, partition: Partition) -> Path:
        return self.partition_dir(partition) / "manifest.json"

    def temp_manifest_path(self, partition: Partition, run_id: str) -> Path:
        return self.partition_dir(partition) / f".manifest.{run_id}.tmp.json"

    def lock_path(self, partition: Partition) -> Path:
        lock_name = partition.key.replace("/", "__") + ".lock"
        return self.base / "_locks" / lock_name

    def catalog_path(self, timeframe: str) -> Path:
        validate_timeframe(timeframe)
        return self.base / "_catalog" / f"timeframe={timeframe}.json"
```

- [ ] **Step 4: Verify tests pass**

Run:

```bash
pytest tests/data/test_canonical_bars.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/data/paths.py tests/data/test_canonical_bars.py
git commit -m "feat: add canonical data paths"
```

## Task 4: Catalog And Export Manifest

**Files:**
- Create: `src/xsignal/data/catalog.py`
- Create: `src/xsignal/runs/manifest.py`
- Test: `tests/data/test_catalog.py`

- [ ] **Step 1: Write failing tests**

Write `tests/data/test_catalog.py`:

```python
from pathlib import Path

from xsignal.data.canonical_bars import Partition
from xsignal.data.catalog import Catalog, PartitionStatus
from xsignal.data.paths import CanonicalPaths
from xsignal.runs.manifest import ExportManifest


def make_manifest(partition: Partition, parquet_path: Path) -> ExportManifest:
    return ExportManifest(
        dataset_version="v1",
        source_table="xgate.klines_1m",
        timeframe=partition.timeframe,
        partition_key=partition.key,
        deduplication_mode="FINAL",
        aggregation_semantics_version="ohlcv-v1",
        query_hash="abc123",
        row_count=10,
        parquet_path=str(parquet_path),
        exported_at="2026-05-06T00:00:00Z",
    )


def test_catalog_marks_partition_complete(tmp_path):
    paths = CanonicalPaths(root=tmp_path)
    catalog = Catalog(paths=paths)
    partition = Partition(timeframe="1h", year=2026, month=5)
    parquet_path = paths.parquet_path(partition)
    parquet_path.parent.mkdir(parents=True)
    parquet_path.write_bytes(b"fake-parquet")
    manifest = make_manifest(partition, parquet_path)
    paths.manifest_path(partition).write_text(manifest.model_dump_json(indent=2))

    catalog.mark_complete(manifest)

    status = catalog.status(partition, dataset_version="v1")
    assert status == PartitionStatus.COMPLETE


def test_catalog_treats_missing_manifest_as_missing(tmp_path):
    paths = CanonicalPaths(root=tmp_path)
    catalog = Catalog(paths=paths)
    partition = Partition(timeframe="1h", year=2026, month=5)
    paths.parquet_path(partition).parent.mkdir(parents=True)
    paths.parquet_path(partition).write_bytes(b"fake-parquet")

    assert catalog.status(partition, dataset_version="v1") == PartitionStatus.MISSING
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/data/test_catalog.py -q
```

Expected: FAIL because catalog and manifest modules do not exist.

- [ ] **Step 3: Implement manifest model**

Write `src/xsignal/runs/manifest.py`:

```python
from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ExportManifest(BaseModel):
    model_config = ConfigDict(frozen=True)

    dataset_version: str
    source_table: str
    timeframe: str
    partition_key: str
    deduplication_mode: str
    aggregation_semantics_version: str
    query_hash: str
    row_count: int
    parquet_path: str
    exported_at: str
```

- [ ] **Step 4: Implement catalog**

Write `src/xsignal/data/catalog.py`:

```python
from __future__ import annotations

import json
from enum import StrEnum

from xsignal.data.canonical_bars import Partition
from xsignal.data.paths import CanonicalPaths
from xsignal.runs.manifest import ExportManifest


class PartitionStatus(StrEnum):
    COMPLETE = "complete"
    MISSING = "missing"
    STALE = "stale"


class Catalog:
    def __init__(self, paths: CanonicalPaths) -> None:
        self.paths = paths

    def status(self, partition: Partition, dataset_version: str) -> PartitionStatus:
        parquet_path = self.paths.parquet_path(partition)
        manifest_path = self.paths.manifest_path(partition)
        if not parquet_path.exists() or not manifest_path.exists():
            return PartitionStatus.MISSING

        manifest = ExportManifest.model_validate_json(manifest_path.read_text())
        if manifest.dataset_version != dataset_version:
            return PartitionStatus.STALE
        if manifest.timeframe != partition.timeframe:
            return PartitionStatus.STALE
        if manifest.partition_key != partition.key:
            return PartitionStatus.STALE
        if manifest.parquet_path != str(parquet_path):
            return PartitionStatus.STALE
        if manifest.row_count <= 0:
            return PartitionStatus.STALE
        return PartitionStatus.COMPLETE

    def mark_complete(self, manifest: ExportManifest) -> None:
        catalog_path = self.paths.catalog_path(manifest.timeframe)
        catalog_path.parent.mkdir(parents=True, exist_ok=True)
        if catalog_path.exists():
            catalog = json.loads(catalog_path.read_text())
        else:
            catalog = {"timeframe": manifest.timeframe, "partitions": {}}
        catalog["partitions"][manifest.partition_key] = {
            "dataset_version": manifest.dataset_version,
            "row_count": manifest.row_count,
            "query_hash": manifest.query_hash,
            "parquet_path": manifest.parquet_path,
            "exported_at": manifest.exported_at,
        }
        catalog_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n")
```

- [ ] **Step 5: Verify tests pass**

Run:

```bash
pytest tests/data/test_catalog.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/xsignal/data/catalog.py src/xsignal/runs/manifest.py tests/data/test_catalog.py
git commit -m "feat: add canonical export catalog"
```

## Task 5: Locks And Atomic Publication

**Files:**
- Create: `src/xsignal/data/locks.py`
- Test: `tests/data/test_locks.py`

- [ ] **Step 1: Write failing tests**

Write `tests/data/test_locks.py`:

```python
from pathlib import Path

from xsignal.data.locks import ExportLock, atomic_publish


def test_export_lock_creates_parent_directory(tmp_path):
    lock_path = tmp_path / "canonical_bars" / "_locks" / "partition.lock"

    with ExportLock(lock_path):
        assert lock_path.parent.exists()


def test_atomic_publish_replaces_target(tmp_path):
    temp_path = tmp_path / ".bars.tmp.parquet"
    target_path = tmp_path / "bars.parquet"
    temp_path.write_bytes(b"new")
    target_path.write_bytes(b"old")

    atomic_publish(temp_path, target_path)

    assert target_path.read_bytes() == b"new"
    assert not temp_path.exists()
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/data/test_locks.py -q
```

Expected: FAIL because `xsignal.data.locks` does not exist.

- [ ] **Step 3: Implement locks and atomic publish**

Write `src/xsignal/data/locks.py`:

```python
from __future__ import annotations

from pathlib import Path

from filelock import FileLock


class ExportLock:
    def __init__(self, lock_path: Path, timeout_seconds: int = 3600) -> None:
        self.lock_path = lock_path
        self.timeout_seconds = timeout_seconds
        self._lock = FileLock(str(lock_path), timeout=timeout_seconds)

    def __enter__(self) -> "ExportLock":
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._lock.release()


def atomic_publish(temp_path: Path, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.replace(target_path)
```

- [ ] **Step 4: Verify tests pass**

Run:

```bash
pytest tests/data/test_locks.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/data/locks.py tests/data/test_locks.py
git commit -m "feat: add export locks and atomic publish"
```

## Task 6: ClickHouse Query Template

**Files:**
- Create: `src/xsignal/data/query_templates.py`
- Test: `tests/data/test_query_templates.py`

- [ ] **Step 1: Write failing tests**

Write `tests/data/test_query_templates.py`:

```python
from datetime import datetime, timezone

from xsignal.data.query_templates import build_aggregate_query, query_hash


def test_build_aggregate_query_uses_final_and_expected_interval():
    sql = build_aggregate_query(
        timeframe="1h",
        start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    assert "FROM xgate.klines_1m FINAL" in sql
    assert "INTERVAL 1 hour" in sql
    assert "bar_count" in sql
    assert "is_complete" in sql
    assert "2026-05-01 00:00:00" in sql
    assert "2026-06-01 00:00:00" in sql


def test_query_hash_is_stable():
    assert query_hash("select 1") == query_hash("select 1")
    assert query_hash("select 1") != query_hash("select 2")
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/data/test_query_templates.py -q
```

Expected: FAIL because `query_templates.py` does not exist.

- [ ] **Step 3: Implement query template**

Write `src/xsignal/data/query_templates.py`:

```python
from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from xsignal.data.canonical_bars import expected_1m_count, validate_timeframe


CLICKHOUSE_SOURCE_TABLE = "xgate.klines_1m"


def _format_clickhouse_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    utc_value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return utc_value.strftime("%Y-%m-%d %H:%M:%S")


def _interval_sql(timeframe: str) -> str:
    validate_timeframe(timeframe)
    if timeframe == "1h":
        return "INTERVAL 1 hour"
    if timeframe == "4h":
        return "INTERVAL 4 hour"
    return "INTERVAL 1 day"


def build_aggregate_query(timeframe: str, start: datetime, end: datetime) -> str:
    interval = _interval_sql(timeframe)
    expected_count = expected_1m_count(timeframe)
    start_sql = _format_clickhouse_datetime(start)
    end_sql = _format_clickhouse_datetime(end)
    return f"""
SELECT
    symbol,
    toStartOfInterval(open_time, {interval}, 'UTC') AS open_time,
    toFloat64(argMin(open, open_time)) AS open,
    toFloat64(max(high)) AS high,
    toFloat64(min(low)) AS low,
    toFloat64(argMax(close, open_time)) AS close,
    toFloat64(sum(volume)) AS volume,
    toFloat64(sum(quote_volume)) AS quote_volume,
    toUInt64(sum(trade_count)) AS trade_count,
    toFloat64(sum(taker_buy_volume)) AS taker_buy_volume,
    toFloat64(sum(taker_buy_quote_volume)) AS taker_buy_quote_volume,
    toUInt16(count()) AS bar_count,
    bar_count = {expected_count} AS is_complete
FROM {CLICKHOUSE_SOURCE_TABLE} FINAL
WHERE open_time >= toDateTime('{start_sql}', 'UTC')
  AND open_time < toDateTime('{end_sql}', 'UTC')
GROUP BY
    symbol,
    open_time
ORDER BY
    open_time,
    symbol
""".strip()


def query_hash(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()
```

- [ ] **Step 4: Verify tests pass**

Run:

```bash
pytest tests/data/test_query_templates.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/data/query_templates.py tests/data/test_query_templates.py
git commit -m "feat: add canonical bar query template"
```

## Task 7: ClickHouse Client Wrapper

**Files:**
- Create: `src/xsignal/data/clickhouse.py`

- [ ] **Step 1: Implement minimal ClickHouse wrapper**

Write `src/xsignal/data/clickhouse.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import clickhouse_connect


@dataclass(frozen=True)
class ClickHouseConfig:
    host: str = "127.0.0.1"
    port: int = 8123
    username: str = "default"
    password: str = ""
    database: str = "xgate"


class ClickHouseClient:
    def __init__(self, config: ClickHouseConfig) -> None:
        self.config = config
        self._client = clickhouse_connect.get_client(
            host=config.host,
            port=config.port,
            username=config.username,
            password=config.password,
            database=config.database,
        )

    def query_arrow(self, sql: str):
        return self._client.query_arrow(sql)

    def write_parquet(self, sql: str, path: Path) -> int:
        table = self.query_arrow(sql)
        path.parent.mkdir(parents=True, exist_ok=True)
        import pyarrow.parquet as pq

        pq.write_table(table, path, compression="zstd")
        return table.num_rows
```

- [ ] **Step 2: Run import smoke test**

Run:

```bash
python3 - <<'PY'
from xsignal.data.clickhouse import ClickHouseConfig
print(ClickHouseConfig())
PY
```

Expected: prints a `ClickHouseConfig(...)` object.

- [ ] **Step 3: Commit**

```bash
git add src/xsignal/data/clickhouse.py
git commit -m "feat: add clickhouse client wrapper"
```

## Task 8: Canonical Export Orchestration

**Files:**
- Create: `src/xsignal/data/canonical_export.py`
- Test: `tests/data/test_canonical_export.py`

- [ ] **Step 1: Write failing tests with a fake exporter**

Write `tests/data/test_canonical_export.py`:

```python
from datetime import datetime, timezone
from pathlib import Path

from xsignal.data.canonical_bars import CanonicalRequest, Partition
from xsignal.data.canonical_export import ensure_canonical_bars
from xsignal.data.paths import CanonicalPaths


class FakeExporter:
    def __init__(self) -> None:
        self.calls = []

    def export(self, sql: str, path: Path) -> int:
        self.calls.append((sql, path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fake-parquet")
        return 3


def test_ensure_exports_missing_partition(tmp_path):
    exporter = FakeExporter()
    paths = CanonicalPaths(root=tmp_path)
    partition = Partition(timeframe="1h", year=2026, month=5)

    result = ensure_canonical_bars(
        request=CanonicalRequest(timeframe="1h"),
        paths=paths,
        partitions=[partition],
        exporter=exporter,
        now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
    )

    assert len(exporter.calls) == 1
    assert paths.parquet_path(partition).exists()
    assert paths.manifest_path(partition).exists()
    assert result.partitions == [partition]


def test_ensure_reuses_complete_partition(tmp_path):
    exporter = FakeExporter()
    paths = CanonicalPaths(root=tmp_path)
    partition = Partition(timeframe="1h", year=2026, month=5)

    ensure_canonical_bars(
        request=CanonicalRequest(timeframe="1h"),
        paths=paths,
        partitions=[partition],
        exporter=exporter,
        now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
    )
    ensure_canonical_bars(
        request=CanonicalRequest(timeframe="1h"),
        paths=paths,
        partitions=[partition],
        exporter=exporter,
        now=lambda: datetime(2026, 5, 6, tzinfo=timezone.utc),
    )

    assert len(exporter.calls) == 1
```

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/data/test_canonical_export.py -q
```

Expected: FAIL because `canonical_export.py` does not exist.

- [ ] **Step 3: Implement orchestration**

Write `src/xsignal/data/canonical_export.py`:

```python
from __future__ import annotations

import argparse
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Protocol

from xsignal.data.canonical_bars import CanonicalRequest, Partition
from xsignal.data.catalog import Catalog, PartitionStatus
from xsignal.data.clickhouse import ClickHouseClient, ClickHouseConfig
from xsignal.data.locks import ExportLock, atomic_publish
from xsignal.data.paths import CanonicalPaths
from xsignal.data.query_templates import CLICKHOUSE_SOURCE_TABLE, build_aggregate_query, query_hash
from xsignal.runs.manifest import ExportManifest


class Exporter(Protocol):
    def export(self, sql: str, path: Path) -> int:
        ...


@dataclass(frozen=True)
class CanonicalDataset:
    request: CanonicalRequest
    root: Path
    partitions: list[Partition]


def _partition_bounds(partition: Partition) -> tuple[datetime, datetime]:
    if partition.month is None:
        start = datetime(partition.year, 1, 1, tzinfo=timezone.utc)
        end = datetime(partition.year + 1, 1, 1, tzinfo=timezone.utc)
        return start, end
    start = datetime(partition.year, partition.month, 1, tzinfo=timezone.utc)
    if partition.month == 12:
        end = datetime(partition.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(partition.year, partition.month + 1, 1, tzinfo=timezone.utc)
    return start, end


def ensure_canonical_bars(
    request: CanonicalRequest,
    paths: CanonicalPaths,
    partitions: list[Partition],
    exporter: Exporter,
    now: Callable[[], datetime] | None = None,
) -> CanonicalDataset:
    clock = now or (lambda: datetime.now(timezone.utc))
    catalog = Catalog(paths=paths)

    for partition in partitions:
        if partition.timeframe != request.timeframe:
            raise ValueError("Partition timeframe must match request timeframe")
        if catalog.status(partition, request.dataset_version) == PartitionStatus.COMPLETE:
            continue

        with ExportLock(paths.lock_path(partition)):
            if catalog.status(partition, request.dataset_version) == PartitionStatus.COMPLETE:
                continue

            run_id = uuid.uuid4().hex
            start, end = _partition_bounds(partition)
            sql = build_aggregate_query(request.timeframe, start, end)
            temp_parquet = paths.temp_parquet_path(partition, run_id)
            target_parquet = paths.parquet_path(partition)
            row_count = exporter.export(sql, temp_parquet)
            atomic_publish(temp_parquet, target_parquet)

            manifest = ExportManifest(
                dataset_version=request.dataset_version,
                source_table=CLICKHOUSE_SOURCE_TABLE,
                timeframe=request.timeframe,
                partition_key=partition.key,
                deduplication_mode="FINAL",
                aggregation_semantics_version="ohlcv-v1",
                query_hash=query_hash(sql),
                row_count=row_count,
                parquet_path=str(target_parquet),
                exported_at=clock().astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
            )
            temp_manifest = paths.temp_manifest_path(partition, run_id)
            temp_manifest.write_text(manifest.model_dump_json(indent=2) + "\n")
            atomic_publish(temp_manifest, paths.manifest_path(partition))
            catalog.mark_complete(manifest)

    return CanonicalDataset(request=request, root=paths.base, partitions=partitions)


class ClickHouseExporter:
    def __init__(self, client: ClickHouseClient) -> None:
        self.client = client

    def export(self, sql: str, path: Path) -> int:
        return self.client.write_parquet(sql, path)


def main() -> None:
    parser = argparse.ArgumentParser(prog="xsignal-export")
    subparsers = parser.add_subparsers(dest="command", required=True)
    ensure_parser = subparsers.add_parser("ensure")
    ensure_parser.add_argument("--timeframe", required=True, choices=["1h", "4h", "1d"])
    ensure_parser.add_argument("--root", default="data")
    ensure_parser.add_argument("--year", type=int, required=True)
    ensure_parser.add_argument("--month", type=int)
    args = parser.parse_args()

    request = CanonicalRequest(timeframe=args.timeframe)
    partition = Partition(timeframe=args.timeframe, year=args.year, month=args.month)
    client = ClickHouseClient(ClickHouseConfig())
    exporter = ClickHouseExporter(client)
    dataset = ensure_canonical_bars(
        request=request,
        paths=CanonicalPaths(root=Path(args.root)),
        partitions=[partition],
        exporter=exporter,
    )
    print(dataset)
```

- [ ] **Step 4: Verify tests pass**

Run:

```bash
pytest tests/data/test_canonical_export.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/data/canonical_export.py tests/data/test_canonical_export.py
git commit -m "feat: add canonical export orchestration"
```

## Task 9: Full-History Partition Discovery

**Files:**
- Modify: `src/xsignal/data/canonical_export.py`
- Test: `tests/data/test_canonical_export.py`

- [ ] **Step 1: Add partition discovery tests**

Append to `tests/data/test_canonical_export.py`:

```python
from xsignal.data.canonical_export import partitions_for_full_history


def test_partitions_for_full_history_uses_months_for_intraday_timeframes():
    partitions = partitions_for_full_history(
        timeframe="1h",
        start=datetime(2026, 4, 10, tzinfo=timezone.utc),
        end=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    assert partitions == [
        Partition(timeframe="1h", year=2026, month=4),
        Partition(timeframe="1h", year=2026, month=5),
    ]


def test_partitions_for_full_history_uses_years_for_daily_timeframe():
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

- [ ] **Step 2: Run tests to verify failure**

Run:

```bash
pytest tests/data/test_canonical_export.py::test_partitions_for_full_history_uses_months_for_intraday_timeframes -q
```

Expected: FAIL because `partitions_for_full_history` does not exist.

- [ ] **Step 3: Implement full-history partition discovery**

Append to `src/xsignal/data/canonical_export.py`:

```python
def partitions_for_full_history(
    timeframe: str,
    start: datetime,
    end: datetime,
) -> list[Partition]:
    if start.tzinfo is None or end.tzinfo is None:
        raise ValueError("start and end must be timezone-aware")
    if end <= start:
        raise ValueError("end must be after start")

    partitions: list[Partition] = []
    if timeframe == "1d":
        for year in range(start.year, end.year + 1):
            partitions.append(Partition(timeframe=timeframe, year=year))
        return partitions

    year = start.year
    month = start.month
    while (year, month) <= (end.year, end.month):
        if (year, month) == (end.year, end.month) and end.day == 1 and end.hour == 0 and end.minute == 0:
            break
        partitions.append(Partition(timeframe=timeframe, year=year, month=month))
        if month == 12:
            year += 1
            month = 1
        else:
            month += 1
    return partitions
```

- [ ] **Step 4: Verify tests pass**

Run:

```bash
pytest tests/data/test_canonical_export.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/data/canonical_export.py tests/data/test_canonical_export.py
git commit -m "feat: add full history partition discovery"
```

## Task 10: CLI Smoke Path

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add CLI usage docs**

Append to `README.md`:

```markdown
## Canonical Bar Export

Export or reuse canonical bars for one partition:

```bash
xsignal-export ensure --timeframe 1h --year 2026 --month 5 --root data
```

The export layer writes:

- `data/canonical_bars/timeframe=1h/year=2026/month=05/bars.parquet`
- `data/canonical_bars/timeframe=1h/year=2026/month=05/manifest.json`
- `data/canonical_bars/_catalog/timeframe=1h.json`

Strategies should call `ensure_canonical_bars` before reading canonical Parquet.
```

- [ ] **Step 2: Run full local verification**

Run:

```bash
ruff check .
pytest -q
```

Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document canonical export usage"
```

## Task 11: Server Validation

**Files:**
- No code changes unless validation reveals a bug.

- [ ] **Step 1: Confirm ClickHouse tunnel or local server access**

Run from local machine:

```bash
ssh -L 8123:127.0.0.1:8123 X 'docker ps --format "{{.Names}} {{.Status}}" | grep xgate-clickhouse'
```

Expected: `xgate-clickhouse Up ...`.

- [ ] **Step 2: In a separate terminal, run one small real export**

Run:

```bash
xsignal-export ensure --timeframe 1h --year 2026 --month 5 --root data
```

Expected:

```text
CanonicalDataset(...)
```

Then verify files:

```bash
ls -lh data/canonical_bars/timeframe=1h/year=2026/month=05/
```

Expected:

```text
bars.parquet
manifest.json
```

- [ ] **Step 3: Re-run the same export and confirm reuse**

Run:

```bash
time xsignal-export ensure --timeframe 1h --year 2026 --month 5 --root data
```

Expected: finishes quickly and does not rewrite `bars.parquet`.

- [ ] **Step 4: Inspect manifest**

Run:

```bash
python3 - <<'PY'
from pathlib import Path
print(Path("data/canonical_bars/timeframe=1h/year=2026/month=05/manifest.json").read_text())
PY
```

Expected fields include:

```text
dataset_version
source_table
timeframe
partition_key
deduplication_mode
query_hash
row_count
parquet_path
exported_at
```

- [ ] **Step 5: Commit validation fixes only if needed**

If validation required code fixes:

```bash
git add src tests README.md
git commit -m "fix: harden canonical export validation"
```

If no fixes were needed, do not create an empty commit.

## Self-Review

Spec coverage:

- Shared deduplicated export function: Tasks 6, 8, 9.
- Default full-universe, full-history behavior: Tasks 2, 8, 9.
- Reuse completed Parquet: Tasks 4 and 8.
- Missing partition export: Task 8.
- Atomic publication and locks: Task 5.
- Catalog and manifest: Tasks 4 and 8.
- Strategy independence: preserved; no strategy engine is introduced.
- Initial standard timeframes `1h`, `4h`, `1d`: Task 2.

Placeholder scan:

- No unresolved marker text or undefined future work remains in the implementation tasks.

Type consistency:

- `CanonicalRequest`, `Partition`, `CanonicalPaths`, `Catalog`, `ExportManifest`, and `ensure_canonical_bars` are introduced before later tasks use them.
