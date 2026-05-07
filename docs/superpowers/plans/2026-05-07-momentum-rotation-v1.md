# Momentum Rotation V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the first strategy-specific high-performance backtest: a long-only, daily-rebalanced, multi-timeframe Top N momentum rotation strategy that consumes canonical Parquet only.

**Architecture:** Keep the strategy under `src/xsignal/strategies/momentum_rotation_v1/` with focused modules for config, canonical data loading, daily alignment, scoring, kernel execution, artifact writing, and CLI orchestration. The strategy calls the canonical export layer before reading data, prepares dense time-major arrays, then runs a narrow NumPy backtest kernel without introducing a generic engine.

**Tech Stack:** Python 3.12, NumPy, PyArrow, Pydantic, pytest, ruff, existing X-Signal canonical export modules.

---

## Review Outcome

The design spec was reviewed before this plan. Two ambiguities were fixed in `docs/superpowers/specs/2026-05-07-momentum-rotation-v1-design.md`:

- `1h` and `4h` signal bars must be aligned to completed UTC daily rebalance timestamps with close time `<= t`.
- Data-quality masks must cover every source bar inside each lookback window, not just return endpoints.

No blocking design issues remain.

## File Structure

Create strategy package:

```text
src/xsignal/strategies/__init__.py
src/xsignal/strategies/momentum_rotation_v1/__init__.py
src/xsignal/strategies/momentum_rotation_v1/config.py
src/xsignal/strategies/momentum_rotation_v1/paths.py
src/xsignal/strategies/momentum_rotation_v1/data.py
src/xsignal/strategies/momentum_rotation_v1/prepare.py
src/xsignal/strategies/momentum_rotation_v1/signals.py
src/xsignal/strategies/momentum_rotation_v1/kernel.py
src/xsignal/strategies/momentum_rotation_v1/artifacts.py
src/xsignal/strategies/momentum_rotation_v1/cli.py
```

Create tests:

```text
tests/strategies/__init__.py
tests/strategies/momentum_rotation_v1/__init__.py
tests/strategies/momentum_rotation_v1/test_config.py
tests/strategies/momentum_rotation_v1/test_paths.py
tests/strategies/momentum_rotation_v1/test_data.py
tests/strategies/momentum_rotation_v1/test_prepare.py
tests/strategies/momentum_rotation_v1/test_signals.py
tests/strategies/momentum_rotation_v1/test_kernel.py
tests/strategies/momentum_rotation_v1/test_artifacts.py
tests/strategies/momentum_rotation_v1/test_cli.py
```

Modify existing files:

```text
pyproject.toml
README.md
```

Responsibilities:

- `config.py`: validated strategy config and config hashing.
- `paths.py`: strategy-owned cache and run output paths.
- `data.py`: canonical manifest discovery, Parquet reads, required column validation, and optional canonical export orchestration.
- `prepare.py`: symbol/time alignment and dense daily arrays.
- `signals.py`: multi-timeframe momentum score and eligibility masks.
- `kernel.py`: Top N equal-weight daily portfolio simulation.
- `artifacts.py`: manifest, summary, equity curve, and positions writing.
- `cli.py`: one strategy-specific `run` command.
- `README.md`: short command example and artifact description.

## Task 1: Package Skeleton, NumPy Dependency, And Config

**Files:**
- Modify: `pyproject.toml`
- Create: `src/xsignal/strategies/__init__.py`
- Create: `src/xsignal/strategies/momentum_rotation_v1/__init__.py`
- Create: `src/xsignal/strategies/momentum_rotation_v1/config.py`
- Create: `tests/strategies/__init__.py`
- Create: `tests/strategies/momentum_rotation_v1/__init__.py`
- Create: `tests/strategies/momentum_rotation_v1/test_config.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/strategies/momentum_rotation_v1/test_config.py`:

```python
from __future__ import annotations

import pytest

from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig


def test_default_config_matches_v1_design():
    config = MomentumRotationConfig()

    assert config.strategy_name == "momentum_rotation_v1"
    assert config.timeframes == ("1h", "4h", "1d")
    assert config.fill_policy == "raw"
    assert config.top_n == 10
    assert config.fee_bps == 5.0
    assert config.slippage_bps == 5.0
    assert config.initial_equity == 1.0
    assert config.short_return_weight == 0.4
    assert config.medium_return_weight == 0.4
    assert config.long_return_weight == 0.2
    assert config.short_window_hours == 24
    assert config.medium_window_days == 7
    assert config.long_window_days == 30
    assert config.min_rolling_7d_quote_volume == 0.0


def test_config_hash_is_stable_for_same_payload():
    first = MomentumRotationConfig(top_n=20)
    second = MomentumRotationConfig(top_n=20)

    assert first.config_hash() == second.config_hash()
    assert len(first.config_hash()) == 64


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="top_n"):
        MomentumRotationConfig(top_n=0)
    with pytest.raises(ValueError, match="fee_bps"):
        MomentumRotationConfig(fee_bps=-1)
    with pytest.raises(ValueError, match="slippage_bps"):
        MomentumRotationConfig(slippage_bps=-1)
    with pytest.raises(ValueError, match="initial_equity"):
        MomentumRotationConfig(initial_equity=0)
    with pytest.raises(ValueError, match="return weights"):
        MomentumRotationConfig(short_return_weight=0.5, medium_return_weight=0.5, long_return_weight=0.5)
    with pytest.raises(ValueError, match="fill_policy"):
        MomentumRotationConfig(fill_policy="prev_close_zero_volume")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_config.py -q
```

Expected: FAIL with `ModuleNotFoundError: No module named 'xsignal.strategies'`.

- [ ] **Step 3: Add NumPy dependency and package files**

Update `pyproject.toml` dependencies:

```toml
dependencies = [
  "clickhouse-connect>=0.8.0",
  "filelock>=3.15.0",
  "numpy>=2.0.0",
  "pyarrow>=16.0.0",
  "pydantic>=2.7.0",
]
```

Create empty package markers:

```python
# src/xsignal/strategies/__init__.py
```

```python
# src/xsignal/strategies/momentum_rotation_v1/__init__.py
```

```python
# tests/strategies/__init__.py
```

```python
# tests/strategies/momentum_rotation_v1/__init__.py
```

- [ ] **Step 4: Implement config**

Create `src/xsignal/strategies/momentum_rotation_v1/config.py`:

```python
from __future__ import annotations

import hashlib
from datetime import date

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class MomentumRotationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy_name: str = "momentum_rotation_v1"
    timeframes: tuple[str, str, str] = ("1h", "4h", "1d")
    fill_policy: str = "raw"
    top_n: int = 10
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    initial_equity: float = 1.0
    min_rolling_7d_quote_volume: float = 0.0
    short_return_weight: float = 0.4
    medium_return_weight: float = 0.4
    long_return_weight: float = 0.2
    short_window_hours: int = 24
    medium_window_days: int = 7
    long_window_days: int = 30
    start_date: date | None = None
    end_date: date | None = None

    @field_validator("strategy_name", "fill_policy")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be non-empty")
        return value

    @model_validator(mode="after")
    def _validate_config(self) -> "MomentumRotationConfig":
        if self.strategy_name != "momentum_rotation_v1":
            raise ValueError("strategy_name must be 'momentum_rotation_v1'")
        if self.timeframes != ("1h", "4h", "1d"):
            raise ValueError("timeframes must be ('1h', '4h', '1d')")
        if self.fill_policy != "raw":
            raise ValueError("fill_policy must be 'raw'")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.fee_bps < 0:
            raise ValueError("fee_bps must be non-negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if self.min_rolling_7d_quote_volume < 0:
            raise ValueError("min_rolling_7d_quote_volume must be non-negative")
        if self.short_window_hours != 24:
            raise ValueError("short_window_hours must be 24 for v1")
        if self.medium_window_days != 7:
            raise ValueError("medium_window_days must be 7 for v1")
        if self.long_window_days != 30:
            raise ValueError("long_window_days must be 30 for v1")
        weight_sum = self.short_return_weight + self.medium_return_weight + self.long_return_weight
        if abs(weight_sum - 1.0) > 1e-12:
            raise ValueError("return weights must sum to 1.0")
        if self.start_date is not None and self.end_date is not None and self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        return self

    def config_hash(self) -> str:
        payload = self.model_dump_json(sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
```

- [ ] **Step 5: Run config tests**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_config.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/xsignal/strategies tests/strategies
git commit -m "feat: add momentum rotation config"
```

## Task 2: Strategy Paths

**Files:**
- Create: `src/xsignal/strategies/momentum_rotation_v1/paths.py`
- Create: `tests/strategies/momentum_rotation_v1/test_paths.py`

- [ ] **Step 1: Write failing path tests**

Create `tests/strategies/momentum_rotation_v1/test_paths.py`:

```python
from __future__ import annotations

from xsignal.strategies.momentum_rotation_v1.paths import MomentumRotationPaths


def test_strategy_paths_are_under_strategy_owned_data_dir(tmp_path):
    paths = MomentumRotationPaths(root=tmp_path)

    assert paths.base == tmp_path / "strategies" / "momentum_rotation_v1"
    assert paths.cache == paths.base / "cache"
    assert paths.runs == paths.base / "runs"
    assert paths.cache_file("close_1d.npy") == paths.cache / "close_1d.npy"
    assert paths.run_dir("abc123") == paths.runs / "abc123"


def test_run_id_rejects_path_traversal(tmp_path):
    paths = MomentumRotationPaths(root=tmp_path)

    for bad_run_id in ["", "../abc", "abc/def", "abc\\def"]:
        try:
            paths.run_dir(bad_run_id)
        except ValueError as exc:
            assert "run_id" in str(exc)
        else:
            raise AssertionError(f"run_id {bad_run_id!r} should fail")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_paths.py -q
```

Expected: FAIL because `paths.py` does not exist.

- [ ] **Step 3: Implement paths**

Create `src/xsignal/strategies/momentum_rotation_v1/paths.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _validate_run_id(run_id: str) -> str:
    if not run_id or "/" in run_id or "\\" in run_id or ".." in run_id:
        raise ValueError("run_id must be non-empty and must not contain path separators or '..'")
    return run_id


@dataclass(frozen=True)
class MomentumRotationPaths:
    root: Path

    @property
    def base(self) -> Path:
        return self.root / "strategies" / "momentum_rotation_v1"

    @property
    def cache(self) -> Path:
        return self.base / "cache"

    @property
    def runs(self) -> Path:
        return self.base / "runs"

    def cache_file(self, name: str) -> Path:
        if "/" in name or "\\" in name or not name:
            raise ValueError("cache file name must be a plain file name")
        return self.cache / name

    def run_dir(self, run_id: str) -> Path:
        return self.runs / _validate_run_id(run_id)
```

- [ ] **Step 4: Run path tests**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_paths.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/strategies/momentum_rotation_v1/paths.py tests/strategies/momentum_rotation_v1/test_paths.py
git commit -m "feat: add momentum rotation paths"
```

## Task 3: Canonical Manifest And Parquet Loader

**Files:**
- Create: `src/xsignal/strategies/momentum_rotation_v1/data.py`
- Create: `tests/strategies/momentum_rotation_v1/test_data.py`

- [ ] **Step 1: Write failing loader tests**

Create `tests/strategies/momentum_rotation_v1/test_data.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from xsignal.strategies.momentum_rotation_v1.data import (
    REQUIRED_CANONICAL_COLUMNS,
    CanonicalBarTable,
    load_manifested_table,
)


def write_manifested_partition(root: Path, timeframe: str = "1d", fill_policy: str = "raw") -> Path:
    partition_dir = root / "canonical_bars" / f"timeframe={timeframe}" / f"fill_policy={fill_policy}" / "year=2026"
    parquet_path = partition_dir / "bars.abc.parquet"
    partition_dir.mkdir(parents=True)
    table = pa.table(
        {
            "symbol": ["BTCUSDT", "ETHUSDT"],
            "open_time": pa.array([0, 0], type=pa.timestamp("us", tz="UTC")),
            "close": [100.0, 50.0],
            "quote_volume": [10_000.0, 8_000.0],
            "bar_count": [1440, 1440],
            "expected_1m_count": [1440, 1440],
            "is_complete": [True, True],
            "has_synthetic": [False, False],
            "fill_policy": ["raw", "raw"],
        }
    )
    pq.write_table(table.select(list(REQUIRED_CANONICAL_COLUMNS)), parquet_path)
    manifest = {
        "timeframe": timeframe,
        "fill_policy": fill_policy,
        "parquet_path": str(parquet_path),
        "row_count": 2,
    }
    (partition_dir / "manifest.json").write_text(json.dumps(manifest))
    return partition_dir / "manifest.json"


def test_load_manifested_table_validates_identity_and_columns(tmp_path):
    manifest_path = write_manifested_partition(tmp_path)

    loaded = load_manifested_table(manifest_path, timeframe="1d", fill_policy="raw")

    assert isinstance(loaded, CanonicalBarTable)
    assert loaded.timeframe == "1d"
    assert loaded.fill_policy == "raw"
    assert loaded.manifest_path == manifest_path
    assert loaded.table.num_rows == 2
    assert loaded.table.column_names == list(REQUIRED_CANONICAL_COLUMNS)


def test_load_manifested_table_rejects_fill_policy_mismatch(tmp_path):
    manifest_path = write_manifested_partition(tmp_path, fill_policy="prev_close_zero_volume")

    with pytest.raises(ValueError, match="fill_policy"):
        load_manifested_table(manifest_path, timeframe="1d", fill_policy="raw")


def test_load_manifested_table_rejects_missing_required_column(tmp_path):
    manifest_path = write_manifested_partition(tmp_path)
    manifest = json.loads(manifest_path.read_text())
    pq.write_table(pa.table({"symbol": ["BTCUSDT"]}), Path(manifest["parquet_path"]))

    with pytest.raises(ValueError, match="missing required columns"):
        load_manifested_table(manifest_path, timeframe="1d", fill_policy="raw")
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_data.py -q
```

Expected: FAIL because `data.py` does not exist.

- [ ] **Step 3: Implement manifested table loader**

Create `src/xsignal/strategies/momentum_rotation_v1/data.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


REQUIRED_CANONICAL_COLUMNS = (
    "symbol",
    "open_time",
    "close",
    "quote_volume",
    "bar_count",
    "expected_1m_count",
    "is_complete",
    "has_synthetic",
    "fill_policy",
)


@dataclass(frozen=True)
class CanonicalBarTable:
    timeframe: str
    fill_policy: str
    manifest_path: Path
    parquet_path: Path
    table: pa.Table


def load_manifested_table(
    manifest_path: Path,
    *,
    timeframe: str,
    fill_policy: str,
) -> CanonicalBarTable:
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("timeframe") != timeframe:
        raise ValueError("manifest timeframe does not match request")
    if manifest.get("fill_policy") != fill_policy:
        raise ValueError("manifest fill_policy does not match request")
    parquet_path = Path(manifest["parquet_path"])
    table = pq.ParquetFile(parquet_path).read()
    missing = sorted(set(REQUIRED_CANONICAL_COLUMNS) - set(table.column_names))
    if missing:
        raise ValueError(f"missing required columns: {missing}")
    table = table.select(list(REQUIRED_CANONICAL_COLUMNS))
    if table.num_rows != int(manifest["row_count"]):
        raise ValueError("manifest row_count does not match parquet")
    return CanonicalBarTable(
        timeframe=timeframe,
        fill_policy=fill_policy,
        manifest_path=manifest_path,
        parquet_path=parquet_path,
        table=table,
    )
```

- [ ] **Step 4: Run loader tests**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_data.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/strategies/momentum_rotation_v1/data.py tests/strategies/momentum_rotation_v1/test_data.py
git commit -m "feat: load momentum rotation canonical tables"
```

## Task 4: Daily Alignment And Prepared Arrays

**Files:**
- Create: `src/xsignal/strategies/momentum_rotation_v1/prepare.py`
- Create: `tests/strategies/momentum_rotation_v1/test_prepare.py`

- [ ] **Step 1: Write failing daily alignment tests**

Create `tests/strategies/momentum_rotation_v1/test_prepare.py`:

```python
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pyarrow as pa

from xsignal.strategies.momentum_rotation_v1.data import CanonicalBarTable
from xsignal.strategies.momentum_rotation_v1.prepare import prepare_daily_arrays


def table_for(timeframe: str, rows: list[dict]) -> CanonicalBarTable:
    return CanonicalBarTable(
        timeframe=timeframe,
        fill_policy="raw",
        manifest_path=__file__,
        parquet_path=__file__,
        table=pa.Table.from_pylist(rows),
    )


def rows(symbols: list[str], opens: list[datetime], multiplier: float, expected_count: int) -> list[dict]:
    output = []
    for symbol_index, symbol in enumerate(symbols):
        for time_index, open_time in enumerate(opens):
            output.append(
                {
                    "symbol": symbol,
                    "open_time": open_time,
                    "close": multiplier + symbol_index * 100.0 + time_index,
                    "quote_volume": 1_000_000.0 + time_index,
                    "bar_count": expected_count,
                    "expected_1m_count": expected_count,
                    "is_complete": True,
                    "has_synthetic": False,
                    "fill_policy": "raw",
                }
            )
    return output


def test_prepare_daily_arrays_aligns_completed_intraday_bars_to_daily_close():
    symbols = ["BTCUSDT", "ETHUSDT"]
    day0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    daily_opens = [day0 + timedelta(days=i) for i in range(3)]
    hourly_opens = [day0 + timedelta(hours=i) for i in range(72)]
    four_hour_opens = [day0 + timedelta(hours=4 * i) for i in range(18)]

    prepared = prepare_daily_arrays(
        bars_1h=table_for("1h", rows(symbols, hourly_opens, 10.0, 60)),
        bars_4h=table_for("4h", rows(symbols, four_hour_opens, 20.0, 240)),
        bars_1d=table_for("1d", rows(symbols, daily_opens, 30.0, 1440)),
    )

    assert prepared.symbols == tuple(symbols)
    assert prepared.rebalance_times.tolist() == [
        day0 + timedelta(days=1),
        day0 + timedelta(days=2),
        day0 + timedelta(days=3),
    ]
    assert prepared.close_1h.shape == (3, 2)
    assert prepared.close_4h.shape == (3, 2)
    assert prepared.close_1d.shape == (3, 2)
    assert prepared.close_1h[0, 0] == 10.0 + 23
    assert prepared.close_4h[0, 0] == 20.0 + 5
    assert prepared.close_1d[0, 0] == 30.0
    assert prepared.complete_1h[0, 0]
    assert prepared.complete_4h[0, 0]
    assert prepared.complete_1d[0, 0]
    assert prepared.quality_1h_24h[1, 0]


def test_prepare_daily_arrays_marks_missing_intraday_close_incomplete():
    symbols = ["BTCUSDT"]
    day0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    daily_opens = [day0, day0 + timedelta(days=1)]
    hourly_opens = [day0 + timedelta(hours=i) for i in range(47)]
    four_hour_opens = [day0 + timedelta(hours=4 * i) for i in range(12)]

    prepared = prepare_daily_arrays(
        bars_1h=table_for("1h", rows(symbols, hourly_opens, 10.0, 60)),
        bars_4h=table_for("4h", rows(symbols, four_hour_opens, 20.0, 240)),
        bars_1d=table_for("1d", rows(symbols, daily_opens, 30.0, 1440)),
    )

    assert prepared.complete_1h[0, 0]
    assert not prepared.complete_1h[1, 0]
    assert not prepared.quality_1h_24h[1, 0]
    assert np.isnan(prepared.close_1h[1, 0])
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_prepare.py -q
```

Expected: FAIL because `prepare.py` does not exist.

- [ ] **Step 3: Implement prepared arrays**

Create `src/xsignal/strategies/momentum_rotation_v1/prepare.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np

from xsignal.strategies.momentum_rotation_v1.data import CanonicalBarTable


@dataclass(frozen=True)
class PreparedArrays:
    symbols: tuple[str, ...]
    rebalance_times: np.ndarray
    close_1h: np.ndarray
    close_4h: np.ndarray
    close_1d: np.ndarray
    quote_volume_1d: np.ndarray
    complete_1h: np.ndarray
    complete_4h: np.ndarray
    complete_1d: np.ndarray
    quality_1h_24h: np.ndarray
    quality_4h_7d: np.ndarray
    quality_1d_30d: np.ndarray


def _rows_by_symbol_close_time(table: CanonicalBarTable, close_delta: timedelta) -> dict[tuple[str, object], dict]:
    rows = {}
    for row in table.table.to_pylist():
        close_time = row["open_time"] + close_delta
        rows[(row["symbol"], close_time)] = row
    return rows


def _is_good(row: dict | None) -> bool:
    return bool(
        row
        and row["is_complete"]
        and not row["has_synthetic"]
        and row["bar_count"] == row["expected_1m_count"]
        and row["close"] is not None
        and float(row["close"]) > 0
    )


def _window_quality(
    rows: dict[tuple[str, object], dict],
    symbol: str,
    rebalance_time: object,
    step: timedelta,
    steps_back: int,
) -> bool:
    for offset in range(steps_back + 1):
        close_time = rebalance_time - step * offset
        if not _is_good(rows.get((symbol, close_time))):
            return False
    return True


def prepare_daily_arrays(
    *,
    bars_1h: CanonicalBarTable,
    bars_4h: CanonicalBarTable,
    bars_1d: CanonicalBarTable,
) -> PreparedArrays:
    if bars_1h.timeframe != "1h" or bars_4h.timeframe != "4h" or bars_1d.timeframe != "1d":
        raise ValueError("expected 1h, 4h, and 1d canonical tables")
    symbols = tuple(sorted(set(bars_1d.table.column("symbol").to_pylist())))
    if not symbols:
        raise ValueError("no symbols in daily canonical table")
    daily_rows = _rows_by_symbol_close_time(bars_1d, timedelta(days=1))
    hourly_rows = _rows_by_symbol_close_time(bars_1h, timedelta(hours=1))
    four_hour_rows = _rows_by_symbol_close_time(bars_4h, timedelta(hours=4))
    rebalance_times = tuple(sorted({key[1] for key in daily_rows}))
    shape = (len(rebalance_times), len(symbols))
    close_1h = np.full(shape, np.nan, dtype=np.float64)
    close_4h = np.full(shape, np.nan, dtype=np.float64)
    close_1d = np.full(shape, np.nan, dtype=np.float64)
    quote_volume_1d = np.full(shape, np.nan, dtype=np.float64)
    complete_1h = np.zeros(shape, dtype=bool)
    complete_4h = np.zeros(shape, dtype=bool)
    complete_1d = np.zeros(shape, dtype=bool)
    quality_1h_24h = np.zeros(shape, dtype=bool)
    quality_4h_7d = np.zeros(shape, dtype=bool)
    quality_1d_30d = np.zeros(shape, dtype=bool)
    for t_index, rebalance_time in enumerate(rebalance_times):
        for s_index, symbol in enumerate(symbols):
            h_row = hourly_rows.get((symbol, rebalance_time))
            h4_row = four_hour_rows.get((symbol, rebalance_time))
            d_row = daily_rows.get((symbol, rebalance_time))
            if _is_good(h_row):
                close_1h[t_index, s_index] = float(h_row["close"])
                complete_1h[t_index, s_index] = True
            if _is_good(h4_row):
                close_4h[t_index, s_index] = float(h4_row["close"])
                complete_4h[t_index, s_index] = True
            if _is_good(d_row):
                close_1d[t_index, s_index] = float(d_row["close"])
                quote_volume_1d[t_index, s_index] = float(d_row["quote_volume"])
                complete_1d[t_index, s_index] = True
            quality_1h_24h[t_index, s_index] = _window_quality(
                hourly_rows,
                symbol,
                rebalance_time,
                timedelta(hours=1),
                24,
            )
            quality_4h_7d[t_index, s_index] = _window_quality(
                four_hour_rows,
                symbol,
                rebalance_time,
                timedelta(hours=4),
                42,
            )
            quality_1d_30d[t_index, s_index] = _window_quality(
                daily_rows,
                symbol,
                rebalance_time,
                timedelta(days=1),
                30,
            )
    return PreparedArrays(
        symbols=symbols,
        rebalance_times=np.array(rebalance_times, dtype=object),
        close_1h=close_1h,
        close_4h=close_4h,
        close_1d=close_1d,
        quote_volume_1d=quote_volume_1d,
        complete_1h=complete_1h,
        complete_4h=complete_4h,
        complete_1d=complete_1d,
        quality_1h_24h=quality_1h_24h,
        quality_4h_7d=quality_4h_7d,
        quality_1d_30d=quality_1d_30d,
    )
```

- [ ] **Step 4: Run prepare tests**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_prepare.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/strategies/momentum_rotation_v1/prepare.py tests/strategies/momentum_rotation_v1/test_prepare.py
git commit -m "feat: prepare momentum rotation arrays"
```

## Task 5: Momentum Scores And Eligibility Masks

**Files:**
- Create: `src/xsignal/strategies/momentum_rotation_v1/signals.py`
- Create: `tests/strategies/momentum_rotation_v1/test_signals.py`

- [ ] **Step 1: Write failing signal tests**

Create `tests/strategies/momentum_rotation_v1/test_signals.py`:

```python
from __future__ import annotations

import numpy as np

from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
from xsignal.strategies.momentum_rotation_v1.prepare import PreparedArrays
from xsignal.strategies.momentum_rotation_v1.signals import compute_momentum_signals


def prepared(close_multiplier: float = 1.0) -> PreparedArrays:
    t = 40
    n = 2
    base = np.arange(1, t + 1, dtype=np.float64).reshape(t, 1)
    close_1d = close_multiplier * np.concatenate([base, base * 2], axis=1)
    close_1h = close_1d * 10
    close_4h = close_1d * 20
    return PreparedArrays(
        symbols=("BTCUSDT", "ETHUSDT"),
        rebalance_times=np.array(list(range(t)), dtype=object),
        close_1h=close_1h,
        close_4h=close_4h,
        close_1d=close_1d,
        quote_volume_1d=np.full((t, n), 1_000_000.0),
        complete_1h=np.ones((t, n), dtype=bool),
        complete_4h=np.ones((t, n), dtype=bool),
        complete_1d=np.ones((t, n), dtype=bool),
        quality_1h_24h=np.ones((t, n), dtype=bool),
        quality_4h_7d=np.ones((t, n), dtype=bool),
        quality_1d_30d=np.ones((t, n), dtype=bool),
    )


def test_compute_momentum_signals_uses_configured_windows_and_weights():
    config = MomentumRotationConfig()
    result = compute_momentum_signals(prepared(), config)

    expected_24h = 40 / 39 - 1
    expected_7d = 40 / 33 - 1
    expected_30d = 40 / 10 - 1
    expected = 0.4 * expected_24h + 0.4 * expected_7d + 0.2 * expected_30d

    assert np.isnan(result.score[:30]).all()
    assert result.tradable_mask[39, 0]
    assert result.score[39, 0] == expected


def test_compute_momentum_signals_requires_full_quality_window():
    arrays = prepared()
    arrays.quality_1h_24h[30, 0] = False

    result = compute_momentum_signals(arrays, MomentumRotationConfig())

    assert not result.tradable_mask[30, 0]
    assert np.isnan(result.score[30, 0])


def test_compute_momentum_signals_applies_liquidity_filter():
    arrays = prepared()
    arrays.quote_volume_1d[:, 1] = 1.0
    config = MomentumRotationConfig(min_rolling_7d_quote_volume=10_000.0)

    result = compute_momentum_signals(arrays, config)

    assert result.tradable_mask[39, 0]
    assert not result.tradable_mask[39, 1]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_signals.py -q
```

Expected: FAIL because `signals.py` does not exist.

- [ ] **Step 3: Implement signal computation**

Create `src/xsignal/strategies/momentum_rotation_v1/signals.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
from xsignal.strategies.momentum_rotation_v1.prepare import PreparedArrays


@dataclass(frozen=True)
class SignalArrays:
    score: np.ndarray
    tradable_mask: np.ndarray


def _rolling_sum(values: np.ndarray, end_index: int, window: int) -> np.ndarray:
    start_index = end_index - window + 1
    if start_index < 0:
        return np.full(values.shape[1], np.nan, dtype=np.float64)
    return np.nansum(values[start_index : end_index + 1], axis=0)


def compute_momentum_signals(
    arrays: PreparedArrays,
    config: MomentumRotationConfig,
) -> SignalArrays:
    shape = arrays.close_1d.shape
    if arrays.close_1h.shape != shape or arrays.close_4h.shape != shape:
        raise ValueError("prepared close arrays must have matching shape")
    score = np.full(shape, np.nan, dtype=np.float64)
    tradable_mask = np.zeros(shape, dtype=bool)
    for index in range(shape[0]):
        if index < config.long_window_days:
            continue
        short_good = arrays.quality_1h_24h[index]
        medium_good = arrays.quality_4h_7d[index]
        long_good = arrays.quality_1d_30d[index]
        liquidity = _rolling_sum(arrays.quote_volume_1d, index, 7)
        liquid = liquidity >= config.min_rolling_7d_quote_volume
        current_positive = (
            (arrays.close_1h[index] > 0)
            & (arrays.close_4h[index] > 0)
            & (arrays.close_1d[index] > 0)
            & (arrays.close_1h[index - 1] > 0)
            & (arrays.close_4h[index - config.medium_window_days] > 0)
            & (arrays.close_1d[index - config.long_window_days] > 0)
        )
        eligible = short_good & medium_good & long_good & liquid & current_positive
        short_return = arrays.close_1h[index] / arrays.close_1h[index - 1] - 1.0
        medium_return = arrays.close_4h[index] / arrays.close_4h[index - config.medium_window_days] - 1.0
        long_return = arrays.close_1d[index] / arrays.close_1d[index - config.long_window_days] - 1.0
        combined = (
            config.short_return_weight * short_return
            + config.medium_return_weight * medium_return
            + config.long_return_weight * long_return
        )
        score[index, eligible] = combined[eligible]
        tradable_mask[index, eligible] = True
    return SignalArrays(score=score, tradable_mask=tradable_mask)
```

- [ ] **Step 4: Run signal tests**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_signals.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/strategies/momentum_rotation_v1/signals.py tests/strategies/momentum_rotation_v1/test_signals.py
git commit -m "feat: compute momentum rotation signals"
```

## Task 6: Backtest Kernel

**Files:**
- Create: `src/xsignal/strategies/momentum_rotation_v1/kernel.py`
- Create: `tests/strategies/momentum_rotation_v1/test_kernel.py`

- [ ] **Step 1: Write failing kernel tests**

Create `tests/strategies/momentum_rotation_v1/test_kernel.py`:

```python
from __future__ import annotations

import numpy as np
import pytest

from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
from xsignal.strategies.momentum_rotation_v1.kernel import run_backtest
from xsignal.strategies.momentum_rotation_v1.prepare import PreparedArrays
from xsignal.strategies.momentum_rotation_v1.signals import SignalArrays


def arrays() -> PreparedArrays:
    close = np.array(
        [
            [100.0, 100.0, 100.0],
            [110.0, 90.0, 100.0],
            [121.0, 81.0, 100.0],
        ]
    )
    return PreparedArrays(
        symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        rebalance_times=np.array([0, 1, 2], dtype=object),
        close_1h=close,
        close_4h=close,
        close_1d=close,
        quote_volume_1d=np.ones_like(close),
        complete_1h=np.ones_like(close, dtype=bool),
        complete_4h=np.ones_like(close, dtype=bool),
        complete_1d=np.ones_like(close, dtype=bool),
        quality_1h_24h=np.ones_like(close, dtype=bool),
        quality_4h_7d=np.ones_like(close, dtype=bool),
        quality_1d_30d=np.ones_like(close, dtype=bool),
    )


def test_run_backtest_uses_weights_on_next_period_returns():
    signal = SignalArrays(
        score=np.array(
            [
                [3.0, 2.0, 1.0],
                [1.0, 3.0, 2.0],
                [1.0, 2.0, 3.0],
            ]
        ),
        tradable_mask=np.ones((3, 3), dtype=bool),
    )
    result = run_backtest(arrays(), signal, MomentumRotationConfig(top_n=1, fee_bps=0, slippage_bps=0))

    assert result.weights.shape == (3, 3)
    assert result.weights[0].tolist() == [1.0, 0.0, 0.0]
    assert result.period_returns.tolist() == [0.1, -0.1]
    assert result.equity.tolist() == [1.0, 1.1, 0.99]


def test_run_backtest_applies_turnover_costs():
    signal = SignalArrays(
        score=np.array([[3.0, 2.0, 1.0], [1.0, 3.0, 2.0], [1.0, 2.0, 3.0]]),
        tradable_mask=np.ones((3, 3), dtype=bool),
    )
    result = run_backtest(
        arrays(),
        signal,
        MomentumRotationConfig(top_n=1, fee_bps=10, slippage_bps=0),
    )

    assert result.turnover.tolist() == [1.0, 2.0, 2.0]
    assert result.costs[0] == 0.001


def test_run_backtest_fails_when_no_symbols_are_eligible():
    signal = SignalArrays(score=np.full((3, 3), np.nan), tradable_mask=np.zeros((3, 3), dtype=bool))

    with pytest.raises(ValueError, match="No eligible symbols"):
        run_backtest(arrays(), signal, MomentumRotationConfig())
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_kernel.py -q
```

Expected: FAIL because `kernel.py` does not exist.

- [ ] **Step 3: Implement kernel**

Create `src/xsignal/strategies/momentum_rotation_v1/kernel.py`:

```python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
from xsignal.strategies.momentum_rotation_v1.prepare import PreparedArrays
from xsignal.strategies.momentum_rotation_v1.signals import SignalArrays


@dataclass(frozen=True)
class BacktestResult:
    equity: np.ndarray
    period_returns: np.ndarray
    weights: np.ndarray
    turnover: np.ndarray
    costs: np.ndarray


def _target_weights(score_row: np.ndarray, mask_row: np.ndarray, top_n: int) -> np.ndarray:
    eligible = np.flatnonzero(mask_row & np.isfinite(score_row))
    weights = np.zeros(score_row.shape[0], dtype=np.float64)
    if eligible.size == 0:
        return weights
    selected_count = min(top_n, eligible.size)
    eligible_scores = score_row[eligible]
    selected = eligible[np.argsort(eligible_scores)[-selected_count:]]
    weights[selected] = 1.0 / selected_count
    return weights


def run_backtest(
    arrays: PreparedArrays,
    signals: SignalArrays,
    config: MomentumRotationConfig,
) -> BacktestResult:
    if arrays.close_1d.shape != signals.score.shape or signals.score.shape != signals.tradable_mask.shape:
        raise ValueError("array shapes do not match")
    t_count, n_count = arrays.close_1d.shape
    weights = np.zeros((t_count, n_count), dtype=np.float64)
    turnover = np.zeros(t_count, dtype=np.float64)
    costs = np.zeros(t_count, dtype=np.float64)
    previous = np.zeros(n_count, dtype=np.float64)
    cost_rate = (config.fee_bps + config.slippage_bps) / 10_000.0
    any_eligible = False
    for index in range(t_count):
        target = _target_weights(signals.score[index], signals.tradable_mask[index], config.top_n)
        if np.any(target):
            any_eligible = True
        weights[index] = target
        turnover[index] = np.sum(np.abs(target - previous))
        costs[index] = turnover[index] * cost_rate
        previous = target
    if not any_eligible:
        raise ValueError("No eligible symbols")
    symbol_returns = arrays.close_1d[1:] / arrays.close_1d[:-1] - 1.0
    period_returns = np.sum(weights[:-1] * symbol_returns, axis=1) - costs[:-1]
    if not np.all(np.isfinite(period_returns)):
        raise ValueError("portfolio returns contain NaN or infinite values")
    equity = np.empty(t_count, dtype=np.float64)
    equity[0] = config.initial_equity
    for index, period_return in enumerate(period_returns, start=1):
        equity[index] = equity[index - 1] * (1.0 + period_return)
    if not np.all(np.isfinite(equity)):
        raise ValueError("portfolio equity contains NaN or infinite values")
    return BacktestResult(
        equity=equity,
        period_returns=period_returns,
        weights=weights,
        turnover=turnover,
        costs=costs,
    )
```

- [ ] **Step 4: Run kernel tests**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_kernel.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/strategies/momentum_rotation_v1/kernel.py tests/strategies/momentum_rotation_v1/test_kernel.py
git commit -m "feat: add momentum rotation backtest kernel"
```

## Task 7: Run Artifacts

**Files:**
- Create: `src/xsignal/strategies/momentum_rotation_v1/artifacts.py`
- Create: `tests/strategies/momentum_rotation_v1/test_artifacts.py`

- [ ] **Step 1: Write failing artifact tests**

Create `tests/strategies/momentum_rotation_v1/test_artifacts.py`:

```python
from __future__ import annotations

import json

import numpy as np

from xsignal.strategies.momentum_rotation_v1.artifacts import write_run_artifacts
from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
from xsignal.strategies.momentum_rotation_v1.kernel import BacktestResult
from xsignal.strategies.momentum_rotation_v1.paths import MomentumRotationPaths


def test_write_run_artifacts_creates_manifest_summary_equity_and_positions(tmp_path):
    paths = MomentumRotationPaths(root=tmp_path)
    result = BacktestResult(
        equity=np.array([1.0, 1.1]),
        period_returns=np.array([0.1]),
        weights=np.array([[1.0, 0.0], [0.0, 1.0]]),
        turnover=np.array([1.0, 2.0]),
        costs=np.array([0.001, 0.002]),
    )

    run_dir = write_run_artifacts(
        paths=paths,
        run_id="run123",
        config=MomentumRotationConfig(top_n=1),
        symbols=("BTCUSDT", "ETHUSDT"),
        rebalance_times=np.array(["2026-01-02", "2026-01-03"], dtype=object),
        result=result,
        canonical_manifests=["manifest-1h.json", "manifest-4h.json", "manifest-1d.json"],
        git_commit="abc123",
        runtime_seconds=1.25,
    )

    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "equity_curve.parquet").exists()
    assert (run_dir / "daily_positions.parquet").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert manifest["strategy_name"] == "momentum_rotation_v1"
    assert manifest["config_hash"] == MomentumRotationConfig(top_n=1).config_hash()
    assert manifest["canonical_manifests"] == ["manifest-1h.json", "manifest-4h.json", "manifest-1d.json"]
    assert summary["final_equity"] == 1.1
    assert summary["total_return"] == 0.1
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_artifacts.py -q
```

Expected: FAIL because `artifacts.py` does not exist.

- [ ] **Step 3: Implement artifact writing**

Create `src/xsignal/strategies/momentum_rotation_v1/artifacts.py` with:

```python
from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
from xsignal.strategies.momentum_rotation_v1.kernel import BacktestResult
from xsignal.strategies.momentum_rotation_v1.paths import MomentumRotationPaths


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_run_artifacts(
    *,
    paths: MomentumRotationPaths,
    run_id: str,
    config: MomentumRotationConfig,
    symbols: tuple[str, ...],
    rebalance_times,
    result: BacktestResult,
    canonical_manifests: list[str],
    git_commit: str,
    runtime_seconds: float,
) -> Path:
    run_dir = paths.run_dir(run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "initial_equity": float(result.equity[0]),
        "final_equity": float(result.equity[-1]),
        "total_return": float(result.equity[-1] / result.equity[0] - 1.0),
        "period_count": int(result.period_returns.shape[0]),
        "mean_period_return": float(result.period_returns.mean()) if result.period_returns.size else 0.0,
        "total_cost": float(result.costs.sum()),
    }
    manifest = {
        "strategy_name": config.strategy_name,
        "strategy_version": "v1",
        "git_commit": git_commit,
        "config": config.model_dump(mode="json"),
        "config_hash": config.config_hash(),
        "canonical_manifests": canonical_manifests,
        "symbol_count": len(symbols),
        "symbols": list(symbols),
        "runtime_seconds": runtime_seconds,
        "outputs": {
            "summary": str(run_dir / "summary.json"),
            "equity_curve": str(run_dir / "equity_curve.parquet"),
            "daily_positions": str(run_dir / "daily_positions.parquet"),
        },
    }
    _write_json(run_dir / "manifest.json", manifest)
    _write_json(run_dir / "summary.json", summary)
    equity_table = pa.table(
        {
            "rebalance_time": list(rebalance_times),
            "equity": result.equity.tolist(),
            "turnover": result.turnover.tolist(),
            "cost": result.costs.tolist(),
        }
    )
    pq.write_table(equity_table, run_dir / "equity_curve.parquet")
    position_rows = []
    for t_index, rebalance_time in enumerate(rebalance_times):
        for s_index, symbol in enumerate(symbols):
            weight = float(result.weights[t_index, s_index])
            if weight != 0.0:
                position_rows.append(
                    {
                        "rebalance_time": rebalance_time,
                        "symbol": symbol,
                        "weight": weight,
                    }
                )
    pq.write_table(pa.Table.from_pylist(position_rows), run_dir / "daily_positions.parquet")
    return run_dir
```

- [ ] **Step 4: Run artifact tests**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_artifacts.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/strategies/momentum_rotation_v1/artifacts.py tests/strategies/momentum_rotation_v1/test_artifacts.py
git commit -m "feat: write momentum rotation artifacts"
```

## Task 8: Strategy CLI Orchestration

**Files:**
- Modify: `pyproject.toml`
- Create: `src/xsignal/strategies/momentum_rotation_v1/cli.py`
- Create: `tests/strategies/momentum_rotation_v1/test_cli.py`

- [ ] **Step 1: Write failing CLI smoke test**

Create `tests/strategies/momentum_rotation_v1/test_cli.py`:

```python
from __future__ import annotations

import json

import numpy as np

from xsignal.strategies.momentum_rotation_v1.cli import main
from xsignal.strategies.momentum_rotation_v1.kernel import BacktestResult
from xsignal.strategies.momentum_rotation_v1.prepare import PreparedArrays
from xsignal.strategies.momentum_rotation_v1.signals import SignalArrays


def test_cli_run_writes_artifacts_with_injected_pipeline(tmp_path, monkeypatch):
    arrays = PreparedArrays(
        symbols=("BTCUSDT",),
        rebalance_times=np.array(["2026-01-02", "2026-01-03"], dtype=object),
        close_1h=np.array([[100.0], [101.0]]),
        close_4h=np.array([[100.0], [101.0]]),
        close_1d=np.array([[100.0], [101.0]]),
        quote_volume_1d=np.ones((2, 1)),
        complete_1h=np.ones((2, 1), dtype=bool),
        complete_4h=np.ones((2, 1), dtype=bool),
        complete_1d=np.ones((2, 1), dtype=bool),
        quality_1h_24h=np.ones((2, 1), dtype=bool),
        quality_4h_7d=np.ones((2, 1), dtype=bool),
        quality_1d_30d=np.ones((2, 1), dtype=bool),
    )
    signals = SignalArrays(score=np.array([[1.0], [1.0]]), tradable_mask=np.ones((2, 1), dtype=bool))
    result = BacktestResult(
        equity=np.array([1.0, 1.01]),
        period_returns=np.array([0.01]),
        weights=np.array([[1.0], [1.0]]),
        turnover=np.array([1.0, 0.0]),
        costs=np.array([0.0, 0.0]),
    )

    monkeypatch.setattr("xsignal.strategies.momentum_rotation_v1.cli.prepare_from_canonical", lambda *_args, **_kwargs: (arrays, ["manifest.json"]))
    monkeypatch.setattr("xsignal.strategies.momentum_rotation_v1.cli.compute_momentum_signals", lambda *_args, **_kwargs: signals)
    monkeypatch.setattr("xsignal.strategies.momentum_rotation_v1.cli.run_backtest", lambda *_args, **_kwargs: result)
    monkeypatch.setattr("xsignal.strategies.momentum_rotation_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(["run", "--root", str(tmp_path), "--run-id", "testrun", "--top-n", "1"])

    assert exit_code == 0
    manifest_path = tmp_path / "strategies" / "momentum_rotation_v1" / "runs" / "testrun" / "manifest.json"
    assert json.loads(manifest_path.read_text())["git_commit"] == "abc123"
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_cli.py -q
```

Expected: FAIL because `cli.py` does not exist.

- [ ] **Step 3: Add console script**

Update `pyproject.toml`:

```toml
[project.scripts]
xsignal-export = "xsignal.data.canonical_export:main"
xsignal-momentum-v1 = "xsignal.strategies.momentum_rotation_v1.cli:main"
```

- [ ] **Step 4: Implement CLI and pipeline hook**

Create `src/xsignal/strategies/momentum_rotation_v1/cli.py`:

```python
from __future__ import annotations

import argparse
import subprocess
import time
import uuid
from pathlib import Path

from xsignal.strategies.momentum_rotation_v1.artifacts import write_run_artifacts
from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
from xsignal.strategies.momentum_rotation_v1.kernel import run_backtest
from xsignal.strategies.momentum_rotation_v1.paths import MomentumRotationPaths
from xsignal.strategies.momentum_rotation_v1.prepare import PreparedArrays
from xsignal.strategies.momentum_rotation_v1.signals import compute_momentum_signals


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def prepare_from_canonical(root: Path, config: MomentumRotationConfig) -> tuple[PreparedArrays, list[str]]:
    raise RuntimeError("canonical preparation is not connected")


def _run_command(args: argparse.Namespace) -> Path:
    started = time.perf_counter()
    config = MomentumRotationConfig(
        top_n=args.top_n,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        min_rolling_7d_quote_volume=args.min_rolling_7d_quote_volume,
    )
    arrays, canonical_manifests = prepare_from_canonical(Path(args.root), config)
    signals = compute_momentum_signals(arrays, config)
    result = run_backtest(arrays, signals, config)
    runtime_seconds = time.perf_counter() - started
    paths = MomentumRotationPaths(root=Path(args.root))
    run_id = args.run_id or uuid.uuid4().hex
    return write_run_artifacts(
        paths=paths,
        run_id=run_id,
        config=config,
        symbols=arrays.symbols,
        rebalance_times=arrays.rebalance_times,
        result=result,
        canonical_manifests=canonical_manifests,
        git_commit=_git_commit(),
        runtime_seconds=runtime_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run momentum_rotation_v1")
    subparsers = parser.add_subparsers(dest="command", required=True)
    run_parser = subparsers.add_parser("run")
    run_parser.add_argument("--root", default="data")
    run_parser.add_argument("--run-id")
    run_parser.add_argument("--top-n", type=int, default=10)
    run_parser.add_argument("--fee-bps", type=float, default=5.0)
    run_parser.add_argument("--slippage-bps", type=float, default=5.0)
    run_parser.add_argument("--min-rolling-7d-quote-volume", type=float, default=0.0)
    run_parser.set_defaults(func=_run_command)
    args = parser.parse_args(argv)
    output = args.func(args)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run CLI test**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml src/xsignal/strategies/momentum_rotation_v1/cli.py tests/strategies/momentum_rotation_v1/test_cli.py
git commit -m "feat: add momentum rotation cli shell"
```

## Task 9: Wire Canonical Export To Strategy Preparation

**Files:**
- Modify: `src/xsignal/strategies/momentum_rotation_v1/data.py`
- Modify: `src/xsignal/strategies/momentum_rotation_v1/cli.py`
- Modify: `tests/strategies/momentum_rotation_v1/test_data.py`
- Modify: `tests/strategies/momentum_rotation_v1/test_cli.py`

- [ ] **Step 1: Add failing pipeline tests**

Append to `tests/strategies/momentum_rotation_v1/test_data.py`:

```python
from datetime import datetime, timezone

from xsignal.data.canonical_bars import CanonicalRequest, Partition
from xsignal.data.canonical_export import CanonicalDataset
from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
from xsignal.strategies.momentum_rotation_v1.data import StrategyCanonicalInputs, collect_strategy_inputs


def test_collect_strategy_inputs_calls_export_layer_for_required_timeframes(tmp_path):
    calls = []

    def fake_ensure(request, paths, partitions, exporter):
        calls.append((request, paths, partitions, exporter))
        expected_counts = {"1h": 60, "4h": 240, "1d": 1440}
        for partition in partitions:
            manifest_path = paths.manifest_path(partition)
            parquet_path = paths.published_parquet_path(partition, "abc123")
            parquet_path.parent.mkdir(parents=True, exist_ok=True)
            table = pa.table(
                {
                    "symbol": ["BTCUSDT"],
                    "open_time": pa.array([0], type=pa.timestamp("us", tz="UTC")),
                    "close": [100.0],
                    "quote_volume": [1_000_000.0],
                    "bar_count": [expected_counts[request.timeframe]],
                    "expected_1m_count": [expected_counts[request.timeframe]],
                    "is_complete": [True],
                    "has_synthetic": [False],
                    "fill_policy": ["raw"],
                }
            )
            pq.write_table(table.select(list(REQUIRED_CANONICAL_COLUMNS)), parquet_path)
            manifest_path.write_text(
                json.dumps(
                    {
                        "timeframe": request.timeframe,
                        "fill_policy": request.fill_policy,
                        "parquet_path": str(parquet_path),
                        "row_count": 1,
                    }
                )
            )
        return CanonicalDataset(request=request, root=paths.base, partitions=partitions)

    def fake_bounds(_client):
        return (
            datetime(2026, 1, 1, tzinfo=timezone.utc),
            datetime(2026, 2, 1, tzinfo=timezone.utc),
        )

    result = collect_strategy_inputs(
        root=tmp_path,
        config=MomentumRotationConfig(),
        ensure=fake_ensure,
        discover_bounds=fake_bounds,
        exporter_factory=lambda: object(),
    )

    assert isinstance(result, StrategyCanonicalInputs)
    assert [call[0].timeframe for call in calls] == ["1h", "4h", "1d"]
    assert [call[0].fill_policy for call in calls] == ["raw", "raw", "raw"]
    assert all(call[2] for call in calls)
    assert Partition(timeframe="1h", year=2026, month=1) in calls[0][2]
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_data.py::test_collect_strategy_inputs_calls_export_layer_for_required_timeframes -q
```

Expected: FAIL because `collect_strategy_inputs` does not exist.

- [ ] **Step 3: Implement canonical collection**

Add to `src/xsignal/strategies/momentum_rotation_v1/data.py`:

```python
from collections.abc import Callable

from xsignal.data.canonical_bars import CanonicalRequest
from xsignal.data.canonical_export import (
    ClickHouseExporter,
    discover_full_history_bounds,
    ensure_canonical_bars,
    partitions_for_full_history,
)
from xsignal.data.clickhouse import ClickHouseClient, ClickHouseConfig
from xsignal.data.paths import CanonicalPaths
from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig


@dataclass(frozen=True)
class StrategyCanonicalInputs:
    bars_1h: CanonicalBarTable
    bars_4h: CanonicalBarTable
    bars_1d: CanonicalBarTable
    manifest_paths: tuple[Path, ...]


def collect_strategy_inputs(
    *,
    root: Path,
    config: MomentumRotationConfig,
    ensure: Callable = ensure_canonical_bars,
    discover_bounds: Callable = discover_full_history_bounds,
    exporter_factory: Callable[[], object] | None = None,
) -> StrategyCanonicalInputs:
    client = ClickHouseClient(ClickHouseConfig())
    start, end = discover_bounds(client)
    exporter = exporter_factory() if exporter_factory else ClickHouseExporter(client)
    loaded: dict[str, CanonicalBarTable] = {}
    manifests: list[Path] = []
    for timeframe in config.timeframes:
        request = CanonicalRequest(timeframe=timeframe, fill_policy=config.fill_policy)
        paths = CanonicalPaths(root=root, fill_policy=config.fill_policy)
        partitions = partitions_for_full_history(timeframe, start, end)
        dataset = ensure(request, paths, partitions, exporter)
        timeframe_manifests: list[Path] = []
        for partition in dataset.partitions:
            manifest_path = paths.manifest_path(partition)
            if manifest_path.exists():
                timeframe_manifests.append(manifest_path)
        if not timeframe_manifests:
            raise ValueError(f"no manifests found for timeframe={timeframe}")
        manifests.extend(timeframe_manifests)
        timeframe_tables = [
            load_manifested_table(manifest_path, timeframe=timeframe, fill_policy=config.fill_policy).table
            for manifest_path in timeframe_manifests
        ]
        if not timeframe_tables:
            raise ValueError(f"no canonical tables loaded for timeframe={timeframe}")
        loaded[timeframe] = CanonicalBarTable(
            timeframe=timeframe,
            fill_policy=config.fill_policy,
            manifest_path=paths.manifest_path(dataset.partitions[0]),
            parquet_path=Path("multiple-partitions"),
            table=pa.concat_tables(timeframe_tables, promote_options="default"),
        )
    return StrategyCanonicalInputs(
        bars_1h=loaded["1h"],
        bars_4h=loaded["4h"],
        bars_1d=loaded["1d"],
        manifest_paths=tuple(manifests),
    )
```

- [ ] **Step 4: Wire CLI preparation**

Replace `prepare_from_canonical` in `src/xsignal/strategies/momentum_rotation_v1/cli.py`:

```python
from xsignal.strategies.momentum_rotation_v1.data import collect_strategy_inputs
from xsignal.strategies.momentum_rotation_v1.prepare import prepare_daily_arrays


def prepare_from_canonical(root: Path, config: MomentumRotationConfig) -> tuple[PreparedArrays, list[str]]:
    inputs = collect_strategy_inputs(root=root, config=config)
    arrays = prepare_daily_arrays(
        bars_1h=inputs.bars_1h,
        bars_4h=inputs.bars_4h,
        bars_1d=inputs.bars_1d,
    )
    return arrays, [str(path) for path in inputs.manifest_paths]
```

- [ ] **Step 5: Run data and CLI tests**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_data.py tests/strategies/momentum_rotation_v1/test_cli.py -q
```

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/xsignal/strategies/momentum_rotation_v1/data.py src/xsignal/strategies/momentum_rotation_v1/cli.py tests/strategies/momentum_rotation_v1/test_data.py tests/strategies/momentum_rotation_v1/test_cli.py
git commit -m "feat: wire momentum rotation canonical data"
```

## Task 10: Cache Prepared Arrays

**Files:**
- Modify: `src/xsignal/strategies/momentum_rotation_v1/prepare.py`
- Modify: `tests/strategies/momentum_rotation_v1/test_prepare.py`

- [ ] **Step 1: Add failing cache tests**

Append to `tests/strategies/momentum_rotation_v1/test_prepare.py`:

```python
import json

from xsignal.strategies.momentum_rotation_v1.prepare import load_prepared_arrays, save_prepared_arrays


def test_save_and_load_prepared_arrays_round_trip(tmp_path):
    prepared = prepare_daily_arrays(
        bars_1h=table_for("1h", rows(["BTCUSDT"], [datetime(2026, 1, 1, hour=i, tzinfo=timezone.utc) for i in range(24)], 10.0, 60)),
        bars_4h=table_for("4h", rows(["BTCUSDT"], [datetime(2026, 1, 1, hour=4 * i, tzinfo=timezone.utc) for i in range(6)], 20.0, 240)),
        bars_1d=table_for("1d", rows(["BTCUSDT"], [datetime(2026, 1, 1, tzinfo=timezone.utc)], 30.0, 1440)),
    )

    save_prepared_arrays(tmp_path, prepared)
    loaded = load_prepared_arrays(tmp_path)

    assert json.loads((tmp_path / "symbols.json").read_text()) == ["BTCUSDT"]
    assert loaded.symbols == prepared.symbols
    assert loaded.rebalance_times.tolist() == prepared.rebalance_times.tolist()
    assert np.array_equal(loaded.close_1d, prepared.close_1d, equal_nan=True)
```

- [ ] **Step 2: Run test to verify it fails**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_prepare.py::test_save_and_load_prepared_arrays_round_trip -q
```

Expected: FAIL because cache functions do not exist.

- [ ] **Step 3: Implement cache helpers**

Add to `src/xsignal/strategies/momentum_rotation_v1/prepare.py`:

```python
import json
from pathlib import Path


def save_prepared_arrays(cache_dir: Path, arrays: PreparedArrays) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / "symbols.json").write_text(json.dumps(list(arrays.symbols), indent=2) + "\n")
    np.save(cache_dir / "times_1d.npy", arrays.rebalance_times, allow_pickle=True)
    np.save(cache_dir / "close_1h.npy", arrays.close_1h)
    np.save(cache_dir / "close_4h.npy", arrays.close_4h)
    np.save(cache_dir / "close_1d.npy", arrays.close_1d)
    np.save(cache_dir / "quote_volume_1d.npy", arrays.quote_volume_1d)
    np.save(cache_dir / "complete_1h.npy", arrays.complete_1h)
    np.save(cache_dir / "complete_4h.npy", arrays.complete_4h)
    np.save(cache_dir / "complete_1d.npy", arrays.complete_1d)
    np.save(cache_dir / "quality_1h_24h.npy", arrays.quality_1h_24h)
    np.save(cache_dir / "quality_4h_7d.npy", arrays.quality_4h_7d)
    np.save(cache_dir / "quality_1d_30d.npy", arrays.quality_1d_30d)


def load_prepared_arrays(cache_dir: Path) -> PreparedArrays:
    symbols = tuple(json.loads((cache_dir / "symbols.json").read_text()))
    return PreparedArrays(
        symbols=symbols,
        rebalance_times=np.load(cache_dir / "times_1d.npy", allow_pickle=True),
        close_1h=np.load(cache_dir / "close_1h.npy"),
        close_4h=np.load(cache_dir / "close_4h.npy"),
        close_1d=np.load(cache_dir / "close_1d.npy"),
        quote_volume_1d=np.load(cache_dir / "quote_volume_1d.npy"),
        complete_1h=np.load(cache_dir / "complete_1h.npy"),
        complete_4h=np.load(cache_dir / "complete_4h.npy"),
        complete_1d=np.load(cache_dir / "complete_1d.npy"),
        quality_1h_24h=np.load(cache_dir / "quality_1h_24h.npy"),
        quality_4h_7d=np.load(cache_dir / "quality_4h_7d.npy"),
        quality_1d_30d=np.load(cache_dir / "quality_1d_30d.npy"),
    )
```

- [ ] **Step 4: Run prepare tests**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_prepare.py -q
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/strategies/momentum_rotation_v1/prepare.py tests/strategies/momentum_rotation_v1/test_prepare.py
git commit -m "feat: cache momentum rotation arrays"
```

## Task 11: README And Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document strategy command**

Append to `README.md`:

````markdown
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
````

- [ ] **Step 2: Run focused strategy tests**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1 -q
```

Expected: PASS.

- [ ] **Step 3: Run full test suite**

Run:

```bash
.venv/bin/python -m pytest -q
```

Expected: PASS.

- [ ] **Step 4: Run ruff**

Run:

```bash
.venv/bin/python -m ruff check .
```

Expected: PASS.

- [ ] **Step 5: Run smoke command with injected fixtures or a small local canonical fixture**

If full canonical data is not already exported locally, run the CLI test as the smoke:

```bash
.venv/bin/python -m pytest tests/strategies/momentum_rotation_v1/test_cli.py -q
```

If canonical data is available locally, run:

```bash
.venv/bin/xsignal-momentum-v1 run --root data --run-id smoke-momentum-v1 --top-n 10
```

Expected: command prints `data/strategies/momentum_rotation_v1/runs/smoke-momentum-v1`, and that directory contains all four required artifacts.

- [ ] **Step 6: Inspect git status**

Run:

```bash
git status --short --branch
```

Expected: only intended source, test, and README changes before commit.

- [ ] **Step 7: Commit**

```bash
git add README.md
git commit -m "docs: document momentum rotation strategy"
```

## Self-Review

Spec coverage:

- Strategy-specific scope and non-goals: Tasks 1-11 keep code under `momentum_rotation_v1` and avoid generic engine abstractions.
- Canonical Parquet dependency: Tasks 3 and 9.
- Daily UTC alignment and no lookahead: Tasks 4, 5, and 6.
- Full-window data-quality masks: Task 5.
- Top N long-only equal weight portfolio: Task 6.
- Fixed fee and slippage bps: Tasks 1 and 6.
- Required artifacts and manifests: Task 7.
- CLI and verification: Tasks 8 and 11.

Placeholder scan:

- The plan intentionally uses no vague markers or unspecified "add tests" steps.
- Every code-changing task includes concrete files, code snippets, commands, expected results, and commits.

Type consistency:

- `MomentumRotationConfig`, `MomentumRotationPaths`, `CanonicalBarTable`, `PreparedArrays`, `SignalArrays`, and `BacktestResult` are introduced before use.
- `fill_policy` remains `raw` throughout v1.
- `rebalance_times` are daily UTC close timestamps represented as object arrays for simple serialization.
