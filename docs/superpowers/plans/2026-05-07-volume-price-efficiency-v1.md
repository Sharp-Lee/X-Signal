# Volume Price Efficiency V1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a strategy-specific 4h volume-price-efficiency event study that detects high upward movement per unit of relative volume and measures forward return edge against matched non-signal baselines.

**Architecture:** Keep the strategy isolated under `src/xsignal/strategies/volume_price_efficiency_v1/`. The strategy reads canonical 4h raw Parquet, prepares dense OHLCV arrays, computes normalized efficiency features, emits signal events, samples matched baseline events, and writes event-study artifacts. It does not introduce a generic backtest engine or portfolio simulator.

**Tech Stack:** Python 3.12, NumPy, PyArrow, Pydantic, pytest, ruff, existing X-Signal canonical data modules.

---

## Review Outcome

The design spec was reviewed against the current repository and local canonical
Parquet schema before this plan:

- 4h canonical Parquet contains `open`, `high`, `low`, `close`, and
  `quote_volume`, so the signal can use complete OHLCV.
- `momentum_rotation_v1.data` only loads `close` and `quote_volume`, so this
  strategy needs its own OHLCV loader.
- The spec was amended to add `baseline_events.parquet` and net forward returns
  after round-trip friction. That makes the event study auditable and avoids
  overclaiming tiny gross edges.

No blocking design issues remain.

## File Structure

Create strategy package:

```text
src/xsignal/strategies/volume_price_efficiency_v1/__init__.py
src/xsignal/strategies/volume_price_efficiency_v1/config.py
src/xsignal/strategies/volume_price_efficiency_v1/paths.py
src/xsignal/strategies/volume_price_efficiency_v1/data.py
src/xsignal/strategies/volume_price_efficiency_v1/features.py
src/xsignal/strategies/volume_price_efficiency_v1/events.py
src/xsignal/strategies/volume_price_efficiency_v1/baseline.py
src/xsignal/strategies/volume_price_efficiency_v1/artifacts.py
src/xsignal/strategies/volume_price_efficiency_v1/cli.py
```

Create tests:

```text
tests/strategies/volume_price_efficiency_v1/__init__.py
tests/strategies/volume_price_efficiency_v1/test_config.py
tests/strategies/volume_price_efficiency_v1/test_paths.py
tests/strategies/volume_price_efficiency_v1/test_data.py
tests/strategies/volume_price_efficiency_v1/test_features.py
tests/strategies/volume_price_efficiency_v1/test_events.py
tests/strategies/volume_price_efficiency_v1/test_baseline.py
tests/strategies/volume_price_efficiency_v1/test_artifacts.py
tests/strategies/volume_price_efficiency_v1/test_cli.py
```

Modify existing files:

```text
pyproject.toml
README.md
```

Responsibilities:

- `config.py`: validated event-study parameters and stable config hash.
- `paths.py`: strategy-owned output paths and run-id validation.
- `data.py`: 4h canonical manifest discovery, required OHLCV column validation,
  open-time normalization, and dense OHLCV preparation.
- `features.py`: true range, ATR, volume baseline, move unit, volume unit,
  efficiency, close position, body ratio, and signal mask.
- `events.py`: convert signal mask into event rows with no-lookahead forward
  returns and net forward returns.
- `baseline.py`: deterministic matched non-signal event sampling by symbol and
  month.
- `artifacts.py`: write `manifest.json`, `events.parquet`,
  `baseline_events.parquet`, and `summary.json`.
- `cli.py`: strategy-specific `run` command.

## Task 1: Config And Paths

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/__init__.py`
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/config.py`
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/paths.py`
- Create: `tests/strategies/volume_price_efficiency_v1/__init__.py`
- Create: `tests/strategies/volume_price_efficiency_v1/test_config.py`
- Create: `tests/strategies/volume_price_efficiency_v1/test_paths.py`

- [ ] **Step 1: Write failing config tests**

Create `tests/strategies/volume_price_efficiency_v1/test_config.py`:

```python
from __future__ import annotations

import pytest

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)


def test_default_config_matches_design():
    config = VolumePriceEfficiencyConfig()

    assert config.strategy_name == "volume_price_efficiency_v1"
    assert config.timeframe == "4h"
    assert config.fill_policy == "raw"
    assert config.atr_window == 14
    assert config.volume_window == 60
    assert config.efficiency_lookback == 120
    assert config.efficiency_percentile == 0.90
    assert config.volume_floor == 0.2
    assert config.min_move_unit == 0.5
    assert config.min_volume_unit == 0.3
    assert config.min_close_position == 0.7
    assert config.min_body_ratio == 0.4
    assert config.horizons == (1, 3, 6, 12, 30)
    assert config.fee_bps == 5.0
    assert config.slippage_bps == 5.0
    assert config.baseline_seed == 17


def test_config_hash_is_stable_for_same_payload():
    first = VolumePriceEfficiencyConfig(min_move_unit=0.8)
    second = VolumePriceEfficiencyConfig(min_move_unit=0.8)

    assert first.config_hash() == second.config_hash()
    assert len(first.config_hash()) == 64


def test_config_rejects_invalid_values():
    invalid_kwargs = [
        {"timeframe": "1h"},
        {"fill_policy": "prev_close_zero_volume"},
        {"atr_window": 0},
        {"volume_window": 0},
        {"efficiency_lookback": 0},
        {"efficiency_percentile": 1.0},
        {"efficiency_percentile": 0.0},
        {"volume_floor": 0.0},
        {"min_move_unit": -0.1},
        {"min_volume_unit": -0.1},
        {"min_close_position": 1.1},
        {"min_body_ratio": 1.1},
        {"horizons": ()},
        {"fee_bps": -1},
        {"slippage_bps": -1},
    ]
    for kwargs in invalid_kwargs:
        with pytest.raises(ValueError):
            VolumePriceEfficiencyConfig(**kwargs)
```

- [ ] **Step 2: Write failing path tests**

Create `tests/strategies/volume_price_efficiency_v1/test_paths.py`:

```python
from __future__ import annotations

import pytest

from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)


def test_paths_are_strategy_scoped(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)

    assert paths.base == tmp_path / "strategies" / "volume_price_efficiency_v1"
    assert paths.runs == paths.base / "runs"
    assert paths.run_dir("run123") == paths.runs / "run123"


def test_run_id_rejects_path_traversal(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)

    for bad_run_id in ["", "../abc", "abc/def", "abc\\def"]:
        with pytest.raises(ValueError, match="run_id"):
            paths.run_dir(bad_run_id)
```

- [ ] **Step 3: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/test_config.py tests/strategies/volume_price_efficiency_v1/test_paths.py -q
```

Expected: FAIL with `ModuleNotFoundError`.

- [ ] **Step 4: Implement config and paths**

Create `src/xsignal/strategies/volume_price_efficiency_v1/__init__.py` as an
empty package marker.

Create `src/xsignal/strategies/volume_price_efficiency_v1/config.py`:

```python
from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class VolumePriceEfficiencyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy_name: str = "volume_price_efficiency_v1"
    timeframe: str = "4h"
    fill_policy: str = "raw"
    atr_window: int = 14
    volume_window: int = 60
    efficiency_lookback: int = 120
    efficiency_percentile: float = 0.90
    volume_floor: float = 0.2
    min_move_unit: float = 0.5
    min_volume_unit: float = 0.3
    min_close_position: float = 0.7
    min_body_ratio: float = 0.4
    horizons: tuple[int, ...] = (1, 3, 6, 12, 30)
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    baseline_seed: int = 17

    @field_validator("strategy_name", "timeframe", "fill_policy")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be non-empty")
        return value

    @model_validator(mode="after")
    def _validate(self) -> "VolumePriceEfficiencyConfig":
        if self.strategy_name != "volume_price_efficiency_v1":
            raise ValueError("strategy_name must be volume_price_efficiency_v1")
        if self.timeframe != "4h":
            raise ValueError("timeframe must be 4h")
        if self.fill_policy != "raw":
            raise ValueError("fill_policy must be raw")
        if self.atr_window <= 0:
            raise ValueError("atr_window must be positive")
        if self.volume_window <= 0:
            raise ValueError("volume_window must be positive")
        if self.efficiency_lookback <= 0:
            raise ValueError("efficiency_lookback must be positive")
        if not 0.0 < self.efficiency_percentile < 1.0:
            raise ValueError("efficiency_percentile must be between 0 and 1")
        if self.volume_floor <= 0:
            raise ValueError("volume_floor must be positive")
        if self.min_move_unit < 0:
            raise ValueError("min_move_unit must be non-negative")
        if self.min_volume_unit < 0:
            raise ValueError("min_volume_unit must be non-negative")
        if not 0.0 <= self.min_close_position <= 1.0:
            raise ValueError("min_close_position must be between 0 and 1")
        if not 0.0 <= self.min_body_ratio <= 1.0:
            raise ValueError("min_body_ratio must be between 0 and 1")
        if not self.horizons or any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be unique and sorted")
        if self.fee_bps < 0:
            raise ValueError("fee_bps must be non-negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        return self

    @property
    def round_trip_cost(self) -> float:
        return 2.0 * (self.fee_bps + self.slippage_bps) / 10_000.0

    def config_hash(self) -> str:
        payload = self.model_dump_json(exclude_none=False)
        return hashlib.sha256(payload.encode()).hexdigest()
```

Create `src/xsignal/strategies/volume_price_efficiency_v1/paths.py`:

```python
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


def _validate_plain_id(value: str, field_name: str) -> str:
    if not value or "/" in value or "\\" in value or ".." in value:
        raise ValueError(
            f"{field_name} must be non-empty and must not contain path separators or '..'"
        )
    return value


@dataclass(frozen=True)
class VolumePriceEfficiencyPaths:
    root: Path

    @property
    def base(self) -> Path:
        return self.root / "strategies" / "volume_price_efficiency_v1"

    @property
    def runs(self) -> Path:
        return self.base / "runs"

    def run_dir(self, run_id: str) -> Path:
        return self.runs / _validate_plain_id(run_id, "run_id")
```

Create `tests/strategies/volume_price_efficiency_v1/__init__.py` as an empty
package marker.

- [ ] **Step 5: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/test_config.py tests/strategies/volume_price_efficiency_v1/test_paths.py -q
```

Expected: PASS.

Commit:

```bash
git add src/xsignal/strategies/volume_price_efficiency_v1 tests/strategies/volume_price_efficiency_v1
git commit -m "feat: add volume price efficiency config"
```

## Task 2: 4h OHLCV Data Loader And Dense Arrays

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/data.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_data.py`

- [ ] **Step 1: Write failing data loader tests**

Create `tests/strategies/volume_price_efficiency_v1/test_data.py` with tests
that:

- Build a temporary canonical 4h raw manifest and Parquet file containing all
  required OHLCV columns.
- Assert `load_manifested_table()` validates timeframe and fill policy.
- Assert `load_manifested_table()` normalizes `open_time` to
  `timestamp("s", tz="UTC")`.
- Assert missing required OHLCV columns raise `ValueError`.
- Assert `prepare_ohlcv_arrays()` returns sorted symbols, sorted times, dense
  `open/high/low/close/quote_volume` arrays, and a quality mask.
- Assert incomplete bars and invalid OHLC constraints produce `quality=False`.

Use this test helper:

```python
def write_4h_manifested_partition(tmp_path, rows):
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    import pyarrow as pa
    import pyarrow.parquet as pq

    partition_dir = (
        tmp_path
        / "canonical_bars"
        / "timeframe=4h"
        / "fill_policy=raw"
        / "year=2026"
        / "month=01"
    )
    partition_dir.mkdir(parents=True)
    parquet_path = partition_dir / "bars.abc.parquet"
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, parquet_path)
    manifest = {
        "timeframe": "4h",
        "fill_policy": "raw",
        "parquet_path": str(parquet_path),
        "row_count": len(rows),
    }
    manifest_path = partition_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return manifest_path
```

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/test_data.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `data.py`.

- [ ] **Step 3: Implement data loader**

Create `src/xsignal/strategies/volume_price_efficiency_v1/data.py`:

```python
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


REQUIRED_CANONICAL_COLUMNS = (
    "symbol",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "quote_volume",
    "bar_count",
    "expected_1m_count",
    "is_complete",
    "has_synthetic",
    "fill_policy",
)


@dataclass(frozen=True)
class CanonicalOhlcvTable:
    timeframe: str
    fill_policy: str
    manifest_path: Path
    parquet_path: Path
    table: pa.Table


@dataclass(frozen=True)
class OhlcvArrays:
    symbols: tuple[str, ...]
    open_times: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    quote_volume: np.ndarray
    quality: np.ndarray


def _normalize_open_time_column(table: pa.Table) -> pa.Table:
    field_index = table.schema.get_field_index("open_time")
    open_time = table["open_time"]
    target_type = pa.timestamp("s", tz="UTC")
    if pa.types.is_timestamp(open_time.type):
        normalized = pc.cast(open_time, target_type)
    elif pa.types.is_integer(open_time.type):
        normalized = pa.chunked_array(
            [chunk.cast(pa.int64()).view(target_type) for chunk in open_time.chunks]
        )
    else:
        raise ValueError(f"unsupported open_time type: {open_time.type}")
    return table.set_column(field_index, "open_time", normalized)


def load_manifested_table(
    manifest_path: Path,
    *,
    timeframe: str = "4h",
    fill_policy: str = "raw",
) -> CanonicalOhlcvTable:
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
    table = _normalize_open_time_column(table)
    if table.num_rows != int(manifest["row_count"]):
        raise ValueError("manifest row_count does not match parquet")
    return CanonicalOhlcvTable(
        timeframe=timeframe,
        fill_policy=fill_policy,
        manifest_path=manifest_path,
        parquet_path=parquet_path,
        table=table,
    )


def collect_offline_manifest_paths(root: Path, *, fill_policy: str = "raw") -> tuple[Path, ...]:
    base = root / "canonical_bars" / "timeframe=4h" / f"fill_policy={fill_policy}"
    manifest_paths = tuple(sorted(base.glob("year=*/month=*/manifest.json")))
    if not manifest_paths:
        raise ValueError("offline canonical manifests missing for timeframe=4h")
    return manifest_paths


def load_offline_ohlcv_table(root: Path, *, fill_policy: str = "raw") -> tuple[CanonicalOhlcvTable, tuple[Path, ...]]:
    manifest_paths = collect_offline_manifest_paths(root, fill_policy=fill_policy)
    tables = [
        load_manifested_table(path, timeframe="4h", fill_policy=fill_policy).table
        for path in manifest_paths
    ]
    return (
        CanonicalOhlcvTable(
            timeframe="4h",
            fill_policy=fill_policy,
            manifest_path=manifest_paths[0],
            parquet_path=Path("multiple-partitions"),
            table=pa.concat_tables(tables, promote_options="default"),
        ),
        manifest_paths,
    )


def _is_quality_row(row: dict) -> bool:
    open_ = float(row["open"])
    high = float(row["high"])
    low = float(row["low"])
    close = float(row["close"])
    quote_volume = float(row["quote_volume"])
    return bool(
        row["is_complete"]
        and not row["has_synthetic"]
        and row["bar_count"] == row["expected_1m_count"]
        and open_ > 0
        and high > 0
        and low > 0
        and close > 0
        and high >= max(open_, close)
        and low <= min(open_, close)
        and quote_volume > 0
    )


def prepare_ohlcv_arrays(table: CanonicalOhlcvTable) -> OhlcvArrays:
    rows = table.table.to_pylist()
    symbols = tuple(sorted({row["symbol"] for row in rows}))
    open_times = tuple(sorted({row["open_time"] for row in rows}))
    shape = (len(open_times), len(symbols))
    symbol_index = {symbol: index for index, symbol in enumerate(symbols)}
    time_index = {open_time: index for index, open_time in enumerate(open_times)}
    arrays = {
        name: np.full(shape, np.nan, dtype=np.float64)
        for name in ("open", "high", "low", "close", "quote_volume")
    }
    quality = np.zeros(shape, dtype=bool)
    for row in rows:
        t_index = time_index[row["open_time"]]
        s_index = symbol_index[row["symbol"]]
        for name in arrays:
            arrays[name][t_index, s_index] = float(row[name])
        quality[t_index, s_index] = _is_quality_row(row)
    return OhlcvArrays(
        symbols=symbols,
        open_times=np.array(open_times, dtype=object),
        open=arrays["open"],
        high=arrays["high"],
        low=arrays["low"],
        close=arrays["close"],
        quote_volume=arrays["quote_volume"],
        quality=quality,
    )
```

- [ ] **Step 4: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/test_data.py -q
```

Expected: PASS.

Commit:

```bash
git add src/xsignal/strategies/volume_price_efficiency_v1/data.py tests/strategies/volume_price_efficiency_v1/test_data.py
git commit -m "feat: load volume price efficiency ohlcv data"
```

## Task 3: Feature Calculation And Signal Mask

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/features.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_features.py`

- [ ] **Step 1: Write failing feature tests**

Create tests that build small `OhlcvArrays` and assert:

- `true_range` uses prior close for gaps.
- `volume_baseline[t]` uses previous bars only, not the current bar.
- `efficiency_threshold[t]` uses previous efficiency values only.
- Long upper-wick bars fail because `close_position` or `body_ratio` is too low.
- A designed high-efficiency bar passes the signal rule.
- A row with `quality=False` never signals.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/test_features.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `features.py`.

- [ ] **Step 3: Implement features**

Create `FeatureArrays` with:

```python
true_range
atr
move_unit
volume_baseline
volume_unit
efficiency
efficiency_threshold
close_position
body_ratio
signal
```

Implementation rules:

- Loop over time and vectorize over symbols.
- Use `np.nanmean`, `np.nanmedian`, and `np.nanpercentile`.
- Exclude current bar from volume baseline and efficiency threshold.
- Set all derived values to `np.nan` when required lookback values are missing.
- Signal requires the full boolean rule from the spec.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/test_features.py -q
```

Expected: PASS.

Commit:

```bash
git add src/xsignal/strategies/volume_price_efficiency_v1/features.py tests/strategies/volume_price_efficiency_v1/test_features.py
git commit -m "feat: detect volume price efficiency signals"
```

## Task 4: Signal Events And Forward Returns

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/events.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_events.py`

- [ ] **Step 1: Write failing event tests**

Create tests that assert:

- Signal at index `t` enters at `open[t + 1]`.
- `forward_return_H = close[t + H] / open[t + 1] - 1`.
- Net returns subtract `config.round_trip_cost`.
- A horizon is `None` or `NaN` when `t + H` is outside the array.
- Missing entry open or forward close makes that horizon unavailable.
- Signal event rows include all required feature columns.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/test_events.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `events.py`.

- [ ] **Step 3: Implement event extraction**

Create:

```python
def build_signal_events(arrays: OhlcvArrays, features: FeatureArrays, config: VolumePriceEfficiencyConfig) -> list[dict]:
    rows: list[dict] = []
    # Iterate over true values in features.signal and return one event row per
    # signal. Each row includes symbol/time identifiers, feature values, gross
    # forward returns, and net forward returns for every configured horizon.
    return rows
```

Use `signal_open_time = arrays.open_times[t]`, `decision_time = signal_open_time
+ 4h`, and `entry_open_time = arrays.open_times[t + 1]`.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/test_events.py -q
```

Expected: PASS.

Commit:

```bash
git add src/xsignal/strategies/volume_price_efficiency_v1/events.py tests/strategies/volume_price_efficiency_v1/test_events.py
git commit -m "feat: build volume price efficiency event returns"
```

## Task 5: Matched Baseline Sampling

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/baseline.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_baseline.py`

- [ ] **Step 1: Write failing baseline tests**

Create tests that assert:

- Baseline rows are non-signal rows.
- Matching is by same symbol and same `YYYY-MM` month as the signal event.
- The baseline count per symbol-month does not exceed signal count when enough
  non-signal rows exist.
- Baseline sampling is deterministic with the same `baseline_seed`.
- Baseline rows include `matched_signal_month` and
  `matched_signal_count_for_symbol_month`.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/test_baseline.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `baseline.py`.

- [ ] **Step 3: Implement baseline sampling**

Create:

```python
def build_baseline_events(arrays: OhlcvArrays, features: FeatureArrays, config: VolumePriceEfficiencyConfig) -> list[dict]:
    rows: list[dict] = []
    # Count signal rows per symbol-month, sample deterministic quality
    # non-signal rows from the same groups, and return event-shaped rows with
    # matching metadata.
    return rows
```

Implementation:

- Count signal events per `(symbol, YYYY-MM)`.
- For each group, collect quality non-signal candidate bars with enough next-open
  and at least one forward horizon available.
- Shuffle candidates with `np.random.default_rng(config.baseline_seed)`.
- Take up to the group signal count.
- Use the same event row builder as signal events, with added matching fields.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/test_baseline.py -q
```

Expected: PASS.

Commit:

```bash
git add src/xsignal/strategies/volume_price_efficiency_v1/baseline.py tests/strategies/volume_price_efficiency_v1/test_baseline.py
git commit -m "feat: add matched non-signal baseline events"
```

## Task 6: Artifacts And Summary Metrics

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/artifacts.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_artifacts.py`

- [ ] **Step 1: Write failing artifact tests**

Create tests that assert:

- `build_event_study_summary()` reports event count, baseline count, symbol
  count, first/last signal time, per-horizon gross metrics, net metrics,
  baseline metrics, and event-minus-baseline deltas.
- Empty returns do not crash; their metrics are `None`.
- `write_run_artifacts()` writes `manifest.json`, `summary.json`,
  `events.parquet`, and `baseline_events.parquet`.
- Manifest includes config, config hash, git commit, canonical manifests, symbol
  count, and output paths.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/test_artifacts.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `artifacts.py`.

- [ ] **Step 3: Implement artifacts**

Create `artifacts.py` with:

```python
from pathlib import Path
from typing import Any

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)


def build_event_study_summary(
    events: list[dict],
    baseline_events: list[dict],
    horizons: tuple[int, ...],
) -> dict[str, Any]:
    return {
        "event_count": len(events),
        "baseline_event_count": len(baseline_events),
        "horizons": {},
    }


def write_run_artifacts(
    *,
    paths: VolumePriceEfficiencyPaths,
    run_id: str,
    config: VolumePriceEfficiencyConfig,
    events: list[dict],
    baseline_events: list[dict],
    symbols: tuple[str, ...],
    canonical_manifests: list[str],
    git_commit: str,
    runtime_seconds: float,
) -> Path:
    run_dir = paths.run_dir(run_id)
    return run_dir
```

Use `pyarrow.Table.from_pylist()` for event files. For empty events, write an
empty table with an explicit schema that includes all required columns.

- [ ] **Step 4: Run tests and commit**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/test_artifacts.py -q
```

Expected: PASS.

Commit:

```bash
git add src/xsignal/strategies/volume_price_efficiency_v1/artifacts.py tests/strategies/volume_price_efficiency_v1/test_artifacts.py
git commit -m "feat: write volume price efficiency artifacts"
```

## Task 7: CLI, Entry Point, README, And Smoke Run

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/cli.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_cli.py`
- Modify: `pyproject.toml`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI tests**

Create tests that monkeypatch data loading and assert:

- `main(["run", "--root", tmp_path, "--run-id", "testrun", "--offline"])`
  writes a run directory.
- CLI passes config values from flags into the pipeline.
- CLI refuses non-offline mode in the first version with a clear error, or
  explicitly documents and tests export support if implemented.

- [ ] **Step 2: Run tests to verify they fail**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/test_cli.py -q
```

Expected: FAIL with `ModuleNotFoundError` for `cli.py`.

- [ ] **Step 3: Implement CLI**

Create `cli.py` with:

```python
from pathlib import Path


def run_event_study(args: argparse.Namespace) -> Path:
    config = VolumePriceEfficiencyConfig(
        atr_window=args.atr_window,
        volume_window=args.volume_window,
        efficiency_lookback=args.efficiency_lookback,
        efficiency_percentile=args.efficiency_percentile,
        volume_floor=args.volume_floor,
        min_move_unit=args.min_move_unit,
        min_volume_unit=args.min_volume_unit,
        min_close_position=args.min_close_position,
        min_body_ratio=args.min_body_ratio,
        fee_bps=args.fee_bps,
        slippage_bps=args.slippage_bps,
        baseline_seed=args.baseline_seed,
    )
    table, manifests = load_offline_ohlcv_table(Path(args.root), fill_policy=config.fill_policy)
    arrays = prepare_ohlcv_arrays(table)
    features = compute_features(arrays, config)
    events = build_signal_events(arrays, features, config)
    baseline_events = build_baseline_events(arrays, features, config)
    return write_run_artifacts(
        paths=VolumePriceEfficiencyPaths(root=Path(args.root)),
        run_id=args.run_id,
        config=config,
        events=events,
        baseline_events=baseline_events,
        symbols=arrays.symbols,
        canonical_manifests=[str(path) for path in manifests],
        git_commit=_git_commit(),
        runtime_seconds=runtime_seconds,
    )
```

Add script entry to `pyproject.toml`:

```toml
xsignal-vpe-v1 = "xsignal.strategies.volume_price_efficiency_v1.cli:main"
```

Update README with:

```bash
xsignal-vpe-v1 run --root data --offline --run-id smoke-vpe-v1
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1 -q
```

Expected: PASS.

- [ ] **Step 5: Run full verification**

Run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
git diff --check
```

Expected: all pass.

- [ ] **Step 6: Run real offline smoke**

Run:

```bash
.venv/bin/xsignal-vpe-v1 run --root data --offline --run-id smoke-vpe-v1-20260507
```

Expected:

```text
data/strategies/volume_price_efficiency_v1/runs/smoke-vpe-v1-20260507
```

Then inspect:

```bash
sed -n '1,220p' data/strategies/volume_price_efficiency_v1/runs/smoke-vpe-v1-20260507/summary.json
```

- [ ] **Step 7: Commit**

Commit:

```bash
git add src/xsignal/strategies/volume_price_efficiency_v1/cli.py tests/strategies/volume_price_efficiency_v1/test_cli.py pyproject.toml README.md
git commit -m "feat: add volume price efficiency event study cli"
```

## Final Verification

Before opening a PR or merging, run:

```bash
.venv/bin/python -m pytest -q
.venv/bin/python -m ruff check .
git diff --check HEAD~7 HEAD
git status --short --branch
```

Expected:

- Full pytest passes.
- Ruff passes.
- Diff check prints no whitespace errors.
- Working tree is clean except expected ignored smoke artifacts under
  `data/strategies/`.

## Execution Notes

- Use TDD for every task: write the failing test first, verify the failure, then
  implement the smallest passing code.
- Do not reuse `momentum_rotation_v1.data`; it drops OHLC columns required by
  this signal.
- Do not add bottom filters, portfolio allocation, trailing stops, or scan
  selectors in this phase.
- Do not forward-fill OHLCV inputs for signal calculation or forward returns.
- Keep artifacts strategy-owned under
  `data/strategies/volume_price_efficiency_v1/`.
