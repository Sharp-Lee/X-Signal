# Volume Price Efficiency Scan Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add research-only parameter scanning and stratified diagnostics for `volume_price_efficiency_v1`, with a reserved holdout window.

**Architecture:** Keep the scan inside the strategy module. Add an OHLCV split helper, scan metric builders, scan artifact writers, and a `scan` CLI subcommand. The scan computes events and matched baselines per compact parameter combination, writes summaries, and never includes holdout rows in ranking.

**Tech Stack:** Python 3.12, NumPy, PyArrow, Parquet, pydantic, pytest, ruff.

---

### Task 1: Holdout Split

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/splits.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_splits.py`

- [ ] Write tests for default tail-window split, disabled holdout, negative holdout, and holdout consuming all rows.
- [ ] Implement `split_research_and_holdout(arrays, holdout_days)` for `OhlcvArrays`.
- [ ] Run `pytest tests/strategies/volume_price_efficiency_v1/test_splits.py -q`.
- [ ] Commit with `feat: add volume price efficiency holdout split`.

### Task 2: Scan Metrics and Buckets

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/scan.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_scan.py`

- [ ] Write tests for scan-row metrics, ranking score, top configs, and feature bucket diagnostics.
- [ ] Implement compact config-grid creation, event/baseline summary flattening, and bucket summaries.
- [ ] Run `pytest tests/strategies/volume_price_efficiency_v1/test_scan.py -q`.
- [ ] Commit with `feat: add volume price efficiency scan metrics`.

### Task 3: Scan Artifacts and Paths

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/paths.py`
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/artifacts.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_paths.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_artifacts.py`

- [ ] Write tests for scan directory validation and scan artifact outputs.
- [ ] Add `scans` and `scan_dir(scan_id)` paths.
- [ ] Add `write_scan_artifacts(...)` for manifest, summary, CSV, top configs, and bucket Parquet.
- [ ] Run artifact/path tests.
- [ ] Commit with `feat: write volume price efficiency scan artifacts`.

### Task 4: CLI Scan Command

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/cli.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_cli.py`
- Modify: `README.md`

- [ ] Write tests that the CLI scan splits holdout, passes grid parameters, and writes scan artifacts.
- [ ] Add `scan` subcommand with `--holdout-days`, grid args, `--ranking-horizon`, and `--top-k`.
- [ ] Update README with the scan command and holdout rule.
- [ ] Run strategy tests and ruff.
- [ ] Commit with `feat: add volume price efficiency scan cli`.

### Task 5: Real Data Smoke and PR

**Files:**
- No code files expected.

- [ ] Run `xsignal-vpe-v1 scan --root data --offline --scan-id smoke-vpe-scan-YYYYMMDD`.
- [ ] Inspect row counts, top configs, holdout metadata, and bucket Parquet.
- [ ] Run full `pytest -q`, `ruff check .`, and `git diff --check`.
- [ ] Push branch, create PR, confirm mergeability, and merge if checks pass.
