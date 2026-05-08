# VPE Seed Efficiency Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated `seed_efficiency` signal mode that captures low-area, low-volume bars whose price-per-volume efficiency is sharply higher than the previous 3-4 bars.

**Architecture:** Keep the existing `classic` VPE signal untouched and add a second mask builder selected by `VolumePriceEfficiencyConfig.signal_mode`. The new mode reuses existing ATR, volume baseline, efficiency, close position, and body ratio features, then adds no-lookahead local efficiency and bottom-context gates.

**Tech Stack:** Python, NumPy, Pydantic config validation, existing VPE CLI and pytest suite.

---

### Task 1: Config Surface

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/config.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_config.py`

- [ ] Add `signal_mode` with allowed values `classic` and `seed_efficiency`.
- [ ] Add seed controls: `seed_efficiency_lookback`, `seed_min_efficiency_ratio_to_max`, `seed_min_efficiency_ratio_to_mean`, `seed_max_volume_unit`, `seed_bottom_lookback`, and `seed_max_close_position_in_range`.
- [ ] Validate positive lookbacks and ratios, positive max volume unit, and bottom range position between 0 and 1.
- [ ] Extend default-config and invalid-value tests.

### Task 2: Seed Signal Mask

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/features.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_features.py`

- [ ] Write a failing RAVE-like unit test using the 2026-04-03 through 2026-04-07 OHLCV shape.
- [ ] Verify that `classic` strict parameters reject the same bar while `seed_efficiency` accepts it.
- [ ] Implement `build_seed_efficiency_signal_mask` using only bars before or at the signal close:
  - current quality row must be valid
  - current efficiency must exceed previous local max by the configured ratio
  - current efficiency must exceed previous local mean by the configured ratio
  - existing move, volume, close-position, and body filters still apply
  - current volume must not exceed `seed_max_volume_unit`
  - current close must be in the lower configured part of the prior price range
- [ ] Route `build_signal_mask` to either classic or seed mode.

### Task 3: CLI And Scan Plumbing

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/cli.py`
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/scan.py`
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/trailing_scan.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_cli.py`
- Test: `tests/strategies/volume_price_efficiency_v1/test_scan.py`

- [ ] Add CLI arguments for `--signal-mode` and seed controls to `run`, `scan`, `trail`, `trail-scan`, `trail-diagnose`, walk-forward, and regime commands that construct `VolumePriceEfficiencyConfig`.
- [ ] Pass seed config through `build_scan_configs`.
- [ ] Include seed config values in scan rows and trailing scan rows so artifacts are interpretable.
- [ ] Extend CLI tests to assert seed arguments reach `VolumePriceEfficiencyConfig`.

### Task 4: Verification And Local Diagnostic

**Files:**
- No production code beyond Tasks 1-3.

- [ ] Run targeted tests for config, features, CLI, scan, and trailing scan.
- [ ] Run full pytest, ruff, and diff whitespace checks.
- [ ] Run a local diagnostic on canonical RAVEUSDT data to confirm 2026-04-07 signals in `seed_efficiency` mode.
- [ ] Summarize how to run a research-only seed trailing scan without touching holdout.
