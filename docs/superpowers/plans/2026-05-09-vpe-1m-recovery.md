# VPE 1m Recovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace multi-interval WebSocket subscriptions with a 1m-only market stream that locally aggregates configured signal intervals and can recover safely after downtime.

**Architecture:** The daemon subscribes only to `<symbol>@kline_1m`, persists closed 1m bars and cursors, fills reconnect gaps through REST, and feeds locally aggregated closed bars into the existing realtime strategy service. Active-position maintenance uses 1m realtime high/close while looking up the position's original strategy interval for ATR.

**Tech Stack:** Python 3.12, SQLite, Binance USD-M Futures REST/WebSocket, pytest, ruff, systemd.

---

### Task 1: Persistent Market Cursor And Bars

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/store.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_store.py`

- [ ] Add tests for storing closed market bars and advancing per-symbol interval cursors.
- [ ] Add SQLite tables `market_bars` and `market_cursors`.
- [ ] Add `upsert_market_bar`, `list_market_bars`, `get_market_cursor`, and `advance_market_cursor`.
- [ ] Run targeted store tests.

### Task 2: 1m Aggregator

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/bar_aggregator.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_bar_aggregator.py`

- [ ] Add tests for UTC-aligned 1h, 4h, 1d aggregation from closed 1m bars.
- [ ] Add tests proving incomplete buckets do not emit closed bars.
- [ ] Implement interval boundary helpers and `MultiIntervalAggregator`.
- [ ] Run targeted aggregator tests.

### Task 3: Recovery Gap Fill

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/market_data.py`
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/recovery.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_recovery.py`

- [ ] Add tests for REST fetching closed 1m bars from a cursor.
- [ ] Add tests that first startup seeds from the earliest active bucket start.
- [ ] Add recovery replay that persists 1m bars, advances cursors, feeds the aggregator, and marks recovered closed bars as not eligible for new entries.
- [ ] Run targeted recovery tests.

### Task 4: Realtime Service 1m Price Maintenance

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/models.py`
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/store.py`
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/position_store.py`
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/execution.py`
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/realtime.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_realtime.py`

- [ ] Add `strategy_interval` to live positions.
- [ ] Persist the signal interval when opening a position.
- [ ] Add a realtime price-maintenance path for 1m events that uses the active position's strategy interval ATR.
- [ ] Add a catch-up mode that updates stops but does not retroactively pyramid-add.
- [ ] Run targeted realtime tests.

### Task 5: Stream Daemon 1m-Only Wiring

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/stream_daemon.py`
- Modify: `tests/strategies/volume_price_efficiency_v1/live/test_stream_daemon.py`
- Modify: `README.md`

- [ ] Change daemon WebSocket URLs to use only `1m` streams.
- [ ] Seed configured signal interval buffers as before, then recover 1m gaps before opening live streams.
- [ ] On each 1m event, persist closed bars, maintain active positions from realtime high/close, emit locally aggregated closed bars, and screen only newly closed live bars for entries.
- [ ] Document 1m-only stream and recovery semantics.
- [ ] Run targeted daemon tests, full pytest, and ruff.
