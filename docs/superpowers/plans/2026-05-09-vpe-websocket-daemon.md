# VPE WebSocket Daemon Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deploy a persistent Binance USD-M Futures WebSocket daemon for realtime VPE signal screening, order entry, trailing stops, and pyramid adds.

**Architecture:** Add a thin realtime layer beside the existing audited live stack. WebSocket parsing and rolling bar state are isolated from execution; the orchestrator reuses existing store, risk, reconcile, and broker methods.

**Tech Stack:** Python 3.12, `websockets`, SQLite, Binance USD-M Futures REST/WebSocket, pytest, ruff, systemd on alpha.

---

## Files

- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/ws_market.py`
  - Validate Binance intervals, chunk stream names, construct WebSocket URLs, parse kline stream events.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/bar_buffer.py`
  - Maintain rolling OHLCV rows keyed by interval/symbol/open time and expose `OhlcvArrays`.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/realtime.py`
  - Process kline events, run realtime stop/add logic on unclosed bars, and run signal/entry logic on closed bars.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/stream_daemon.py`
  - Async WebSocket supervision, startup seeding, periodic reconciliation, and daemon loop.
- Modify `src/xsignal/strategies/volume_price_efficiency_v1/live/market_data.py`
  - Generalize daily kline helpers to arbitrary Binance intervals while preserving existing daily wrappers.
- Modify `src/xsignal/strategies/volume_price_efficiency_v1/live/cli.py`
  - Add `stream-daemon` command with testnet/live guards.
- Modify `pyproject.toml`
  - Add `websockets` dependency.
- Add `deploy/systemd/xsignal-vpe-testnet-stream-daemon.service`
- Add `deploy/systemd/xsignal-vpe-live-stream-daemon.service`
- Add tests under `tests/strategies/volume_price_efficiency_v1/live/`.

## Tasks

### Task 1: Market Stream Primitives

- [ ] Write tests for interval validation, stream URL chunking, and kline payload parsing.
- [ ] Implement `ws_market.py`.
- [ ] Run the targeted tests and commit.

### Task 2: Generic REST Kline Seeding and Rolling Buffer

- [ ] Write tests proving arbitrary intervals can be fetched and forming bars are excluded.
- [ ] Write tests proving rolling buffers keep bounded history and emit `OhlcvArrays`.
- [ ] Generalize `market_data.py` and implement `bar_buffer.py`.
- [ ] Run targeted tests and commit.

### Task 3: Realtime Event Orchestrator

- [ ] Write tests proving unclosed realtime high/latest close moves stops and triggers add logic.
- [ ] Write tests proving closed bars run signal screening and open entries only after `is_closed`.
- [ ] Write tests proving active symbol lock, shared risk, and reconciliation errors block unsafe entries.
- [ ] Implement `realtime.py`.
- [ ] Run targeted tests and commit.

### Task 4: Async Daemon and CLI

- [ ] Write tests for stream chunk creation, startup seeding, live guard refusal, and CLI parsing.
- [ ] Add `websockets`, implement `stream_daemon.py`, and wire `stream-daemon` into CLI.
- [ ] Run targeted tests and commit.

### Task 5: Systemd, Docs, and Deployment

- [ ] Add testnet/live stream daemon unit files.
- [ ] Update README with realtime daemon semantics and operator commands.
- [ ] Run full local verification.
- [ ] Merge to main, push, deploy to alpha.
- [ ] Enable testnet stream daemon; keep live stream daemon disabled and guarded.
- [ ] Verify alpha service, logs, exchange positions, open orders, and health endpoint.
