# Full Universe WebSocket Ingest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace steady-state full-market REST polling with a full-universe Binance USD-M 1m kline WebSocket ingest pipeline.

**Architecture:** The WebSocket reader only parses and enqueues events. Separate workers route unclosed events to active-position maintenance, batch closed 1m persistence, aggregate higher intervals, and run signal checks from closed aggregate batches. REST remains for metadata, reconciliation, order execution, and gap recovery only.

**Tech Stack:** Python 3.12, `asyncio`, `websockets`, SQLite, Binance USD-M Futures WebSocket/REST, pytest, ruff, systemd on alpha.

---

## File Structure

- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/market_pipeline.py`
  - Own event queues, full-universe routing, closed-bar batching, and worker orchestration helpers.
- Modify `src/xsignal/strategies/volume_price_efficiency_v1/live/stream_daemon.py`
  - Replace steady-state `_poll_closed_1m_loop` plus `_active_position_stream_manager` with full-universe WebSocket reader and pipeline workers.
  - Keep startup recovery, reconnect recovery, reconciliation, rate-limit backoff, and daemon config.
- Modify `src/xsignal/strategies/volume_price_efficiency_v1/live/realtime.py`
  - Add a batch closed-bar processing API so finalized aggregate bars can be processed by `(interval, open_time)` batch.
- Modify `src/xsignal/strategies/volume_price_efficiency_v1/live/recovery.py`
  - Reuse existing cursor-based replay for startup and reconnect gap recovery.
- Modify `src/xsignal/strategies/volume_price_efficiency_v1/live/cli.py`
  - Remove steady-state closed-poll CLI semantics or make them recovery-only names.
- Modify `tests/strategies/volume_price_efficiency_v1/live/test_stream_daemon.py`
  - Cover daemon orchestration and prove steady-state REST polling is not started.
- Create `tests/strategies/volume_price_efficiency_v1/live/test_market_pipeline.py`
  - Cover routing, queue behavior, closed-bar batching, and active-position unclosed updates.
- Modify `tests/strategies/volume_price_efficiency_v1/live/test_realtime.py`
  - Cover batch closed-bar signal behavior.
- Modify `README.md`
  - Document full-universe WebSocket mode, REST recovery role, and operator verification commands.

## Task 1: Market Pipeline Queues And Router

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/market_pipeline.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_market_pipeline.py`

- [ ] **Step 1: Write failing tests for unclosed routing**

Add this test skeleton:

```python
from datetime import datetime, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.market_pipeline import MarketEventRouter
from xsignal.strategies.volume_price_efficiency_v1.live.ws_market import KlineStreamEvent


NOW = datetime(2026, 5, 10, 8, tzinfo=timezone.utc)


class ActiveService:
    def __init__(self):
        self.active = {"BTCUSDT"}
        self.price_events = []

    def has_active_symbol_position(self, symbol: str) -> bool:
        return symbol in self.active

    def process_price_event(self, event, *, allow_pyramid_add=True, allow_stop_replace=True):
        self.price_events.append((event.symbol, event.high, event.close))
        return type("Result", (), {"entries": 0, "adds": 0, "stop_updates": 0})()


def event(symbol: str, *, closed: bool) -> KlineStreamEvent:
    return KlineStreamEvent(
        symbol=symbol,
        interval="1m",
        event_time=NOW,
        open_time=NOW,
        close_time=NOW,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        quote_volume=10.0,
        is_closed=closed,
    )


def test_unclosed_events_update_latest_and_only_active_symbols_are_maintained():
    service = ActiveService()
    router = MarketEventRouter(service=service)

    router.route(event("ETHUSDT", closed=False))
    router.route(event("BTCUSDT", closed=False))

    assert router.latest_unclosed("ETHUSDT").close == 100.5
    assert router.latest_unclosed("BTCUSDT").high == 101.0
    assert service.price_events == [("BTCUSDT", 101.0, 100.5)]
```

- [ ] **Step 2: Verify the test fails**

Run:

```bash
.venv/bin/python -m pytest -q tests/strategies/volume_price_efficiency_v1/live/test_market_pipeline.py::test_unclosed_events_update_latest_and_only_active_symbols_are_maintained
```

Expected: import failure for `MarketEventRouter`.

- [ ] **Step 3: Implement `MarketEventRouter`**

Add this implementation:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from queue import SimpleQueue

from xsignal.strategies.volume_price_efficiency_v1.live.ws_market import KlineStreamEvent


@dataclass
class MarketEventRouter:
    service: object
    closed_queue: SimpleQueue[KlineStreamEvent] = field(default_factory=SimpleQueue)
    _latest_unclosed: dict[str, KlineStreamEvent] = field(default_factory=dict)

    def route(self, event: KlineStreamEvent) -> None:
        if event.is_closed:
            self.closed_queue.put(event)
            return
        self._latest_unclosed[event.symbol] = event
        if self.service.has_active_symbol_position(event.symbol):
            self.service.process_price_event(
                event,
                allow_pyramid_add=True,
                allow_stop_replace=True,
            )

    def latest_unclosed(self, symbol: str) -> KlineStreamEvent | None:
        return self._latest_unclosed.get(symbol)
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/strategies/volume_price_efficiency_v1/live/test_market_pipeline.py
```

Expected: the new routing test passes.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/strategies/volume_price_efficiency_v1/live/market_pipeline.py tests/strategies/volume_price_efficiency_v1/live/test_market_pipeline.py
git commit -m "feat: route websocket market events off the recv path"
```

## Task 2: Closed 1m Batch Worker

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/market_pipeline.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_market_pipeline.py`

- [ ] **Step 1: Write failing test for batch closed-bar persistence**

Add a fake store, fake aggregator, and test:

```python
class FakeStore:
    def __init__(self):
        self.bars = []
        self.cursors = []
        self.commits = 0
        self.connection = self

    def upsert_market_bar(self, row, *, commit=True):
        self.bars.append((row["symbol"], row["interval"], row["open_time"], commit))

    def advance_market_cursor(self, *, symbol, interval, open_time, commit=True):
        self.cursors.append((symbol, interval, open_time, commit))

    def commit(self):
        self.commits += 1


class FakeAggregator:
    def __init__(self, aggregate):
        self.aggregate = aggregate

    def apply_1m_event(self, item):
        return [self.aggregate]


class BatchService(ActiveService):
    def __init__(self):
        super().__init__()
        self.closed_batches = []

    def process_closed_bar_batch(self, events, *, allow_entry, allow_pyramid_add, allow_stop_replace):
        self.closed_batches.append((tuple(item.symbol for item in events), allow_entry))
        return []


def test_closed_worker_batches_store_writes_and_publishes_aggregates():
    store = FakeStore()
    aggregate = event("BTCUSDT", closed=True)
    service = BatchService()
    worker = ClosedBarBatchWorker(
        store=store,
        aggregator=FakeAggregator(aggregate),
        service=service,
    )

    worker.process_many([event("BTCUSDT", closed=True), event("ETHUSDT", closed=True)])

    assert store.commits == 1
    assert ("BTCUSDT", "1m", NOW, False) in store.bars
    assert ("ETHUSDT", "1m", NOW, False) in store.bars
    assert service.closed_batches == [(("BTCUSDT", "BTCUSDT"), True)]
```

- [ ] **Step 2: Verify the test fails**

Run:

```bash
.venv/bin/python -m pytest -q tests/strategies/volume_price_efficiency_v1/live/test_market_pipeline.py::test_closed_worker_batches_store_writes_and_publishes_aggregates
```

Expected: import failure for `ClosedBarBatchWorker`.

- [ ] **Step 3: Implement `ClosedBarBatchWorker`**

Add a worker that mirrors `_process_closed_1m_event` semantics but processes a list:

```python
from collections import defaultdict

from xsignal.strategies.volume_price_efficiency_v1.live.recovery import market_bar_from_event


@dataclass
class ClosedBarBatchWorker:
    store: object
    aggregator: object
    service: object

    def process_many(self, events: list[KlineStreamEvent], *, allow_entry: bool = True) -> None:
        aggregates_by_key: dict[tuple[str, object], list[KlineStreamEvent]] = defaultdict(list)
        for event in events:
            if event.interval != "1m" or not event.is_closed:
                continue
            self.store.upsert_market_bar(market_bar_from_event(event), commit=False)
            self.store.advance_market_cursor(
                symbol=event.symbol,
                interval="1m",
                open_time=event.open_time,
                commit=False,
            )
            self.service.process_price_event(
                event,
                allow_pyramid_add=True,
                allow_stop_replace=True,
            )
            for aggregate in self.aggregator.apply_1m_event(event):
                self.store.upsert_market_bar(market_bar_from_event(aggregate), commit=False)
                aggregates_by_key[(aggregate.interval, aggregate.open_time)].append(aggregate)
        for batch in aggregates_by_key.values():
            self.service.process_closed_bar_batch(
                batch,
                allow_entry=allow_entry,
                allow_pyramid_add=True,
                allow_stop_replace=True,
            )
        self.store.connection.commit()
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/strategies/volume_price_efficiency_v1/live/test_market_pipeline.py
```

Expected: router and batch worker tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/strategies/volume_price_efficiency_v1/live/market_pipeline.py tests/strategies/volume_price_efficiency_v1/live/test_market_pipeline.py
git commit -m "feat: batch closed websocket market bars"
```

## Task 3: Batch Signal API

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/realtime.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_realtime.py`

- [ ] **Step 1: Write failing test for closed-bar batch signal processing**

Add a test using the existing fake broker helpers:

```python
def test_closed_bar_batch_runs_signal_checks_once_per_batch(tmp_path):
    service, store, broker = _service(tmp_path, signal_value=True)
    calls = []

    def signal_mask(arrays, config):
        calls.append(arrays.symbols)
        return np.full(arrays.open.shape, True)

    service.signal_mask_builder = signal_mask

    result = service.process_closed_bar_batch(
        [
            _event(closed=True, high=110.0, close=106.0),
        ],
        allow_entry=True,
        allow_pyramid_add=True,
        allow_stop_replace=True,
    )

    assert result.entries == 1
    assert calls == [("BTCUSDT",)]
    assert [call[0] for call in broker.calls] == ["market_buy", "place_stop_market_close"]
```

- [ ] **Step 2: Verify the test fails**

Run:

```bash
.venv/bin/python -m pytest -q tests/strategies/volume_price_efficiency_v1/live/test_realtime.py::test_closed_bar_batch_runs_signal_checks_once_per_batch
```

Expected: `RealtimeStrategyService` has no `process_closed_bar_batch`.

- [ ] **Step 3: Implement batch API with existing single-bar behavior**

Add this method first, preserving existing behavior:

```python
def process_closed_bar_batch(
    self,
    events: list[KlineStreamEvent],
    *,
    allow_entry: bool = True,
    allow_pyramid_add: bool = True,
    allow_stop_replace: bool = True,
) -> RealtimeEventResult:
    entries = 0
    stop_updates = 0
    adds = 0
    checked = False
    for event in events:
        result = self.process_closed_bar(
            event,
            allow_entry=allow_entry,
            allow_pyramid_add=allow_pyramid_add,
            allow_stop_replace=allow_stop_replace,
        )
        checked = checked or result.closed_signal_checked
        entries += result.entries
        stop_updates += result.stop_updates
        adds += result.adds
    return RealtimeEventResult(
        closed_signal_checked=checked,
        entries=entries,
        stop_updates=stop_updates,
        adds=adds,
    )
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/strategies/volume_price_efficiency_v1/live/test_realtime.py tests/strategies/volume_price_efficiency_v1/live/test_market_pipeline.py
```

Expected: both test files pass.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/strategies/volume_price_efficiency_v1/live/realtime.py tests/strategies/volume_price_efficiency_v1/live/test_realtime.py
git commit -m "feat: process closed bars through a batch API"
```

## Task 4: Full Universe WebSocket Orchestration

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/stream_daemon.py`
- Modify: `tests/strategies/volume_price_efficiency_v1/live/test_stream_daemon.py`

- [ ] **Step 1: Write failing test proving steady-state REST closed polling is not scheduled**

Add a test around task construction:

```python
def test_stream_daemon_starts_full_universe_stream_without_closed_poll_task(monkeypatch):
    created = []

    async def fake_full_universe(**kwargs):
        created.append("full_universe_ws")

    async def fake_closed_poll(**kwargs):
        created.append("closed_poll")

    monkeypatch.setattr(stream_daemon_module, "_full_universe_stream_manager", fake_full_universe)
    monkeypatch.setattr(stream_daemon_module, "_poll_closed_1m_loop", fake_closed_poll)

    tasks = stream_daemon_module._build_market_data_tasks(
        store=object(),
        rest_client=object(),
        aggregator=object(),
        service=object(),
        entry_gate=object(),
        symbols=["BTCUSDT"],
        stop_event=asyncio.Event(),
        counter=stream_daemon_module._EventCounter(limit=1),
        config=StreamDaemonConfig(mode="testnet", db_path="live.sqlite"),
        recovery_lock=asyncio.Lock(),
    )

    for task in tasks:
        task.cancel()

    assert len(tasks) == 1
```

- [ ] **Step 2: Verify the test fails**

Run:

```bash
.venv/bin/python -m pytest -q tests/strategies/volume_price_efficiency_v1/live/test_stream_daemon.py::test_stream_daemon_starts_full_universe_stream_without_closed_poll_task
```

Expected: `_build_market_data_tasks` is missing.

- [ ] **Step 3: Implement task builder and full universe stream manager**

Add `_build_market_data_tasks` that creates one full-universe manager task and
does not create `_poll_closed_1m_loop` or `_active_position_stream_manager`.

```python
def _build_market_data_tasks(
    *,
    store: LiveStore,
    rest_client: BinanceRestClient,
    aggregator: MultiIntervalAggregator,
    service: RealtimeStrategyService,
    entry_gate: EntryHealthGate,
    symbols: list[str],
    stop_event: asyncio.Event,
    counter: "_EventCounter",
    config: StreamDaemonConfig,
    recovery_lock: asyncio.Lock,
) -> list[asyncio.Task]:
    return [
        asyncio.create_task(
            _full_universe_stream_manager(
                store=store,
                rest_client=rest_client,
                aggregator=aggregator,
                service=service,
                entry_gate=entry_gate,
                symbols=symbols,
                stop_event=stop_event,
                counter=counter,
                config=config,
                recovery_lock=recovery_lock,
            )
        )
    ]
```

- [ ] **Step 4: Replace the daemon task list**

In `run_stream_daemon_async`, replace direct creation of `_poll_closed_1m_loop`
and `_active_position_stream_manager` with:

```python
tasks = _build_market_data_tasks(
    store=store,
    rest_client=broker.rest_client,
    aggregator=aggregator,
    service=service,
    entry_gate=entry_gate,
    symbols=symbols,
    stop_event=stop_event,
    counter=counter,
    config=config,
    recovery_lock=recovery_lock,
)
```

- [ ] **Step 5: Run targeted tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/strategies/volume_price_efficiency_v1/live/test_stream_daemon.py
```

Expected: stream daemon tests pass.

- [ ] **Step 6: Commit**

```bash
git add src/xsignal/strategies/volume_price_efficiency_v1/live/stream_daemon.py tests/strategies/volume_price_efficiency_v1/live/test_stream_daemon.py
git commit -m "refactor: run live market data from full universe websocket"
```

## Task 5: Reconnect Gap Recovery

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/stream_daemon.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_stream_daemon.py`

- [ ] **Step 1: Write failing test for reconnect recovery before stream reconnect**

Add this test:

```python
def test_full_universe_stream_recovers_gaps_before_connect(monkeypatch):
    events = []

    async def fake_recover(**kwargs):
        events.append(("recover", tuple(kwargs["symbols"])))

    class OneMessageSocket:
        def __init__(self):
            self.sent = False

        async def __aenter__(self):
            events.append(("connect",))
            return self

        async def __aexit__(self, exc_type, exc, tb):
            return False

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.sent:
                raise StopAsyncIteration
            self.sent = True
            return (
                '{"stream":"btcusdt@kline_1m","data":{"e":"kline","E":1778318492123,'
                '"s":"BTCUSDT","k":{"t":1778313600000,"T":1778313659999,"s":"BTCUSDT",'
                '"i":"1m","o":"100","c":"101","h":"102","l":"99","q":"10","x":false}}}'
            )

    async def fake_connect(url, ping_interval, ping_timeout):
        return OneMessageSocket()

    monkeypatch.setattr(stream_daemon_module, "_recover_symbols_1m_gap_async", fake_recover)
    monkeypatch.setitem(
        sys.modules,
        "websockets",
        types.SimpleNamespace(connect=fake_connect),
    )

    class Service:
        def has_active_symbol_position(self, symbol):
            return False

    async def run():
        stop_event = asyncio.Event()
        counter = stream_daemon_module._EventCounter(limit=1)
        await stream_daemon_module._consume_full_universe_stream_url(
            spec=StreamUrlSpec(url="wss://example", symbols=("BTCUSDT",)),
            store=object(),
            rest_client=object(),
            aggregator=object(),
            service=Service(),
            entry_gate=type("Gate", (), {"mark_stream_error": lambda self, error: None, "snapshot": lambda self: {}})(),
            stop_event=stop_event,
            counter=counter,
            config=StreamDaemonConfig(mode="testnet", db_path="live.sqlite"),
            recovery_lock=asyncio.Lock(),
            recover_before_connect=True,
        )

    asyncio.run(run())

    assert events[0] == ("recover", ("BTCUSDT",))
    assert events[1] == ("connect",)
```

- [ ] **Step 2: Verify the test fails**

Run:

```bash
.venv/bin/python -m pytest -q tests/strategies/volume_price_efficiency_v1/live/test_stream_daemon.py -k reconnect
```

Expected: the new reconnect recovery assertion fails.

- [ ] **Step 3: Wire recovery into `_full_universe_stream_manager`**

Make the manager reuse `_consume_stream_url` semantics:

```python
async def _full_universe_stream_manager(
    *,
    store: LiveStore,
    rest_client: BinanceRestClient,
    aggregator: MultiIntervalAggregator,
    service: RealtimeStrategyService,
    entry_gate: EntryHealthGate,
    symbols: list[str],
    stop_event: asyncio.Event,
    counter: "_EventCounter",
    config: StreamDaemonConfig,
    recovery_lock: asyncio.Lock,
) -> None:
    specs = build_daemon_stream_specs(
        mode=config.mode,
        symbols=symbols,
        max_streams=config.max_streams,
    )
    await asyncio.gather(
        *[
            _consume_full_universe_stream_url(
                spec=spec,
                store=store,
                rest_client=rest_client,
                aggregator=aggregator,
                service=service,
                entry_gate=entry_gate,
                stop_event=stop_event,
                counter=counter,
                config=config,
                recovery_lock=recovery_lock,
            )
            for spec in specs
        ]
    )
```

- [ ] **Step 4: Run targeted tests**

Run:

```bash
.venv/bin/python -m pytest -q tests/strategies/volume_price_efficiency_v1/live/test_stream_daemon.py
```

Expected: reconnect and existing recovery tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/xsignal/strategies/volume_price_efficiency_v1/live/stream_daemon.py tests/strategies/volume_price_efficiency_v1/live/test_stream_daemon.py
git commit -m "fix: recover websocket gaps on reconnect"
```

## Task 6: Operator Docs And Alpha Verification

**Files:**
- Modify: `README.md`
- Verify: alpha service `xsignal-vpe-testnet-stream-daemon.service`

- [ ] **Step 1: Update README**

Document that steady-state market data uses full-universe 1m WebSocket and REST is recovery-only:

```markdown
The stream daemon uses full-universe 1m kline WebSocket streams as the steady-state market feed. REST kline calls are used only during startup and reconnect gap recovery. Unclosed kline updates are memory-only and only active positions consume them for trailing stop and pyramid maintenance.
```

- [ ] **Step 2: Run full local verification**

Run:

```bash
.venv/bin/python -m pytest -q tests/strategies/volume_price_efficiency_v1/live
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
git diff --check
```

Expected:

```text
All tests pass
All checks passed!
git diff --check prints no output
```

- [ ] **Step 3: Commit docs and any final fixes**

```bash
git add README.md
git commit -m "docs: describe full universe websocket ingest"
```

- [ ] **Step 4: Deploy to alpha**

Run:

```bash
rsync -az --delete \
  --exclude '.git/' --exclude '.venv/' --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' --exclude '.secrets/' --exclude '.worktrees/' \
  --exclude 'data/' ./ alpha:/opt/x-signal/
sha="$(git rev-parse --short HEAD)"
ssh alpha "printf '%s\n' '$sha' > /opt/x-signal/DEPLOY_REVISION"
ssh alpha 'systemctl restart xsignal-vpe-testnet-stream-daemon.service'
```

Expected: service restarts and stays active.

- [ ] **Step 5: Verify alpha**

Run:

```bash
ssh alpha 'systemctl show -p ActiveState -p SubState -p NRestarts xsignal-vpe-testnet-stream-daemon.service'
ssh alpha '/opt/x-signal/.venv/bin/xsignal-vpe-status --db /var/lib/xsignal/live/vpe-testnet.sqlite'
ssh alpha 'journalctl -u xsignal-vpe-testnet-stream-daemon.service --since "10 minutes ago" --no-pager | tail -240'
```

Expected:

```text
ActiveState=active
SubState=running
NRestarts does not increase during observation
stream_connected events show purpose="full_universe_market_data"
No repeated recent_rest_429 after prior ban expires
No socket recv_q backlog growth
```

## Self-Review

- Spec coverage: The plan covers full-universe WebSocket ingestion, closed 1m
  persistence, active-position unclosed updates, batch signal processing,
  reconnect recovery, and alpha verification.
- Placeholder scan: The plan contains no TBD markers and every task has exact
  files, commands, and expected outcomes.
- Type consistency: `MarketEventRouter`, `ClosedBarBatchWorker`,
  `process_closed_bar_batch`, `_build_market_data_tasks`, and
  `_full_universe_stream_manager` are introduced before they are referenced by
  subsequent tasks.
