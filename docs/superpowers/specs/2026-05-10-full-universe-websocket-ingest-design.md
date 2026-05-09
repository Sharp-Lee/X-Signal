# Full Universe WebSocket Ingest Design

## Goal

Replace the current closed-bar REST polling path with a full-universe Binance
USD-M 1m kline WebSocket market data path.

The live service must monitor every eligible TRADING USDT perpetual symbol in
realtime, generate strategy signals only from locally finalized closed bars, and
maintain active positions from realtime unclosed high/last prices without
allowing market ingestion to block order execution.

## Current Problem

The live daemon currently uses two market data paths:

- Closed-bar market data comes from REST gap recovery and periodic closed 1m
  polling.
- Active positions use a small WebSocket subscription for unclosed high/last
  updates.

This protects the service from full-universe WebSocket recv backlog, but it puts
too much pressure on REST. REST is also the trading control plane, so market
polling, startup recovery, reconciliation, stop replacement, and pyramid adds
can compete for the same external limit.

The previous full-universe WebSocket path failed because the WebSocket receive
loop directly performed SQLite writes, aggregation, feature calculation, signal
checks, and trading maintenance. That coupled IO and strategy work to the socket
read path and allowed recv backlog to build.

## Design Decision

Use full-universe 1m kline WebSocket as the primary market data source.

REST remains only for:

- exchange metadata and account state
- reconciliation
- exchange order placement/cancel/repair
- cold startup gap recovery
- reconnect gap recovery

REST must not be used as the steady-state full-market closed-bar feed.

## Architecture

```text
Binance full-universe 1m kline WebSocket
    -> FullUniverseWsReader
    -> MarketEventRouter
        -> unclosed latest state in memory
        -> active-position price maintenance queue
        -> closed 1m queue
    -> ClosedBarWorker
        -> batch SQLite market_bars writes
        -> market_cursors update
        -> local MultiIntervalAggregator
    -> SignalBatchWorker
        -> batch process finalized 1h/4h/1d or requested interval bars
        -> strategy entry intents
    -> ExecutionWorker / existing broker path
        -> shared capital risk gate
        -> entry, stop replacement, pyramid add
        -> REST rate limit and ban-until backoff
```

## Components

### FullUniverseWsReader

Owns Binance WebSocket connections for all selected symbols.

Responsibilities:

- build combined stream URLs from all selected symbols and 1m kline interval
- keep WebSocket messages flowing with minimal work in the receive loop
- parse only the kline event fields required for routing
- push events into an asyncio queue
- reconnect with backoff
- expose connection and queue-lag telemetry

It must not write SQLite, aggregate bars, calculate features, or submit orders.

### MarketEventRouter

Consumes parsed kline events from the reader.

Responsibilities:

- for unclosed events, update one latest in-memory event per symbol
- for unclosed events on symbols with active positions, forward price events to
  the existing active-position maintenance path
- for closed 1m events, push to the closed-bar worker queue
- drop stale or duplicate unclosed events when the queue is under pressure
- never drop closed 1m events unless the service is shutting down

### ClosedBarWorker

Consumes closed 1m events and owns local market state updates.

Responsibilities:

- batch upsert closed 1m rows into `market_bars`
- advance `market_cursors`
- apply `MultiIntervalAggregator`
- batch upsert finalized aggregate bars
- publish finalized aggregate batches to the signal worker

### SignalBatchWorker

Runs strategy signals on finalized aggregate bars.

Responsibilities:

- group aggregate events by `(interval, open_time)`
- call strategy signal logic once per batch instead of once per symbol
- open entries only when the aggregate bar is closed
- keep the existing per-symbol active-position lock
- preserve the current no-lookahead behavior

### Execution Safety

Existing execution protections remain required:

- `last_stop_replace_at`
- stop replacement minimum interval
- stop replacement minimum price improvement
- Binance 429/418 ban-until backoff
- read-only default reconciliation
- live mode guard file and acknowledgement

Market data workers must never bypass the execution risk gate.

## Restart And Gap Recovery

On startup:

1. Load recent aggregate bars from local SQLite into rolling buffers.
2. Run reconciliation with Binance under startup rate-limit backoff.
3. Use REST only to recover missing closed 1m bars from each symbol cursor to
   the latest closed 1m open time.
4. Start full-universe WebSocket after startup recovery finishes.

On WebSocket reconnect:

1. Record the disconnect timestamp.
2. Reconnect.
3. Recover missing closed 1m bars by cursor using REST, with the existing
   recovery lock and rate-limit backoff.
4. Resume live ingestion.

If REST is temporarily banned, the service remains running, entry gate stays
closed, and the process sleeps until ban-until instead of letting systemd
restart it.

## Storage

SQLite remains the live store for positions, order intents, audit, market
cursors, and closed local market bars.

Closed 1m rows are small enough for SQLite WAL and batched writes. Unclosed
events are not persisted. The latest unclosed state is memory-only.

Parquet and ClickHouse remain research/history systems and are not part of the
live hot path.

## Acceptance Criteria

- The live daemon no longer runs steady-state full-market REST closed-bar
  polling.
- Full-universe 1m WebSocket is the steady-state market feed.
- Closed 1m bars are not lost during normal operation.
- Unclosed events only maintain active positions.
- Signal checks use closed aggregate bars only.
- WebSocket receive loop does not call SQLite, feature calculation, or broker
  methods.
- Execution REST calls remain throttled and independent from market ingestion.
- Startup and reconnect recovery obey Binance 429/418 backoff.
- Local tests cover queue routing, full-universe stream startup, batch closed-bar
  processing, active-position unclosed updates, and REST polling removal.
