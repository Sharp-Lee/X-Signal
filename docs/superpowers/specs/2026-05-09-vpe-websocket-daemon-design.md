# VPE WebSocket Daemon Design

## Goal

Build a persistent Binance USD-M Futures WebSocket daemon for `volume_price_efficiency_v1`.
It must subscribe to all `TRADING` USDT perpetual symbols, screen closed kline signals for
one or more configured intervals, submit testnet/live orders through the existing audited
execution stack, and maintain active positions continuously.

## Non-Negotiable Trading Semantics

- Signal generation uses closed klines only. A kline event is eligible for signal screening
  only when Binance marks it closed.
- Trailing stop and pyramid-add triggers use realtime forming kline `high` and latest close
  price. This deliberately uses unclosed bar updates for position management.
- The strategy remains long-only, one-way mode, single-asset USDT, isolated margin, 1x leverage.
- Per-symbol lock remains in force: while a symbol has an active position, new entry signals
  for that symbol are ignored.
- All positions share one account capital snapshot and one risk budget.
- Production mode remains guarded by both an explicit CLI acknowledgement and the
  `/etc/xsignal/enable-live-trading` file.

## Binance Stream Model

The daemon subscribes to combined kline streams named `<symbol>@kline_<interval>`.
Intervals follow Binance USD-M Futures kline intervals:
`1m, 3m, 5m, 15m, 30m, 1h, 2h, 4h, 6h, 8h, 12h, 1d, 3d, 1w, 1M`.

Each combined WebSocket connection is chunked below Binance's documented stream limit.
The daemon reconnects forever with bounded backoff and reseeds/reconciles after reconnect.

## Runtime Flow

1. Load credentials, mode, DB path, intervals, and live guard.
2. Fetch all `TRADING` USDT perpetual symbols from `exchangeInfo`.
3. Seed each `(interval, symbol)` with recent closed klines through REST.
4. Open combined WebSocket connections for all selected streams.
5. For every kline update:
   - parse and validate the event;
   - update active-position trailing stop and pyramid logic immediately using realtime
     high/latest close;
   - if the kline is closed, append it to the rolling buffer and run closed-bar signal
     screening for that event's symbol and interval;
   - if a signal is accepted, execute entry plus protective stop through persisted order
     intents before exchange submission.
6. Run reconciliation on startup and periodically while the daemon is alive.

## Implementation Boundaries

- `ws_market.py`: stream URL construction, interval validation, and kline event parsing.
- `bar_buffer.py`: rolling seeded OHLCV state that can emit `OhlcvArrays`.
- `realtime.py`: event-driven strategy orchestration and risk-gated execution.
- `stream_daemon.py`: async WebSocket connection supervision and CLI-facing daemon runner.
- Existing execution, reconciliation, store, broker, and risk modules stay authoritative.

## Deployment

Testnet runs as a `systemd` service, not a timer. Live service is installed but guarded.
The existing daily `run-cycle` timer can remain available as a fallback/manual tool, but
the WebSocket daemon is the primary automatic trading task.
