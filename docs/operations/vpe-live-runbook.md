# VPE Live Operations Runbook

This runbook covers the VPE automatic trading service on alpha. It is written
for testnet operations first. Production order submission must stay disabled
unless the operator deliberately enables the live guard.

## Safety Boundaries

- Testnet service: `xsignal-vpe-testnet-stream-daemon.service`
- Production service: `xsignal-vpe-live-stream-daemon.service`
- Testnet database: `/var/lib/xsignal/live/vpe-testnet.sqlite`
- Production database: `/var/lib/xsignal/live/vpe-live.sqlite`
- Production guard file: `/etc/xsignal/enable-live-trading`

Do not create `/etc/xsignal/enable-live-trading` during testnet rehearsals.
The production systemd unit also requires `XSIGNAL_ENABLE_LIVE_TRADING=1` and
`--i-understand-live-order`, but the guard file is the server-side switch that
allows the live unit to start.

The status and default reconciliation commands are read-only. Commands using
`--repair`, lifecycle order commands, or manual exchange actions can change
account state and should be run only when that is the explicit goal.

## Healthy Alpha Check

Run:

```bash
ssh alpha '/opt/x-signal/.venv/bin/xsignal-vpe-status --db /var/lib/xsignal/live/vpe-testnet.sqlite'
ssh alpha 'systemctl show -p ActiveState -p SubState -p NRestarts -p ExecMainPID xsignal-vpe-testnet-stream-daemon.service'
ssh alpha 'journalctl -u xsignal-vpe-testnet-stream-daemon.service --since "30 minutes ago" --no-pager | tail -240'
```

Expected healthy state:

- `OVERALL OK`
- `SERVICE active=True live_active=False live_guard=False`
- `SOCKETS` matches the full-universe stream chunks, currently usually `3`
- socket `recv_q=0` and `send_q=0`
- `UNRESOLVED_ORDER_INTENTS 0`
- `ERROR_LOCKED_POSITIONS 0`
- no recent `rest_429`, `stream_errors`, or reconciliation errors
- recent logs include `reconcile_pass` with `status=clean`

The deployed revision should match `/opt/x-signal/DEPLOY_REVISION`:

```bash
ssh alpha 'cat /opt/x-signal/DEPLOY_REVISION'
```

## Service Model

The stream daemon uses full-universe Binance USD-M `1m` kline WebSocket streams
for steady-state market data. It subscribes to every selected `TRADING` USDT
perpetual symbol, then locally aggregates configured strategy intervals such as
`1h`, `4h`, and `1d`.

REST klines are recovery-only:

- cold startup recovery
- reconnect gap recovery
- exchange metadata and account reconciliation
- order placement, cancellation, and stop replacement

Unclosed `1m` updates are memory-only. They are used for active positions so
trailing stops and pyramid-add checks can react to realtime high/last price.
Signal entries use only locally finalized closed aggregate bars.

Default full-universe chunking is `--max-streams 200`. With 527 symbols this
creates three combined WebSocket connections. Each connection rotates before
Binance's 24-hour hard disconnect using the defaults:

- `--stream-max-lifetime-seconds 82800`
- `--stream-rotation-jitter-seconds 1800`

A daily `stream_rotation_due` log line is normal.

## Startup Recovery Semantics

On startup, the daemon seeds recent aggregate buffers from local SQLite, runs
read-only reconciliation, then recovers missing closed `1m` bars only for
symbols with active strategy positions.

Symbols without active positions are treated like a fresh start:

- their missed downtime bars are not backfilled through REST
- delayed entry signals are not replayed
- delayed pyramid adds are not submitted
- fresh signal checks resume after realtime WebSocket ingestion restarts

This is intentional. The live SQLite history exists mainly for position
maintenance, restart handoff, and audit. Full-market historical research remains
in ClickHouse and canonical Parquet.

## 带仓重启演练

Use this rehearsal when there are small protected testnet positions.

1. Capture the pre-restart state:

```bash
ssh alpha '/opt/x-signal/.venv/bin/xsignal-vpe-status --db /var/lib/xsignal/live/vpe-testnet.sqlite'
```

2. Restart the daemon:

```bash
ssh alpha 'systemctl restart xsignal-vpe-testnet-stream-daemon.service'
```

3. Confirm startup recovery only touched active symbols:

```bash
ssh alpha 'journalctl -u xsignal-vpe-testnet-stream-daemon.service --since "5 minutes ago" --no-pager | grep -E "startup_recovery_|full_universe_streams_started|stream_connected|reconcile_pass"'
```

Expected log shape:

```text
startup_recovery_started symbols=<active_positions> skipped_symbols=<universe-active> universe_symbols=<universe>
startup_recovery_finished symbols=<active_positions> skipped_symbols=<universe-active> universe_symbols=<universe>
full_universe_streams_started streams=3 symbols=<universe>
stream_connected purpose=full_universe_market_data symbols=200
stream_connected purpose=full_universe_market_data symbols=200
stream_connected purpose=full_universe_market_data symbols=<remaining>
```

4. Confirm the post-restart state is still clean:

```bash
ssh alpha '/opt/x-signal/.venv/bin/xsignal-vpe-status --db /var/lib/xsignal/live/vpe-testnet.sqlite'
```

The active position count should be unchanged. `ERROR_LOCKED_POSITIONS` and
`UNRESOLVED_ORDER_INTENTS` should both be `0`.

## 主动轮换演练

This rehearsal proves the service can reconnect before Binance's 24-hour hard
disconnect. It temporarily shortens the WebSocket lifetime, then restores the
normal unit.

1. Install a temporary systemd override:

```bash
ssh alpha 'mkdir -p /etc/systemd/system/xsignal-vpe-testnet-stream-daemon.service.d'
ssh alpha 'cat > /etc/systemd/system/xsignal-vpe-testnet-stream-daemon.service.d/rotation-rehearsal.conf <<EOF
[Service]
ExecStart=
ExecStart=/opt/x-signal/.venv/bin/xsignal-vpe-live stream-daemon --mode testnet --db /var/lib/xsignal/live/vpe-testnet.sqlite --interval 1h --interval 4h --interval 1d --lookback-bars 120 --stream-max-lifetime-seconds 20 --stream-rotation-jitter-seconds 0
EOF'
ssh alpha 'systemctl daemon-reload'
ssh alpha 'systemctl restart xsignal-vpe-testnet-stream-daemon.service'
```

2. Observe rotation:

```bash
ssh alpha 'journalctl -u xsignal-vpe-testnet-stream-daemon.service --since "3 minutes ago" --no-pager | grep -E "stream_rotation_due|stream_connected|market_gap_recovery|rest_429|stream_error"'
ssh alpha '/opt/x-signal/.venv/bin/xsignal-vpe-status --db /var/lib/xsignal/live/vpe-testnet.sqlite'
```

Expected:

- `stream_rotation_due` appears for each chunk
- new `stream_connected` events appear after rotation
- no persistent `stream_error`
- no `rest_429`
- status remains `OVERALL OK`

3. Restore the normal unit:

```bash
ssh alpha 'rm -f /etc/systemd/system/xsignal-vpe-testnet-stream-daemon.service.d/rotation-rehearsal.conf'
ssh alpha 'systemctl daemon-reload'
ssh alpha 'systemctl restart xsignal-vpe-testnet-stream-daemon.service'
ssh alpha 'systemctl cat xsignal-vpe-testnet-stream-daemon.service | grep -E "ExecStart|stream-max-lifetime|stream-rotation"'
```

The final `ExecStart` should not include the short lifetime rehearsal flags.

4. Run the normal health check again.

## Active Position And Stop Checks

List active local positions without submitting orders:

```bash
ssh alpha "sqlite3 /var/lib/xsignal/live/vpe-testnet.sqlite \"select symbol,state,quantity,entry_price,highest_high,stop_price,active_stop_client_order_id,updated_at from positions where state in ('OPEN','ADD_ARMED','ADD_SUBMITTED','STOP_REPLACING','EXITING') order by symbol;\""
```

Run read-only reconciliation for active symbols:

```bash
ssh alpha '/opt/x-signal/.venv/bin/xsignal-vpe-live testnet-reconcile --db /var/lib/xsignal/live/vpe-testnet.sqlite --symbol BTCUSDT'
```

Repeat `--symbol` for each active local position. The expected result is
`error_count=0`; any `ERROR_LOCKED` local state means the symbol must be
inspected before allowing new entries.

Do not close a local strategy position only on the exchange UI. If a position
must be closed manually, keep the exchange action and local SQLite reconciliation
as one audited procedure so the daemon does not restart with stale local state.

## 受控测试网开平仓演练

Use the dedicated testnet commands instead of ad hoc Python snippets. Both
commands require `--i-understand-testnet-order` because they submit real Binance
testnet orders.

Before opening a rehearsal position, confirm the daemon is healthy and note the
current active position count:

```bash
ssh alpha '/opt/x-signal/.venv/bin/xsignal-vpe-status --db /var/lib/xsignal/live/vpe-testnet.sqlite'
```

Open a small protected rehearsal position:

```bash
ssh alpha '/opt/x-signal/.venv/bin/xsignal-vpe-live testnet-open-protected --db /var/lib/xsignal/live/vpe-testnet.sqlite --symbol SOLUSDT --notional 8 --stop-offset-pct 0.05 --i-understand-testnet-order'
```

The command persists the local position and deterministic entry/stop client ids
before submitting orders. It sets isolated margin and `1x` leverage, calculates
a market quantity from the requested notional using Binance symbol rules, opens
a long, and immediately places a close-position stop.

After opening, run read-only reconciliation for the same symbol and confirm the
status remains clean:

```bash
ssh alpha '/opt/x-signal/.venv/bin/xsignal-vpe-live testnet-reconcile --db /var/lib/xsignal/live/vpe-testnet.sqlite --symbol SOLUSDT'
ssh alpha '/opt/x-signal/.venv/bin/xsignal-vpe-status --db /var/lib/xsignal/live/vpe-testnet.sqlite'
```

Close the rehearsal position through the audited close path:

```bash
ssh alpha '/opt/x-signal/.venv/bin/xsignal-vpe-live testnet-close-protected --db /var/lib/xsignal/live/vpe-testnet.sqlite --symbol SOLUSDT --position-id SOLUSDT-1 --i-understand-testnet-order'
```

The close command cancels the strategy stop, records a `MANUAL_RECONCILE`
reduce-only close intent, submits the reduce-only market close, verifies the
symbol is flat, and marks the local position `CLOSED`.

After closing, run status and read-only reconciliation again. Expected state:

- no unresolved order intents
- no error-locked positions
- the closed symbol is absent from active local positions
- the daemon remains `OVERALL OK`

## Deployment To Alpha

From the active worktree:

```bash
rsync -az --delete \
  --exclude '.git/' --exclude '.venv/' --exclude '.pytest_cache/' \
  --exclude '.ruff_cache/' --exclude '.secrets/' --exclude '.worktrees/' \
  --exclude 'data/' ./ alpha:/opt/x-signal/
sha="$(git rev-parse --short HEAD)"
ssh alpha "printf '%s\n' '$sha' > /opt/x-signal/DEPLOY_REVISION"
ssh alpha 'systemctl restart xsignal-vpe-testnet-stream-daemon.service'
```

Then run the healthy alpha check.

## Incident Checklist

If `xsignal-vpe-status` returns `WARN`, inspect in this order:

1. `UNRESOLVED_ORDER_INTENTS` or `ERROR_LOCKED_POSITIONS`
2. recent `reconcile_pass status=error`
3. recent `rest_429` or `-1003`
4. recent `stream_error`
5. socket `recv_q` growth
6. `NRestarts` changes

For REST 429 or 418 style rate limits, do not force repeated restarts. The
daemon is designed to keep running, close the entry gate, and wait for the
backoff window. Restart only after reading the journal and confirming the
service is not already recovering.

If socket backlog grows, first verify the WebSocket chunks and journal:

```bash
ssh alpha '/opt/x-signal/.venv/bin/xsignal-vpe-status --db /var/lib/xsignal/live/vpe-testnet.sqlite'
ssh alpha 'journalctl -u xsignal-vpe-testnet-stream-daemon.service --since "10 minutes ago" --no-pager | tail -240'
```

If the backlog is reproducible, lower `--max-streams` in a temporary systemd
override to split the universe across more sockets, then rerun the normal health
check and remove the override after the test.

## Last Verified Alpha Evidence

On 2026-05-10, alpha was verified on deployed revision `de75d4d`:

- service active and running
- production live service inactive
- production live guard absent
- full universe: 527 symbols in 3 WebSocket connections
- no socket backlog
- no recent REST 429
- no stream errors
- no unresolved order intents
- no error-locked positions
- three small protected testnet positions remained open for rehearsal
