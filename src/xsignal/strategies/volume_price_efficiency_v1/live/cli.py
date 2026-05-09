from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from xsignal.strategies.volume_price_efficiency_v1.live.binance_adapter import (
    BINANCE_USD_FUTURES_TESTNET_BASE_URL,
    BinanceUsdFuturesTestnetBroker,
    build_usd_futures_broker,
)
from xsignal.strategies.volume_price_efficiency_v1.live.binance_rest import (
    BinanceCredentials,
    BinanceRestClient,
)
from xsignal.strategies.volume_price_efficiency_v1.live.ids import build_client_order_id
from xsignal.strategies.volume_price_efficiency_v1.live.market_data import load_recent_daily_arrays
from xsignal.strategies.volume_price_efficiency_v1.live.order_normalizer import SymbolRules
from xsignal.strategies.volume_price_efficiency_v1.live.reconcile import run_reconciliation_pass
from xsignal.strategies.volume_price_efficiency_v1.live.runner import run_live_cycle
from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig
from xsignal.strategies.volume_price_efficiency_v1.live.status import (
    build_status_snapshot,
    collect_system_snapshot,
    render_status_text,
)
from xsignal.strategies.volume_price_efficiency_v1.live.stream_daemon import (
    DEFAULT_REALTIME_INTERVALS,
    StreamDaemonConfig,
    run_stream_daemon,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore
from xsignal.strategies.volume_price_efficiency_v1.live.testnet_lifecycle import (
    run_testnet_lifecycle,
)
from xsignal.strategies.volume_price_efficiency_v1.live.testnet_rehearsal import (
    close_rehearsal_position,
    open_protected_rehearsal_position,
)


LOCAL_TESTNET_ENV_FILE = Path(".secrets/binance-testnet.env")
LOCAL_LIVE_ENV_FILE = Path(".secrets/binance-live.env")
SYSTEM_LIVE_ENABLE_FILE = Path("/etc/xsignal/enable-live-trading")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xsignal-vpe-live")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay")
    replay.add_argument("--root", type=Path, default=Path("data"))
    replay.add_argument("--db", type=Path, required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--db", type=Path, required=True)
    status.add_argument("--json", action="store_true")
    status.add_argument("--no-system", action="store_true")

    reconcile = subparsers.add_parser("reconcile")
    reconcile.add_argument("--db", type=Path, required=True)

    smoke = subparsers.add_parser("testnet-smoke")
    smoke.add_argument("--symbol", required=True)
    smoke.add_argument("--quantity", type=float, default=0.001)
    smoke.add_argument("--submit-test-order", action="store_true")

    lifecycle = subparsers.add_parser("testnet-lifecycle")
    lifecycle.add_argument("--symbol", required=True)
    lifecycle.add_argument("--quantity", type=float, default=0.001)
    lifecycle.add_argument("--stop-offset-pct", type=float, default=0.05)
    lifecycle.add_argument("--db", type=Path)
    lifecycle.add_argument("--i-understand-testnet-order", action="store_true")

    testnet_reconcile = subparsers.add_parser("testnet-reconcile")
    testnet_reconcile.add_argument("--db", type=Path, required=True)
    testnet_reconcile.add_argument("--symbol", action="append", required=True)
    testnet_reconcile.add_argument("--repair", action="store_true")
    testnet_reconcile.add_argument("--i-understand-testnet-order", action="store_true")

    testnet_open = subparsers.add_parser("testnet-open-protected")
    testnet_open.add_argument("--db", type=Path, required=True)
    testnet_open.add_argument("--symbol", required=True)
    testnet_open.add_argument("--notional", type=float, required=True)
    testnet_open.add_argument("--stop-offset-pct", type=float, default=0.05)
    testnet_open.add_argument("--i-understand-testnet-order", action="store_true")

    testnet_close = subparsers.add_parser("testnet-close-protected")
    testnet_close.add_argument("--db", type=Path, required=True)
    testnet_close.add_argument("--symbol", required=True)
    testnet_close.add_argument("--position-id")
    testnet_close.add_argument("--i-understand-testnet-order", action="store_true")

    run_cycle = subparsers.add_parser("run-cycle")
    run_cycle.add_argument("--mode", choices=["testnet", "live"], required=True)
    run_cycle.add_argument("--db", type=Path, required=True)
    run_cycle.add_argument("--symbol", action="append")
    run_cycle.add_argument("--max-symbols", type=int)
    run_cycle.add_argument("--lookback-bars", type=int, default=120)
    run_cycle.add_argument("--env-file", type=Path)
    run_cycle.add_argument("--i-understand-live-order", action="store_true")

    stream_daemon = subparsers.add_parser("stream-daemon")
    stream_daemon.add_argument("--mode", choices=["testnet", "live"], required=True)
    stream_daemon.add_argument("--db", type=Path, required=True)
    stream_daemon.add_argument("--interval", action="append", default=[])
    stream_daemon.add_argument("--max-symbols", type=int)
    stream_daemon.add_argument("--max-streams", type=int)
    stream_daemon.add_argument("--stream-max-lifetime-seconds", type=float)
    stream_daemon.add_argument("--stream-rotation-jitter-seconds", type=float)
    stream_daemon.add_argument("--lookback-bars", type=int, default=120)
    stream_daemon.add_argument("--seed-sleep-ms", type=int, default=20)
    stream_daemon.add_argument("--recovery-sleep-ms", type=int, default=500)
    stream_daemon.add_argument("--closed-poll-sleep-ms", type=int, default=25)
    stream_daemon.add_argument("--closed-poll-grace-seconds", type=float, default=2.0)
    stream_daemon.add_argument("--closed-poll-fetch-limit", type=int, default=99)
    stream_daemon.add_argument("--reconcile-interval-seconds", type=float, default=300.0)
    stream_daemon.add_argument("--env-file", type=Path)
    stream_daemon.add_argument("--stop-after-events", type=int)
    stream_daemon.add_argument("--i-understand-live-order", action="store_true")

    live_smoke = subparsers.add_parser("live-smoke")
    live_smoke.add_argument("--symbol", default="BTCUSDT")
    live_smoke.add_argument("--env-file", type=Path, default=LOCAL_LIVE_ENV_FILE)
    return parser


def _credentials_from_env(env_file: Path | None = LOCAL_TESTNET_ENV_FILE) -> BinanceCredentials | None:
    file_values = _read_env_file(env_file)
    api_key = os.environ.get("BINANCE_API_KEY") or file_values.get("BINANCE_API_KEY")
    secret_key = os.environ.get("BINANCE_SECRET_KEY") or file_values.get("BINANCE_SECRET_KEY")
    if not api_key or not secret_key:
        return None
    return BinanceCredentials(api_key=api_key, secret_key=secret_key)


def _read_env_file(env_file: Path | None) -> dict[str, str]:
    if env_file is None or not env_file.exists():
        return {}
    values: dict[str, str] = {}
    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip().strip("\"'")
    return values


def _build_testnet_rest_client(env_file: Path | None = LOCAL_TESTNET_ENV_FILE) -> BinanceRestClient | None:
    credentials = _credentials_from_env(env_file=env_file)
    if credentials is None:
        return None
    return BinanceRestClient(
        base_url=BINANCE_USD_FUTURES_TESTNET_BASE_URL,
        credentials=credentials,
    )


def _default_env_file(mode: str, env_file: Path | None) -> Path | None:
    if env_file is not None:
        return env_file
    return LOCAL_TESTNET_ENV_FILE if mode == "testnet" else LOCAL_LIVE_ENV_FILE


def _live_enabled() -> bool:
    return os.environ.get("XSIGNAL_ENABLE_LIVE_TRADING") == "1" or SYSTEM_LIVE_ENABLE_FILE.exists()


def run_testnet_smoke(
    *,
    symbol: str,
    submit_test_order: bool,
    quantity: float,
    rest_client=None,
    broker=None,
) -> int:
    rest_client = rest_client or _build_testnet_rest_client()
    if rest_client is None:
        print(
            "BINANCE_API_KEY and BINANCE_SECRET_KEY are required for testnet-smoke",
            file=sys.stderr,
        )
        return 2
    broker = broker or BinanceUsdFuturesTestnetBroker(rest_client)
    server_time = rest_client.request("GET", "/fapi/v1/time")
    exchange_info = rest_client.request("GET", "/fapi/v1/exchangeInfo")
    account = rest_client.request("GET", "/fapi/v3/account", signed=True)
    position_mode = broker.get_position_mode()
    asset_mode = broker.get_multi_assets_mode()
    matching_symbols = [
        item for item in exchange_info.get("symbols", []) if item.get("symbol") == symbol
    ]
    output = {
        "mode": "testnet",
        "symbol": symbol,
        "server_time": server_time.get("serverTime"),
        "symbol_found": bool(matching_symbols),
        "symbol_status": matching_symbols[0].get("status") if matching_symbols else None,
        "position_mode": position_mode,
        "asset_mode": asset_mode,
        "equity": account.get("totalMarginBalance"),
        "available_balance": account.get("availableBalance"),
        "test_order_submitted": False,
    }
    if submit_test_order:
        client_order_id = build_client_order_id(
            env="testnet",
            intent="ENTRY",
            symbol=symbol,
            position_id="smoke",
            sequence=1,
        )
        broker.test_order(
            symbol=symbol,
            side="BUY",
            order_type="MARKET",
            quantity=quantity,
            client_order_id=client_order_id,
        )
        output["test_order_submitted"] = True
        output["test_order_client_id"] = client_order_id
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def run_testnet_lifecycle_command(
    *,
    symbol: str,
    quantity: float,
    stop_offset_pct: float,
    acknowledge: bool,
    db: Path | None = None,
    rest_client=None,
    broker=None,
    lifecycle_runner=run_testnet_lifecycle,
) -> int:
    if not acknowledge:
        print(
            "testnet-lifecycle requires --i-understand-testnet-order",
            file=sys.stderr,
        )
        return 2
    rest_client = rest_client or (None if broker is not None else _build_testnet_rest_client())
    if rest_client is None and broker is None:
        print(
            "BINANCE_API_KEY and BINANCE_SECRET_KEY are required for testnet-lifecycle",
            file=sys.stderr,
        )
        return 2

    broker = broker or BinanceUsdFuturesTestnetBroker(rest_client)
    metadata = broker.get_symbol_metadata(symbol)
    if metadata.status != "TRADING":
        print(f"{symbol} is not TRADING on Binance USD-M testnet", file=sys.stderr)
        return 2

    store = None
    if db is not None:
        store = LiveStore.open(db)
        store.initialize()

    result = lifecycle_runner(
        broker=broker,
        symbol=symbol,
        quantity=quantity,
        stop_offset_pct=stop_offset_pct,
        store=store,
        price_tick=metadata.price_tick,
        symbol_rules=SymbolRules.from_metadata(metadata),
    )
    print(json.dumps(vars(result), indent=2, sort_keys=True))
    return 0


def run_testnet_reconcile_command(
    *,
    db: Path,
    symbols: list[str],
    repair: bool,
    acknowledge: bool,
    rest_client=None,
    broker=None,
    reconcile_runner=run_reconciliation_pass,
) -> int:
    if repair and not acknowledge:
        print(
            "testnet-reconcile --repair requires --i-understand-testnet-order",
            file=sys.stderr,
        )
        return 2
    rest_client = rest_client or (None if broker is not None else _build_testnet_rest_client())
    if rest_client is None and broker is None:
        print(
            "BINANCE_API_KEY and BINANCE_SECRET_KEY are required for testnet-reconcile",
            file=sys.stderr,
        )
        return 2

    broker = broker or BinanceUsdFuturesTestnetBroker(rest_client)
    store = LiveStore.open(db)
    store.initialize()
    summary = reconcile_runner(
        store=store,
        broker=broker,
        symbols=symbols,
        environment="testnet",
        allow_repair=repair,
    )
    print(json.dumps(summary.to_dict(), indent=2, sort_keys=True))
    return 1 if summary.error_count else 0


def run_testnet_open_protected_command(
    *,
    db: Path,
    symbol: str,
    notional: float,
    stop_offset_pct: float,
    acknowledge: bool,
    rest_client=None,
    broker=None,
    rehearsal_runner=open_protected_rehearsal_position,
) -> int:
    if not acknowledge:
        print(
            "testnet-open-protected requires --i-understand-testnet-order",
            file=sys.stderr,
        )
        return 2
    rest_client = rest_client or (None if broker is not None else _build_testnet_rest_client())
    if rest_client is None and broker is None:
        print(
            "BINANCE_API_KEY and BINANCE_SECRET_KEY are required for testnet-open-protected",
            file=sys.stderr,
        )
        return 2

    broker = broker or BinanceUsdFuturesTestnetBroker(rest_client)
    store = LiveStore.open(db)
    store.initialize()
    result = rehearsal_runner(
        store=store,
        broker=broker,
        symbol=symbol,
        notional=notional,
        stop_offset_pct=stop_offset_pct,
    )
    print(json.dumps(vars(result), indent=2, sort_keys=True))
    return 0


def run_testnet_close_protected_command(
    *,
    db: Path,
    symbol: str,
    position_id: str | None,
    acknowledge: bool,
    rest_client=None,
    broker=None,
    rehearsal_runner=close_rehearsal_position,
) -> int:
    if not acknowledge:
        print(
            "testnet-close-protected requires --i-understand-testnet-order",
            file=sys.stderr,
        )
        return 2
    rest_client = rest_client or (None if broker is not None else _build_testnet_rest_client())
    if rest_client is None and broker is None:
        print(
            "BINANCE_API_KEY and BINANCE_SECRET_KEY are required for testnet-close-protected",
            file=sys.stderr,
        )
        return 2

    broker = broker or BinanceUsdFuturesTestnetBroker(rest_client)
    store = LiveStore.open(db)
    store.initialize()
    result = rehearsal_runner(
        store=store,
        broker=broker,
        symbol=symbol,
        position_id=position_id,
    )
    print(json.dumps(vars(result), indent=2, sort_keys=True))
    return 0


def run_status_command(
    *,
    db: Path,
    json_output: bool,
    collect_system: bool = True,
    system_snapshot: dict[str, object] | None = None,
) -> int:
    if system_snapshot is None:
        system_snapshot = (
            collect_system_snapshot()
            if collect_system
            else {"system_available": False, "sockets": [], "journal": {}}
        )
    snapshot = build_status_snapshot(db_path=db, system_snapshot=system_snapshot)
    if json_output:
        print(json.dumps(snapshot, indent=2, sort_keys=True))
    else:
        print(render_status_text(snapshot))
    return 1 if snapshot["overall"] == "WARN" else 0


def run_live_cycle_command(
    *,
    mode: str,
    db: Path,
    symbols: list[str] | None,
    max_symbols: int | None,
    lookback_bars: int,
    env_file: Path | None,
    acknowledge_live: bool,
    live_enabled: bool | None = None,
    broker=None,
    arrays=None,
    account=None,
    metadata_by_symbol=None,
    prices_by_symbol=None,
    cycle_runner=run_live_cycle,
) -> int:
    live_enabled = _live_enabled() if live_enabled is None else live_enabled
    if mode == "live" and (not acknowledge_live or not live_enabled):
        print(
            "live trading requires --i-understand-live-order and XSIGNAL_ENABLE_LIVE_TRADING=1 "
            "or /etc/xsignal/enable-live-trading",
            file=sys.stderr,
        )
        return 2

    credentials = _credentials_from_env(env_file=_default_env_file(mode, env_file))
    if broker is None:
        if credentials is None:
            print(
                "BINANCE_API_KEY and BINANCE_SECRET_KEY are required for run-cycle",
                file=sys.stderr,
            )
            return 2
        broker = build_usd_futures_broker(mode=mode, credentials=credentials)

    store = LiveStore.open(db)
    store.initialize()
    selected_symbols = list(symbols or [])
    if not selected_symbols:
        selected_symbols = broker.list_trading_usdt_perpetual_symbols()
    if max_symbols is not None:
        selected_symbols = selected_symbols[:max_symbols]
    if not selected_symbols:
        print("run-cycle requires at least one symbol", file=sys.stderr)
        return 2

    if arrays is None or account is None or metadata_by_symbol is None or prices_by_symbol is None:
        server_time = broker.rest_client.request("GET", "/fapi/v1/time")["serverTime"]
        arrays = load_recent_daily_arrays(
            broker.rest_client,
            symbols=selected_symbols,
            limit=lookback_bars,
            server_time_ms=int(server_time),
        )
        account = broker.get_account_snapshot(mode=mode, daily_realized_pnl=0.0)
        metadata_by_symbol = {
            symbol: broker.get_symbol_metadata(symbol)
            for symbol in selected_symbols
            if symbol in arrays.symbols
        }
        prices_by_symbol = {
            symbol: broker.get_symbol_price(symbol)
            for symbol in selected_symbols
            if symbol in arrays.symbols
        }

    config = LiveTradingConfig(
        mode=mode,
        live_acknowledgement=(mode != "live" or (acknowledge_live and live_enabled)),
    )
    result = cycle_runner(
        store=store,
        broker=broker,
        config=config,
        environment=mode,
        arrays=arrays,
        account=account,
        metadata_by_symbol=metadata_by_symbol,
        prices_by_symbol=prices_by_symbol,
        now=datetime.now(timezone.utc),
    )
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 1 if getattr(result, "blocked", False) else 0


def run_live_smoke_command(*, symbol: str, env_file: Path | None = LOCAL_LIVE_ENV_FILE) -> int:
    credentials = _credentials_from_env(env_file=env_file)
    if credentials is None:
        print(
            "BINANCE_API_KEY and BINANCE_SECRET_KEY are required for live-smoke",
            file=sys.stderr,
        )
        return 2
    broker = build_usd_futures_broker(mode="live", credentials=credentials)
    server_time = broker.rest_client.request("GET", "/fapi/v1/time")
    metadata = broker.get_symbol_metadata(symbol)
    account = broker.get_account_snapshot(mode="live", daily_realized_pnl=0.0)
    output = {
        "mode": "live",
        "symbol": symbol,
        "server_time": server_time.get("serverTime"),
        "symbol_status": metadata.status,
        "position_mode": account.account_mode,
        "asset_mode": account.asset_mode,
        "equity": account.equity,
        "available_balance": account.available_balance,
        "open_position_count": account.open_position_count,
        "orders_submitted": False,
    }
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


def run_stream_daemon_command(
    *,
    mode: str,
    db: Path,
    intervals: list[str],
    max_symbols: int | None,
    lookback_bars: int,
    env_file: Path | None,
    acknowledge_live: bool,
    live_enabled: bool | None = None,
    seed_sleep_ms: int = 20,
    recovery_sleep_ms: int = 500,
    closed_poll_sleep_ms: int = 25,
    closed_poll_grace_seconds: float = 2.0,
    closed_poll_fetch_limit: int = 99,
    reconcile_interval_seconds: float = 300.0,
    stop_after_events: int | None = None,
    max_streams: int | None = None,
    stream_max_lifetime_seconds: float | None = None,
    stream_rotation_jitter_seconds: float | None = None,
    credentials=None,
    daemon_runner=run_stream_daemon,
) -> int:
    live_enabled = _live_enabled() if live_enabled is None else live_enabled
    if mode == "live" and (not acknowledge_live or not live_enabled):
        print(
            "live trading requires --i-understand-live-order and XSIGNAL_ENABLE_LIVE_TRADING=1 "
            "or /etc/xsignal/enable-live-trading",
            file=sys.stderr,
        )
        return 2
    credentials = credentials or _credentials_from_env(env_file=_default_env_file(mode, env_file))
    if credentials is None:
        print(
            "BINANCE_API_KEY and BINANCE_SECRET_KEY are required for stream-daemon",
            file=sys.stderr,
        )
        return 2
    config = StreamDaemonConfig(
        mode=mode,
        db_path=db,
        intervals=tuple(intervals or DEFAULT_REALTIME_INTERVALS),
        lookback_bars=lookback_bars,
        max_symbols=max_symbols,
        **({"max_streams": max_streams} if max_streams is not None else {}),
        **(
            {"stream_max_lifetime_seconds": stream_max_lifetime_seconds}
            if stream_max_lifetime_seconds is not None
            else {}
        ),
        **(
            {"stream_rotation_jitter_seconds": stream_rotation_jitter_seconds}
            if stream_rotation_jitter_seconds is not None
            else {}
        ),
        seed_sleep_ms=seed_sleep_ms,
        recovery_sleep_ms=recovery_sleep_ms,
        closed_poll_sleep_ms=closed_poll_sleep_ms,
        closed_poll_grace_seconds=closed_poll_grace_seconds,
        closed_poll_fetch_limit=closed_poll_fetch_limit,
        reconcile_interval_seconds=reconcile_interval_seconds,
        stop_after_events=stop_after_events,
    )
    return daemon_runner(config=config, credentials=credentials)


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "testnet-smoke":
        return run_testnet_smoke(
            symbol=args.symbol,
            submit_test_order=args.submit_test_order,
            quantity=args.quantity,
        )
    if args.command == "testnet-lifecycle":
        return run_testnet_lifecycle_command(
            symbol=args.symbol,
            quantity=args.quantity,
            stop_offset_pct=args.stop_offset_pct,
            acknowledge=args.i_understand_testnet_order,
            db=args.db,
        )
    if args.command == "testnet-reconcile":
        return run_testnet_reconcile_command(
            db=args.db,
            symbols=args.symbol,
            repair=args.repair,
            acknowledge=args.i_understand_testnet_order,
        )
    if args.command == "testnet-open-protected":
        return run_testnet_open_protected_command(
            db=args.db,
            symbol=args.symbol,
            notional=args.notional,
            stop_offset_pct=args.stop_offset_pct,
            acknowledge=args.i_understand_testnet_order,
        )
    if args.command == "testnet-close-protected":
        return run_testnet_close_protected_command(
            db=args.db,
            symbol=args.symbol,
            position_id=args.position_id,
            acknowledge=args.i_understand_testnet_order,
        )
    if args.command == "status":
        return run_status_command(
            db=args.db,
            json_output=args.json,
            collect_system=not args.no_system,
        )
    if args.command == "run-cycle":
        return run_live_cycle_command(
            mode=args.mode,
            db=args.db,
            symbols=args.symbol,
            max_symbols=args.max_symbols,
            lookback_bars=args.lookback_bars,
            env_file=args.env_file,
            acknowledge_live=args.i_understand_live_order,
        )
    if args.command == "stream-daemon":
        return run_stream_daemon_command(
            mode=args.mode,
            db=args.db,
            intervals=args.interval,
            max_symbols=args.max_symbols,
            lookback_bars=args.lookback_bars,
            env_file=args.env_file,
            acknowledge_live=args.i_understand_live_order,
            seed_sleep_ms=args.seed_sleep_ms,
            recovery_sleep_ms=args.recovery_sleep_ms,
            closed_poll_sleep_ms=args.closed_poll_sleep_ms,
            closed_poll_grace_seconds=args.closed_poll_grace_seconds,
            closed_poll_fetch_limit=args.closed_poll_fetch_limit,
            reconcile_interval_seconds=args.reconcile_interval_seconds,
            stop_after_events=args.stop_after_events,
            max_streams=args.max_streams,
            stream_max_lifetime_seconds=args.stream_max_lifetime_seconds,
            stream_rotation_jitter_seconds=args.stream_rotation_jitter_seconds,
        )
    if args.command == "live-smoke":
        return run_live_smoke_command(symbol=args.symbol, env_file=args.env_file)
    return 0
