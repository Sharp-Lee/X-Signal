from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

from xsignal.strategies.volume_price_efficiency_v1.live.binance_adapter import (
    BINANCE_USD_FUTURES_TESTNET_BASE_URL,
    BinanceUsdFuturesTestnetBroker,
)
from xsignal.strategies.volume_price_efficiency_v1.live.binance_rest import (
    BinanceCredentials,
    BinanceRestClient,
)
from xsignal.strategies.volume_price_efficiency_v1.live.ids import build_client_order_id


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="xsignal-vpe-live")
    subparsers = parser.add_subparsers(dest="command", required=True)

    replay = subparsers.add_parser("replay")
    replay.add_argument("--root", type=Path, default=Path("data"))
    replay.add_argument("--db", type=Path, required=True)

    status = subparsers.add_parser("status")
    status.add_argument("--db", type=Path, required=True)

    reconcile = subparsers.add_parser("reconcile")
    reconcile.add_argument("--db", type=Path, required=True)

    smoke = subparsers.add_parser("testnet-smoke")
    smoke.add_argument("--symbol", required=True)
    smoke.add_argument("--quantity", type=float, default=0.001)
    smoke.add_argument("--submit-test-order", action="store_true")
    return parser


def _credentials_from_env() -> BinanceCredentials | None:
    api_key = os.environ.get("BINANCE_API_KEY")
    secret_key = os.environ.get("BINANCE_SECRET_KEY")
    if not api_key or not secret_key:
        return None
    return BinanceCredentials(api_key=api_key, secret_key=secret_key)


def _build_testnet_rest_client() -> BinanceRestClient | None:
    credentials = _credentials_from_env()
    if credentials is None:
        return None
    return BinanceRestClient(
        base_url=BINANCE_USD_FUTURES_TESTNET_BASE_URL,
        credentials=credentials,
    )


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


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "testnet-smoke":
        return run_testnet_smoke(
            symbol=args.symbol,
            submit_test_order=args.submit_test_order,
            quantity=args.quantity,
        )
    return 0
