from __future__ import annotations

from datetime import datetime
from typing import Any

from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    AccountSnapshot,
    SymbolMetadata,
)


BINANCE_USD_FUTURES_TESTNET_BASE_URL = "https://testnet.binancefuture.com"


def _filter_by_type(symbol_payload: dict[str, Any], filter_type: str) -> dict[str, Any]:
    for item in symbol_payload.get("filters", []):
        if item.get("filterType") == filter_type:
            return item
    raise ValueError(f"missing Binance filter {filter_type}")


def parse_symbol_metadata(
    symbol_payload: dict[str, Any],
    *,
    updated_at: datetime,
) -> SymbolMetadata:
    order_types = set(symbol_payload.get("orderTypes", []))
    if "STOP_MARKET" not in order_types:
        raise ValueError("symbol does not support STOP_MARKET")
    price_filter = _filter_by_type(symbol_payload, "PRICE_FILTER")
    lot_size = _filter_by_type(symbol_payload, "LOT_SIZE")
    min_notional = _filter_by_type(symbol_payload, "MIN_NOTIONAL")
    return SymbolMetadata(
        symbol=str(symbol_payload["symbol"]),
        status=str(symbol_payload["status"]),
        min_notional=float(min_notional["notional"]),
        quantity_step=float(lot_size["stepSize"]),
        price_tick=float(price_filter["tickSize"]),
        supports_stop_market=True,
        trigger_protect=float(symbol_payload.get("triggerProtect", 0.0)),
        updated_at=updated_at,
    )


def parse_account_snapshot(
    payload: dict[str, Any],
    *,
    mode: str,
    account_mode: str,
    asset_mode: str,
    open_position_count: int,
    daily_realized_pnl: float,
    captured_at: datetime,
) -> AccountSnapshot:
    return AccountSnapshot(
        mode=mode,
        account_mode=account_mode,
        asset_mode=asset_mode,
        equity=float(payload["totalMarginBalance"]),
        available_balance=float(payload["availableBalance"]),
        open_notional=float(payload.get("totalInitialMargin", 0.0)),
        open_position_count=open_position_count,
        daily_realized_pnl=daily_realized_pnl,
        captured_at=captured_at,
    )


class BinanceUsdFuturesTestnetBroker:
    def __init__(self, rest_client) -> None:
        self.rest_client = rest_client

    def get_position_mode(self) -> str:
        payload = self.rest_client.request("GET", "/fapi/v1/positionSide/dual", signed=True)
        return "hedge" if payload.get("dualSidePosition") else "one_way"

    def get_multi_assets_mode(self) -> str:
        payload = self.rest_client.request("GET", "/fapi/v1/multiAssetsMargin", signed=True)
        return "multi_asset" if payload.get("multiAssetsMargin") else "single_asset_usdt"
