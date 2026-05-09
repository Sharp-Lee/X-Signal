from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    AccountSnapshot,
    SymbolMetadata,
)
from xsignal.strategies.volume_price_efficiency_v1.live.order_normalizer import format_decimal
from xsignal.strategies.volume_price_efficiency_v1.live.binance_rest import BinanceRestClient


BINANCE_USD_FUTURES_TESTNET_BASE_URL = "https://testnet.binancefuture.com"
BINANCE_USD_FUTURES_LIVE_BASE_URL = "https://fapi.binance.com"


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
    market_lot_size = _filter_by_type(symbol_payload, "MARKET_LOT_SIZE")
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
        min_quantity=float(lot_size["minQty"]),
        max_quantity=float(lot_size["maxQty"]),
        market_min_quantity=float(market_lot_size["minQty"]),
        market_max_quantity=float(market_lot_size["maxQty"]),
        market_quantity_step=float(market_lot_size["stepSize"]),
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

    def get_symbol_metadata(self, symbol: str) -> SymbolMetadata:
        payload = self.rest_client.request("GET", "/fapi/v1/exchangeInfo")
        for item in payload.get("symbols", []):
            if item.get("symbol") == symbol:
                return parse_symbol_metadata(item, updated_at=datetime.now(timezone.utc))
        raise ValueError(f"missing Binance symbol metadata for {symbol}")

    def list_trading_usdt_perpetual_symbols(self) -> list[str]:
        return sorted(self.list_trading_usdt_perpetual_metadata())

    def list_trading_usdt_perpetual_metadata(self) -> dict[str, SymbolMetadata]:
        payload = self.rest_client.request("GET", "/fapi/v1/exchangeInfo")
        updated_at = datetime.now(timezone.utc)
        metadata = {}
        for item in payload.get("symbols", []):
            if (
                item.get("status") == "TRADING"
                and item.get("quoteAsset") == "USDT"
                and item.get("contractType") == "PERPETUAL"
            ):
                parsed = parse_symbol_metadata(item, updated_at=updated_at)
                metadata[parsed.symbol] = parsed
        return dict(sorted(metadata.items()))

    def get_symbol_price(self, symbol: str) -> float:
        payload = self.rest_client.request(
            "GET",
            "/fapi/v1/ticker/price",
            params={"symbol": symbol},
        )
        return float(payload["price"])

    def change_margin_type(self, symbol: str, margin_mode: str) -> dict[str, Any]:
        if margin_mode != "isolated":
            raise ValueError("only isolated margin is supported")
        return self.rest_client.request(
            "POST",
            "/fapi/v1/marginType",
            signed=True,
            params={"symbol": symbol, "marginType": "ISOLATED"},
        )

    def change_leverage(self, symbol: str, leverage: int) -> dict[str, Any]:
        if leverage != 1:
            raise ValueError("only 1x leverage is supported")
        return self.rest_client.request(
            "POST",
            "/fapi/v1/leverage",
            signed=True,
            params={"symbol": symbol, "leverage": leverage},
        )

    def market_buy(self, *, symbol: str, quantity: float, client_order_id: str) -> dict[str, Any]:
        return self.rest_client.request(
            "POST",
            "/fapi/v1/order",
            signed=True,
            params={
                "symbol": symbol,
                "side": "BUY",
                "type": "MARKET",
                "quantity": _format_decimal(quantity),
                "newClientOrderId": client_order_id,
                "positionSide": "BOTH",
            },
        )

    def place_stop_market_close(
        self,
        *,
        symbol: str,
        stop_price: float,
        client_order_id: str,
    ) -> dict[str, Any]:
        return self.rest_client.request(
            "POST",
            "/fapi/v1/algoOrder",
            signed=True,
            params={
                "algoType": "CONDITIONAL",
                "symbol": symbol,
                "side": "SELL",
                "type": "STOP_MARKET",
                "triggerPrice": _format_decimal(stop_price),
                "closePosition": "true",
                "workingType": "CONTRACT_PRICE",
                "clientAlgoId": client_order_id,
                "positionSide": "BOTH",
            },
        )

    def cancel_order(self, *, symbol: str, client_order_id: str) -> dict[str, Any]:
        return self.rest_client.request(
            "DELETE",
            "/fapi/v1/algoOrder",
            signed=True,
            params={"clientAlgoId": client_order_id},
        )

    def get_position_risk(self, *, symbol: str) -> list[dict[str, Any]]:
        return self.rest_client.request(
            "GET",
            "/fapi/v3/positionRisk",
            signed=True,
            params={"symbol": symbol},
        )

    def get_all_position_risk(self) -> list[dict[str, Any]]:
        return self.rest_client.request("GET", "/fapi/v3/positionRisk", signed=True)

    def get_account_snapshot(
        self,
        *,
        mode: str,
        daily_realized_pnl: float,
    ) -> AccountSnapshot:
        account = self.rest_client.request("GET", "/fapi/v3/account", signed=True)
        positions = self.get_all_position_risk()
        open_position_count = sum(
            1 for position in positions if abs(float(position.get("positionAmt", 0.0))) > 0.0
        )
        return parse_account_snapshot(
            account,
            mode=mode,
            account_mode=self.get_position_mode(),
            asset_mode=self.get_multi_assets_mode(),
            open_position_count=open_position_count,
            daily_realized_pnl=daily_realized_pnl,
            captured_at=datetime.now(timezone.utc),
        )

    def get_order(self, *, symbol: str, client_order_id: str) -> dict[str, Any]:
        return self.rest_client.request(
            "GET",
            "/fapi/v1/order",
            signed=True,
            params={"symbol": symbol, "origClientOrderId": client_order_id},
        )

    def get_open_order(self, *, symbol: str, client_order_id: str) -> dict[str, Any]:
        return self.rest_client.request(
            "GET",
            "/fapi/v1/algoOrder",
            signed=True,
            params={"clientAlgoId": client_order_id},
        )

    def get_open_orders(self, *, symbol: str) -> list[dict[str, Any]]:
        return self.rest_client.request(
            "GET",
            "/fapi/v1/openAlgoOrders",
            signed=True,
            params={"symbol": symbol},
        )

    def market_sell_reduce_only(
        self,
        *,
        symbol: str,
        quantity: float,
        client_order_id: str,
    ) -> dict[str, Any]:
        return self.rest_client.request(
            "POST",
            "/fapi/v1/order",
            signed=True,
            params={
                "symbol": symbol,
                "side": "SELL",
                "type": "MARKET",
                "quantity": _format_decimal(quantity),
                "reduceOnly": "true",
                "newClientOrderId": client_order_id,
                "positionSide": "BOTH",
            },
        )

    def test_order(
        self,
        *,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        client_order_id: str,
    ) -> dict[str, Any]:
        return self.rest_client.request(
            "POST",
            "/fapi/v1/order/test",
            signed=True,
            params={
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "quantity": _format_decimal(quantity),
                "newClientOrderId": client_order_id,
                "positionSide": "BOTH",
            },
        )


def _format_decimal(value) -> str:
    return format_decimal(value)


def build_usd_futures_broker(
    *,
    mode: str,
    credentials,
    transport=None,
    now_ms=None,
) -> BinanceUsdFuturesTestnetBroker:
    if mode == "testnet":
        base_url = BINANCE_USD_FUTURES_TESTNET_BASE_URL
    elif mode == "live":
        base_url = BINANCE_USD_FUTURES_LIVE_BASE_URL
    else:
        raise ValueError(f"unsupported Binance USD-M mode: {mode}")
    return BinanceUsdFuturesTestnetBroker(
        BinanceRestClient(
            base_url=base_url,
            credentials=credentials,
            transport=transport,
            now_ms=now_ms,
        )
    )
