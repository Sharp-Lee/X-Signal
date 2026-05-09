from datetime import datetime, timezone

import pytest

from xsignal.strategies.volume_price_efficiency_v1.live.binance_adapter import (
    BINANCE_USD_FUTURES_LIVE_BASE_URL,
    BinanceUsdFuturesTestnetBroker,
    build_usd_futures_broker,
    parse_account_snapshot,
    parse_symbol_metadata,
)
from xsignal.strategies.volume_price_efficiency_v1.live.models import SymbolMetadata


def _symbol_payload(**overrides):
    payload = {
        "symbol": "BTCUSDT",
        "status": "TRADING",
        "orderTypes": ["LIMIT", "MARKET", "STOP_MARKET"],
        "triggerProtect": "0.0500",
        "filters": [
            {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
            {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "1000", "stepSize": "0.001"},
            {
                "filterType": "MARKET_LOT_SIZE",
                "minQty": "0.001",
                "maxQty": "100",
                "stepSize": "0.001",
            },
            {"filterType": "MIN_NOTIONAL", "notional": "5"},
        ],
    }
    payload.update(overrides)
    return payload


def test_parse_symbol_metadata_from_exchange_info_symbol():
    updated_at = datetime(2026, 5, 9, tzinfo=timezone.utc)

    metadata = parse_symbol_metadata(_symbol_payload(), updated_at=updated_at)

    assert metadata == SymbolMetadata(
        symbol="BTCUSDT",
        status="TRADING",
        min_notional=5.0,
        quantity_step=0.001,
        price_tick=0.1,
        supports_stop_market=True,
        trigger_protect=0.05,
        updated_at=updated_at,
        min_quantity=0.001,
        max_quantity=1000.0,
        market_min_quantity=0.001,
        market_max_quantity=100.0,
        market_quantity_step=0.001,
    )


@pytest.mark.parametrize(
    "payload",
    [
        _symbol_payload(orderTypes=["MARKET"]),
        _symbol_payload(filters=[{"filterType": "LOT_SIZE", "stepSize": "0.001"}]),
        _symbol_payload(
            filters=[
                {"filterType": "PRICE_FILTER", "tickSize": "0.10"},
                {"filterType": "LOT_SIZE", "minQty": "0.001", "maxQty": "1000", "stepSize": "0.001"},
                {"filterType": "MIN_NOTIONAL", "notional": "5"},
            ],
        ),
    ],
)
def test_parse_symbol_metadata_rejects_missing_required_trading_rules(payload):
    with pytest.raises(ValueError):
        parse_symbol_metadata(payload, updated_at=datetime(2026, 5, 9, tzinfo=timezone.utc))


def test_parse_account_snapshot_from_account_v3():
    captured_at = datetime(2026, 5, 9, tzinfo=timezone.utc)
    snapshot = parse_account_snapshot(
        {
            "totalMarginBalance": "1000.5",
            "availableBalance": "800.25",
            "totalInitialMargin": "50.0",
            "totalWalletBalance": "995.0",
        },
        mode="testnet",
        account_mode="one_way",
        asset_mode="single_asset_usdt",
        open_position_count=3,
        daily_realized_pnl=-7.0,
        captured_at=captured_at,
    )

    assert snapshot.mode == "testnet"
    assert snapshot.account_mode == "one_way"
    assert snapshot.asset_mode == "single_asset_usdt"
    assert snapshot.equity == 1000.5
    assert snapshot.available_balance == 800.25
    assert snapshot.open_notional == 50.0
    assert snapshot.open_position_count == 3
    assert snapshot.daily_realized_pnl == -7.0
    assert snapshot.captured_at == captured_at


def test_build_broker_selects_testnet_and_live_base_urls():
    testnet = build_usd_futures_broker(
        mode="testnet",
        credentials=None,
        transport=object(),
    )
    live = build_usd_futures_broker(
        mode="live",
        credentials=None,
        transport=object(),
    )

    assert testnet.rest_client.base_url == "https://testnet.binancefuture.com"
    assert live.rest_client.base_url == BINANCE_USD_FUTURES_LIVE_BASE_URL
    assert live.rest_client.base_url == "https://fapi.binance.com"


class FakeRestClient:
    def __init__(self) -> None:
        self.calls = []

    def request(self, method, path, *, signed=False, params=None):
        self.calls.append((method, path, signed, params or {}))
        return {}


def test_broker_maps_position_and_multi_asset_modes():
    class ModeRestClient(FakeRestClient):
        def request(self, method, path, *, signed=False, params=None):
            super().request(method, path, signed=signed, params=params)
            if path.endswith("positionSide/dual"):
                return {"dualSidePosition": False}
            if path.endswith("multiAssetsMargin"):
                return {"multiAssetsMargin": False}
            return {}

    broker = BinanceUsdFuturesTestnetBroker(ModeRestClient())

    assert broker.get_position_mode() == "one_way"
    assert broker.get_multi_assets_mode() == "single_asset_usdt"


def test_broker_fetches_symbol_metadata_from_exchange_info():
    class ExchangeInfoRestClient(FakeRestClient):
        def request(self, method, path, *, signed=False, params=None):
            super().request(method, path, signed=signed, params=params)
            return {"symbols": [_symbol_payload(symbol="ETHUSDT"), _symbol_payload()]}

    broker = BinanceUsdFuturesTestnetBroker(ExchangeInfoRestClient())

    metadata = broker.get_symbol_metadata("BTCUSDT")

    assert metadata.symbol == "BTCUSDT"
    assert metadata.price_tick == 0.1
    assert metadata.quantity_step == 0.001


def test_broker_lists_trading_usdt_perpetual_symbols_and_prices():
    class ExchangeInfoRestClient(FakeRestClient):
        def request(self, method, path, *, signed=False, params=None):
            super().request(method, path, signed=signed, params=params)
            if path == "/fapi/v1/exchangeInfo":
                return {
                    "symbols": [
                        _symbol_payload(symbol="BTCUSDT", quoteAsset="USDT", contractType="PERPETUAL"),
                        _symbol_payload(symbol="ETHUSDT", quoteAsset="USDT", contractType="PERPETUAL"),
                        _symbol_payload(symbol="OLDUSDT", quoteAsset="USDT", contractType="PERPETUAL", status="SETTLING"),
                        _symbol_payload(symbol="BTCBUSD", quoteAsset="BUSD", contractType="PERPETUAL"),
                    ]
                }
            if path == "/fapi/v1/ticker/price":
                return {"symbol": params["symbol"], "price": "123.45"}
            return {}

    broker = BinanceUsdFuturesTestnetBroker(ExchangeInfoRestClient())

    assert broker.list_trading_usdt_perpetual_symbols() == ["BTCUSDT", "ETHUSDT"]
    assert broker.get_symbol_price("BTCUSDT") == 123.45


def test_broker_lists_trading_usdt_perpetual_metadata_with_one_exchange_info_call():
    class ExchangeInfoRestClient(FakeRestClient):
        def request(self, method, path, *, signed=False, params=None):
            super().request(method, path, signed=signed, params=params)
            return {
                "symbols": [
                    _symbol_payload(symbol="BTCUSDT", quoteAsset="USDT", contractType="PERPETUAL"),
                    _symbol_payload(symbol="ETHUSDT", quoteAsset="USDT", contractType="PERPETUAL"),
                    _symbol_payload(
                        symbol="OLDUSDT",
                        quoteAsset="USDT",
                        contractType="PERPETUAL",
                        status="SETTLING",
                    ),
                ]
            }

    broker = BinanceUsdFuturesTestnetBroker(ExchangeInfoRestClient())

    metadata = broker.list_trading_usdt_perpetual_metadata()

    assert sorted(metadata) == ["BTCUSDT", "ETHUSDT"]
    assert metadata["BTCUSDT"].status == "TRADING"
    assert broker.rest_client.calls == [("GET", "/fapi/v1/exchangeInfo", False, {})]


def test_broker_changes_margin_type_and_leverage():
    rest_client = FakeRestClient()
    broker = BinanceUsdFuturesTestnetBroker(rest_client)

    broker.change_margin_type("BTCUSDT", "isolated")
    broker.change_leverage("BTCUSDT", 1)

    assert rest_client.calls == [
        (
            "POST",
            "/fapi/v1/marginType",
            True,
            {"symbol": "BTCUSDT", "marginType": "ISOLATED"},
        ),
        ("POST", "/fapi/v1/leverage", True, {"symbol": "BTCUSDT", "leverage": 1}),
    ]


def test_broker_market_buy_uses_compact_client_order_id():
    rest_client = FakeRestClient()
    broker = BinanceUsdFuturesTestnetBroker(rest_client)

    broker.market_buy(
        symbol="BTCUSDT",
        quantity=0.001,
        client_order_id="XV1TEBTC123",
    )

    assert rest_client.calls == [
        (
            "POST",
            "/fapi/v1/order",
            True,
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "MARKET",
                "quantity": "0.001",
                "newClientOrderId": "XV1TEBTC123",
                "positionSide": "BOTH",
            },
        )
    ]


def test_broker_places_stop_market_close_without_quantity():
    rest_client = FakeRestClient()
    broker = BinanceUsdFuturesTestnetBroker(rest_client)

    broker.place_stop_market_close(
        symbol="BTCUSDT",
        stop_price=90.100000000006,
        client_order_id="XV1TSBTC123",
    )

    assert rest_client.calls == [
        (
            "POST",
            "/fapi/v1/algoOrder",
            True,
            {
                "algoType": "CONDITIONAL",
                "symbol": "BTCUSDT",
                "side": "SELL",
                "type": "STOP_MARKET",
                "triggerPrice": "90.1",
                "closePosition": "true",
                "workingType": "CONTRACT_PRICE",
                "clientAlgoId": "XV1TSBTC123",
                "positionSide": "BOTH",
            },
        )
    ]


def test_broker_cancels_and_validates_test_order():
    rest_client = FakeRestClient()
    broker = BinanceUsdFuturesTestnetBroker(rest_client)

    broker.cancel_order(symbol="BTCUSDT", client_order_id="XV1TSBTC123")
    broker.test_order(
        symbol="BTCUSDT",
        side="BUY",
        order_type="MARKET",
        quantity=0.001,
        client_order_id="XV1TEBTC123",
    )

    assert rest_client.calls == [
        (
            "DELETE",
            "/fapi/v1/algoOrder",
            True,
            {"clientAlgoId": "XV1TSBTC123"},
        ),
        (
            "POST",
            "/fapi/v1/order/test",
            True,
            {
                "symbol": "BTCUSDT",
                "side": "BUY",
                "type": "MARKET",
                "quantity": "0.001",
                "newClientOrderId": "XV1TEBTC123",
                "positionSide": "BOTH",
            },
        ),
    ]


def test_broker_queries_position_risk_and_open_orders():
    rest_client = FakeRestClient()
    broker = BinanceUsdFuturesTestnetBroker(rest_client)

    broker.get_position_risk(symbol="BTCUSDT")
    broker.get_order(symbol="BTCUSDT", client_order_id="XV1TEBTC123")
    broker.get_open_order(symbol="BTCUSDT", client_order_id="XV1TSBTC123")
    broker.get_open_orders(symbol="BTCUSDT")

    assert rest_client.calls == [
        (
            "GET",
            "/fapi/v3/positionRisk",
            True,
            {"symbol": "BTCUSDT"},
        ),
        (
            "GET",
            "/fapi/v1/order",
            True,
            {"symbol": "BTCUSDT", "origClientOrderId": "XV1TEBTC123"},
        ),
        (
            "GET",
            "/fapi/v1/algoOrder",
            True,
            {"clientAlgoId": "XV1TSBTC123"},
        ),
        (
            "GET",
            "/fapi/v1/openAlgoOrders",
            True,
            {"symbol": "BTCUSDT"},
        ),
    ]


def test_broker_market_sell_reduce_only_closes_existing_position():
    rest_client = FakeRestClient()
    broker = BinanceUsdFuturesTestnetBroker(rest_client)

    broker.market_sell_reduce_only(
        symbol="BTCUSDT",
        quantity=0.001,
        client_order_id="XV1TCBTC123",
    )

    assert rest_client.calls == [
        (
            "POST",
            "/fapi/v1/order",
            True,
            {
                "symbol": "BTCUSDT",
                "side": "SELL",
                "type": "MARKET",
                "quantity": "0.001",
                "reduceOnly": "true",
                "newClientOrderId": "XV1TCBTC123",
                "positionSide": "BOTH",
            },
        )
    ]


def test_broker_manages_user_data_listen_key():
    class ListenKeyRestClient(FakeRestClient):
        def request(self, method, path, *, signed=False, params=None):
            super().request(method, path, signed=signed, params=params)
            if method == "POST" and path == "/fapi/v1/listenKey":
                return {"listenKey": "abc-listen-key"}
            return {}

    rest_client = ListenKeyRestClient()
    broker = BinanceUsdFuturesTestnetBroker(rest_client)

    listen_key = broker.start_user_data_stream()
    broker.keepalive_user_data_stream(listen_key)
    broker.close_user_data_stream(listen_key)

    assert listen_key == "abc-listen-key"
    assert rest_client.calls == [
        ("POST", "/fapi/v1/listenKey", False, {}),
        ("PUT", "/fapi/v1/listenKey", False, {}),
        ("DELETE", "/fapi/v1/listenKey", False, {}),
    ]


def test_broker_builds_account_snapshot_from_live_account_and_positions():
    class AccountRestClient(FakeRestClient):
        def request(self, method, path, *, signed=False, params=None):
            super().request(method, path, signed=signed, params=params)
            if path == "/fapi/v3/account":
                return {
                    "totalMarginBalance": "1000",
                    "availableBalance": "900",
                    "totalInitialMargin": "50",
                }
            if path == "/fapi/v1/positionSide/dual":
                return {"dualSidePosition": False}
            if path == "/fapi/v1/multiAssetsMargin":
                return {"multiAssetsMargin": False}
            if path == "/fapi/v3/positionRisk":
                return [
                    {"symbol": "BTCUSDT", "positionAmt": "0.1"},
                    {"symbol": "ETHUSDT", "positionAmt": "0"},
                ]
            return {}

    broker = BinanceUsdFuturesTestnetBroker(AccountRestClient())

    snapshot = broker.get_account_snapshot(mode="live", daily_realized_pnl=0.0)

    assert snapshot.mode == "live"
    assert snapshot.open_position_count == 1
    assert snapshot.equity == 1000.0
