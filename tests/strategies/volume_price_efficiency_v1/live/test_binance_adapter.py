from datetime import datetime, timezone

import pytest

from xsignal.strategies.volume_price_efficiency_v1.live.binance_adapter import (
    BinanceUsdFuturesTestnetBroker,
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
            {"filterType": "LOT_SIZE", "stepSize": "0.001"},
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
    )


@pytest.mark.parametrize(
    "payload",
    [
        _symbol_payload(orderTypes=["MARKET"]),
        _symbol_payload(filters=[{"filterType": "LOT_SIZE", "stepSize": "0.001"}]),
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
        stop_price=90.5,
        client_order_id="XV1TSBTC123",
    )

    assert rest_client.calls == [
        (
            "POST",
            "/fapi/v1/order",
            True,
            {
                "symbol": "BTCUSDT",
                "side": "SELL",
                "type": "STOP_MARKET",
                "stopPrice": "90.5",
                "closePosition": "true",
                "workingType": "CONTRACT_PRICE",
                "newClientOrderId": "XV1TSBTC123",
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
            "/fapi/v1/order",
            True,
            {"symbol": "BTCUSDT", "origClientOrderId": "XV1TSBTC123"},
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
