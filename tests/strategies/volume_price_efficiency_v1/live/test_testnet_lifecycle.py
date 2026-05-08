import pytest

from xsignal.strategies.volume_price_efficiency_v1.live.binance_rest import BinanceApiError
from xsignal.strategies.volume_price_efficiency_v1.live.testnet_lifecycle import (
    run_testnet_lifecycle,
)


class FakeLifecycleBroker:
    def __init__(self) -> None:
        self.calls = []
        self.position_risk_payloads = [
            [
                {
                    "symbol": "BTCUSDT",
                    "positionSide": "BOTH",
                    "positionAmt": "0.001",
                    "entryPrice": "100",
                    "markPrice": "101",
                }
            ],
            [
                {
                    "symbol": "BTCUSDT",
                    "positionSide": "BOTH",
                    "positionAmt": "0.001",
                    "entryPrice": "100",
                    "markPrice": "101",
                }
            ],
            [
                {
                    "symbol": "BTCUSDT",
                    "positionSide": "BOTH",
                    "positionAmt": "0",
                    "entryPrice": "0",
                    "markPrice": "101",
                }
            ],
        ]

    def change_margin_type(self, symbol, margin_mode):
        self.calls.append(("change_margin_type", symbol, margin_mode))
        return {}

    def change_leverage(self, symbol, leverage):
        self.calls.append(("change_leverage", symbol, leverage))
        return {}

    def market_buy(self, *, symbol, quantity, client_order_id):
        self.calls.append(("market_buy", symbol, quantity, client_order_id))
        return {}

    def get_position_risk(self, *, symbol):
        self.calls.append(("get_position_risk", symbol))
        return self.position_risk_payloads.pop(0)

    def place_stop_market_close(self, *, symbol, stop_price, client_order_id):
        self.calls.append(("place_stop_market_close", symbol, stop_price, client_order_id))
        return {}

    def get_open_order(self, *, symbol, client_order_id):
        self.calls.append(("get_open_order", symbol, client_order_id))
        return {"symbol": symbol, "clientAlgoId": client_order_id, "algoStatus": "NEW"}

    def cancel_order(self, *, symbol, client_order_id):
        self.calls.append(("cancel_order", symbol, client_order_id))
        return {}

    def market_sell_reduce_only(self, *, symbol, quantity, client_order_id):
        self.calls.append(("market_sell_reduce_only", symbol, quantity, client_order_id))
        return {}


def test_run_testnet_lifecycle_opens_protects_closes_and_verifies_flat():
    broker = FakeLifecycleBroker()

    result = run_testnet_lifecycle(
        broker=broker,
        symbol="BTCUSDT",
        quantity=0.001,
        stop_offset_pct=0.05,
        position_id="test-position",
    )

    assert result.symbol == "BTCUSDT"
    assert result.opened_position_amount == pytest.approx(0.001)
    assert result.final_position_amount == pytest.approx(0.0)
    assert result.stop_price == pytest.approx(95.0)
    assert result.entry_client_order_id.startswith("XV1TE")
    assert result.stop_client_order_id.startswith("XV1TS")
    assert result.close_client_order_id.startswith("XV1TM")
    assert [call[0] for call in broker.calls] == [
        "change_margin_type",
        "change_leverage",
        "market_buy",
        "get_position_risk",
        "place_stop_market_close",
        "get_position_risk",
        "get_open_order",
        "cancel_order",
        "market_sell_reduce_only",
        "get_position_risk",
    ]


def test_run_testnet_lifecycle_cleans_up_stop_and_position_after_late_failure():
    class FailingBroker(FakeLifecycleBroker):
        def __init__(self) -> None:
            super().__init__()
            self.position_risk_payloads = [
                [
                    {
                        "symbol": "BTCUSDT",
                        "positionSide": "BOTH",
                        "positionAmt": "0.001",
                        "entryPrice": "100",
                        "markPrice": "101",
                    }
                ],
                [
                    {
                        "symbol": "BTCUSDT",
                        "positionSide": "BOTH",
                        "positionAmt": "0.001",
                        "entryPrice": "100",
                        "markPrice": "101",
                    }
                ],
                [
                    {
                        "symbol": "BTCUSDT",
                        "positionSide": "BOTH",
                        "positionAmt": "0.001",
                        "entryPrice": "100",
                        "markPrice": "101",
                    }
                ],
            ]

        def get_open_order(self, *, symbol, client_order_id):
            self.calls.append(("get_open_order", symbol, client_order_id))
            raise RuntimeError("open order check failed")

    broker = FailingBroker()

    with pytest.raises(RuntimeError, match="open order check failed"):
        run_testnet_lifecycle(
            broker=broker,
            symbol="BTCUSDT",
            quantity=0.001,
            stop_offset_pct=0.05,
            position_id="test-position",
        )

    call_names = [call[0] for call in broker.calls]
    assert call_names[-4:] == [
        "get_open_order",
        "cancel_order",
        "get_position_risk",
        "market_sell_reduce_only",
    ]


def test_run_testnet_lifecycle_retries_until_position_is_visible():
    broker = FakeLifecycleBroker()
    broker.position_risk_payloads = [
        [
            {
                "symbol": "BTCUSDT",
                "positionSide": "BOTH",
                "positionAmt": "0",
                "entryPrice": "0",
                "markPrice": "100",
            }
        ],
        [
            {
                "symbol": "BTCUSDT",
                "positionSide": "BOTH",
                "positionAmt": "0.001",
                "entryPrice": "100",
                "markPrice": "101",
            }
        ],
        [
            {
                "symbol": "BTCUSDT",
                "positionSide": "BOTH",
                "positionAmt": "0.001",
                "entryPrice": "100",
                "markPrice": "101",
            }
        ],
        [
            {
                "symbol": "BTCUSDT",
                "positionSide": "BOTH",
                "positionAmt": "0",
                "entryPrice": "0",
                "markPrice": "101",
            }
        ],
    ]

    result = run_testnet_lifecycle(
        broker=broker,
        symbol="BTCUSDT",
        quantity=0.001,
        stop_offset_pct=0.05,
        position_id="test-position",
        poll_attempts=3,
        poll_sleep_seconds=0,
    )

    assert result.opened_position_amount == pytest.approx(0.001)
    assert [call[0] for call in broker.calls].count("get_position_risk") == 4


def test_run_testnet_lifecycle_treats_empty_final_position_risk_as_flat():
    broker = FakeLifecycleBroker()
    broker.position_risk_payloads = [
        [
            {
                "symbol": "BTCUSDT",
                "positionSide": "BOTH",
                "positionAmt": "0.001",
                "entryPrice": "100",
                "markPrice": "101",
            }
        ],
        [
            {
                "symbol": "BTCUSDT",
                "positionSide": "BOTH",
                "positionAmt": "0.001",
                "entryPrice": "100",
                "markPrice": "101",
            }
        ],
        [],
    ]

    result = run_testnet_lifecycle(
        broker=broker,
        symbol="BTCUSDT",
        quantity=0.001,
        stop_offset_pct=0.05,
        position_id="test-position",
    )

    assert result.final_position_amount == pytest.approx(0.0)


def test_run_testnet_lifecycle_rounds_stop_price_down_to_tick():
    broker = FakeLifecycleBroker()
    broker.position_risk_payloads[0][0]["entryPrice"] = "100.07"
    broker.position_risk_payloads[1][0]["entryPrice"] = "100.07"

    result = run_testnet_lifecycle(
        broker=broker,
        symbol="BTCUSDT",
        quantity=0.001,
        stop_offset_pct=0.05,
        position_id="test-position",
        price_tick=0.1,
    )

    assert result.stop_price == pytest.approx(95.0)


def test_run_testnet_lifecycle_treats_already_isolated_margin_as_success():
    class AlreadyIsolatedBroker(FakeLifecycleBroker):
        def change_margin_type(self, symbol, margin_mode):
            self.calls.append(("change_margin_type", symbol, margin_mode))
            raise BinanceApiError(
                status=400,
                code=-4046,
                message="No need to change margin type.",
            )

    broker = AlreadyIsolatedBroker()

    result = run_testnet_lifecycle(
        broker=broker,
        symbol="BTCUSDT",
        quantity=0.001,
        stop_offset_pct=0.05,
        position_id="test-position",
    )

    assert result.final_position_amount == pytest.approx(0.0)
    assert [call[0] for call in broker.calls][0:3] == [
        "change_margin_type",
        "change_leverage",
        "market_buy",
    ]
