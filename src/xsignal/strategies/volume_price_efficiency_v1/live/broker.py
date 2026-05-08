from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BrokerOrder:
    symbol: str
    side: str
    order_type: str
    quantity: float | None
    stop_price: float | None
    close_position: bool


class FakeBroker:
    def __init__(self) -> None:
        self.orders: list[BrokerOrder] = []
        self.cancelled_order_ids: list[str] = []

    def market_buy(self, *, symbol: str, quantity: float) -> BrokerOrder:
        order = BrokerOrder(symbol, "BUY", "MARKET", quantity, None, False)
        self.orders.append(order)
        return order

    def place_stop_market_close(self, *, symbol: str, stop_price: float) -> BrokerOrder:
        order = BrokerOrder(symbol, "SELL", "STOP_MARKET", None, stop_price, True)
        self.orders.append(order)
        return order

    def cancel_order(self, order_id: str) -> None:
        self.cancelled_order_ids.append(order_id)
