from datetime import datetime, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.market_pipeline import MarketEventRouter
from xsignal.strategies.volume_price_efficiency_v1.live.ws_market import KlineStreamEvent


NOW = datetime(2026, 5, 10, 8, tzinfo=timezone.utc)


class ActiveService:
    def __init__(self):
        self.active = {"BTCUSDT"}
        self.price_events = []

    def has_active_symbol_position(self, symbol: str) -> bool:
        return symbol in self.active

    def process_price_event(self, event, *, allow_pyramid_add=True, allow_stop_replace=True):
        self.price_events.append((event.symbol, event.high, event.close))
        return type("Result", (), {"entries": 0, "adds": 0, "stop_updates": 0})()


def event(symbol: str, *, closed: bool) -> KlineStreamEvent:
    return KlineStreamEvent(
        symbol=symbol,
        interval="1m",
        event_time=NOW,
        open_time=NOW,
        close_time=NOW,
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        quote_volume=10.0,
        is_closed=closed,
    )


def test_unclosed_events_update_latest_and_only_active_symbols_are_maintained():
    service = ActiveService()
    router = MarketEventRouter(service=service)

    router.route(event("ETHUSDT", closed=False))
    router.route(event("BTCUSDT", closed=False))

    assert router.latest_unclosed("ETHUSDT").close == 100.5
    assert router.latest_unclosed("BTCUSDT").high == 101.0
    assert service.price_events == [("BTCUSDT", 101.0, 100.5)]
