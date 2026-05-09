from datetime import datetime, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.market_pipeline import MarketEventRouter
from xsignal.strategies.volume_price_efficiency_v1.live.market_pipeline import ClosedBarBatchWorker
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


class FakeStore:
    def __init__(self):
        self.bars = []
        self.cursors = []
        self.commits = 0
        self.connection = self

    def upsert_market_bar(self, row, *, commit=True):
        self.bars.append((row["symbol"], row["interval"], row["open_time"], commit))

    def advance_market_cursor(self, *, symbol, interval, open_time, commit=True):
        self.cursors.append((symbol, interval, open_time, commit))

    def commit(self):
        self.commits += 1


class FakeAggregator:
    def __init__(self, aggregate):
        self.aggregate = aggregate

    def apply_1m_event(self, item):
        return [self.aggregate]


class BatchService(ActiveService):
    def __init__(self):
        super().__init__()
        self.closed_batches = []

    def process_closed_bar_batch(self, events, *, allow_entry, allow_pyramid_add, allow_stop_replace):
        self.closed_batches.append((tuple(item.symbol for item in events), allow_entry))
        return []


def test_closed_worker_batches_store_writes_and_publishes_aggregates():
    store = FakeStore()
    aggregate = event("BTCUSDT", closed=True)
    service = BatchService()
    worker = ClosedBarBatchWorker(
        store=store,
        aggregator=FakeAggregator(aggregate),
        service=service,
    )

    worker.process_many([event("BTCUSDT", closed=True), event("ETHUSDT", closed=True)])

    assert store.commits == 1
    assert ("BTCUSDT", "1m", NOW, False) in store.bars
    assert ("ETHUSDT", "1m", NOW, False) in store.bars
    assert service.closed_batches == [(("BTCUSDT", "BTCUSDT"), True)]
