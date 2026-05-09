from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from queue import SimpleQueue

from xsignal.strategies.volume_price_efficiency_v1.live.recovery import market_bar_from_event
from xsignal.strategies.volume_price_efficiency_v1.live.ws_market import KlineStreamEvent


@dataclass
class MarketEventRouter:
    service: object
    closed_queue: SimpleQueue[KlineStreamEvent] = field(default_factory=SimpleQueue)
    _latest_unclosed: dict[str, KlineStreamEvent] = field(default_factory=dict)

    def route(self, event: KlineStreamEvent) -> None:
        if event.is_closed:
            self.closed_queue.put(event)
            return
        self._latest_unclosed[event.symbol] = event
        if self.service.has_active_symbol_position(event.symbol):
            self.service.process_price_event(
                event,
                allow_pyramid_add=True,
                allow_stop_replace=True,
            )

    def latest_unclosed(self, symbol: str) -> KlineStreamEvent | None:
        return self._latest_unclosed.get(symbol)


@dataclass
class ClosedBarBatchWorker:
    store: object
    aggregator: object
    service: object

    def process_many(self, events: list[KlineStreamEvent], *, allow_entry: bool = True) -> None:
        aggregates_by_key: dict[tuple[str, object], list[KlineStreamEvent]] = defaultdict(list)
        for event in events:
            if event.interval != "1m" or not event.is_closed:
                continue
            self.store.upsert_market_bar(market_bar_from_event(event), commit=False)
            self.store.advance_market_cursor(
                symbol=event.symbol,
                interval="1m",
                open_time=event.open_time,
                commit=False,
            )
            self.service.process_price_event(
                event,
                allow_pyramid_add=True,
                allow_stop_replace=True,
            )
            for aggregate in self.aggregator.apply_1m_event(event):
                self.store.upsert_market_bar(market_bar_from_event(aggregate), commit=False)
                aggregates_by_key[(aggregate.interval, aggregate.open_time)].append(aggregate)
        for batch in aggregates_by_key.values():
            self.service.process_closed_bar_batch(
                batch,
                allow_entry=allow_entry,
                allow_pyramid_add=True,
                allow_stop_replace=True,
            )
        self.store.connection.commit()
