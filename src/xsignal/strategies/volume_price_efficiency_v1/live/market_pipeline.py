from __future__ import annotations

from dataclasses import dataclass, field
from queue import SimpleQueue

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
