from __future__ import annotations

from collections import defaultdict
from datetime import datetime

from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.live.market_data import (
    build_arrays_from_klines,
)
from xsignal.strategies.volume_price_efficiency_v1.live.ws_market import (
    KlineStreamEvent,
    validate_interval,
)


class RollingBarBuffer:
    def __init__(self, *, interval: str, max_bars: int) -> None:
        self.interval = validate_interval(interval)
        if max_bars <= 0:
            raise ValueError("max_bars must be positive")
        self.max_bars = max_bars
        self._rows: dict[tuple[str, datetime], dict[str, object]] = {}
        self._times_by_symbol: dict[str, set[datetime]] = defaultdict(set)

    def seed_rows(self, rows: list[dict[str, object]]) -> None:
        for row in rows:
            if row.get("interval", self.interval) != self.interval:
                raise ValueError("row interval does not match buffer interval")
            self._put_row(row)

    def apply_event(self, event: KlineStreamEvent) -> None:
        if event.interval != self.interval:
            raise ValueError("event interval does not match buffer interval")
        if not event.is_closed:
            return
        self._put_row(
            {
                "symbol": event.symbol,
                "interval": event.interval,
                "open_time": event.open_time,
                "open": event.open,
                "high": event.high,
                "low": event.low,
                "close": event.close,
                "quote_volume": event.quote_volume,
                "is_complete": True,
            }
        )

    def to_arrays(self) -> OhlcvArrays:
        return build_arrays_from_klines(list(self._rows.values()))

    def _put_row(self, row: dict[str, object]) -> None:
        symbol = str(row["symbol"])
        open_time = row["open_time"]
        if not isinstance(open_time, datetime):
            raise ValueError("open_time must be a datetime")
        key = (symbol, open_time)
        self._rows[key] = dict(row)
        self._times_by_symbol[symbol].add(open_time)
        self._trim_symbol(symbol)

    def _trim_symbol(self, symbol: str) -> None:
        times = sorted(self._times_by_symbol[symbol])
        stale = times[: max(len(times) - self.max_bars, 0)]
        for open_time in stale:
            self._rows.pop((symbol, open_time), None)
            self._times_by_symbol[symbol].discard(open_time)
