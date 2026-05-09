from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Protocol

from xsignal.strategies.volume_price_efficiency_v1.live.bar_aggregator import (
    MultiIntervalAggregator,
    bucket_open_time,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore
from xsignal.strategies.volume_price_efficiency_v1.live.ws_market import KlineStreamEvent


class ReplaySink(Protocol):
    def process_price_event(
        self,
        event: KlineStreamEvent,
        *,
        allow_pyramid_add: bool,
        allow_stop_replace: bool,
    ):
        ...

    def process_closed_bar(
        self,
        event: KlineStreamEvent,
        *,
        allow_entry: bool,
        allow_pyramid_add: bool,
        allow_stop_replace: bool,
    ):
        ...


@dataclass(frozen=True)
class ReplayResult:
    source_bars: int
    aggregated_bars: int


def recovery_start_time(
    *,
    store: LiveStore,
    symbol: str,
    target_intervals: tuple[str, ...] | list[str],
    server_time_ms: int,
) -> datetime:
    cursor = store.get_market_cursor(symbol=symbol, interval="1m")
    if cursor is not None:
        return cursor + timedelta(minutes=1)
    latest = latest_closed_1m_open_time(server_time_ms)
    if not target_intervals:
        return latest
    return min(bucket_open_time(latest, interval) for interval in target_intervals)


def latest_closed_1m_open_time(server_time_ms: int) -> datetime:
    minute_start_ms = (server_time_ms // 60_000) * 60_000
    return datetime.fromtimestamp((minute_start_ms - 60_000) / 1000, tz=timezone.utc)


def replay_closed_1m_events(
    *,
    store: LiveStore,
    aggregator: MultiIntervalAggregator,
    sink: ReplaySink | None,
    events: list[KlineStreamEvent],
    allow_entries: bool,
    allow_pyramid_add: bool,
) -> ReplayResult:
    source_bars = 0
    aggregated_bars = 0
    latest_cursor_by_symbol: dict[str, datetime] = {}
    for event in sorted(events, key=lambda item: (item.symbol, item.open_time)):
        if event.interval != "1m" or not event.is_closed:
            continue
        source_bars += 1
        store.upsert_market_bar(market_bar_from_event(event), commit=False)
        latest_cursor_by_symbol[event.symbol] = event.open_time
        if sink is not None:
            sink.process_price_event(
                event,
                allow_pyramid_add=allow_pyramid_add,
                allow_stop_replace=False,
            )
        for aggregate in aggregator.apply_1m_event(event):
            aggregated_bars += 1
            store.upsert_market_bar(market_bar_from_event(aggregate), commit=False)
            if sink is not None:
                sink.process_closed_bar(
                    aggregate,
                    allow_entry=allow_entries,
                    allow_pyramid_add=allow_pyramid_add,
                    allow_stop_replace=False,
                )
    for symbol, open_time in latest_cursor_by_symbol.items():
        store.advance_market_cursor(
            symbol=symbol,
            interval="1m",
            open_time=open_time,
            commit=False,
        )
    if source_bars or aggregated_bars or latest_cursor_by_symbol:
        store.connection.commit()
    return ReplayResult(source_bars=source_bars, aggregated_bars=aggregated_bars)


def market_bar_from_event(event: KlineStreamEvent) -> dict[str, object]:
    return {
        "symbol": event.symbol,
        "interval": event.interval,
        "open_time": event.open_time,
        "open": event.open,
        "high": event.high,
        "low": event.low,
        "close": event.close,
        "quote_volume": event.quote_volume,
        "is_complete": event.is_closed,
    }


def event_from_market_bar(row: dict[str, object]) -> KlineStreamEvent:
    open_time = row["open_time"]
    if not isinstance(open_time, datetime):
        raise ValueError("market bar open_time must be a datetime")
    interval = str(row["interval"])
    return KlineStreamEvent(
        symbol=str(row["symbol"]),
        interval=interval,
        event_time=open_time + _interval_delta(interval) - timedelta(milliseconds=1),
        open_time=open_time,
        close_time=open_time + _interval_delta(interval) - timedelta(milliseconds=1),
        open=float(row["open"]),
        high=float(row["high"]),
        low=float(row["low"]),
        close=float(row["close"]),
        quote_volume=float(row["quote_volume"]),
        is_closed=bool(row["is_complete"]),
    )


def _interval_delta(interval: str) -> timedelta:
    if interval.endswith("m"):
        return timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return timedelta(hours=int(interval[:-1]))
    if interval.endswith("d"):
        return timedelta(days=int(interval[:-1]))
    if interval == "1w":
        return timedelta(days=7)
    raise ValueError(f"interval delta is not fixed for {interval}")
