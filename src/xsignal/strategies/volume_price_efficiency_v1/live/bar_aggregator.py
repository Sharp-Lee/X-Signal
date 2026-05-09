from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.ws_market import (
    KlineStreamEvent,
    validate_interval,
)


@dataclass
class _Bucket:
    open_time: datetime
    close_time: datetime
    expected_minutes: int
    rows: dict[datetime, KlineStreamEvent]
    emitted: bool = False


class MultiIntervalAggregator:
    def __init__(self, *, intervals: tuple[str, ...] | list[str]) -> None:
        self.intervals = tuple(validate_interval(interval) for interval in intervals)
        self._buckets: dict[tuple[str, str, datetime], _Bucket] = {}

    def apply_1m_event(self, event: KlineStreamEvent) -> list[KlineStreamEvent]:
        if event.interval != "1m":
            raise ValueError("MultiIntervalAggregator only accepts 1m events")
        if not event.is_closed:
            return []
        emitted: list[KlineStreamEvent] = []
        for interval in self.intervals:
            if interval == "1m":
                emitted.append(event)
                continue
            bucket_start = bucket_open_time(event.open_time, interval)
            bucket_end = next_bucket_open_time(bucket_start, interval)
            key = (event.symbol, interval, bucket_start)
            bucket = self._buckets.get(key)
            if bucket is None:
                bucket = _Bucket(
                    open_time=bucket_start,
                    close_time=bucket_end - timedelta(milliseconds=1),
                    expected_minutes=expected_minutes(bucket_start, interval),
                    rows={},
                )
                self._buckets[key] = bucket
            bucket.rows[event.open_time] = event
            if _bucket_complete(bucket) and not bucket.emitted:
                bucket.emitted = True
                emitted.append(_aggregate_bucket(event.symbol, interval, bucket))
        self._trim_emitted()
        return emitted

    def _trim_emitted(self) -> None:
        stale = [
            key
            for key, bucket in self._buckets.items()
            if bucket.emitted and len(bucket.rows) == bucket.expected_minutes
        ]
        for key in stale[:-8]:
            self._buckets.pop(key, None)


def bucket_open_time(open_time: datetime, interval: str) -> datetime:
    interval = validate_interval(interval)
    normalized = open_time.astimezone(timezone.utc)
    if interval == "1M":
        return normalized.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    if interval == "1w":
        midnight = normalized.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight - timedelta(days=midnight.weekday())
    minutes = _fixed_interval_minutes(interval)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    elapsed_minutes = int((normalized - epoch).total_seconds() // 60)
    bucket_minutes = (elapsed_minutes // minutes) * minutes
    return epoch + timedelta(minutes=bucket_minutes)


def next_bucket_open_time(bucket_start: datetime, interval: str) -> datetime:
    interval = validate_interval(interval)
    if interval == "1M":
        year = bucket_start.year + (1 if bucket_start.month == 12 else 0)
        month = 1 if bucket_start.month == 12 else bucket_start.month + 1
        return bucket_start.replace(year=year, month=month)
    return bucket_start + timedelta(minutes=_fixed_interval_minutes(interval))


def expected_minutes(bucket_start: datetime, interval: str) -> int:
    return int((next_bucket_open_time(bucket_start, interval) - bucket_start).total_seconds() // 60)


def _fixed_interval_minutes(interval: str) -> int:
    interval = validate_interval(interval)
    if interval.endswith("m"):
        return int(interval[:-1])
    if interval.endswith("h"):
        return int(interval[:-1]) * 60
    if interval.endswith("d"):
        return int(interval[:-1]) * 24 * 60
    if interval == "1w":
        return 7 * 24 * 60
    raise ValueError(f"interval {interval} does not have fixed minutes")


def _bucket_complete(bucket: _Bucket) -> bool:
    if len(bucket.rows) != bucket.expected_minutes:
        return False
    last_minute = bucket.close_time + timedelta(milliseconds=1) - timedelta(minutes=1)
    return last_minute in bucket.rows


def _aggregate_bucket(symbol: str, interval: str, bucket: _Bucket) -> KlineStreamEvent:
    rows = [bucket.rows[open_time] for open_time in sorted(bucket.rows)]
    return KlineStreamEvent(
        symbol=symbol,
        interval=interval,
        event_time=rows[-1].event_time,
        open_time=bucket.open_time,
        close_time=bucket.close_time,
        open=rows[0].open,
        high=max(row.high for row in rows),
        low=min(row.low for row in rows),
        close=rows[-1].close,
        quote_volume=sum(row.quote_volume for row in rows),
        is_closed=True,
    )
