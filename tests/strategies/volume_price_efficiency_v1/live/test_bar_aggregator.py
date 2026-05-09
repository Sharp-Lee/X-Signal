from datetime import datetime, timedelta, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.bar_aggregator import (
    MultiIntervalAggregator,
)
from xsignal.strategies.volume_price_efficiency_v1.live.ws_market import KlineStreamEvent


def _minute_event(open_time: datetime, *, close: float = 100.0) -> KlineStreamEvent:
    minute = int((open_time.minute + open_time.hour * 60) % 50)
    high = close + minute + 1
    low = close - minute - 1
    return KlineStreamEvent(
        symbol="BTCUSDT",
        interval="1m",
        event_time=open_time + timedelta(seconds=59, milliseconds=900),
        open_time=open_time,
        close_time=open_time + timedelta(seconds=59, milliseconds=999),
        open=close,
        high=high,
        low=low,
        close=close + 0.5,
        quote_volume=10.0 + minute,
        is_closed=True,
    )


def test_aggregator_emits_1h_bar_when_all_closed_minutes_are_present():
    aggregator = MultiIntervalAggregator(intervals=("1h",))
    start = datetime(2026, 5, 9, 8, tzinfo=timezone.utc)
    emitted = []

    for offset in range(60):
        emitted.extend(aggregator.apply_1m_event(_minute_event(start + timedelta(minutes=offset))))

    assert len(emitted) == 1
    bar = emitted[0]
    assert bar.symbol == "BTCUSDT"
    assert bar.interval == "1h"
    assert bar.open_time == start
    assert bar.close_time == start + timedelta(hours=1) - timedelta(milliseconds=1)
    assert bar.open == 100.0
    assert bar.high == max(_minute_event(start + timedelta(minutes=i)).high for i in range(60))
    assert bar.low == min(_minute_event(start + timedelta(minutes=i)).low for i in range(60))
    assert bar.close == 100.5
    assert bar.quote_volume == sum(
        _minute_event(start + timedelta(minutes=i)).quote_volume for i in range(60)
    )
    assert bar.is_closed is True


def test_aggregator_does_not_emit_incomplete_4h_bucket():
    aggregator = MultiIntervalAggregator(intervals=("4h",))
    start = datetime(2026, 5, 9, 0, tzinfo=timezone.utc)
    emitted = []

    for offset in range(240):
        if offset == 137:
            continue
        emitted.extend(aggregator.apply_1m_event(_minute_event(start + timedelta(minutes=offset))))

    assert emitted == []


def test_aggregator_emits_4h_and_1d_on_utc_boundaries():
    aggregator = MultiIntervalAggregator(intervals=("4h", "1d"))
    start = datetime(2026, 5, 9, tzinfo=timezone.utc)
    emitted = []

    for offset in range(24 * 60):
        emitted.extend(aggregator.apply_1m_event(_minute_event(start + timedelta(minutes=offset))))

    emitted_keys = [(bar.interval, bar.open_time) for bar in emitted]
    assert emitted_keys[:6] == [
        ("4h", datetime(2026, 5, 9, hour, tzinfo=timezone.utc))
        for hour in (0, 4, 8, 12, 16, 20)
    ]
    assert emitted_keys[-1] == ("1d", start)
