from datetime import datetime, timedelta, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.bar_aggregator import (
    MultiIntervalAggregator,
)
from xsignal.strategies.volume_price_efficiency_v1.live.recovery import (
    ReplaySink,
    event_from_market_bar,
    recovery_start_time,
    replay_closed_1m_events,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore
from xsignal.strategies.volume_price_efficiency_v1.live.ws_market import KlineStreamEvent


def _minute_event(open_time: datetime) -> KlineStreamEvent:
    return KlineStreamEvent(
        symbol="BTCUSDT",
        interval="1m",
        event_time=open_time + timedelta(seconds=59),
        open_time=open_time,
        close_time=open_time + timedelta(seconds=59, milliseconds=999),
        open=100.0,
        high=101.0,
        low=99.0,
        close=100.5,
        quote_volume=10.0,
        is_closed=True,
    )


class RecordingSink(ReplaySink):
    def __init__(self) -> None:
        self.price_events = []
        self.closed_bars = []

    def process_price_event(
        self,
        event,
        *,
        allow_pyramid_add: bool,
        allow_stop_replace: bool,
    ):
        self.price_events.append(
            (event.interval, event.open_time, allow_pyramid_add, allow_stop_replace)
        )

    def process_closed_bar(
        self,
        event,
        *,
        allow_entry: bool,
        allow_pyramid_add: bool,
        allow_stop_replace: bool,
    ):
        self.closed_bars.append(
            (event.interval, event.open_time, allow_entry, allow_pyramid_add, allow_stop_replace)
        )


def test_recovery_start_uses_cursor_plus_one_minute_when_available(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    cursor = datetime(2026, 5, 9, 8, 5, tzinfo=timezone.utc)
    store.advance_market_cursor(symbol="BTCUSDT", interval="1m", open_time=cursor)

    assert recovery_start_time(
        store=store,
        symbol="BTCUSDT",
        target_intervals=("4h", "1d"),
        server_time_ms=1778322600000,
    ) == cursor + timedelta(minutes=1)


def test_recovery_start_without_cursor_starts_after_latest_closed_1m(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()

    assert recovery_start_time(
        store=store,
        symbol="BTCUSDT",
        target_intervals=("4h", "1d"),
        server_time_ms=1778347800000,
    ) == datetime(2026, 5, 9, 17, 30, tzinfo=timezone.utc)


def test_replay_persists_1m_bars_advances_cursor_and_blocks_historical_entries(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    aggregator = MultiIntervalAggregator(intervals=("1h",))
    sink = RecordingSink()
    start = datetime(2026, 5, 9, 8, tzinfo=timezone.utc)

    result = replay_closed_1m_events(
        store=store,
        aggregator=aggregator,
        sink=sink,
        events=[_minute_event(start + timedelta(minutes=offset)) for offset in range(60)],
        allow_entries=False,
        allow_pyramid_add=False,
    )

    assert result.source_bars == 60
    assert result.aggregated_bars == 1
    assert store.get_market_cursor(symbol="BTCUSDT", interval="1m") == start + timedelta(minutes=59)
    assert len(store.list_market_bars(symbol="BTCUSDT", interval="1m")) == 60
    assert store.list_market_bars(symbol="BTCUSDT", interval="1h")[0]["open_time"] == start
    assert all(item[2:] == (False, False) for item in sink.price_events)
    assert sink.closed_bars == [("1h", start, False, False, False)]


def test_event_from_market_bar_round_trips_store_rows():
    row = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "open_time": datetime(2026, 5, 9, 8, tzinfo=timezone.utc),
        "open": 100.0,
        "high": 101.0,
        "low": 99.0,
        "close": 100.5,
        "quote_volume": 10.0,
        "is_complete": True,
    }

    event = event_from_market_bar(row)

    assert event.symbol == "BTCUSDT"
    assert event.interval == "1m"
    assert event.close_time == row["open_time"] + timedelta(minutes=1) - timedelta(milliseconds=1)
