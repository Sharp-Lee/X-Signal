from datetime import datetime, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.bar_buffer import RollingBarBuffer
from xsignal.strategies.volume_price_efficiency_v1.live.ws_market import KlineStreamEvent


def _event(symbol: str, minute: int, close: float, *, closed: bool = True) -> KlineStreamEvent:
    return KlineStreamEvent(
        symbol=symbol,
        interval="1h",
        event_time=datetime(2026, 5, 9, 8, minute, tzinfo=timezone.utc),
        open_time=datetime(2026, 5, 9, 8 + minute, tzinfo=timezone.utc),
        close_time=datetime(2026, 5, 9, 8 + minute, 59, 59, tzinfo=timezone.utc),
        open=close - 1,
        high=close + 1,
        low=close - 2,
        close=close,
        quote_volume=1000.0,
        is_closed=closed,
    )


def test_rolling_bar_buffer_seeds_rows_and_emits_arrays():
    buffer = RollingBarBuffer(interval="1h", max_bars=3)
    buffer.seed_rows(
        [
            {
                "symbol": "ETHUSDT",
                "interval": "1h",
                "open_time": datetime(2026, 5, 9, 8, tzinfo=timezone.utc),
                "open": 200.0,
                "high": 210.0,
                "low": 190.0,
                "close": 205.0,
                "quote_volume": 1000.0,
                "is_complete": True,
            },
            {
                "symbol": "BTCUSDT",
                "interval": "1h",
                "open_time": datetime(2026, 5, 9, 8, tzinfo=timezone.utc),
                "open": 100.0,
                "high": 110.0,
                "low": 90.0,
                "close": 105.0,
                "quote_volume": 1000.0,
                "is_complete": True,
            },
        ]
    )

    arrays = buffer.to_arrays()

    assert arrays.symbols == ("BTCUSDT", "ETHUSDT")
    assert arrays.open.shape == (1, 2)
    assert arrays.close[0, 0] == 105.0
    assert arrays.close[0, 1] == 205.0


def test_rolling_bar_buffer_appends_only_closed_events_and_trims_history():
    buffer = RollingBarBuffer(interval="1h", max_bars=2)

    buffer.apply_event(_event("BTCUSDT", 0, 100.0, closed=True))
    buffer.apply_event(_event("BTCUSDT", 1, 101.0, closed=False))
    buffer.apply_event(_event("BTCUSDT", 1, 102.0, closed=True))
    buffer.apply_event(_event("BTCUSDT", 2, 103.0, closed=True))

    arrays = buffer.to_arrays()

    assert [value.hour for value in arrays.open_times] == [9, 10]
    assert arrays.close[:, 0].tolist() == [102.0, 103.0]
