from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.splits import (
    holdout_mask_for_open_times,
    split_research_and_holdout,
)


def _arrays(day_count: int = 10) -> OhlcvArrays:
    open_times = np.array(
        [
            datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(days=index)
            for index in range(day_count)
        ],
        dtype=object,
    )
    values = np.arange(day_count, dtype=np.float64).reshape(day_count, 1) + 100.0
    return OhlcvArrays(
        symbols=("BTCUSDT",),
        open_times=open_times,
        open=values.copy(),
        high=values.copy() + 1.0,
        low=values.copy() - 1.0,
        close=values.copy() + 0.5,
        quote_volume=values.copy() * 1_000.0,
        quality=np.ones((day_count, 1), dtype=bool),
    )


def test_split_research_and_holdout_reserves_tail_window():
    research, holdout, metadata = split_research_and_holdout(_arrays(), holdout_days=3)

    assert research.open_times.tolist() == [
        datetime(2026, 1, day, tzinfo=timezone.utc) for day in range(1, 7)
    ]
    assert holdout is not None
    assert holdout.open_times.tolist() == [
        datetime(2026, 1, day, tzinfo=timezone.utc) for day in range(7, 11)
    ]
    assert metadata == {
        "holdout_days": 3,
        "research_start": "2026-01-01T00:00:00Z",
        "research_end": "2026-01-06T00:00:00Z",
        "holdout_start": "2026-01-07T00:00:00Z",
        "holdout_end": "2026-01-10T00:00:00Z",
    }


def test_split_research_and_holdout_can_disable_holdout():
    arrays = _arrays(2)

    research, holdout, metadata = split_research_and_holdout(arrays, holdout_days=0)

    assert research is arrays
    assert holdout is None
    assert metadata == {
        "holdout_days": 0,
        "research_start": "2026-01-01T00:00:00Z",
        "research_end": "2026-01-02T00:00:00Z",
        "holdout_start": None,
        "holdout_end": None,
    }


def test_split_research_and_holdout_rejects_negative_days():
    with pytest.raises(ValueError, match="holdout_days must be non-negative"):
        split_research_and_holdout(_arrays(), holdout_days=-1)


def test_split_research_and_holdout_rejects_window_that_consumes_all_rows():
    with pytest.raises(ValueError, match="holdout window leaves no research rows"):
        split_research_and_holdout(_arrays(3), holdout_days=30)


def test_holdout_mask_for_open_times_matches_tail_window():
    arrays = _arrays()

    mask = holdout_mask_for_open_times(arrays.open_times, holdout_days=3)

    assert mask.tolist() == [False, False, False, False, False, False, True, True, True, True]
