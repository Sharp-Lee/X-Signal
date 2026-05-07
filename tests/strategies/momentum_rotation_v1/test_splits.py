from __future__ import annotations

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from xsignal.strategies.momentum_rotation_v1.prepare import PreparedArrays
from xsignal.strategies.momentum_rotation_v1.splits import (
    filter_by_rebalance_date,
    split_research_and_holdout,
)


def arrays_with_days(day_count: int) -> PreparedArrays:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    times = np.array([start + timedelta(days=index) for index in range(day_count)], dtype=object)
    close = np.arange(1, day_count + 1, dtype=np.float64).reshape(day_count, 1)
    return PreparedArrays(
        symbols=("BTCUSDT",),
        rebalance_times=times,
        close_1h=close,
        close_4h=close,
        close_1d=close,
        quote_volume_1d=np.ones_like(close),
        complete_1h=np.ones_like(close, dtype=bool),
        complete_4h=np.ones_like(close, dtype=bool),
        complete_1d=np.ones_like(close, dtype=bool),
        quality_1h_24h=np.ones_like(close, dtype=bool),
        quality_4h_7d=np.ones_like(close, dtype=bool),
        quality_1d_30d=np.ones_like(close, dtype=bool),
    )


def test_filter_by_rebalance_date_uses_inclusive_start_exclusive_end():
    arrays = arrays_with_days(5)

    filtered = filter_by_rebalance_date(
        arrays,
        start_date=datetime(2026, 1, 2, tzinfo=timezone.utc).date(),
        end_date=datetime(2026, 1, 5, tzinfo=timezone.utc).date(),
    )

    assert filtered.rebalance_times.tolist() == arrays.rebalance_times[1:4].tolist()
    assert filtered.close_1d[:, 0].tolist() == [2.0, 3.0, 4.0]


def test_split_research_and_holdout_reserves_tail_window():
    arrays = arrays_with_days(10)

    research, holdout, metadata = split_research_and_holdout(arrays, holdout_days=3)

    assert research.rebalance_times.tolist() == arrays.rebalance_times[:6].tolist()
    assert holdout is not None
    assert holdout.rebalance_times.tolist() == arrays.rebalance_times[6:].tolist()
    assert metadata == {
        "holdout_days": 3,
        "research_start": "2026-01-01T00:00:00Z",
        "research_end": "2026-01-06T00:00:00Z",
        "holdout_start": "2026-01-07T00:00:00Z",
        "holdout_end": "2026-01-10T00:00:00Z",
    }


def test_split_research_and_holdout_can_disable_holdout():
    arrays = arrays_with_days(3)

    research, holdout, metadata = split_research_and_holdout(arrays, holdout_days=0)

    assert research is arrays
    assert holdout is None
    assert metadata["holdout_days"] == 0
    assert metadata["holdout_start"] is None


def test_split_research_and_holdout_rejects_window_that_consumes_all_rows():
    arrays = arrays_with_days(3)

    with pytest.raises(ValueError, match="leaves no research rows"):
        split_research_and_holdout(arrays, holdout_days=30)
