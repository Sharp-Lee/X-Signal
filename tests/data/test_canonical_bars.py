from datetime import datetime, timezone

import pytest

from xsignal.data.canonical_bars import (
    SUPPORTED_TIMEFRAMES,
    CanonicalRequest,
    Partition,
    expected_1m_count,
)


def test_supported_timeframes_are_explicit():
    assert SUPPORTED_TIMEFRAMES == {"1h", "4h", "1d"}


def test_canonical_request_defaults_to_all_symbols_full_history():
    request = CanonicalRequest(timeframe="1h")

    assert request.timeframe == "1h"
    assert request.universe == "all"
    assert request.range_name == "full_history"


def test_rejects_unsupported_timeframe():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        CanonicalRequest(timeframe="15m")


def test_expected_bar_counts():
    assert expected_1m_count("1h") == 60
    assert expected_1m_count("4h") == 240
    assert expected_1m_count("1d") == 1440


def test_partition_from_datetime():
    partition = Partition.from_datetime(
        timeframe="1h",
        value=datetime(2026, 5, 6, 11, 36, tzinfo=timezone.utc),
    )

    assert partition.timeframe == "1h"
    assert partition.year == 2026
    assert partition.month == 5
    assert partition.key == "timeframe=1h/year=2026/month=05"
