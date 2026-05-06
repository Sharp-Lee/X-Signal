from datetime import datetime, timedelta, timezone

import pytest

from xsignal.data.canonical_bars import (
    FILL_POLICIES,
    FIXED_TIMEFRAME_SPECS,
    SUPPORTED_TIMEFRAMES,
    CanonicalRequest,
    FillPolicy,
    Partition,
    TimeframeSpec,
    expected_1m_count,
    timeframe_spec,
    validate_fill_policy,
)
from xsignal.data.paths import CanonicalPaths


def test_supported_timeframes_are_binance_fixed_length_intervals():
    assert set(FIXED_TIMEFRAME_SPECS) == {
        "1m",
        "3m",
        "5m",
        "15m",
        "30m",
        "1h",
        "2h",
        "4h",
        "6h",
        "8h",
        "12h",
        "1d",
    }
    assert SUPPORTED_TIMEFRAMES == set(FIXED_TIMEFRAME_SPECS)


def test_timeframe_specs_define_clickhouse_interval_and_partition_grain():
    assert timeframe_spec("15m") == TimeframeSpec(
        name="15m",
        minutes=15,
        clickhouse_interval="INTERVAL 15 minute",
        partition_grain="month",
    )
    assert timeframe_spec("1d") == TimeframeSpec(
        name="1d",
        minutes=1440,
        clickhouse_interval="INTERVAL 1 day",
        partition_grain="year",
    )


def test_canonical_request_defaults_to_all_symbols_full_history():
    request = CanonicalRequest(timeframe="1h")

    assert request.timeframe == "1h"
    assert request.universe == "all"
    assert request.range_name == "full_history"
    assert request.fill_policy == "raw"


def test_canonical_request_validates_fill_policy():
    with pytest.raises(ValueError, match="Unsupported fill_policy"):
        CanonicalRequest(timeframe="1h", fill_policy="forward_volume")


def test_rejects_unsupported_timeframe():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        CanonicalRequest(timeframe="1w")


def test_expected_bar_counts_cover_all_fixed_intervals():
    assert expected_1m_count("1m") == 1
    assert expected_1m_count("30m") == 30
    assert expected_1m_count("12h") == 720
    assert expected_1m_count("1d") == 1440


def test_fill_policy_defaults_and_validation():
    assert FILL_POLICIES == {"raw", "prev_close_zero_volume"}
    policy: FillPolicy = validate_fill_policy("raw")
    assert policy == "raw"
    assert validate_fill_policy("prev_close_zero_volume") == "prev_close_zero_volume"


def test_fill_policy_rejects_unknown_policy():
    with pytest.raises(ValueError, match="Unsupported fill_policy"):
        validate_fill_policy("forward_volume")


def test_partition_from_datetime():
    partition = Partition.from_datetime(
        timeframe="1h",
        value=datetime(2026, 5, 6, 11, 36, tzinfo=timezone.utc),
    )

    assert partition.timeframe == "1h"
    assert partition.year == 2026
    assert partition.month == 5
    assert partition.key == "timeframe=1h/year=2026/month=05"


def test_partition_rejects_unsupported_timeframe():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        Partition(timeframe="1w", year=2026, month=5)


def test_yearly_partition_rejects_month():
    with pytest.raises(ValueError, match="Yearly partitions"):
        Partition(timeframe="1d", year=2026, month=5)


def test_three_day_timeframe_is_not_supported_in_first_phase():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        Partition(timeframe="3d", year=2026, month=5)


def test_monthly_partition_requires_month():
    with pytest.raises(ValueError, match="Monthly partitions"):
        Partition(timeframe="1h", year=2026)


def test_intraday_partition_rejects_invalid_month():
    with pytest.raises(ValueError, match="month"):
        Partition(timeframe="1h", year=2026, month=13)


def test_partition_rejects_non_positive_year():
    with pytest.raises(ValueError, match="year"):
        Partition(timeframe="1h", year=0, month=5)


def test_partition_from_datetime_requires_timezone_aware_value():
    with pytest.raises(ValueError, match="timezone-aware"):
        Partition.from_datetime("1h", datetime(2026, 5, 1, 0, 30))


def test_partition_from_datetime_normalizes_to_utc():
    partition = Partition.from_datetime(
        "1h",
        datetime(2026, 5, 1, 0, 30, tzinfo=timezone(timedelta(hours=8))),
    )

    assert partition.year == 2026
    assert partition.month == 4
    assert partition.key == "timeframe=1h/year=2026/month=04"


def test_canonical_paths_are_deterministic(tmp_path):
    partition = Partition(timeframe="1h", year=2026, month=5)
    paths = CanonicalPaths(root=tmp_path)

    assert paths.parquet_path(partition) == (
        tmp_path / "canonical_bars" / "timeframe=1h" / "year=2026" / "month=05" / "bars.parquet"
    )
    assert paths.published_parquet_path(partition, "abc123") == (
        tmp_path / "canonical_bars" / "timeframe=1h" / "year=2026" / "month=05" / "bars.abc123.parquet"
    )
    assert paths.manifest_path(partition) == (
        tmp_path / "canonical_bars" / "timeframe=1h" / "year=2026" / "month=05" / "manifest.json"
    )
    assert paths.lock_path(partition) == (
        tmp_path / "canonical_bars" / "_locks" / "timeframe=1h__year=2026__month=05.lock"
    )
    assert paths.catalog_path("1h") == tmp_path / "canonical_bars" / "_catalog" / "timeframe=1h.json"


def test_temp_paths_accept_safe_run_ids(tmp_path):
    partition = Partition(timeframe="1h", year=2026, month=5)
    paths = CanonicalPaths(root=tmp_path)

    assert paths.temp_parquet_path(partition, "abc123") == (
        tmp_path / "canonical_bars" / "timeframe=1h" / "year=2026" / "month=05" / ".bars.abc123.tmp.parquet"
    )
    assert paths.temp_manifest_path(partition, "abc123") == (
        tmp_path / "canonical_bars" / "timeframe=1h" / "year=2026" / "month=05" / ".manifest.abc123.tmp.json"
    )


def test_temp_parquet_path_rejects_unsafe_run_id(tmp_path):
    partition = Partition(timeframe="1h", year=2026, month=5)
    paths = CanonicalPaths(root=tmp_path)

    with pytest.raises(ValueError, match="run_id"):
        paths.temp_parquet_path(partition, "x/../../escape")


def test_published_parquet_path_rejects_unsafe_run_id(tmp_path):
    partition = Partition(timeframe="1h", year=2026, month=5)
    paths = CanonicalPaths(root=tmp_path)

    with pytest.raises(ValueError, match="run_id"):
        paths.published_parquet_path(partition, "x/../../escape")


def test_temp_manifest_path_rejects_unsafe_run_id(tmp_path):
    partition = Partition(timeframe="1h", year=2026, month=5)
    paths = CanonicalPaths(root=tmp_path)

    with pytest.raises(ValueError, match="run_id"):
        paths.temp_manifest_path(partition, "..")
