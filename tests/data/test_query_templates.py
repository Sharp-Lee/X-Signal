from datetime import datetime, timedelta, timezone

import pytest

from xsignal.data.query_templates import build_aggregate_query, query_hash


def test_build_aggregate_query_uses_final_and_expected_interval():
    sql = build_aggregate_query(
        timeframe="1h",
        start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    assert sql == """
SELECT
    symbol,
    bucket_open_time AS open_time,
    open,
    high,
    low,
    close,
    volume,
    quote_volume,
    trade_count,
    taker_buy_volume,
    taker_buy_quote_volume,
    bar_count,
    is_complete
FROM
(
    SELECT
        symbol,
        toStartOfInterval(k.open_time, INTERVAL 1 hour, 'UTC') AS bucket_open_time,
        toFloat64(argMin(open, k.open_time)) AS open,
        toFloat64(max(high)) AS high,
        toFloat64(min(low)) AS low,
        toFloat64(argMax(close, k.open_time)) AS close,
        toFloat64(sum(volume)) AS volume,
        toFloat64(sum(quote_volume)) AS quote_volume,
        toUInt64(sum(trade_count)) AS trade_count,
        toFloat64(sum(taker_buy_volume)) AS taker_buy_volume,
        toFloat64(sum(taker_buy_quote_volume)) AS taker_buy_quote_volume,
        toUInt16(count()) AS bar_count,
        bar_count = 60 AS is_complete
    FROM xgate.klines_1m AS k FINAL
    WHERE k.open_time >= toDateTime('2026-05-01 00:00:00', 'UTC')
      AND k.open_time < toDateTime('2026-06-01 00:00:00', 'UTC')
    GROUP BY
        symbol,
        bucket_open_time
)
ORDER BY
    bucket_open_time,
    symbol
""".strip()


def test_build_aggregate_query_groups_by_non_conflicting_bucket_alias():
    sql = build_aggregate_query(
        timeframe="1h",
        start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    assert "AS bucket_open_time" in sql
    assert "bucket_open_time AS open_time" in sql
    assert "GROUP BY\n        symbol,\n        bucket_open_time" in sql
    assert "ORDER BY\n    bucket_open_time,\n    symbol" in sql
    assert "GROUP BY\n    symbol,\n    open_time" not in sql


@pytest.mark.parametrize(
    ("timeframe", "interval_sql", "expected_count"),
    [
        ("1h", "INTERVAL 1 hour", 60),
        ("4h", "INTERVAL 4 hour", 240),
        ("1d", "INTERVAL 1 day", 1440),
    ],
)
def test_build_aggregate_query_uses_supported_intervals(timeframe, interval_sql, expected_count):
    sql = build_aggregate_query(
        timeframe=timeframe,
        start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    assert f"toStartOfInterval(k.open_time, {interval_sql}, 'UTC') AS bucket_open_time" in sql
    assert f"bar_count = {expected_count} AS is_complete" in sql
    assert "argMin(open, k.open_time)" in sql
    assert "argMax(close, k.open_time)" in sql
    assert "FROM xgate.klines_1m AS k FINAL" in sql
    assert "WHERE k.open_time >=" in sql
    assert "  AND k.open_time <" in sql


def test_build_aggregate_query_rejects_unsupported_timeframe():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        build_aggregate_query(
            timeframe="2h",
            start=datetime(2026, 5, 1, tzinfo=timezone.utc),
            end=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )


def test_build_aggregate_query_rejects_naive_datetime():
    with pytest.raises(ValueError, match="timezone-aware"):
        build_aggregate_query(
            timeframe="1h",
            start=datetime(2026, 5, 1),
            end=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )


def test_build_aggregate_query_converts_datetimes_to_utc():
    utc_plus_8 = timezone(timedelta(hours=8))

    sql = build_aggregate_query(
        timeframe="1h",
        start=datetime(2026, 5, 1, 0, 30, tzinfo=utc_plus_8),
        end=datetime(2026, 5, 1, 1, 30, tzinfo=utc_plus_8),
    )

    assert "toDateTime('2026-04-30 16:30:00', 'UTC')" in sql


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (
            datetime(2026, 5, 1, tzinfo=timezone.utc),
            datetime(2026, 5, 1, tzinfo=timezone.utc),
        ),
        (
            datetime(2026, 5, 2, tzinfo=timezone.utc),
            datetime(2026, 5, 1, tzinfo=timezone.utc),
        ),
    ],
)
def test_build_aggregate_query_rejects_non_increasing_ranges(start, end):
    with pytest.raises(ValueError, match="end"):
        build_aggregate_query(timeframe="1h", start=start, end=end)


def test_query_hash_is_stable():
    assert query_hash("select 1") == query_hash("select 1")
    assert query_hash("select 1") != query_hash("select 2")
