from datetime import datetime, timedelta, timezone

import pytest

from xsignal.data.query_templates import build_aggregate_query, query_hash


def test_build_aggregate_query_uses_final_and_expected_interval():
    sql = build_aggregate_query(
        timeframe="1h",
        start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end=datetime(2026, 6, 1, tzinfo=timezone.utc),
        fill_policy="raw",
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
    toUInt16(0) AS synthetic_1m_count,
    toUInt16(60) AS expected_1m_count,
    is_complete,
    toUInt8(0) AS has_synthetic,
    'raw' AS fill_policy
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
        fill_policy="raw",
    )

    assert "AS bucket_open_time" in sql
    assert "bucket_open_time AS open_time" in sql
    assert "GROUP BY\n        symbol,\n        bucket_open_time" in sql
    assert "ORDER BY\n    bucket_open_time,\n    symbol" in sql
    assert "GROUP BY\n    symbol,\n    open_time" not in sql


@pytest.mark.parametrize(
    ("timeframe", "interval_sql", "expected_count"),
    [
        ("1m", "INTERVAL 1 minute", 1),
        ("3m", "INTERVAL 3 minute", 3),
        ("5m", "INTERVAL 5 minute", 5),
        ("15m", "INTERVAL 15 minute", 15),
        ("30m", "INTERVAL 30 minute", 30),
        ("1h", "INTERVAL 1 hour", 60),
        ("2h", "INTERVAL 2 hour", 120),
        ("4h", "INTERVAL 4 hour", 240),
        ("6h", "INTERVAL 6 hour", 360),
        ("8h", "INTERVAL 8 hour", 480),
        ("12h", "INTERVAL 12 hour", 720),
        ("1d", "INTERVAL 1 day", 1440),
    ],
)
def test_build_aggregate_query_uses_supported_intervals(timeframe, interval_sql, expected_count):
    sql = build_aggregate_query(
        timeframe=timeframe,
        start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end=datetime(2026, 6, 1, tzinfo=timezone.utc),
        fill_policy="raw",
    )

    assert f"toStartOfInterval(k.open_time, {interval_sql}, 'UTC') AS bucket_open_time" in sql
    assert f"bar_count = {expected_count} AS is_complete" in sql
    assert f"toUInt16({expected_count}) AS expected_1m_count" in sql
    assert "toUInt16(0) AS synthetic_1m_count" in sql
    assert "toUInt8(0) AS has_synthetic" in sql
    assert "'raw' AS fill_policy" in sql
    assert "argMin(open, k.open_time)" in sql
    assert "argMax(close, k.open_time)" in sql
    assert "FROM xgate.klines_1m AS k FINAL" in sql
    assert "WHERE k.open_time >=" in sql
    assert "  AND k.open_time <" in sql


def test_build_aggregate_query_rejects_unsupported_timeframe():
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        build_aggregate_query(
            timeframe="1w",
            start=datetime(2026, 5, 1, tzinfo=timezone.utc),
            end=datetime(2026, 6, 1, tzinfo=timezone.utc),
        )


def test_build_aggregate_query_rejects_unsupported_fill_policy():
    with pytest.raises(ValueError, match="Unsupported fill_policy"):
        build_aggregate_query(
            timeframe="1h",
            start=datetime(2026, 5, 1, tzinfo=timezone.utc),
            end=datetime(2026, 6, 1, tzinfo=timezone.utc),
            fill_policy="forward_volume",
        )


def test_build_aggregate_query_builds_prev_close_zero_volume_query_shape():
    sql = build_aggregate_query(
        timeframe="1h",
        start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end=datetime(2026, 6, 1, tzinfo=timezone.utc),
        fill_policy="prev_close_zero_volume",
    )

    assert """
SELECT
    symbol,
    bucket_open_time AS open_time,
    toFloat64(argMin(open, minute_open_time)) AS open,
    toFloat64(max(high)) AS high,
    toFloat64(min(low)) AS low,
    toFloat64(argMax(close, minute_open_time)) AS close,
    toFloat64(sum(volume)) AS volume,
    toFloat64(sum(quote_volume)) AS quote_volume,
    toUInt64(sum(trade_count)) AS trade_count,
    toFloat64(sum(taker_buy_volume)) AS taker_buy_volume,
    toFloat64(sum(taker_buy_quote_volume)) AS taker_buy_quote_volume,
    toUInt16(sum(is_real_1m)) AS bar_count,
    toUInt16(sum(is_synthetic_1m)) AS synthetic_1m_count,
    toUInt16(expected_count) AS expected_1m_count,
    bar_count + synthetic_1m_count = expected_count AS is_complete,
    toUInt8(synthetic_1m_count > 0) AS has_synthetic,
    'prev_close_zero_volume' AS fill_policy
""".strip() in sql
    for fragment in [
        "raw_1m AS",
        "symbol_bounds AS",
        "minute_grid AS",
        "min(open_time) AS first_open_time",
        "max(open_time) AS last_open_time",
        "ARRAY JOIN range(toUInt64(dateDiff('minute', first_open_time, last_open_time) + 1))",
        "addMinutes(first_open_time, toInt64(minute_offset)) AS minute_open_time",
        "previous_real_close",
        "toUInt8(1) AS is_real_row",
        "r.is_real_row = 1 AS is_real_1m",
        "anyLast(if(r.is_real_row = 1, r.close, NULL)) OVER",
        "toUInt16(sum(is_real_1m)) AS bar_count",
        "toUInt16(sum(is_synthetic_1m)) AS synthetic_1m_count",
        "60 AS expected_count",
        "toUInt16(expected_count) AS expected_1m_count",
        "toUInt8(synthetic_1m_count > 0) AS has_synthetic",
        "'prev_close_zero_volume' AS fill_policy",
        "FROM xgate.klines_1m AS k FINAL",
        "WHERE previous_real_close IS NOT NULL",
    ]:
        assert fragment in sql
    assert "r.close IS NOT NULL AS is_real_1m" not in sql
    assert "numbers(dateDiff('minute', start_time, end_time))" not in sql


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


def test_build_aggregate_query_filled_policy_converts_datetimes_to_utc():
    utc_plus_8 = timezone(timedelta(hours=8))

    sql = build_aggregate_query(
        timeframe="1h",
        start=datetime(2026, 5, 1, 0, 30, tzinfo=utc_plus_8),
        end=datetime(2026, 5, 1, 1, 30, tzinfo=utc_plus_8),
        fill_policy="prev_close_zero_volume",
    )

    assert "toDateTime('2026-04-30 16:30:00', 'UTC') AS start_time" in sql
    assert "toDateTime('2026-04-30 17:30:00', 'UTC') AS end_time" in sql


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
def test_build_aggregate_query_filled_policy_rejects_non_increasing_ranges(start, end):
    with pytest.raises(ValueError, match="end"):
        build_aggregate_query(
            timeframe="1h",
            start=start,
            end=end,
            fill_policy="prev_close_zero_volume",
        )


def test_query_hash_is_stable():
    assert query_hash("select 1") == query_hash("select 1")
    assert query_hash("select 1") != query_hash("select 2")
