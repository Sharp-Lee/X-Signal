from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from xsignal.data.canonical_bars import expected_1m_count, timeframe_spec, validate_fill_policy


CLICKHOUSE_SOURCE_TABLE = "xgate.klines_1m"


def _normalize_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _format_clickhouse_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _interval_sql(timeframe: str) -> str:
    return timeframe_spec(timeframe).clickhouse_interval


def build_aggregate_query(timeframe: str, start: datetime, end: datetime, fill_policy: str = "raw") -> str:
    fill_policy = validate_fill_policy(fill_policy)
    if fill_policy == "prev_close_zero_volume":
        return build_filled_aggregate_query(timeframe, start, end)

    start_utc = _normalize_utc_datetime(start)
    end_utc = _normalize_utc_datetime(end)
    if end_utc <= start_utc:
        raise ValueError("end must be after start")

    interval = _interval_sql(timeframe)
    expected_count = expected_1m_count(timeframe)
    start_sql = _format_clickhouse_datetime(start_utc)
    end_sql = _format_clickhouse_datetime(end_utc)
    return f"""
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
    toUInt16({expected_count}) AS expected_1m_count,
    is_complete,
    toUInt8(0) AS has_synthetic,
    'raw' AS fill_policy
FROM
(
    SELECT
        symbol,
        toStartOfInterval(k.open_time, {interval}, 'UTC') AS bucket_open_time,
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
        bar_count = {expected_count} AS is_complete
    FROM {CLICKHOUSE_SOURCE_TABLE} AS k FINAL
    WHERE k.open_time >= toDateTime('{start_sql}', 'UTC')
      AND k.open_time < toDateTime('{end_sql}', 'UTC')
    GROUP BY
        symbol,
        bucket_open_time
)
ORDER BY
    bucket_open_time,
    symbol
""".strip()


def build_filled_aggregate_query(timeframe: str, start: datetime, end: datetime) -> str:
    start_utc = _normalize_utc_datetime(start)
    end_utc = _normalize_utc_datetime(end)
    if end_utc <= start_utc:
        raise ValueError("end must be after start")

    interval = _interval_sql(timeframe)
    expected_count = expected_1m_count(timeframe)
    start_sql = _format_clickhouse_datetime(start_utc)
    end_sql = _format_clickhouse_datetime(end_utc)
    return f"""
WITH
    toDateTime('{start_sql}', 'UTC') AS start_time,
    toDateTime('{end_sql}', 'UTC') AS end_time,
    {expected_count} AS expected_count,
    raw_1m AS
    (
        SELECT
            symbol,
            open_time,
            open,
            high,
            low,
            close,
            volume,
            quote_volume,
            trade_count,
            taker_buy_volume,
            taker_buy_quote_volume,
            toUInt8(1) AS is_real_row
        FROM {CLICKHOUSE_SOURCE_TABLE} AS k FINAL
        WHERE k.open_time >= start_time
          AND k.open_time < end_time
    ),
    symbol_bounds AS
    (
        SELECT
            symbol,
            min(open_time) AS first_open_time,
            max(open_time) AS last_open_time
        FROM raw_1m
        GROUP BY symbol
    ),
    minute_grid AS
    (
        SELECT
            symbol,
            addMinutes(first_open_time, toInt64(minute_offset)) AS minute_open_time
        FROM symbol_bounds
        ARRAY JOIN range(toUInt64(dateDiff('minute', first_open_time, last_open_time) + 1)) AS minute_offset
    )
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
FROM
(
    SELECT
        symbol,
        minute_open_time,
        toStartOfInterval(minute_open_time, {interval}, 'UTC') AS bucket_open_time,
        if(is_real_1m, open, previous_real_close) AS open,
        if(is_real_1m, high, previous_real_close) AS high,
        if(is_real_1m, low, previous_real_close) AS low,
        if(is_real_1m, close, previous_real_close) AS close,
        if(is_real_1m, volume, 0) AS volume,
        if(is_real_1m, quote_volume, 0) AS quote_volume,
        if(is_real_1m, trade_count, 0) AS trade_count,
        if(is_real_1m, taker_buy_volume, 0) AS taker_buy_volume,
        if(is_real_1m, taker_buy_quote_volume, 0) AS taker_buy_quote_volume,
        is_real_1m,
        NOT is_real_1m AS is_synthetic_1m
    FROM
    (
        SELECT
            g.symbol,
            g.minute_open_time,
            r.open,
            r.high,
            r.low,
            r.close,
            r.volume,
            r.quote_volume,
            r.trade_count,
            r.taker_buy_volume,
            r.taker_buy_quote_volume,
            r.is_real_row = 1 AS is_real_1m,
            anyLast(if(r.is_real_row = 1, r.close, NULL)) OVER (
                PARTITION BY g.symbol
                ORDER BY g.minute_open_time
                ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
            ) AS previous_real_close
        FROM minute_grid AS g
        LEFT JOIN raw_1m AS r
            ON g.symbol = r.symbol
           AND g.minute_open_time = r.open_time
    )
    WHERE previous_real_close IS NOT NULL
)
GROUP BY
    symbol,
    bucket_open_time
ORDER BY
    bucket_open_time,
    symbol
""".strip()


def query_hash(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()
