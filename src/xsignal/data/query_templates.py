from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from xsignal.data.canonical_bars import expected_1m_count, validate_timeframe


CLICKHOUSE_SOURCE_TABLE = "xgate.klines_1m"


def _format_clickhouse_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        raise ValueError("datetime must be timezone-aware")
    utc_value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return utc_value.strftime("%Y-%m-%d %H:%M:%S")


def _interval_sql(timeframe: str) -> str:
    validate_timeframe(timeframe)
    if timeframe == "1h":
        return "INTERVAL 1 hour"
    if timeframe == "4h":
        return "INTERVAL 4 hour"
    return "INTERVAL 1 day"


def build_aggregate_query(timeframe: str, start: datetime, end: datetime) -> str:
    interval = _interval_sql(timeframe)
    expected_count = expected_1m_count(timeframe)
    start_sql = _format_clickhouse_datetime(start)
    end_sql = _format_clickhouse_datetime(end)
    return f"""
SELECT
    symbol,
    toStartOfInterval(open_time, {interval}, 'UTC') AS open_time,
    toFloat64(argMin(open, open_time)) AS open,
    toFloat64(max(high)) AS high,
    toFloat64(min(low)) AS low,
    toFloat64(argMax(close, open_time)) AS close,
    toFloat64(sum(volume)) AS volume,
    toFloat64(sum(quote_volume)) AS quote_volume,
    toUInt64(sum(trade_count)) AS trade_count,
    toFloat64(sum(taker_buy_volume)) AS taker_buy_volume,
    toFloat64(sum(taker_buy_quote_volume)) AS taker_buy_quote_volume,
    toUInt16(count()) AS bar_count,
    bar_count = {expected_count} AS is_complete
FROM {CLICKHOUSE_SOURCE_TABLE} FINAL
WHERE open_time >= toDateTime('{start_sql}', 'UTC')
  AND open_time < toDateTime('{end_sql}', 'UTC')
GROUP BY
    symbol,
    open_time
ORDER BY
    open_time,
    symbol
""".strip()


def query_hash(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()
