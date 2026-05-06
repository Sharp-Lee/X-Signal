from __future__ import annotations

import hashlib
from datetime import datetime, timezone

from xsignal.data.canonical_bars import expected_1m_count, validate_timeframe


CLICKHOUSE_SOURCE_TABLE = "xgate.klines_1m"


def _normalize_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value.astimezone(timezone.utc).replace(tzinfo=None)


def _format_clickhouse_datetime(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _interval_sql(timeframe: str) -> str:
    validate_timeframe(timeframe)
    if timeframe == "1h":
        return "INTERVAL 1 hour"
    if timeframe == "4h":
        return "INTERVAL 4 hour"
    return "INTERVAL 1 day"


def build_aggregate_query(timeframe: str, start: datetime, end: datetime) -> str:
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
    toStartOfInterval(k.open_time, {interval}, 'UTC') AS open_time,
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
    open_time
ORDER BY
    open_time,
    symbol
""".strip()


def query_hash(sql: str) -> str:
    return hashlib.sha256(sql.encode("utf-8")).hexdigest()
