from datetime import datetime, timezone

from xsignal.data.query_templates import build_aggregate_query, query_hash


def test_build_aggregate_query_uses_final_and_expected_interval():
    sql = build_aggregate_query(
        timeframe="1h",
        start=datetime(2026, 5, 1, tzinfo=timezone.utc),
        end=datetime(2026, 6, 1, tzinfo=timezone.utc),
    )

    assert "FROM xgate.klines_1m FINAL" in sql
    assert "INTERVAL 1 hour" in sql
    assert "bar_count" in sql
    assert "is_complete" in sql
    assert "2026-05-01 00:00:00" in sql
    assert "2026-06-01 00:00:00" in sql


def test_query_hash_is_stable():
    assert query_hash("select 1") == query_hash("select 1")
    assert query_hash("select 1") != query_hash("select 2")
