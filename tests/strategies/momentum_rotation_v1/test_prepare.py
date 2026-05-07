from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import numpy as np
import pyarrow as pa

from xsignal.strategies.momentum_rotation_v1.data import CanonicalBarTable
from xsignal.strategies.momentum_rotation_v1.prepare import (
    load_prepared_arrays,
    prepare_daily_arrays,
    save_prepared_arrays,
)


def table_for(timeframe: str, rows: list[dict]) -> CanonicalBarTable:
    return CanonicalBarTable(
        timeframe=timeframe,
        fill_policy="raw",
        manifest_path=__file__,
        parquet_path=__file__,
        table=pa.Table.from_pylist(rows),
    )


def rows(
    symbols: list[str],
    opens: list[datetime],
    multiplier: float,
    expected_count: int,
) -> list[dict]:
    output = []
    for symbol_index, symbol in enumerate(symbols):
        for time_index, open_time in enumerate(opens):
            output.append(
                {
                    "symbol": symbol,
                    "open_time": open_time,
                    "close": multiplier + symbol_index * 100.0 + time_index,
                    "quote_volume": 1_000_000.0 + time_index,
                    "bar_count": expected_count,
                    "expected_1m_count": expected_count,
                    "is_complete": True,
                    "has_synthetic": False,
                    "fill_policy": "raw",
                }
            )
    return output


def test_prepare_daily_arrays_aligns_completed_intraday_bars_to_daily_close():
    symbols = ["BTCUSDT", "ETHUSDT"]
    day0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    daily_opens = [day0 + timedelta(days=i) for i in range(3)]
    hourly_opens = [day0 + timedelta(hours=i) for i in range(72)]
    four_hour_opens = [day0 + timedelta(hours=4 * i) for i in range(18)]

    prepared = prepare_daily_arrays(
        bars_1h=table_for("1h", rows(symbols, hourly_opens, 10.0, 60)),
        bars_4h=table_for("4h", rows(symbols, four_hour_opens, 20.0, 240)),
        bars_1d=table_for("1d", rows(symbols, daily_opens, 30.0, 1440)),
    )

    assert prepared.symbols == tuple(symbols)
    assert prepared.rebalance_times.tolist() == [
        day0 + timedelta(days=1),
        day0 + timedelta(days=2),
        day0 + timedelta(days=3),
    ]
    assert prepared.close_1h.shape == (3, 2)
    assert prepared.close_4h.shape == (3, 2)
    assert prepared.close_1d.shape == (3, 2)
    assert prepared.close_1h[0, 0] == 10.0 + 23
    assert prepared.close_4h[0, 0] == 20.0 + 5
    assert prepared.close_1d[0, 0] == 30.0
    assert prepared.complete_1h[0, 0]
    assert prepared.complete_4h[0, 0]
    assert prepared.complete_1d[0, 0]
    assert prepared.quality_1h_24h[1, 0]


def test_prepare_daily_arrays_marks_missing_intraday_close_incomplete():
    symbols = ["BTCUSDT"]
    day0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    daily_opens = [day0, day0 + timedelta(days=1)]
    hourly_opens = [day0 + timedelta(hours=i) for i in range(47)]
    four_hour_opens = [day0 + timedelta(hours=4 * i) for i in range(12)]

    prepared = prepare_daily_arrays(
        bars_1h=table_for("1h", rows(symbols, hourly_opens, 10.0, 60)),
        bars_4h=table_for("4h", rows(symbols, four_hour_opens, 20.0, 240)),
        bars_1d=table_for("1d", rows(symbols, daily_opens, 30.0, 1440)),
    )

    assert prepared.complete_1h[0, 0]
    assert not prepared.complete_1h[1, 0]
    assert not prepared.quality_1h_24h[1, 0]
    assert np.isnan(prepared.close_1h[1, 0])


def test_prepare_daily_arrays_preserves_incomplete_daily_price_for_pnl():
    symbols = ["BTCUSDT"]
    day0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    daily_rows = rows(
        symbols,
        [day0, day0 + timedelta(days=1)],
        30.0,
        1440,
    )
    daily_rows[1]["bar_count"] = 1439
    daily_rows[1]["is_complete"] = False
    hourly_opens = [day0 + timedelta(hours=i) for i in range(48)]
    four_hour_opens = [day0 + timedelta(hours=4 * i) for i in range(12)]

    prepared = prepare_daily_arrays(
        bars_1h=table_for("1h", rows(symbols, hourly_opens, 10.0, 60)),
        bars_4h=table_for("4h", rows(symbols, four_hour_opens, 20.0, 240)),
        bars_1d=table_for("1d", daily_rows),
    )

    assert prepared.close_1d[1, 0] == 31.0
    assert not prepared.complete_1d[1, 0]
    assert not prepared.quality_1d_30d[1, 0]


def test_prepare_daily_arrays_forward_fills_missing_daily_price_for_pnl_only():
    day0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    daily_rows = rows(["BTCUSDT"], [day0], 30.0, 1440)
    daily_rows.extend(rows(["ETHUSDT"], [day0, day0 + timedelta(days=1)], 50.0, 1440))
    hourly_opens = [day0 + timedelta(hours=i) for i in range(48)]
    four_hour_opens = [day0 + timedelta(hours=4 * i) for i in range(12)]

    prepared = prepare_daily_arrays(
        bars_1h=table_for("1h", rows(["BTCUSDT", "ETHUSDT"], hourly_opens, 10.0, 60)),
        bars_4h=table_for("4h", rows(["BTCUSDT", "ETHUSDT"], four_hour_opens, 20.0, 240)),
        bars_1d=table_for("1d", daily_rows),
    )

    btc_index = prepared.symbols.index("BTCUSDT")
    assert prepared.rebalance_times.tolist() == [day0 + timedelta(days=1), day0 + timedelta(days=2)]
    assert prepared.close_1d[1, btc_index] == 30.0
    assert not prepared.complete_1d[1, btc_index]
    assert not prepared.quality_1d_30d[1, btc_index]


def test_save_and_load_prepared_arrays_round_trip(tmp_path):
    prepared = prepare_daily_arrays(
        bars_1h=table_for(
            "1h",
            rows(
                ["BTCUSDT"],
                [datetime(2026, 1, 1, hour=i, tzinfo=timezone.utc) for i in range(24)],
                10.0,
                60,
            ),
        ),
        bars_4h=table_for(
            "4h",
            rows(
                ["BTCUSDT"],
                [datetime(2026, 1, 1, hour=4 * i, tzinfo=timezone.utc) for i in range(6)],
                20.0,
                240,
            ),
        ),
        bars_1d=table_for(
            "1d",
            rows(["BTCUSDT"], [datetime(2026, 1, 1, tzinfo=timezone.utc)], 30.0, 1440),
        ),
    )

    save_prepared_arrays(tmp_path, prepared)
    loaded = load_prepared_arrays(tmp_path)

    assert json.loads((tmp_path / "symbols.json").read_text()) == ["BTCUSDT"]
    assert loaded.symbols == prepared.symbols
    assert loaded.rebalance_times.tolist() == prepared.rebalance_times.tolist()
    assert np.array_equal(loaded.close_1d, prepared.close_1d, equal_nan=True)
