from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_diagnostics import (
    build_trailing_bucket_summary_rows,
    build_trailing_time_summary_rows,
    enrich_trades_with_market_context,
    write_trailing_diagnostic_artifacts,
)


def _arrays() -> OhlcvArrays:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    open_times = np.array([start + timedelta(hours=4 * index) for index in range(4)], dtype=object)
    close = np.array(
        [
            [100.0, 10.0],
            [110.0, 12.0],
            [120.0, 15.0],
            [90.0, 18.0],
        ],
        dtype=np.float64,
    )
    return OhlcvArrays(
        symbols=("BTCUSDT", "ETHUSDT"),
        open_times=open_times,
        open=close.copy(),
        high=close.copy() + 1.0,
        low=close.copy() - 1.0,
        close=close,
        quote_volume=np.full(close.shape, 1000.0),
        quality=np.ones(close.shape, dtype=bool),
    )


def _trade(
    *,
    signal_open_time: str = "2026-01-01T08:00:00+00:00",
    symbol: str = "ETHUSDT",
    net_return: float = 0.05,
    move_unit: float = 1.0,
) -> dict:
    return {
        "symbol": symbol,
        "signal_open_time": signal_open_time,
        "net_realized_return": net_return,
        "realized_return": net_return + 0.002,
        "holding_bars": 3,
        "ignored_signal_count": 1,
        "move_unit": move_unit,
        "volume_unit": 1.5,
    }


def test_enrich_trades_with_market_context_adds_causal_lookback_returns():
    rows = enrich_trades_with_market_context(
        [_trade()],
        _arrays(),
        lookback_bars=2,
    )

    assert rows[0]["btc_lookback_return"] == pytest.approx(0.2)
    assert rows[0]["symbol_lookback_return"] == pytest.approx(0.5)
    assert rows[0]["market_lookback_return"] == pytest.approx(0.35)


def test_build_trailing_time_summary_rows_groups_by_year_and_month():
    rows = build_trailing_time_summary_rows(
        [
            _trade(signal_open_time="2026-01-01T00:00:00+00:00", net_return=0.05),
            _trade(signal_open_time="2026-01-02T00:00:00+00:00", net_return=-0.03),
            _trade(signal_open_time="2026-02-01T00:00:00+00:00", net_return=0.01),
        ],
        data_set="research",
    )

    month_rows = [row for row in rows if row["period_type"] == "month"]
    assert month_rows == [
        {
            "data_set": "research",
            "period_type": "month",
            "period": "2026-01",
            "trade_count": 2,
            "win_rate": 0.5,
            "mean_net_realized_return": 0.01,
            "median_net_realized_return": 0.01,
            "average_holding_bars": 3.0,
            "total_ignored_signal_count": 2,
        },
        {
            "data_set": "research",
            "period_type": "month",
            "period": "2026-02",
            "trade_count": 1,
            "win_rate": 1.0,
            "mean_net_realized_return": 0.01,
            "median_net_realized_return": 0.01,
            "average_holding_bars": 3.0,
            "total_ignored_signal_count": 1,
        },
    ]


def test_build_trailing_bucket_summary_rows_groups_by_feature_quantiles():
    rows = build_trailing_bucket_summary_rows(
        [
            _trade(net_return=0.01, move_unit=1.0),
            _trade(net_return=0.03, move_unit=2.0),
            _trade(net_return=-0.01, move_unit=3.0),
            _trade(net_return=0.05, move_unit=4.0),
        ],
        data_set="research",
        feature_names=("move_unit",),
        bucket_count=2,
    )

    assert rows == [
        {
            "data_set": "research",
            "feature_name": "move_unit",
            "bucket_index": 0,
            "bucket_count": 2,
            "lower_bound": 1.0,
            "upper_bound": 2.0,
            "trade_count": 2,
            "win_rate": 1.0,
            "mean_net_realized_return": 0.02,
            "median_net_realized_return": 0.02,
            "average_holding_bars": 3.0,
            "total_ignored_signal_count": 2,
        },
        {
            "data_set": "research",
            "feature_name": "move_unit",
            "bucket_index": 1,
            "bucket_count": 2,
            "lower_bound": 3.0,
            "upper_bound": 4.0,
            "trade_count": 2,
            "win_rate": 0.5,
            "mean_net_realized_return": 0.02,
            "median_net_realized_return": 0.02,
            "average_holding_bars": 3.0,
            "total_ignored_signal_count": 2,
        },
    ]


def test_write_trailing_diagnostic_artifacts(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)

    output = write_trailing_diagnostic_artifacts(
        paths=paths,
        diagnostic_id="diag",
        config=VolumePriceEfficiencyConfig(),
        time_rows=build_trailing_time_summary_rows([_trade()], data_set="research"),
        bucket_rows=build_trailing_bucket_summary_rows(
            [_trade(), _trade(move_unit=2.0)],
            data_set="research",
            feature_names=("move_unit",),
            bucket_count=2,
        ),
        canonical_manifests=["manifest.json"],
        git_commit="abc123",
        runtime_seconds=1.5,
        data_split={"holdout_days": 180},
        atr_multiplier=2.0,
        lookback_bars=30,
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert output == paths.trailing_diagnostic_dir("diag")
    assert manifest["run_type"] == "trailing_stop_diagnostics"
    assert set(manifest["outputs"]) == {"time_summary", "bucket_summary"}
    assert pq.read_table(output / "time_summary.parquet").num_rows == 2
    assert pq.read_table(output / "bucket_summary.parquet").num_rows == 2
