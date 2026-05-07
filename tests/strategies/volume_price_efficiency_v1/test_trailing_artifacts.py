from __future__ import annotations

import json

import numpy as np
import pyarrow.parquet as pq
import pytest

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing import (
    TrailingStopResult,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_artifacts import (
    build_trailing_summary,
    write_trailing_run_artifacts,
)


def _result() -> TrailingStopResult:
    return TrailingStopResult(
        trades=[
            {
                "symbol": "BTCUSDT",
                "signal_open_time": "2026-01-01T00:00:00+00:00",
                "decision_time": "2026-01-01T04:00:00+00:00",
                "entry_open_time": "2026-01-01T04:00:00+00:00",
                "exit_time": "2026-01-01T08:00:00+00:00",
                "entry_price": 100.0,
                "exit_price": 108.0,
                "stop_price_at_exit": 108.0,
                "atr_at_entry": 5.0,
                "atr_at_exit": 1.0,
                "highest_high": 110.0,
                "realized_return": 0.08,
                "net_realized_return": 0.078,
                "holding_bars": 2,
                "ignored_signal_count": 1,
            },
            {
                "symbol": "ETHUSDT",
                "signal_open_time": "2026-01-02T00:00:00+00:00",
                "decision_time": "2026-01-02T04:00:00+00:00",
                "entry_open_time": "2026-01-02T04:00:00+00:00",
                "exit_time": "2026-01-02T08:00:00+00:00",
                "entry_price": 100.0,
                "exit_price": 95.0,
                "stop_price_at_exit": 95.0,
                "atr_at_entry": 5.0,
                "atr_at_exit": 5.0,
                "highest_high": 105.0,
                "realized_return": -0.05,
                "net_realized_return": -0.052,
                "holding_bars": 2,
                "ignored_signal_count": 0,
            },
        ],
        equity=np.array([1.0, 1.08, 1.026], dtype=np.float64),
        period_returns=np.array([0.08, -0.05], dtype=np.float64),
        positions=np.array([[False, False], [True, False], [False, True]], dtype=bool),
        stop_prices=np.array([[np.nan, np.nan], [90.0, np.nan], [np.nan, 95.0]]),
    )


def test_build_trailing_summary_reports_trade_metrics():
    summary = build_trailing_summary(_result())

    assert summary["trade_count"] == 2
    assert summary["winning_trade_count"] == 1
    assert summary["win_rate"] == 0.5
    assert summary["mean_realized_return"] == pytest.approx(0.015)
    assert summary["mean_net_realized_return"] == pytest.approx(0.013)
    assert summary["final_equity"] == pytest.approx(1.026)
    assert summary["total_return"] == pytest.approx(0.026)
    assert summary["max_drawdown"] == pytest.approx(0.05)
    assert summary["total_ignored_signal_count"] == 1
    assert summary["average_holding_bars"] == pytest.approx(2.0)


def test_write_trailing_run_artifacts_creates_manifest_summary_and_tables(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)
    config = VolumePriceEfficiencyConfig(fee_bps=5, slippage_bps=5)
    result = _result()

    run_dir = write_trailing_run_artifacts(
        paths=paths,
        run_id="trailrun",
        config=config,
        result=result,
        symbols=("BTCUSDT", "ETHUSDT"),
        open_times=np.array(
            [
                "2026-01-01T00:00:00+00:00",
                "2026-01-01T04:00:00+00:00",
                "2026-01-01T08:00:00+00:00",
            ],
            dtype=object,
        ),
        canonical_manifests=["manifest.json"],
        git_commit="abc123",
        runtime_seconds=1.5,
        data_split={"holdout_days": 180},
        atr_multiplier=2.0,
    )

    assert run_dir == paths.trailing_run_dir("trailrun")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert manifest["strategy_name"] == "volume_price_efficiency_v1"
    assert manifest["run_type"] == "trailing_stop_holdout"
    assert manifest["config_hash"] == config.config_hash()
    assert manifest["canonical_manifests"] == ["manifest.json"]
    assert manifest["data_split"]["holdout_days"] == 180
    assert manifest["atr_multiplier"] == 2.0
    assert set(manifest["outputs"]) == {
        "summary",
        "trades",
        "equity_curve",
        "daily_positions",
    }
    assert summary["trade_count"] == 2

    trades = pq.read_table(run_dir / "trades.parquet")
    equity = pq.read_table(run_dir / "equity_curve.parquet")
    positions = pq.read_table(run_dir / "daily_positions.parquet")
    assert trades.num_rows == 2
    assert set(
        [
            "symbol",
            "signal_open_time",
            "entry_open_time",
            "exit_time",
            "exit_price",
            "stop_price_at_exit",
            "realized_return",
            "net_realized_return",
        ]
    ).issubset(set(trades.column_names))
    assert equity.num_rows == 3
    assert positions.num_rows == 2


def test_write_trailing_run_artifacts_writes_empty_trade_table(tmp_path):
    result = TrailingStopResult(
        trades=[],
        equity=np.array([1.0], dtype=np.float64),
        period_returns=np.array([], dtype=np.float64),
        positions=np.zeros((1, 1), dtype=bool),
        stop_prices=np.full((1, 1), np.nan),
    )

    run_dir = write_trailing_run_artifacts(
        paths=VolumePriceEfficiencyPaths(root=tmp_path),
        run_id="empty",
        config=VolumePriceEfficiencyConfig(),
        result=result,
        symbols=("BTCUSDT",),
        open_times=np.array(["2026-01-01T00:00:00+00:00"], dtype=object),
        canonical_manifests=[],
        git_commit="abc123",
        runtime_seconds=0.1,
        data_split={"holdout_days": 180},
        atr_multiplier=2.0,
    )

    assert pq.read_table(run_dir / "trades.parquet").num_rows == 0
    assert pq.read_table(run_dir / "daily_positions.parquet").num_rows == 0
