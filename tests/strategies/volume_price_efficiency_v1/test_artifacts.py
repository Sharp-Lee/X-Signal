from __future__ import annotations

import json

import pyarrow.parquet as pq

from xsignal.strategies.volume_price_efficiency_v1.artifacts import (
    build_event_study_summary,
    write_run_artifacts,
)
from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)


def _event(symbol="BTCUSDT", signal_time="2026-01-01T00:00:00+00:00", ret=0.1, net=0.098):
    return {
        "symbol": symbol,
        "signal_open_time": signal_time,
        "decision_time": "2026-01-01T04:00:00+00:00",
        "entry_open_time": "2026-01-01T04:00:00+00:00",
        "entry_price": 100.0,
        "move_unit": 1.0,
        "volume_unit": 1.0,
        "efficiency": 1.0,
        "efficiency_threshold": 0.5,
        "close_position": 0.8,
        "body_ratio": 0.5,
        "quote_volume": 1_000_000.0,
        "volume_baseline": 900_000.0,
        "atr": 10.0,
        "forward_return_1": ret,
        "net_forward_return_1": net,
    }


def test_build_event_study_summary_reports_gross_net_and_baseline_metrics():
    events = [
        _event(symbol="BTCUSDT", signal_time="2026-01-01T00:00:00+00:00", ret=0.1, net=0.098),
        _event(symbol="ETHUSDT", signal_time="2026-01-02T00:00:00+00:00", ret=-0.02, net=-0.022),
    ]
    baseline_events = [
        _event(symbol="BTCUSDT", ret=0.03, net=0.028),
        _event(symbol="ETHUSDT", ret=0.01, net=0.008),
    ]

    summary = build_event_study_summary(events, baseline_events, horizons=(1,))

    assert summary["event_count"] == 2
    assert summary["baseline_event_count"] == 2
    assert summary["symbol_count"] == 2
    assert summary["first_signal_time"] == "2026-01-01T00:00:00+00:00"
    assert summary["last_signal_time"] == "2026-01-02T00:00:00+00:00"
    horizon = summary["horizons"]["1"]
    assert horizon["available_event_count"] == 2
    assert horizon["mean_return"] == 0.04
    assert horizon["median_return"] == 0.04
    assert horizon["win_rate"] == 0.5
    assert horizon["net_mean_return"] == 0.038
    assert horizon["net_median_return"] == 0.038
    assert horizon["net_win_rate"] == 0.5
    assert horizon["baseline_mean_return"] == 0.02
    assert horizon["baseline_median_return"] == 0.02
    assert horizon["baseline_win_rate"] == 1.0
    assert horizon["event_minus_baseline_mean_return"] == 0.02
    assert horizon["event_minus_baseline_median_return"] == 0.02
    assert summary["top_symbols_by_event_count"][0]["symbol"] == "BTCUSDT"
    assert summary["top_symbols_by_average_return"][0]["symbol"] == "BTCUSDT"


def test_build_event_study_summary_handles_empty_returns():
    event = _event()
    event["forward_return_1"] = None
    event["net_forward_return_1"] = None

    summary = build_event_study_summary([event], [], horizons=(1,))

    assert summary["horizons"]["1"]["available_event_count"] == 0
    assert summary["horizons"]["1"]["mean_return"] is None
    assert summary["horizons"]["1"]["baseline_mean_return"] is None
    assert summary["horizons"]["1"]["event_minus_baseline_mean_return"] is None


def test_write_run_artifacts_creates_manifest_summary_and_parquet(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)
    config = VolumePriceEfficiencyConfig(horizons=(1,))
    events = [_event()]
    baseline_events = [_event(symbol="ETHUSDT")]

    run_dir = write_run_artifacts(
        paths=paths,
        run_id="testrun",
        config=config,
        events=events,
        baseline_events=baseline_events,
        symbols=("BTCUSDT", "ETHUSDT"),
        canonical_manifests=["manifest.json"],
        git_commit="abc123",
        runtime_seconds=1.5,
    )

    assert run_dir == paths.run_dir("testrun")
    manifest = json.loads((run_dir / "manifest.json").read_text())
    summary = json.loads((run_dir / "summary.json").read_text())
    assert manifest["strategy_name"] == "volume_price_efficiency_v1"
    assert manifest["git_commit"] == "abc123"
    assert manifest["config_hash"] == config.config_hash()
    assert manifest["canonical_manifests"] == ["manifest.json"]
    assert manifest["symbol_count"] == 2
    assert set(manifest["outputs"]) == {"summary", "events", "baseline_events"}
    assert summary["event_count"] == 1
    assert pq.read_table(run_dir / "events.parquet").num_rows == 1
    assert pq.read_table(run_dir / "baseline_events.parquet").num_rows == 1


def test_write_run_artifacts_writes_empty_event_tables(tmp_path):
    run_dir = write_run_artifacts(
        paths=VolumePriceEfficiencyPaths(root=tmp_path),
        run_id="emptyrun",
        config=VolumePriceEfficiencyConfig(horizons=(1,)),
        events=[],
        baseline_events=[],
        symbols=(),
        canonical_manifests=[],
        git_commit="abc123",
        runtime_seconds=0.1,
    )

    assert pq.read_table(run_dir / "events.parquet").num_rows == 0
    assert pq.read_table(run_dir / "baseline_events.parquet").num_rows == 0
