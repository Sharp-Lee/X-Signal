from __future__ import annotations

import json

import pyarrow.parquet as pq

from xsignal.strategies.volume_price_efficiency_v1.artifacts import (
    build_event_study_summary,
    write_scan_artifacts,
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


def test_write_scan_artifacts_creates_manifest_summaries_csv_and_buckets(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)
    config = VolumePriceEfficiencyConfig(horizons=(1,))
    rows = [
        {
            "scan_id": "scan123",
            "config_hash": "hash1",
            "efficiency_percentile": 0.9,
            "min_move_unit": 0.5,
            "min_volume_unit": 0.3,
            "min_close_position": 0.7,
            "min_body_ratio": 0.4,
            "fee_bps": 5.0,
            "slippage_bps": 5.0,
            "baseline_seed": 17,
            "ranking_horizon": 1,
            "event_count": 10,
            "baseline_event_count": 10,
            "symbol_count": 2,
            "h1_mean_return": 0.02,
            "h1_net_mean_return": 0.018,
            "h1_baseline_mean_return": 0.01,
            "h1_event_minus_baseline_mean_return": 0.01,
            "h1_win_rate": 0.6,
            "score": 0.01,
        }
    ]
    top_configs = [rows[0]]
    bucket_rows = [
        {
            "config_hash": "hash1",
            "feature_name": "efficiency",
            "bucket_index": 0,
            "bucket_count": 5,
            "lower_bound": 1.0,
            "upper_bound": 2.0,
            "event_count": 3,
            "h1_mean_return": 0.02,
            "h1_net_mean_return": 0.018,
            "h1_win_rate": 0.6,
        }
    ]

    scan_dir = write_scan_artifacts(
        paths=paths,
        scan_id="scan123",
        base_config=config,
        rows=rows,
        top_configs=top_configs,
        bucket_rows=bucket_rows,
        canonical_manifests=["manifest.json"],
        git_commit="abc123",
        runtime_seconds=2.5,
        symbol_count=2,
        data_split={
            "holdout_days": 180,
            "research_start": "2020-01-01T00:00:00Z",
            "research_end": "2025-11-08T20:00:00Z",
            "holdout_start": "2025-11-09T00:00:00Z",
            "holdout_end": "2026-05-07T04:00:00Z",
        },
    )

    assert scan_dir == paths.scan_dir("scan123")
    manifest = json.loads((scan_dir / "manifest.json").read_text())
    summary = json.loads((scan_dir / "summary.json").read_text())
    top = json.loads((scan_dir / "top_configs.json").read_text())
    assert manifest["strategy_name"] == "volume_price_efficiency_v1"
    assert manifest["scan_id"] == "scan123"
    assert manifest["base_config_hash"] == config.config_hash()
    assert manifest["data_split"]["holdout_days"] == 180
    assert set(manifest["outputs"]) == {
        "summary",
        "summary_csv",
        "top_configs",
        "bucket_summary",
    }
    assert summary["combination_count"] == 1
    assert summary["best_score"] == 0.01
    assert top == top_configs
    assert "hash1" in (scan_dir / "summary.csv").read_text()
    assert pq.read_table(scan_dir / "bucket_summary.parquet").num_rows == 1
