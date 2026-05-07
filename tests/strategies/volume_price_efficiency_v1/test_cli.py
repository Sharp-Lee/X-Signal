from __future__ import annotations

from datetime import datetime, timezone
import json

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.cli import main
from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.data import (
    CanonicalOhlcvTable,
    OhlcvArrays,
)
from xsignal.strategies.volume_price_efficiency_v1.features import FeatureArrays


def _arrays() -> OhlcvArrays:
    return OhlcvArrays(
        symbols=("BTCUSDT",),
        open_times=np.array([datetime(2026, 1, 1, tzinfo=timezone.utc)], dtype=object),
        open=np.array([[100.0]]),
        high=np.array([[101.0]]),
        low=np.array([[99.0]]),
        close=np.array([[100.5]]),
        quote_volume=np.array([[1_000_000.0]]),
        quality=np.array([[True]]),
    )


def _features() -> FeatureArrays:
    return FeatureArrays(
        true_range=np.array([[2.0]]),
        atr=np.array([[2.0]]),
        move_unit=np.array([[0.25]]),
        volume_baseline=np.array([[1_000_000.0]]),
        volume_unit=np.array([[1.0]]),
        efficiency=np.array([[0.25]]),
        efficiency_threshold=np.array([[0.2]]),
        close_position=np.array([[0.75]]),
        body_ratio=np.array([[0.25]]),
        signal=np.array([[False]]),
    )


def _event(ret30: float = 0.10, baseline: bool = False) -> dict:
    ret1 = 0.02 if not baseline else 0.01
    net1 = ret1 - 0.002
    return {
        "symbol": "BTCUSDT",
        "signal_open_time": "2026-01-01T00:00:00+00:00",
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
        "forward_return_1": ret1,
        "net_forward_return_1": net1,
        "forward_return_3": ret1,
        "net_forward_return_3": net1,
        "forward_return_6": ret1,
        "net_forward_return_6": net1,
        "forward_return_12": ret1,
        "net_forward_return_12": net1,
        "forward_return_30": ret30,
        "net_forward_return_30": ret30 - 0.002,
    }


def test_cli_run_writes_artifacts_with_injected_pipeline(tmp_path, monkeypatch):
    arrays = _arrays()
    features = _features()

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("4h", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
            (tmp_path / "manifest.json",),
        ),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.prepare_ohlcv_arrays",
        lambda _table: arrays,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.compute_features",
        lambda _arrays, _config: features,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.build_event_rows",
        lambda _arrays, _features, _config: [],
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.build_baseline_events",
        lambda _arrays, _features, _config: [],
    )
    monkeypatch.setattr("xsignal.strategies.volume_price_efficiency_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(["run", "--root", str(tmp_path), "--run-id", "testrun", "--offline"])

    assert exit_code == 0
    assert (
        tmp_path
        / "strategies"
        / "volume_price_efficiency_v1"
        / "runs"
        / "testrun"
        / "manifest.json"
    ).exists()


def test_cli_passes_config_values_to_pipeline(tmp_path, monkeypatch):
    arrays = _arrays()
    captured = []

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("4h", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
            (),
        ),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.prepare_ohlcv_arrays",
        lambda _table: arrays,
    )

    def fake_features(_arrays, config: VolumePriceEfficiencyConfig):
        captured.append(config)
        return _features()

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.compute_features",
        fake_features,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.build_event_rows",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.build_baseline_events",
        lambda *_args, **_kwargs: [],
    )
    monkeypatch.setattr("xsignal.strategies.volume_price_efficiency_v1.cli._git_commit", lambda: "abc123")

    main(
        [
            "run",
            "--root",
            str(tmp_path),
            "--run-id",
            "configrun",
            "--offline",
            "--atr-window",
            "20",
            "--efficiency-percentile",
            "0.95",
            "--min-move-unit",
            "0.8",
            "--fee-bps",
            "2.5",
            "--baseline-seed",
            "99",
        ]
    )

    assert captured[0].atr_window == 20
    assert captured[0].efficiency_percentile == 0.95
    assert captured[0].min_move_unit == 0.8
    assert captured[0].fee_bps == 2.5
    assert captured[0].baseline_seed == 99


def test_cli_refuses_non_offline_mode(tmp_path):
    try:
        main(["run", "--root", str(tmp_path), "--run-id", "online"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("non-offline mode should fail")


def test_cli_scan_writes_research_only_artifacts_with_holdout_metadata(tmp_path, monkeypatch):
    arrays = _arrays()
    captured_holdout_days = []
    captured_configs = []

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.load_offline_ohlcv_table",
        lambda *_args, **_kwargs: (
            CanonicalOhlcvTable("4h", "raw", tmp_path / "manifest.json", tmp_path / "bars.parquet", None),
            (tmp_path / "manifest.json",),
        ),
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.prepare_ohlcv_arrays",
        lambda _table: arrays,
    )

    def fake_split(_arrays, *, holdout_days):
        captured_holdout_days.append(holdout_days)
        return (
            arrays,
            None,
            {
                "holdout_days": holdout_days,
                "research_start": "2020-01-01T00:00:00Z",
                "research_end": "2025-11-08T20:00:00Z",
                "holdout_start": "2025-11-09T00:00:00Z",
                "holdout_end": "2026-05-07T04:00:00Z",
            },
        )

    def fake_features(_arrays, config: VolumePriceEfficiencyConfig):
        captured_configs.append(config)
        return _features()

    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.split_research_and_holdout",
        fake_split,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.compute_features",
        fake_features,
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.build_event_rows",
        lambda _arrays, _features, config: [_event(ret30=config.efficiency_percentile / 10.0)],
    )
    monkeypatch.setattr(
        "xsignal.strategies.volume_price_efficiency_v1.cli.build_baseline_events",
        lambda *_args, **_kwargs: [_event(ret30=0.01, baseline=True)],
    )
    monkeypatch.setattr("xsignal.strategies.volume_price_efficiency_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        [
            "scan",
            "--root",
            str(tmp_path),
            "--scan-id",
            "scanrun",
            "--offline",
            "--holdout-days",
            "7",
            "--efficiency-percentile",
            "0.9,0.95",
            "--min-move-unit",
            "0.5",
            "--min-volume-unit",
            "0.3",
            "--min-close-position",
            "0.7",
            "--min-body-ratio",
            "0.4",
            "--ranking-horizon",
            "30",
            "--top-k",
            "1",
        ]
    )

    scan_dir = tmp_path / "strategies" / "volume_price_efficiency_v1" / "scans" / "scanrun"
    manifest = json.loads((scan_dir / "manifest.json").read_text())
    top_configs = json.loads((scan_dir / "top_configs.json").read_text())
    assert exit_code == 0
    assert captured_holdout_days == [7]
    assert [config.efficiency_percentile for config in captured_configs] == [0.9, 0.95]
    assert manifest["data_split"]["holdout_days"] == 7
    assert manifest["combination_count"] == 2
    assert len(top_configs) == 1
    assert top_configs[0]["efficiency_percentile"] == 0.95
    assert (scan_dir / "summary.csv").exists()
    assert (scan_dir / "bucket_summary.parquet").exists()


def test_cli_scan_refuses_non_offline_mode(tmp_path):
    try:
        main(["scan", "--root", str(tmp_path), "--scan-id", "online"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("non-offline scan should fail")
