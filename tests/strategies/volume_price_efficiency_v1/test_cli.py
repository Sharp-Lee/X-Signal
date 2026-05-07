from __future__ import annotations

from datetime import datetime, timezone

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
