from __future__ import annotations

import json

import numpy as np

from xsignal.strategies.momentum_rotation_v1.cli import main
from xsignal.strategies.momentum_rotation_v1.kernel import BacktestResult
from xsignal.strategies.momentum_rotation_v1.prepare import PreparedArrays, save_prepared_arrays
from xsignal.strategies.momentum_rotation_v1.signals import SignalArrays


def test_cli_run_writes_artifacts_with_injected_pipeline(tmp_path, monkeypatch):
    arrays = PreparedArrays(
        symbols=("BTCUSDT",),
        rebalance_times=np.array(["2026-01-02", "2026-01-03"], dtype=object),
        close_1h=np.array([[100.0], [101.0]]),
        close_4h=np.array([[100.0], [101.0]]),
        close_1d=np.array([[100.0], [101.0]]),
        quote_volume_1d=np.ones((2, 1)),
        complete_1h=np.ones((2, 1), dtype=bool),
        complete_4h=np.ones((2, 1), dtype=bool),
        complete_1d=np.ones((2, 1), dtype=bool),
        quality_1h_24h=np.ones((2, 1), dtype=bool),
        quality_4h_7d=np.ones((2, 1), dtype=bool),
        quality_1d_30d=np.ones((2, 1), dtype=bool),
    )
    signals = SignalArrays(score=np.array([[1.0], [1.0]]), tradable_mask=np.ones((2, 1), dtype=bool))
    result = BacktestResult(
        equity=np.array([1.0, 1.01]),
        period_returns=np.array([0.01]),
        weights=np.array([[1.0], [1.0]]),
        turnover=np.array([1.0, 0.0]),
        costs=np.array([0.0, 0.0]),
    )

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.prepare_from_canonical",
        lambda *_args, **_kwargs: (arrays, ["manifest.json"]),
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.compute_momentum_signals",
        lambda *_args, **_kwargs: signals,
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.run_backtest",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr("xsignal.strategies.momentum_rotation_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(["run", "--root", str(tmp_path), "--run-id", "testrun", "--top-n", "1"])

    assert exit_code == 0
    manifest_path = (
        tmp_path / "strategies" / "momentum_rotation_v1" / "runs" / "testrun" / "manifest.json"
    )
    assert json.loads(manifest_path.read_text())["git_commit"] == "abc123"


def test_cli_run_passes_offline_mode_to_canonical_preparation(tmp_path, monkeypatch):
    arrays = PreparedArrays(
        symbols=("BTCUSDT",),
        rebalance_times=np.array(["2026-01-02", "2026-01-03"], dtype=object),
        close_1h=np.array([[100.0], [101.0]]),
        close_4h=np.array([[100.0], [101.0]]),
        close_1d=np.array([[100.0], [101.0]]),
        quote_volume_1d=np.ones((2, 1)),
        complete_1h=np.ones((2, 1), dtype=bool),
        complete_4h=np.ones((2, 1), dtype=bool),
        complete_1d=np.ones((2, 1), dtype=bool),
        quality_1h_24h=np.ones((2, 1), dtype=bool),
        quality_4h_7d=np.ones((2, 1), dtype=bool),
        quality_1d_30d=np.ones((2, 1), dtype=bool),
    )
    signals = SignalArrays(score=np.array([[1.0], [1.0]]), tradable_mask=np.ones((2, 1), dtype=bool))
    result = BacktestResult(
        equity=np.array([1.0, 1.01]),
        period_returns=np.array([0.01]),
        weights=np.array([[1.0], [1.0]]),
        turnover=np.array([1.0, 0.0]),
        costs=np.array([0.0, 0.0]),
    )
    calls = []

    def fake_prepare(root, config, *, offline, use_cache):
        calls.append((root, config, offline, use_cache))
        return arrays, ["manifest.json"]

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.prepare_from_canonical",
        fake_prepare,
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.compute_momentum_signals",
        lambda *_args, **_kwargs: signals,
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.run_backtest",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr("xsignal.strategies.momentum_rotation_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        ["run", "--root", str(tmp_path), "--run-id", "offline-run", "--top-n", "1", "--offline"]
    )

    assert exit_code == 0
    assert calls[0][0] == tmp_path
    assert calls[0][2] is True
    assert calls[0][3] is True


def test_cli_run_passes_no_cache_to_canonical_preparation(tmp_path, monkeypatch):
    arrays = PreparedArrays(
        symbols=("BTCUSDT",),
        rebalance_times=np.array(["2026-01-02", "2026-01-03"], dtype=object),
        close_1h=np.array([[100.0], [101.0]]),
        close_4h=np.array([[100.0], [101.0]]),
        close_1d=np.array([[100.0], [101.0]]),
        quote_volume_1d=np.ones((2, 1)),
        complete_1h=np.ones((2, 1), dtype=bool),
        complete_4h=np.ones((2, 1), dtype=bool),
        complete_1d=np.ones((2, 1), dtype=bool),
        quality_1h_24h=np.ones((2, 1), dtype=bool),
        quality_4h_7d=np.ones((2, 1), dtype=bool),
        quality_1d_30d=np.ones((2, 1), dtype=bool),
    )
    signals = SignalArrays(score=np.array([[1.0], [1.0]]), tradable_mask=np.ones((2, 1), dtype=bool))
    result = BacktestResult(
        equity=np.array([1.0, 1.01]),
        period_returns=np.array([0.01]),
        weights=np.array([[1.0], [1.0]]),
        turnover=np.array([1.0, 0.0]),
        costs=np.array([0.0, 0.0]),
    )
    calls = []

    def fake_prepare(root, config, *, offline, use_cache):
        calls.append((root, config, offline, use_cache))
        return arrays, ["manifest.json"]

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.prepare_from_canonical",
        fake_prepare,
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.compute_momentum_signals",
        lambda *_args, **_kwargs: signals,
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.run_backtest",
        lambda *_args, **_kwargs: result,
    )
    monkeypatch.setattr("xsignal.strategies.momentum_rotation_v1.cli._git_commit", lambda: "abc123")

    exit_code = main(
        ["run", "--root", str(tmp_path), "--run-id", "nocache-run", "--top-n", "1", "--no-cache"]
    )

    assert exit_code == 0
    assert calls[0][3] is False


def test_cli_prepare_from_canonical_uses_prepared_cache(tmp_path, monkeypatch):
    from xsignal.strategies.momentum_rotation_v1.cli import prepare_from_canonical
    from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
    from xsignal.strategies.momentum_rotation_v1.data import StrategyCanonicalInputs

    arrays = PreparedArrays(
        symbols=("BTCUSDT",),
        rebalance_times=np.array(["2026-01-02", "2026-01-03"], dtype=object),
        close_1h=np.array([[100.0], [101.0]]),
        close_4h=np.array([[100.0], [101.0]]),
        close_1d=np.array([[100.0], [101.0]]),
        quote_volume_1d=np.ones((2, 1)),
        complete_1h=np.ones((2, 1), dtype=bool),
        complete_4h=np.ones((2, 1), dtype=bool),
        complete_1d=np.ones((2, 1), dtype=bool),
        quality_1h_24h=np.ones((2, 1), dtype=bool),
        quality_4h_7d=np.ones((2, 1), dtype=bool),
        quality_1d_30d=np.ones((2, 1), dtype=bool),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps({"row_count": 2}, sort_keys=True))
    cache_dir = (
        tmp_path
        / "strategies"
        / "momentum_rotation_v1"
        / "cache"
        / "prepared"
        / "cached-key"
    )
    save_prepared_arrays(cache_dir, arrays)
    (cache_dir / "cache_manifest.json").write_text(
        json.dumps(
            {
                "cache_key": "cached-key",
                "cache_version": "momentum-rotation-prepared-v2",
                "canonical_manifests": [str(manifest_path)],
            }
        )
    )

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli._prepared_cache_key",
        lambda *_args, **_kwargs: "cached-key",
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.collect_strategy_inputs",
        lambda *_args, **_kwargs: StrategyCanonicalInputs(
            bars_1h=None,
            bars_4h=None,
            bars_1d=None,
            manifest_paths=(manifest_path,),
        ),
    )
    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.prepare_daily_arrays",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("cache miss")),
    )

    loaded, manifests = prepare_from_canonical(tmp_path, MomentumRotationConfig())

    assert loaded.symbols == arrays.symbols
    assert manifests == [str(manifest_path)]


def test_cli_prepare_from_canonical_can_disable_prepared_cache(tmp_path, monkeypatch):
    from xsignal.strategies.momentum_rotation_v1.cli import prepare_from_canonical
    from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig
    from xsignal.strategies.momentum_rotation_v1.data import StrategyCanonicalInputs

    arrays = PreparedArrays(
        symbols=("BTCUSDT",),
        rebalance_times=np.array(["2026-01-02"], dtype=object),
        close_1h=np.array([[100.0]]),
        close_4h=np.array([[100.0]]),
        close_1d=np.array([[100.0]]),
        quote_volume_1d=np.ones((1, 1)),
        complete_1h=np.ones((1, 1), dtype=bool),
        complete_4h=np.ones((1, 1), dtype=bool),
        complete_1d=np.ones((1, 1), dtype=bool),
        quality_1h_24h=np.ones((1, 1), dtype=bool),
        quality_4h_7d=np.ones((1, 1), dtype=bool),
        quality_1d_30d=np.ones((1, 1), dtype=bool),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}")

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.collect_strategy_inputs",
        lambda *_args, **_kwargs: StrategyCanonicalInputs(
            bars_1h=object(),
            bars_4h=object(),
            bars_1d=object(),
            manifest_paths=(manifest_path,),
        ),
    )
    calls = []

    def fake_prepare_daily_arrays(**_kwargs):
        calls.append(True)
        return arrays

    monkeypatch.setattr(
        "xsignal.strategies.momentum_rotation_v1.cli.prepare_daily_arrays",
        fake_prepare_daily_arrays,
    )

    loaded, manifests = prepare_from_canonical(
        tmp_path,
        MomentumRotationConfig(),
        use_cache=False,
    )

    assert loaded is arrays
    assert manifests == [str(manifest_path)]
    assert calls == [True]
