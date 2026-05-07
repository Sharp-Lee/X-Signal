from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pyarrow.parquet as pq
import pytest

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_walk_forward import (
    build_walk_forward_fold_row,
    build_walk_forward_folds,
    write_trailing_walk_forward_artifacts,
)


def _open_times(day_count: int):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [start + timedelta(days=index) for index in range(day_count)]


def test_build_walk_forward_folds_keeps_train_before_validation():
    folds = build_walk_forward_folds(
        _open_times(9),
        train_days=3,
        test_days=2,
        step_days=2,
    )

    assert [(fold.train_indices, fold.test_indices) for fold in folds] == [
        ((0, 1, 2), (3, 4)),
        ((2, 3, 4), (5, 6)),
        ((4, 5, 6), (7, 8)),
    ]
    assert folds[0].train_end < folds[0].test_start
    assert folds[-1].test_end == datetime(2026, 1, 10, tzinfo=timezone.utc)


def test_build_walk_forward_folds_requires_positive_windows():
    with pytest.raises(ValueError, match="train_days must be positive"):
        build_walk_forward_folds(_open_times(5), train_days=0, test_days=2)
    with pytest.raises(ValueError, match="test_days must be positive"):
        build_walk_forward_folds(_open_times(5), train_days=3, test_days=0)
    with pytest.raises(ValueError, match="step_days must be positive"):
        build_walk_forward_folds(_open_times(5), train_days=3, test_days=1, step_days=0)


def test_build_walk_forward_fold_row_records_selected_train_and_validation_metrics():
    config = VolumePriceEfficiencyConfig(
        efficiency_percentile=0.9,
        min_move_unit=1.2,
        min_volume_unit=1.0,
        min_close_position=0.94,
        min_body_ratio=0.85,
    )
    row = build_walk_forward_fold_row(
        walk_forward_id="wf",
        fold_index=2,
        fold=build_walk_forward_folds(_open_times(6), train_days=3, test_days=2)[0],
        selected_train_row={
            "config_hash": config.config_hash(),
            "score": 0.04,
            "trade_count": 20,
            "total_return": 0.05,
            "max_drawdown": 0.01,
        },
        validation_row={
            "trade_count": 7,
            "total_return": -0.02,
            "max_drawdown": 0.03,
            "score": -0.05,
            "win_rate": 0.4,
            "mean_net_realized_return": -0.003,
            "median_net_realized_return": -0.002,
            "average_holding_bars": 4.0,
            "total_ignored_signal_count": 3,
            "final_equity": 0.98,
        },
        selected_config=config,
    )

    assert row["walk_forward_id"] == "wf"
    assert row["fold_index"] == 2
    assert row["selected_config_hash"] == config.config_hash()
    assert row["train_score"] == 0.04
    assert row["train_trade_count"] == 20
    assert row["validation_score"] == -0.05
    assert row["validation_trade_count"] == 7
    assert row["efficiency_percentile"] == 0.9


def test_write_trailing_walk_forward_artifacts(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)
    config = VolumePriceEfficiencyConfig()
    fold_rows = [
        {
            "walk_forward_id": "wf",
            "fold_index": 0,
            "selected_config_hash": config.config_hash(),
            "train_score": 0.02,
            "validation_score": -0.01,
        }
    ]
    selection_rows = [
        {
            "walk_forward_id": "wf",
            "fold_index": 0,
            "config_hash": config.config_hash(),
            "score": 0.02,
            "trade_count": 10,
        }
    ]

    output = write_trailing_walk_forward_artifacts(
        paths=paths,
        walk_forward_id="wf",
        base_config=config,
        fold_rows=fold_rows,
        selection_rows=selection_rows,
        top_configs=[selection_rows[0]],
        canonical_manifests=["manifest.json"],
        git_commit="abc123",
        runtime_seconds=1.25,
        symbol_count=3,
        data_split={"holdout_days": 180},
        atr_multiplier=2.0,
        train_days=720,
        test_days=90,
        step_days=90,
        min_trades=50,
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert output == paths.trailing_walk_forward_dir("wf")
    assert manifest["run_type"] == "trailing_stop_research_walk_forward"
    assert manifest["fold_count"] == 1
    assert manifest["data_scope"] == "research_only"
    assert set(manifest["outputs"]) == {
        "fold_summary",
        "fold_summary_csv",
        "selection_summary",
        "top_configs",
    }
    assert pq.read_table(output / "fold_summary.parquet").num_rows == 1
    assert pq.read_table(output / "selection_summary.parquet").num_rows == 1
    assert (output / "fold_summary.csv").exists()
