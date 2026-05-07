from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json

import pyarrow.parquet as pq

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_regime_scan import (
    RegimeFilterRule,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_regime_walk_forward import (
    build_regime_walk_forward_fold_row,
    write_trailing_regime_walk_forward_artifacts,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_walk_forward import (
    build_walk_forward_folds,
)


def _open_times(day_count: int):
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return [start + timedelta(days=index) for index in range(day_count)]


def test_build_regime_walk_forward_fold_row_records_train_threshold_and_validation_metrics():
    config = VolumePriceEfficiencyConfig(min_move_unit=1.2)
    rule = RegimeFilterRule(
        rule_id="market_lookback_return_lt_p80",
        feature_name="market_lookback_return",
        direction="lt",
        quantile=0.8,
        threshold=0.12,
    )

    row = build_regime_walk_forward_fold_row(
        regime_walk_forward_id="rwf",
        fold_index=1,
        fold=build_walk_forward_folds(_open_times(6), train_days=3, test_days=2)[0],
        selected_train_row={
            "rule_id": rule.rule_id,
            "score": 0.04,
            "trade_count": 22,
            "total_return": 0.05,
            "max_drawdown": 0.01,
            "filtered_signal_count": 30,
            "base_signal_count": 60,
        },
        validation_row={
            "score": -0.02,
            "trade_count": 8,
            "total_return": -0.01,
            "max_drawdown": 0.01,
            "win_rate": 0.375,
            "mean_net_realized_return": -0.002,
            "median_net_realized_return": -0.001,
            "average_holding_bars": 5.0,
            "total_ignored_signal_count": 2,
            "final_equity": 0.99,
            "filtered_signal_count": 9,
            "base_signal_count": 18,
        },
        selected_rule=rule,
        selected_config=config,
    )

    assert row["regime_walk_forward_id"] == "rwf"
    assert row["fold_index"] == 1
    assert row["rule_id"] == "market_lookback_return_lt_p80"
    assert row["threshold"] == 0.12
    assert row["threshold_source"] == "train_fold_signal_distribution"
    assert row["train_score"] == 0.04
    assert row["train_signal_keep_rate"] == 0.5
    assert row["validation_score"] == -0.02
    assert row["validation_signal_keep_rate"] == 0.5
    assert row["min_move_unit"] == 1.2


def test_write_trailing_regime_walk_forward_artifacts(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)
    config = VolumePriceEfficiencyConfig()
    rule = RegimeFilterRule(
        rule_id="market_lookback_return_lt_p80",
        feature_name="market_lookback_return",
        direction="lt",
        quantile=0.8,
        threshold=0.12,
    )
    fold_rows = [
        build_regime_walk_forward_fold_row(
            regime_walk_forward_id="rwf",
            fold_index=0,
            fold=build_walk_forward_folds(_open_times(6), train_days=3, test_days=2)[0],
            selected_train_row={"score": 0.04, "trade_count": 22},
            validation_row={"score": -0.02, "trade_count": 8},
            selected_rule=rule,
            selected_config=config,
        )
    ]
    selection_rows = [
        {
            "regime_walk_forward_id": "rwf",
            "fold_index": 0,
            "rule_id": rule.rule_id,
            "score": 0.04,
            "trade_count": 22,
        }
    ]

    output = write_trailing_regime_walk_forward_artifacts(
        paths=paths,
        regime_walk_forward_id="rwf",
        config=config,
        fold_rows=fold_rows,
        selection_rows=selection_rows,
        top_filters=selection_rows,
        canonical_manifests=["manifest.json"],
        git_commit="abc123",
        runtime_seconds=1.5,
        symbol_count=5,
        data_split={"holdout_days": 180},
        atr_multiplier=2.0,
        lookback_bars=30,
        train_days=720,
        test_days=90,
        step_days=90,
        min_trades=50,
        quantiles=(0.8,),
        feature_names=("market_lookback_return",),
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert output == paths.trailing_regime_walk_forward_dir("rwf")
    assert manifest["run_type"] == "trailing_stop_research_regime_walk_forward"
    assert manifest["data_scope"] == "research_only"
    assert manifest["threshold_scope"] == "per_fold_train_signal_distribution"
    assert set(manifest["outputs"]) == {
        "fold_summary",
        "fold_summary_csv",
        "selection_summary",
        "top_filters",
    }
    assert pq.read_table(output / "fold_summary.parquet").num_rows == 1
    assert pq.read_table(output / "selection_summary.parquet").num_rows == 1
