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
from xsignal.strategies.volume_price_efficiency_v1.features import FeatureArrays
from xsignal.strategies.volume_price_efficiency_v1.paths import (
    VolumePriceEfficiencyPaths,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing import (
    TrailingStopResult,
)
from xsignal.strategies.volume_price_efficiency_v1.trailing_regime_scan import (
    RegimeFilterRule,
    apply_regime_filter_rule,
    build_composite_regime_filter_rules,
    build_regime_filter_rules,
    build_regime_scan_row,
    build_regime_value_arrays,
    select_top_regime_filters,
    write_trailing_regime_scan_artifacts,
)


def _arrays() -> OhlcvArrays:
    start = datetime(2026, 1, 1, tzinfo=timezone.utc)
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
        open_times=np.array([start + timedelta(hours=4 * index) for index in range(4)], dtype=object),
        open=close.copy(),
        high=close.copy() + 1.0,
        low=close.copy() - 1.0,
        close=close,
        quote_volume=np.array(
            [
                [100.0, 1_000.0],
                [200.0, 2_000.0],
                [300.0, 3_000.0],
                [400.0, 4_000.0],
            ],
            dtype=np.float64,
        ),
        quality=np.ones(close.shape, dtype=bool),
    )


def _features() -> FeatureArrays:
    values = np.arange(8, dtype=np.float64).reshape(4, 2) + 1.0
    signal = np.zeros((4, 2), dtype=bool)
    signal[1, 0] = True
    signal[2, 1] = True
    signal[3, 1] = True
    return FeatureArrays(
        true_range=values.copy(),
        atr=values.copy(),
        move_unit=values.copy(),
        volume_baseline=values.copy(),
        volume_unit=values.copy() / 2.0,
        efficiency=values.copy() / 3.0,
        efficiency_threshold=values.copy() / 4.0,
        close_position=np.full((4, 2), 0.8),
        body_ratio=np.full((4, 2), 0.5),
        signal=signal,
    )


def _result() -> TrailingStopResult:
    return TrailingStopResult(
        trades=[
            {
                "symbol": "BTCUSDT",
                "realized_return": 0.04,
                "net_realized_return": 0.038,
                "holding_bars": 2,
                "ignored_signal_count": 1,
            },
            {
                "symbol": "ETHUSDT",
                "realized_return": -0.01,
                "net_realized_return": -0.012,
                "holding_bars": 1,
                "ignored_signal_count": 0,
            },
        ],
        equity=np.array([1.0, 0.99, 1.02], dtype=np.float64),
        period_returns=np.array([-0.01, 1.02 / 0.99 - 1.0], dtype=np.float64),
        positions=np.zeros((3, 2), dtype=bool),
        stop_prices=np.full((3, 2), np.nan),
    )


def test_build_regime_value_arrays_adds_causal_market_context_and_signal_features():
    values = build_regime_value_arrays(_arrays(), _features(), lookback_bars=2)

    assert values["btc_lookback_return"][2, 0] == pytest.approx(0.2)
    assert values["btc_lookback_return"][2, 1] == pytest.approx(0.2)
    assert values["symbol_lookback_return"][2, 1] == pytest.approx(0.5)
    assert values["market_lookback_return"][2, 0] == pytest.approx(0.35)
    assert values["signal_quote_volume"][3, 1] == 4_000.0
    assert values["move_unit"][3, 1] == 8.0


def test_build_regime_value_arrays_adds_symbol_age_and_stop_distance_features():
    arrays = _arrays()
    arrays.quality[0, 1] = False
    arrays.quality[1, 1] = False
    values = build_regime_value_arrays(arrays, _features(), lookback_bars=2)

    assert values["symbol_age_days"][0, 0] == 0.0
    assert values["symbol_age_days"][3, 0] == pytest.approx(0.5)
    assert np.isnan(values["symbol_age_days"][1, 1])
    assert values["symbol_age_days"][2, 1] == 0.0
    assert values["symbol_age_days"][3, 1] == pytest.approx(1 / 6)
    assert values["signal_atr_pct"][3, 1] == pytest.approx(8.0 / 18.0)
    assert values["signal_stop_distance_pct"][3, 1] == pytest.approx(2.0 * 8.0 / 18.0)


def test_build_regime_value_arrays_adds_causal_bottom_and_contraction_features():
    values = build_regime_value_arrays(_arrays(), _features(), lookback_bars=2)

    assert values["symbol_drawdown_from_lookback_high"][3, 0] == pytest.approx(-0.25)
    assert values["symbol_drawdown_from_lookback_high"][3, 1] == pytest.approx(0.0)
    assert values["btc_drawdown_from_lookback_high"][3, 0] == pytest.approx(-0.25)
    assert values["btc_drawdown_from_lookback_high"][3, 1] == pytest.approx(-0.25)
    assert values["market_drawdown_from_lookback_high"][3, 0] == pytest.approx(-0.125)
    assert values["market_drawdown_from_lookback_high"][3, 1] == pytest.approx(-0.125)
    assert values["symbol_range_position"][3, 0] == pytest.approx(0.0)
    assert values["symbol_range_position"][3, 1] == pytest.approx(1.0)
    assert values["btc_range_position"][3, 0] == pytest.approx(0.0)
    assert values["btc_range_position"][3, 1] == pytest.approx(0.0)
    assert values["market_range_position"][3, 0] == pytest.approx(0.5)
    assert values["market_range_position"][3, 1] == pytest.approx(0.5)

    assert values["pre_signal_atr_contraction"][3, 0] == pytest.approx(1.25)
    assert values["pre_signal_true_range_contraction"][3, 0] == pytest.approx(1.25)
    assert values["pre_signal_volume_contraction"][3, 0] == pytest.approx(1.2)


def test_build_regime_filter_rules_uses_signal_only_quantiles():
    rules = build_regime_filter_rules(
        _features().signal,
        {"move_unit": _features().move_unit},
        feature_names=("move_unit",),
        quantiles=(0.5,),
    )

    assert rules == (
        RegimeFilterRule(
            rule_id="move_unit_gte_p50",
            feature_name="move_unit",
            direction="gte",
            quantile=0.5,
            threshold=6.0,
        ),
        RegimeFilterRule(
            rule_id="move_unit_lt_p50",
            feature_name="move_unit",
            direction="lt",
            quantile=0.5,
            threshold=6.0,
        ),
    )


def test_apply_regime_filter_rule_keeps_only_matching_signals():
    features = _features()
    rule = RegimeFilterRule(
        rule_id="move_unit_gte_p50",
        feature_name="move_unit",
        direction="gte",
        quantile=0.5,
        threshold=5.0,
    )

    filtered = apply_regime_filter_rule(
        features.signal,
        {"move_unit": features.move_unit},
        rule,
    )

    assert filtered.tolist() == [
        [False, False],
        [False, False],
        [False, True],
        [False, True],
    ]


def test_composite_regime_filter_rule_combines_component_rules_with_and():
    features = _features()
    move_rule = RegimeFilterRule(
        rule_id="move_unit_gte_p50",
        feature_name="move_unit",
        direction="gte",
        quantile=0.5,
        threshold=5.0,
    )
    volume_rule = RegimeFilterRule(
        rule_id="volume_unit_gte_p50",
        feature_name="volume_unit",
        direction="gte",
        quantile=0.5,
        threshold=3.5,
    )

    combo = build_composite_regime_filter_rules((move_rule, volume_rule), combo_size=2)[0]
    filtered = apply_regime_filter_rule(
        features.signal,
        {
            "move_unit": features.move_unit,
            "volume_unit": features.volume_unit,
        },
        combo,
    )

    assert combo.rule_id == "move_unit_gte_p50__and__volume_unit_gte_p50"
    assert combo.feature_name == "move_unit+volume_unit"
    assert combo.direction == "and"
    assert combo.quantile is None
    assert combo.threshold is None
    assert filtered.tolist() == [
        [False, False],
        [False, False],
        [False, False],
        [False, True],
    ]


def test_build_regime_scan_row_extends_trailing_metrics_with_filter_metadata():
    config = VolumePriceEfficiencyConfig()
    rule = RegimeFilterRule(
        rule_id="market_lookback_return_gte_p80",
        feature_name="market_lookback_return",
        direction="gte",
        quantile=0.8,
        threshold=0.03,
    )

    row = build_regime_scan_row(
        regime_scan_id="regime",
        config=config,
        result=_result(),
        symbols=("BTCUSDT", "ETHUSDT"),
        rule=rule,
        base_signal_count=100,
        filtered_signal_count=25,
    )

    assert row["regime_scan_id"] == "regime"
    assert row["rule_id"] == "market_lookback_return_gte_p80"
    assert row["feature_name"] == "market_lookback_return"
    assert row["direction"] == "gte"
    assert row["threshold"] == 0.03
    assert row["base_signal_count"] == 100
    assert row["filtered_signal_count"] == 25
    assert row["signal_keep_rate"] == 0.25
    assert row["trade_count"] == 2
    assert row["score"] == pytest.approx(0.01)


def test_select_top_regime_filters_filters_low_trade_counts_and_unfiltered_baseline():
    rows = [
        {"rule_id": "unfiltered", "trade_count": 100, "score": 0.0},
        {"rule_id": "a", "trade_count": 5, "score": 1.0},
        {"rule_id": "b", "trade_count": 10, "score": 0.02},
        {"rule_id": "c", "trade_count": 11, "score": 0.04},
    ]

    assert select_top_regime_filters(rows, top_k=2, min_trades=10) == [
        {"rule_id": "c", "trade_count": 11, "score": 0.04},
        {"rule_id": "b", "trade_count": 10, "score": 0.02},
    ]


def test_write_trailing_regime_scan_artifacts(tmp_path):
    paths = VolumePriceEfficiencyPaths(root=tmp_path)
    config = VolumePriceEfficiencyConfig()
    rows = [
        build_regime_scan_row(
            regime_scan_id="regime",
            config=config,
            result=_result(),
            symbols=("BTCUSDT", "ETHUSDT"),
            rule=None,
            base_signal_count=100,
            filtered_signal_count=100,
        )
    ]

    output = write_trailing_regime_scan_artifacts(
        paths=paths,
        regime_scan_id="regime",
        config=config,
        rows=rows,
        top_filters=rows,
        canonical_manifests=["manifest.json"],
        git_commit="abc123",
        runtime_seconds=1.2,
        symbol_count=2,
        data_split={"holdout_days": 180},
        atr_multiplier=2.0,
        pyramid_add_step_atr=1.0,
        pyramid_max_adds=1,
        lookback_bars=30,
        quantiles=(0.2, 0.8),
        feature_names=("market_lookback_return",),
    )

    manifest = json.loads((output / "manifest.json").read_text())
    assert output == paths.trailing_regime_scan_dir("regime")
    assert manifest["run_type"] == "trailing_stop_research_regime_scan"
    assert manifest["data_scope"] == "research_only"
    assert manifest["threshold_scope"] == "full_research_signal_distribution_diagnostic_only"
    assert manifest["pyramid_add_step_atr"] == 1.0
    assert manifest["pyramid_max_adds"] == 1
    assert set(manifest["outputs"]) == {"summary", "summary_csv", "top_filters"}
    assert pq.read_table(output / "summary.parquet").num_rows == 1
    assert json.loads((output / "top_filters.json").read_text())[0]["rule_id"] == "unfiltered"
