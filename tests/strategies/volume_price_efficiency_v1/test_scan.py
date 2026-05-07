from __future__ import annotations

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.scan import (
    build_bucket_summary_rows,
    build_scan_configs,
    build_scan_row,
    select_top_configs,
)


def _event(
    *,
    symbol: str = "BTCUSDT",
    efficiency: float = 1.0,
    move_unit: float = 1.0,
    volume_unit: float = 1.0,
    close_position: float = 0.8,
    body_ratio: float = 0.5,
    ret1: float = 0.02,
    net1: float = 0.018,
    ret30: float = 0.10,
    net30: float = 0.098,
) -> dict:
    return {
        "symbol": symbol,
        "signal_open_time": "2026-01-01T00:00:00+00:00",
        "decision_time": "2026-01-01T04:00:00+00:00",
        "entry_open_time": "2026-01-01T04:00:00+00:00",
        "entry_price": 100.0,
        "move_unit": move_unit,
        "volume_unit": volume_unit,
        "efficiency": efficiency,
        "efficiency_threshold": 0.5,
        "close_position": close_position,
        "body_ratio": body_ratio,
        "quote_volume": 1_000_000.0,
        "volume_baseline": 900_000.0,
        "atr": 10.0,
        "forward_return_1": ret1,
        "net_forward_return_1": net1,
        "forward_return_30": ret30,
        "net_forward_return_30": net30,
    }


def test_build_scan_configs_expands_compact_grid():
    configs = build_scan_configs(
        efficiency_percentiles=(0.9, 0.95),
        min_move_units=(0.5,),
        min_volume_units=(0.3,),
        min_close_positions=(0.7, 0.8),
        min_body_ratios=(0.4,),
        fee_bps=5.0,
        slippage_bps=5.0,
        baseline_seed=17,
    )

    assert len(configs) == 4
    assert [config.efficiency_percentile for config in configs] == [0.9, 0.9, 0.95, 0.95]
    assert [config.min_close_position for config in configs] == [0.7, 0.8, 0.7, 0.8]


def test_build_scan_row_flattens_summary_and_scores_ranking_horizon():
    config = VolumePriceEfficiencyConfig(horizons=(1, 30), efficiency_percentile=0.95)
    events = [_event(ret30=0.12, net30=0.118), _event(symbol="ETHUSDT", ret30=0.08, net30=0.078)]
    baseline = [_event(ret30=0.04, net30=0.038), _event(symbol="ETHUSDT", ret30=0.02, net30=0.018)]

    row = build_scan_row(
        scan_id="scan123",
        config=config,
        events=events,
        baseline_events=baseline,
        symbols=("BTCUSDT", "ETHUSDT"),
        ranking_horizon=30,
    )

    assert row["scan_id"] == "scan123"
    assert row["config_hash"] == config.config_hash()
    assert row["efficiency_percentile"] == 0.95
    assert row["event_count"] == 2
    assert row["baseline_event_count"] == 2
    assert row["symbol_count"] == 2
    assert row["h30_mean_return"] == 0.1
    assert row["h30_baseline_mean_return"] == 0.03
    assert row["h30_event_minus_baseline_mean_return"] == 0.07
    assert row["score"] == 0.07


def test_select_top_configs_orders_by_score_then_hash():
    rows = [
        {"config_hash": "b", "score": 0.01},
        {"config_hash": "a", "score": 0.02},
        {"config_hash": "c", "score": 0.02},
    ]

    assert select_top_configs(rows, top_k=2) == [
        {"config_hash": "a", "score": 0.02},
        {"config_hash": "c", "score": 0.02},
    ]


def test_build_bucket_summary_rows_groups_events_by_feature_quantiles():
    config = VolumePriceEfficiencyConfig(horizons=(1,))
    events = [
        _event(efficiency=1.0, ret1=0.01, net1=0.008),
        _event(efficiency=2.0, ret1=0.03, net1=0.028),
        _event(efficiency=3.0, ret1=-0.01, net1=-0.012),
        _event(efficiency=4.0, ret1=0.05, net1=0.048),
    ]

    rows = build_bucket_summary_rows(
        config=config,
        events=events,
        horizons=(1,),
        feature_names=("efficiency",),
        bucket_count=2,
    )

    assert rows == [
        {
            "config_hash": config.config_hash(),
            "feature_name": "efficiency",
            "bucket_index": 0,
            "bucket_count": 2,
            "lower_bound": 1.0,
            "upper_bound": 2.0,
            "event_count": 2,
            "h1_mean_return": 0.02,
            "h1_net_mean_return": 0.018,
            "h1_win_rate": 1.0,
        },
        {
            "config_hash": config.config_hash(),
            "feature_name": "efficiency",
            "bucket_index": 1,
            "bucket_count": 2,
            "lower_bound": 3.0,
            "upper_bound": 4.0,
            "event_count": 2,
            "h1_mean_return": 0.02,
            "h1_net_mean_return": 0.018,
            "h1_win_rate": 0.5,
        },
    ]
