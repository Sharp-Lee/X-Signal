from __future__ import annotations

import pytest

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)


def test_default_config_matches_design():
    config = VolumePriceEfficiencyConfig()

    assert config.strategy_name == "volume_price_efficiency_v1"
    assert config.timeframe == "1d"
    assert config.fill_policy == "raw"
    assert config.atr_window == 14
    assert config.volume_window == 60
    assert config.efficiency_lookback == 120
    assert config.efficiency_percentile == 0.90
    assert config.volume_floor == 0.2
    assert config.signal_mode == "classic"
    assert config.min_move_unit == 0.5
    assert config.min_volume_unit == 0.3
    assert config.min_close_position == 0.7
    assert config.min_body_ratio == 0.4
    assert config.seed_efficiency_lookback == 4
    assert config.seed_min_efficiency_ratio_to_max == 1.5
    assert config.seed_min_efficiency_ratio_to_mean == 3.0
    assert config.seed_max_volume_unit == 1.2
    assert config.seed_bottom_lookback == 30
    assert config.seed_max_close_position_in_range == 0.6
    assert config.horizons == (1, 3, 6, 12, 30)
    assert config.fee_bps == 5.0
    assert config.slippage_bps == 5.0
    assert config.baseline_seed == 17


def test_config_hash_is_stable_for_same_payload():
    first = VolumePriceEfficiencyConfig(min_move_unit=0.8)
    second = VolumePriceEfficiencyConfig(min_move_unit=0.8)

    assert first.config_hash() == second.config_hash()
    assert len(first.config_hash()) == 64


def test_config_round_trip_cost_includes_entry_and_exit_friction():
    config = VolumePriceEfficiencyConfig(fee_bps=5, slippage_bps=10)

    assert config.round_trip_cost == 0.003


def test_config_rejects_invalid_values():
    invalid_kwargs = [
        {"timeframe": "4h"},
        {"fill_policy": "prev_close_zero_volume"},
        {"atr_window": 0},
        {"volume_window": 0},
        {"efficiency_lookback": 0},
        {"efficiency_percentile": 1.0},
        {"efficiency_percentile": 0.0},
        {"volume_floor": 0.0},
        {"signal_mode": "burst"},
        {"min_move_unit": -0.1},
        {"min_volume_unit": -0.1},
        {"min_close_position": 1.1},
        {"min_body_ratio": 1.1},
        {"seed_efficiency_lookback": 0},
        {"seed_min_efficiency_ratio_to_max": 1.0},
        {"seed_min_efficiency_ratio_to_mean": 1.0},
        {"seed_max_volume_unit": 0.0},
        {"seed_bottom_lookback": 0},
        {"seed_max_close_position_in_range": 1.1},
        {"horizons": ()},
        {"fee_bps": -1},
        {"slippage_bps": -1},
    ]
    for kwargs in invalid_kwargs:
        with pytest.raises(ValueError):
            VolumePriceEfficiencyConfig(**kwargs)
