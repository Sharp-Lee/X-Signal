from __future__ import annotations

import pytest

from xsignal.strategies.momentum_rotation_v1.config import MomentumRotationConfig


def test_default_config_matches_v1_design():
    config = MomentumRotationConfig()

    assert config.strategy_name == "momentum_rotation_v1"
    assert config.timeframes == ("1h", "4h", "1d")
    assert config.fill_policy == "raw"
    assert config.top_n == 10
    assert config.fee_bps == 5.0
    assert config.slippage_bps == 5.0
    assert config.initial_equity == 1.0
    assert config.short_return_weight == 0.4
    assert config.medium_return_weight == 0.4
    assert config.long_return_weight == 0.2
    assert config.short_window_hours == 24
    assert config.medium_window_days == 7
    assert config.long_window_days == 30
    assert config.min_rolling_7d_quote_volume == 0.0


def test_config_hash_is_stable_for_same_payload():
    first = MomentumRotationConfig(top_n=20)
    second = MomentumRotationConfig(top_n=20)

    assert first.config_hash() == second.config_hash()
    assert len(first.config_hash()) == 64


def test_config_rejects_invalid_values():
    with pytest.raises(ValueError, match="top_n"):
        MomentumRotationConfig(top_n=0)
    with pytest.raises(ValueError, match="fee_bps"):
        MomentumRotationConfig(fee_bps=-1)
    with pytest.raises(ValueError, match="slippage_bps"):
        MomentumRotationConfig(slippage_bps=-1)
    with pytest.raises(ValueError, match="initial_equity"):
        MomentumRotationConfig(initial_equity=0)
    with pytest.raises(ValueError, match="return weights"):
        MomentumRotationConfig(
            short_return_weight=0.5,
            medium_return_weight=0.5,
            long_return_weight=0.5,
        )
    with pytest.raises(ValueError, match="fill_policy"):
        MomentumRotationConfig(fill_policy="prev_close_zero_volume")
