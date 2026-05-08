import pytest

from xsignal.strategies.volume_price_efficiency_v1.live.config import (
    LiveTradingConfig,
    build_vpe_live_strategy_config,
)


def test_live_defaults_match_design_spec():
    config = LiveTradingConfig()
    assert config.mode == "testnet"
    assert config.account_mode == "one_way"
    assert config.margin_mode == "isolated"
    assert config.asset_mode == "single_asset_usdt"
    assert config.direction == "long_only"
    assert config.leverage == 1
    assert config.base_position_fraction == 0.05
    assert config.per_symbol_notional_cap == 20.0
    assert config.total_open_notional_cap == 100.0
    assert config.max_open_positions == 5
    assert config.max_daily_realized_loss == 50.0


def test_live_mode_requires_acknowledgement():
    with pytest.raises(ValueError, match="live acknowledgement"):
        LiveTradingConfig(mode="live")
    assert LiveTradingConfig(mode="live", live_acknowledgement=True).mode == "live"


def test_strategy_preset_matches_final_holdout_values():
    strategy_config = build_vpe_live_strategy_config()
    assert strategy_config.timeframe == "1d"
    assert strategy_config.fill_policy == "raw"
    assert strategy_config.signal_mode == "seed_efficiency"
    assert strategy_config.min_move_unit == 0.7
    assert strategy_config.min_volume_unit == 0.3
    assert strategy_config.min_close_position == 0.7
    assert strategy_config.min_body_ratio == 0.4
    assert strategy_config.seed_efficiency_lookback == 4
    assert strategy_config.seed_min_efficiency_ratio_to_max == 2.0
    assert strategy_config.seed_min_efficiency_ratio_to_mean == 5.0
    assert strategy_config.seed_max_volume_unit == 0.8
    assert strategy_config.seed_bottom_lookback == 60
    assert strategy_config.seed_max_close_position_in_range == 0.6
