from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)


LiveMode = Literal["testnet", "live", "reconcile-only"]


class LiveTradingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: LiveMode = "testnet"
    live_acknowledgement: bool = False
    account_mode: str = "one_way"
    margin_mode: str = "isolated"
    asset_mode: str = "single_asset_usdt"
    direction: str = "long_only"
    leverage: int = 1
    base_position_fraction: float = 0.05
    per_symbol_notional_cap: float = 20.0
    total_open_notional_cap: float = 100.0
    max_open_positions: int = 5
    max_daily_realized_loss: float = 50.0
    min_quote_notional: float = 5.0
    atr_multiplier: float = 3.0
    stop_replace_min_interval_seconds: float = 30.0
    stop_replace_min_improvement_ticks: int = 10
    stop_replace_min_improvement_fraction: float = 0.0001
    pyramid_add_step_atr: float = 1.0
    pyramid_max_adds: int = 1
    market_regime_lookback_bars: int = 30
    market_regime_min_return: float = -0.10

    @model_validator(mode="after")
    def _validate(self) -> "LiveTradingConfig":
        if self.mode == "live" and not self.live_acknowledgement:
            raise ValueError("live acknowledgement is required for live mode")
        if self.account_mode != "one_way":
            raise ValueError("account_mode must be one_way")
        if self.margin_mode != "isolated":
            raise ValueError("margin_mode must be isolated")
        if self.asset_mode != "single_asset_usdt":
            raise ValueError("asset_mode must be single_asset_usdt")
        if self.direction != "long_only":
            raise ValueError("direction must be long_only")
        if self.leverage != 1:
            raise ValueError("leverage must be 1")
        if not 0.0 < self.base_position_fraction <= 1.0:
            raise ValueError("base_position_fraction must be in (0, 1]")
        if self.per_symbol_notional_cap <= 0.0:
            raise ValueError("per_symbol_notional_cap must be positive")
        if self.total_open_notional_cap <= 0.0:
            raise ValueError("total_open_notional_cap must be positive")
        if self.max_open_positions <= 0:
            raise ValueError("max_open_positions must be positive")
        if self.max_daily_realized_loss <= 0.0:
            raise ValueError("max_daily_realized_loss must be positive")
        if self.min_quote_notional <= 0.0:
            raise ValueError("min_quote_notional must be positive")
        if self.atr_multiplier <= 0.0:
            raise ValueError("atr_multiplier must be positive")
        if self.stop_replace_min_interval_seconds < 0.0:
            raise ValueError("stop_replace_min_interval_seconds must be non-negative")
        if self.stop_replace_min_improvement_ticks < 0:
            raise ValueError("stop_replace_min_improvement_ticks must be non-negative")
        if self.stop_replace_min_improvement_fraction < 0.0:
            raise ValueError("stop_replace_min_improvement_fraction must be non-negative")
        if self.pyramid_add_step_atr <= 0.0:
            raise ValueError("pyramid_add_step_atr must be positive")
        if self.pyramid_max_adds != 1:
            raise ValueError("pyramid_max_adds must be 1 for the first live preset")
        return self


def build_vpe_live_strategy_config() -> VolumePriceEfficiencyConfig:
    return VolumePriceEfficiencyConfig(
        signal_mode="seed_efficiency",
        min_move_unit=0.7,
        min_volume_unit=0.3,
        min_close_position=0.7,
        min_body_ratio=0.4,
        seed_efficiency_lookback=4,
        seed_min_efficiency_ratio_to_max=2.0,
        seed_min_efficiency_ratio_to_mean=5.0,
        seed_max_volume_unit=0.8,
        seed_bottom_lookback=60,
        seed_max_close_position_in_range=0.6,
    )
