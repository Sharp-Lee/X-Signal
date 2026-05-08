from __future__ import annotations

import hashlib
from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


SignalMode = Literal["classic", "seed_efficiency"]


class VolumePriceEfficiencyConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy_name: str = "volume_price_efficiency_v1"
    timeframe: str = "1d"
    fill_policy: str = "raw"
    atr_window: int = 14
    volume_window: int = 60
    efficiency_lookback: int = 120
    efficiency_percentile: float = 0.90
    volume_floor: float = 0.2
    signal_mode: SignalMode = "classic"
    min_move_unit: float = 0.5
    min_volume_unit: float = 0.3
    min_close_position: float = 0.7
    min_body_ratio: float = 0.4
    seed_efficiency_lookback: int = 4
    seed_min_efficiency_ratio_to_max: float = 1.5
    seed_min_efficiency_ratio_to_mean: float = 3.0
    seed_max_volume_unit: float = 1.2
    seed_bottom_lookback: int = 30
    seed_max_close_position_in_range: float = 0.6
    horizons: tuple[int, ...] = (1, 3, 6, 12, 30)
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    baseline_seed: int = 17

    @field_validator("strategy_name", "timeframe", "fill_policy")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be non-empty")
        return value

    @model_validator(mode="after")
    def _validate(self) -> "VolumePriceEfficiencyConfig":
        if self.strategy_name != "volume_price_efficiency_v1":
            raise ValueError("strategy_name must be volume_price_efficiency_v1")
        if self.timeframe != "1d":
            raise ValueError("timeframe must be 1d")
        if self.fill_policy != "raw":
            raise ValueError("fill_policy must be raw")
        if self.atr_window <= 0:
            raise ValueError("atr_window must be positive")
        if self.volume_window <= 0:
            raise ValueError("volume_window must be positive")
        if self.efficiency_lookback <= 0:
            raise ValueError("efficiency_lookback must be positive")
        if not 0.0 < self.efficiency_percentile < 1.0:
            raise ValueError("efficiency_percentile must be between 0 and 1")
        if self.volume_floor <= 0:
            raise ValueError("volume_floor must be positive")
        if self.min_move_unit < 0:
            raise ValueError("min_move_unit must be non-negative")
        if self.min_volume_unit < 0:
            raise ValueError("min_volume_unit must be non-negative")
        if not 0.0 <= self.min_close_position <= 1.0:
            raise ValueError("min_close_position must be between 0 and 1")
        if not 0.0 <= self.min_body_ratio <= 1.0:
            raise ValueError("min_body_ratio must be between 0 and 1")
        if self.seed_efficiency_lookback <= 0:
            raise ValueError("seed_efficiency_lookback must be positive")
        if self.seed_min_efficiency_ratio_to_max <= 1.0:
            raise ValueError("seed_min_efficiency_ratio_to_max must be greater than 1")
        if self.seed_min_efficiency_ratio_to_mean <= 1.0:
            raise ValueError("seed_min_efficiency_ratio_to_mean must be greater than 1")
        if self.seed_max_volume_unit <= 0.0:
            raise ValueError("seed_max_volume_unit must be positive")
        if self.seed_bottom_lookback <= 0:
            raise ValueError("seed_bottom_lookback must be positive")
        if not 0.0 <= self.seed_max_close_position_in_range <= 1.0:
            raise ValueError("seed_max_close_position_in_range must be between 0 and 1")
        if not self.horizons or any(horizon <= 0 for horizon in self.horizons):
            raise ValueError("horizons must contain positive integers")
        if tuple(sorted(set(self.horizons))) != self.horizons:
            raise ValueError("horizons must be unique and sorted")
        if self.fee_bps < 0:
            raise ValueError("fee_bps must be non-negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        return self

    @property
    def round_trip_cost(self) -> float:
        return 2.0 * (self.fee_bps + self.slippage_bps) / 10_000.0

    def config_hash(self) -> str:
        payload = self.model_dump_json(exclude_none=False)
        return hashlib.sha256(payload.encode()).hexdigest()
