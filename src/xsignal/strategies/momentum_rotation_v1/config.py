from __future__ import annotations

import hashlib
import json
from datetime import date

from pydantic import BaseModel, ConfigDict, field_validator, model_validator


class MomentumRotationConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    strategy_name: str = "momentum_rotation_v1"
    timeframes: tuple[str, str, str] = ("1h", "4h", "1d")
    fill_policy: str = "raw"
    top_n: int = 10
    fee_bps: float = 5.0
    slippage_bps: float = 5.0
    initial_equity: float = 1.0
    min_rolling_7d_quote_volume: float = 0.0
    short_return_weight: float = 0.4
    medium_return_weight: float = 0.4
    long_return_weight: float = 0.2
    short_window_hours: int = 24
    medium_window_days: int = 7
    long_window_days: int = 30
    start_date: date | None = None
    end_date: date | None = None

    @field_validator("strategy_name", "fill_policy")
    @classmethod
    def _non_empty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("field must be non-empty")
        return value

    @model_validator(mode="after")
    def _validate_config(self) -> "MomentumRotationConfig":
        if self.strategy_name != "momentum_rotation_v1":
            raise ValueError("strategy_name must be 'momentum_rotation_v1'")
        if self.timeframes != ("1h", "4h", "1d"):
            raise ValueError("timeframes must be ('1h', '4h', '1d')")
        if self.fill_policy != "raw":
            raise ValueError("fill_policy must be 'raw'")
        if self.top_n <= 0:
            raise ValueError("top_n must be positive")
        if self.fee_bps < 0:
            raise ValueError("fee_bps must be non-negative")
        if self.slippage_bps < 0:
            raise ValueError("slippage_bps must be non-negative")
        if self.initial_equity <= 0:
            raise ValueError("initial_equity must be positive")
        if self.min_rolling_7d_quote_volume < 0:
            raise ValueError("min_rolling_7d_quote_volume must be non-negative")
        if self.short_window_hours != 24:
            raise ValueError("short_window_hours must be 24 for v1")
        if self.medium_window_days != 7:
            raise ValueError("medium_window_days must be 7 for v1")
        if self.long_window_days != 30:
            raise ValueError("long_window_days must be 30 for v1")
        weight_sum = self.short_return_weight + self.medium_return_weight + self.long_return_weight
        if abs(weight_sum - 1.0) > 1e-12:
            raise ValueError("return weights must sum to 1.0")
        if (
            self.start_date is not None
            and self.end_date is not None
            and self.end_date <= self.start_date
        ):
            raise ValueError("end_date must be after start_date")
        return self

    def config_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()
