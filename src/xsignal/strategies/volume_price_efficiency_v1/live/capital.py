from __future__ import annotations

from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig
from xsignal.strategies.volume_price_efficiency_v1.live.models import AccountSnapshot


def size_entry_notional(config: LiveTradingConfig, account: AccountSnapshot) -> float:
    desired = account.equity * config.base_position_fraction
    capped = min(desired, config.per_symbol_notional_cap)
    remaining_total = max(config.total_open_notional_cap - account.open_notional, 0.0)
    return max(min(capped, account.available_balance, remaining_total), 0.0)
