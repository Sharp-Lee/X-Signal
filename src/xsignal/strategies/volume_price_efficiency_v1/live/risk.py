from __future__ import annotations

from datetime import datetime, timedelta

from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig
from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    AccountSnapshot,
    OrderIntent,
    PositionState,
    RiskResult,
    SymbolMetadata,
)


def evaluate_intent(
    *,
    config: LiveTradingConfig,
    intent: OrderIntent,
    metadata: SymbolMetadata,
    account: AccountSnapshot,
    position_state: PositionState,
    now: datetime,
) -> RiskResult:
    if account.mode != config.mode:
        return RiskResult(False, "mode_mismatch")
    if account.account_mode != config.account_mode:
        return RiskResult(False, "account_mode_mismatch")
    if account.asset_mode != config.asset_mode:
        return RiskResult(False, "asset_mode_mismatch")
    if metadata.updated_at < now - timedelta(minutes=30):
        return RiskResult(False, "metadata_stale")
    if metadata.status != "TRADING":
        return RiskResult(False, "symbol_not_trading")
    if not metadata.supports_stop_market:
        return RiskResult(False, "stop_market_not_supported")
    if intent.notional < metadata.min_notional:
        return RiskResult(False, "below_min_notional")
    if intent.notional > config.per_symbol_notional_cap:
        return RiskResult(False, "per_symbol_cap_exceeded")
    if account.open_notional + intent.notional > config.total_open_notional_cap:
        return RiskResult(False, "total_cap_exceeded")
    if account.open_position_count >= config.max_open_positions and (
        position_state == PositionState.FLAT
    ):
        return RiskResult(False, "max_open_positions_exceeded")
    if account.daily_realized_pnl <= -config.max_daily_realized_loss:
        return RiskResult(False, "daily_loss_limit_exceeded")
    return RiskResult(True, "accepted")
