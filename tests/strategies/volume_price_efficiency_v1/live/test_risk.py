from datetime import datetime, timedelta, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig
from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    AccountSnapshot,
    OrderIntent,
    OrderIntentType,
    PositionState,
    SymbolMetadata,
)
from xsignal.strategies.volume_price_efficiency_v1.live.risk import evaluate_intent


NOW = datetime(2026, 5, 9, tzinfo=timezone.utc)


def _metadata(**overrides) -> SymbolMetadata:
    data = dict(
        symbol="BTCUSDT",
        status="TRADING",
        min_notional=5.0,
        quantity_step=0.001,
        price_tick=0.1,
        supports_stop_market=True,
        trigger_protect=0.05,
        updated_at=NOW,
    )
    data.update(overrides)
    return SymbolMetadata(**data)


def _snapshot(**overrides) -> AccountSnapshot:
    data = dict(
        mode="testnet",
        account_mode="one_way",
        asset_mode="single_asset_usdt",
        equity=1000.0,
        available_balance=1000.0,
        open_notional=0.0,
        open_position_count=0,
        daily_realized_pnl=0.0,
        captured_at=NOW,
    )
    data.update(overrides)
    return AccountSnapshot(**data)


def _intent(**overrides) -> OrderIntent:
    data = dict(
        intent_id="intent-1",
        position_id="BTCUSDT-1",
        symbol="BTCUSDT",
        intent_type=OrderIntentType.ENTRY,
        client_order_id="XV1TEBTC123",
        side="BUY",
        quantity=0.001,
        notional=20.0,
        price=None,
        stop_price=None,
        created_at=NOW,
    )
    data.update(overrides)
    return OrderIntent(**data)


def test_risk_accepts_valid_entry():
    result = evaluate_intent(
        config=LiveTradingConfig(),
        intent=_intent(),
        metadata=_metadata(),
        account=_snapshot(),
        position_state=PositionState.FLAT,
        now=NOW,
    )
    assert result.accepted
    assert result.reason == "accepted"


def test_risk_rejects_wrong_account_mode():
    result = evaluate_intent(
        config=LiveTradingConfig(),
        intent=_intent(),
        metadata=_metadata(),
        account=_snapshot(account_mode="hedge"),
        position_state=PositionState.FLAT,
        now=NOW,
    )
    assert not result.accepted
    assert result.reason == "account_mode_mismatch"


def test_risk_rejects_stale_metadata_and_low_notional():
    result = evaluate_intent(
        config=LiveTradingConfig(),
        intent=_intent(notional=2.0),
        metadata=_metadata(updated_at=NOW - timedelta(hours=2)),
        account=_snapshot(),
        position_state=PositionState.FLAT,
        now=NOW,
    )
    assert not result.accepted
    assert result.reason == "metadata_stale"
