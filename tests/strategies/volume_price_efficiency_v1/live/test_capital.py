from datetime import datetime, timezone

from xsignal.strategies.volume_price_efficiency_v1.live.capital import size_entry_notional
from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig
from xsignal.strategies.volume_price_efficiency_v1.live.models import AccountSnapshot


def _snapshot(equity: float, available: float, open_notional: float = 0.0) -> AccountSnapshot:
    return AccountSnapshot(
        mode="testnet",
        account_mode="one_way",
        asset_mode="single_asset_usdt",
        equity=equity,
        available_balance=available,
        open_notional=open_notional,
        open_position_count=0,
        daily_realized_pnl=0.0,
        captured_at=datetime(2026, 5, 9, tzinfo=timezone.utc),
    )


def test_entry_notional_uses_shared_equity_and_cap():
    config = LiveTradingConfig()
    assert size_entry_notional(config, _snapshot(1000.0, 1000.0)) == 20.0
    assert size_entry_notional(config, _snapshot(200.0, 200.0)) == 10.0


def test_entry_notional_respects_available_balance_and_total_cap():
    config = LiveTradingConfig()
    assert size_entry_notional(config, _snapshot(1000.0, 8.0)) == 8.0
    assert size_entry_notional(config, _snapshot(1000.0, 1000.0, open_notional=95.0)) == 5.0
