from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.live.config import LiveTradingConfig
from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    AccountSnapshot,
    PositionState,
    SymbolMetadata,
)
from xsignal.strategies.volume_price_efficiency_v1.live.position_store import (
    LivePositionRecord,
    get_live_position,
    update_live_position,
)
from xsignal.strategies.volume_price_efficiency_v1.live.runner import run_live_cycle
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore


NOW = datetime(2026, 5, 9, tzinfo=timezone.utc)


class FakeCycleBroker:
    def __init__(self) -> None:
        self.calls = []

    def market_buy(self, *, symbol, quantity, client_order_id):
        self.calls.append(("market_buy", symbol, quantity, client_order_id))
        return {"symbol": symbol, "clientOrderId": client_order_id, "status": "FILLED"}

    def place_stop_market_close(self, *, symbol, stop_price, client_order_id):
        self.calls.append(("place_stop_market_close", symbol, stop_price, client_order_id))
        return {"symbol": symbol, "clientAlgoId": client_order_id, "algoStatus": "NEW"}

    def cancel_order(self, *, symbol, client_order_id):
        self.calls.append(("cancel_order", symbol, client_order_id))
        return {}


def _metadata(symbol: str) -> SymbolMetadata:
    return SymbolMetadata(
        symbol=symbol,
        status="TRADING",
        min_notional=5.0,
        quantity_step=0.001,
        price_tick=0.1,
        supports_stop_market=True,
        trigger_protect=0.05,
        updated_at=NOW,
        min_quantity=0.001,
        max_quantity=1000.0,
        market_min_quantity=0.001,
        market_max_quantity=1000.0,
        market_quantity_step=0.001,
    )


def _account() -> AccountSnapshot:
    return AccountSnapshot(
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


def _account_with(**overrides) -> AccountSnapshot:
    data = {
        "mode": "testnet",
        "account_mode": "one_way",
        "asset_mode": "single_asset_usdt",
        "equity": 1000.0,
        "available_balance": 1000.0,
        "open_notional": 0.0,
        "open_position_count": 0,
        "daily_realized_pnl": 0.0,
        "captured_at": NOW,
    }
    data.update(overrides)
    return AccountSnapshot(**data)


def _arrays(symbols=("BTCUSDT", "ETHUSDT")) -> OhlcvArrays:
    times = np.array(
        [
            datetime(2026, 5, 7, tzinfo=timezone.utc),
            datetime(2026, 5, 8, tzinfo=timezone.utc),
        ],
        dtype=object,
    )
    values = np.full((2, len(symbols)), 100.0)
    return OhlcvArrays(
        symbols=tuple(symbols),
        open_times=times,
        open=values.copy(),
        high=values.copy() + 10.0,
        low=values.copy() - 10.0,
        close=values.copy() + 5.0,
        quote_volume=np.full(values.shape, 1000.0),
        quality=np.ones(values.shape, dtype=bool),
    )


def _features(arrays: OhlcvArrays, atr: float = 5.0):
    return SimpleNamespace(atr=np.full(arrays.open.shape, atr))


def _reconcile_ok(calls):
    def runner(**kwargs):
        calls.append(("reconcile", tuple(kwargs["symbols"])))
        return SimpleNamespace(error_count=0)

    return runner


def test_run_live_cycle_reconciles_before_opening_signal(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    broker = FakeCycleBroker()
    reconcile_calls = []
    signal = np.array([[False, False], [True, False]])

    result = run_live_cycle(
        store=store,
        broker=broker,
        config=LiveTradingConfig(),
        environment="testnet",
        arrays=_arrays(),
        account=_account(),
        metadata_by_symbol={"BTCUSDT": _metadata("BTCUSDT"), "ETHUSDT": _metadata("ETHUSDT")},
        prices_by_symbol={"BTCUSDT": 100.0, "ETHUSDT": 100.0},
        now=NOW,
        reconcile_runner=_reconcile_ok(reconcile_calls),
        signal_mask_builder=lambda arrays, config: signal,
        feature_builder=lambda arrays: _features(arrays),
    )

    assert reconcile_calls == [("reconcile", ("BTCUSDT", "ETHUSDT"))]
    assert result.entries == 1
    assert broker.calls[0][0] == "market_buy"


def test_run_live_cycle_rejects_new_entry_when_max_positions_reached(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    broker = FakeCycleBroker()
    signal = np.array([[False], [True]])

    result = run_live_cycle(
        store=store,
        broker=broker,
        config=LiveTradingConfig(max_open_positions=1),
        environment="testnet",
        arrays=_arrays(("BTCUSDT",)),
        account=_account_with(open_position_count=1),
        metadata_by_symbol={"BTCUSDT": _metadata("BTCUSDT")},
        prices_by_symbol={"BTCUSDT": 100.0},
        now=NOW,
        reconcile_runner=_reconcile_ok([]),
        signal_mask_builder=lambda arrays, config: signal,
        feature_builder=lambda arrays: _features(arrays),
    )

    assert result.signal_count == 1
    assert result.entries == 0
    assert not any(call[0] == "market_buy" for call in broker.calls)


def test_run_live_cycle_rejects_new_entry_after_daily_loss_limit(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    broker = FakeCycleBroker()
    signal = np.array([[False], [True]])

    result = run_live_cycle(
        store=store,
        broker=broker,
        config=LiveTradingConfig(max_daily_realized_loss=50.0),
        environment="testnet",
        arrays=_arrays(("BTCUSDT",)),
        account=_account_with(daily_realized_pnl=-50.0),
        metadata_by_symbol={"BTCUSDT": _metadata("BTCUSDT")},
        prices_by_symbol={"BTCUSDT": 100.0},
        now=NOW,
        reconcile_runner=_reconcile_ok([]),
        signal_mask_builder=lambda arrays, config: signal,
        feature_builder=lambda arrays: _features(arrays),
    )

    assert result.signal_count == 1
    assert result.entries == 0
    assert not any(call[0] == "market_buy" for call in broker.calls)


def test_run_live_cycle_allocates_from_one_shared_snapshot_within_cycle(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    broker = FakeCycleBroker()
    signal = np.array([[False, False], [True, True]])

    result = run_live_cycle(
        store=store,
        broker=broker,
        config=LiveTradingConfig(max_open_positions=5, total_open_notional_cap=100.0),
        environment="testnet",
        arrays=_arrays(("BTCUSDT", "ETHUSDT")),
        account=_account_with(open_notional=80.0, open_position_count=4),
        metadata_by_symbol={"BTCUSDT": _metadata("BTCUSDT"), "ETHUSDT": _metadata("ETHUSDT")},
        prices_by_symbol={"BTCUSDT": 100.0, "ETHUSDT": 100.0},
        now=NOW,
        reconcile_runner=_reconcile_ok([]),
        signal_mask_builder=lambda arrays, config: signal,
        feature_builder=lambda arrays: _features(arrays),
    )

    assert result.signal_count == 2
    assert result.entries == 1
    assert [call[1] for call in broker.calls if call[0] == "market_buy"] == ["BTCUSDT"]


def test_run_live_cycle_ignores_signal_when_symbol_already_open(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    position_id = store.create_position(symbol="BTCUSDT", state=PositionState.OPEN)
    update_live_position(
        store,
        LivePositionRecord(
            position_id=position_id,
            symbol="BTCUSDT",
            state=PositionState.OPEN,
            entry_price=100.0,
            quantity=0.2,
            highest_high=110.0,
            stop_price=90.0,
            atr_at_entry=5.0,
            next_add_trigger=None,
            add_count=1,
            active_stop_client_order_id="stop",
            last_decision_open_time=None,
        ),
    )
    broker = FakeCycleBroker()
    signal = np.array([[False], [True]])

    result = run_live_cycle(
        store=store,
        broker=broker,
        config=LiveTradingConfig(),
        environment="testnet",
        arrays=_arrays(("BTCUSDT",)),
        account=_account(),
        metadata_by_symbol={"BTCUSDT": _metadata("BTCUSDT")},
        prices_by_symbol={"BTCUSDT": 100.0},
        now=NOW,
        reconcile_runner=_reconcile_ok([]),
        signal_mask_builder=lambda arrays, config: signal,
        feature_builder=lambda arrays: _features(arrays),
    )

    assert result.entries == 0
    assert not any(call[0] == "market_buy" for call in broker.calls)


def test_run_live_cycle_trails_existing_stop_upward(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    position_id = store.create_position(symbol="BTCUSDT", state=PositionState.OPEN)
    update_live_position(
        store,
        LivePositionRecord(
            position_id=position_id,
            symbol="BTCUSDT",
            state=PositionState.OPEN,
            entry_price=100.0,
            quantity=0.2,
            highest_high=100.0,
            stop_price=80.0,
            atr_at_entry=5.0,
            next_add_trigger=None,
            add_count=1,
            active_stop_client_order_id="old-stop",
            last_decision_open_time=None,
        ),
    )
    broker = FakeCycleBroker()

    result = run_live_cycle(
        store=store,
        broker=broker,
        config=LiveTradingConfig(),
        environment="testnet",
        arrays=_arrays(("BTCUSDT",)),
        account=_account(),
        metadata_by_symbol={"BTCUSDT": _metadata("BTCUSDT")},
        prices_by_symbol={"BTCUSDT": 100.0},
        now=NOW,
        reconcile_runner=_reconcile_ok([]),
        signal_mask_builder=lambda arrays, config: np.array([[False], [False]]),
        feature_builder=lambda arrays: _features(arrays),
    )

    assert result.stop_updates == 1
    assert get_live_position(store, position_id).stop_price == 95.0
    assert [call[0] for call in broker.calls] == ["place_stop_market_close", "cancel_order"]


def test_run_live_cycle_submits_pyramid_add_when_execution_price_confirms(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    position_id = store.create_position(symbol="BTCUSDT", state=PositionState.OPEN)
    update_live_position(
        store,
        LivePositionRecord(
            position_id=position_id,
            symbol="BTCUSDT",
            state=PositionState.OPEN,
            entry_price=100.0,
            quantity=0.1,
            highest_high=100.0,
            stop_price=80.0,
            atr_at_entry=5.0,
            next_add_trigger=105.0,
            add_count=0,
            active_stop_client_order_id="stop",
            last_decision_open_time=None,
        ),
    )
    broker = FakeCycleBroker()

    result = run_live_cycle(
        store=store,
        broker=broker,
        config=LiveTradingConfig(),
        environment="testnet",
        arrays=_arrays(("BTCUSDT",)),
        account=_account(),
        metadata_by_symbol={"BTCUSDT": _metadata("BTCUSDT")},
        prices_by_symbol={"BTCUSDT": 106.0},
        now=NOW,
        reconcile_runner=_reconcile_ok([]),
        signal_mask_builder=lambda arrays, config: np.array([[False], [False]]),
        feature_builder=lambda arrays: _features(arrays),
    )

    assert result.adds == 1
    assert get_live_position(store, position_id).quantity == 0.2


def test_run_live_cycle_rejects_pyramid_add_when_shared_cap_exhausted(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    position_id = store.create_position(symbol="BTCUSDT", state=PositionState.OPEN)
    update_live_position(
        store,
        LivePositionRecord(
            position_id=position_id,
            symbol="BTCUSDT",
            state=PositionState.OPEN,
            entry_price=100.0,
            quantity=0.1,
            highest_high=100.0,
            stop_price=80.0,
            atr_at_entry=5.0,
            next_add_trigger=105.0,
            add_count=0,
            active_stop_client_order_id="stop",
            last_decision_open_time=None,
        ),
    )
    broker = FakeCycleBroker()

    result = run_live_cycle(
        store=store,
        broker=broker,
        config=LiveTradingConfig(total_open_notional_cap=100.0),
        environment="testnet",
        arrays=_arrays(("BTCUSDT",)),
        account=_account_with(open_notional=98.0),
        metadata_by_symbol={"BTCUSDT": _metadata("BTCUSDT")},
        prices_by_symbol={"BTCUSDT": 106.0},
        now=NOW,
        reconcile_runner=_reconcile_ok([]),
        signal_mask_builder=lambda arrays, config: np.array([[False], [False]]),
        feature_builder=lambda arrays: _features(arrays),
    )

    assert result.adds == 0
    assert get_live_position(store, position_id).quantity == 0.1
    assert not any(call[0] == "market_buy" for call in broker.calls)
