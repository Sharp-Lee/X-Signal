from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.live.bar_buffer import RollingBarBuffer
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
from xsignal.strategies.volume_price_efficiency_v1.live.realtime import RealtimeStrategyService
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore
from xsignal.strategies.volume_price_efficiency_v1.live.ws_market import KlineStreamEvent


NOW = datetime(2026, 5, 9, 8, tzinfo=timezone.utc)


class FakeRealtimeBroker:
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


def _metadata(symbol: str = "BTCUSDT") -> SymbolMetadata:
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


def _account(**overrides) -> AccountSnapshot:
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


def _event(
    *,
    closed: bool,
    high: float = 110.0,
    close: float = 106.0,
    interval: str = "1h",
) -> KlineStreamEvent:
    return KlineStreamEvent(
        symbol="BTCUSDT",
        interval=interval,
        event_time=NOW,
        open_time=datetime(2026, 5, 9, 8, tzinfo=timezone.utc),
        close_time=datetime(2026, 5, 9, 8, 59, 59, tzinfo=timezone.utc),
        open=100.0,
        high=high,
        low=99.0,
        close=close,
        quote_volume=1000.0,
        is_closed=closed,
    )


def _buffer(interval: str = "1h") -> RollingBarBuffer:
    buffer = RollingBarBuffer(interval=interval, max_bars=120)
    buffer.seed_rows(
        [
            {
                "symbol": "BTCUSDT",
                "interval": interval,
                "open_time": datetime(2026, 5, 9, 7, tzinfo=timezone.utc),
                "open": 99.0,
                "high": 101.0,
                "low": 98.0,
                "close": 100.0,
                "quote_volume": 1000.0,
                "is_complete": True,
            }
        ]
    )
    return buffer


def _features(arrays, atr: float = 5.0):
    return SimpleNamespace(atr=np.full(arrays.open.shape, atr))


def _service(tmp_path, *, signal_value=False, account=None):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    broker = FakeRealtimeBroker()
    service = RealtimeStrategyService(
        store=store,
        broker=broker,
        config=LiveTradingConfig(),
        environment="testnet",
        buffers={"1h": _buffer()},
        metadata_by_symbol={"BTCUSDT": _metadata()},
        account_provider=lambda: account or _account(),
        now_provider=lambda: NOW,
        feature_builder=lambda arrays: _features(arrays),
        signal_mask_builder=lambda arrays, config: np.full(arrays.open.shape, signal_value),
    )
    return service, store, broker


def test_unclosed_event_can_move_stop_and_trigger_pyramid_add(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    broker = FakeRealtimeBroker()
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
            active_stop_client_order_id="old-stop",
            last_decision_open_time=None,
        ),
    )
    service = RealtimeStrategyService(
        store=store,
        broker=broker,
        config=LiveTradingConfig(),
        environment="testnet",
        buffers={"1h": _buffer()},
        metadata_by_symbol={"BTCUSDT": _metadata()},
        account_provider=lambda: _account(),
        now_provider=lambda: NOW,
        feature_builder=lambda arrays: _features(arrays),
        signal_mask_builder=lambda arrays, config: np.full(arrays.open.shape, False),
    )

    result = service.process_event(_event(closed=False, high=110.0, close=106.0))

    assert result.closed_signal_checked is False
    assert result.entries == 0
    assert result.stop_updates == 1
    assert result.adds == 1
    assert get_live_position(store, position_id).quantity == 0.2
    assert [call[0] for call in broker.calls] == [
        "cancel_order",
        "place_stop_market_close",
        "market_buy",
    ]


def test_unclosed_event_never_opens_new_entry_even_if_signal_mask_would_be_true(tmp_path):
    service, store, broker = _service(tmp_path, signal_value=True)

    result = service.process_event(_event(closed=False, high=110.0, close=106.0))

    assert result.closed_signal_checked is False
    assert result.entries == 0
    assert not broker.calls
    assert store.list_positions_by_states([PositionState.OPEN]) == []


def test_unclosed_event_without_active_position_does_not_compute_features(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    broker = FakeRealtimeBroker()
    service = RealtimeStrategyService(
        store=store,
        broker=broker,
        config=LiveTradingConfig(),
        environment="testnet",
        buffers={"1h": _buffer()},
        metadata_by_symbol={"BTCUSDT": _metadata()},
        account_provider=lambda: _account(),
        now_provider=lambda: NOW,
        feature_builder=lambda arrays: (_ for _ in ()).throw(AssertionError("no features")),
        signal_mask_builder=lambda arrays, config: np.full(arrays.open.shape, True),
    )

    result = service.process_event(_event(closed=False, high=110.0, close=106.0))

    assert result.closed_signal_checked is False
    assert result.entries == 0
    assert result.stop_updates == 0
    assert result.adds == 0
    assert not broker.calls


def test_1m_price_event_uses_position_strategy_interval_and_can_disable_retroactive_add(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    broker = FakeRealtimeBroker()
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
            active_stop_client_order_id="old-stop",
            last_decision_open_time=None,
            strategy_interval="4h",
        ),
    )
    service = RealtimeStrategyService(
        store=store,
        broker=broker,
        config=LiveTradingConfig(),
        environment="testnet",
        buffers={"4h": _buffer("4h")},
        metadata_by_symbol={"BTCUSDT": _metadata()},
        account_provider=lambda: _account(),
        now_provider=lambda: NOW,
        feature_builder=lambda arrays: _features(arrays),
        signal_mask_builder=lambda arrays, config: np.full(arrays.open.shape, False),
    )

    result = service.process_price_event(
        _event(closed=False, interval="1m", high=110.0, close=106.0),
        allow_pyramid_add=False,
    )

    assert result.closed_signal_checked is False
    assert result.stop_updates == 1
    assert result.adds == 0
    assert get_live_position(store, position_id).highest_high == 110.0
    assert [call[0] for call in broker.calls] == ["cancel_order", "place_stop_market_close"]


def test_1m_price_event_can_update_highest_high_without_replacing_stop_during_recovery(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    broker = FakeRealtimeBroker()
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
            active_stop_client_order_id="old-stop",
            last_decision_open_time=None,
            strategy_interval="4h",
        ),
    )
    service = RealtimeStrategyService(
        store=store,
        broker=broker,
        config=LiveTradingConfig(),
        environment="testnet",
        buffers={"4h": _buffer("4h")},
        metadata_by_symbol={"BTCUSDT": _metadata()},
        account_provider=lambda: _account(),
        now_provider=lambda: NOW,
        feature_builder=lambda arrays: _features(arrays),
        signal_mask_builder=lambda arrays, config: np.full(arrays.open.shape, False),
    )

    result = service.process_price_event(
        _event(closed=False, interval="1m", high=110.0, close=106.0),
        allow_pyramid_add=False,
        allow_stop_replace=False,
    )

    assert result.stop_updates == 0
    assert result.adds == 0
    assert get_live_position(store, position_id).highest_high == 110.0
    assert not broker.calls


def test_closed_event_with_signal_opens_entry_and_protective_stop(tmp_path):
    service, store, broker = _service(tmp_path, signal_value=True)

    result = service.process_event(_event(closed=True, high=110.0, close=106.0))

    assert result.closed_signal_checked is True
    assert result.entries == 1
    assert [call[0] for call in broker.calls] == ["market_buy", "place_stop_market_close"]
    position = store.list_positions_by_states([PositionState.OPEN])[0]
    assert position["symbol"] == "BTCUSDT"
    assert get_live_position(store, position["position_id"]).strategy_interval == "1h"


def test_closed_bar_can_update_buffer_without_opening_historical_entry(tmp_path):
    service, store, broker = _service(tmp_path, signal_value=True)

    result = service.process_closed_bar(
        _event(closed=True, high=110.0, close=106.0),
        allow_entry=False,
    )

    assert result.closed_signal_checked is True
    assert result.entries == 0
    assert not broker.calls
    assert store.list_positions_by_states([PositionState.OPEN]) == []


def test_historical_closed_bar_without_active_position_does_not_compute_features(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    broker = FakeRealtimeBroker()
    service = RealtimeStrategyService(
        store=store,
        broker=broker,
        config=LiveTradingConfig(),
        environment="testnet",
        buffers={"1h": _buffer()},
        metadata_by_symbol={"BTCUSDT": _metadata()},
        account_provider=lambda: _account(),
        now_provider=lambda: NOW,
        feature_builder=lambda arrays: (_ for _ in ()).throw(AssertionError("no features")),
        signal_mask_builder=lambda arrays, config: np.full(arrays.open.shape, True),
    )

    result = service.process_closed_bar(
        _event(closed=True, high=110.0, close=106.0),
        allow_entry=False,
    )

    assert result.closed_signal_checked is True
    assert result.entries == 0
    assert not broker.calls


def test_closed_event_signal_is_rejected_when_shared_risk_is_exhausted(tmp_path):
    service, store, broker = _service(
        tmp_path,
        signal_value=True,
        account=_account(open_position_count=5),
    )

    result = service.process_event(_event(closed=True, high=110.0, close=106.0))

    assert result.closed_signal_checked is True
    assert result.entries == 0
    assert not broker.calls
    assert store.list_positions_by_states([PositionState.OPEN]) == []
