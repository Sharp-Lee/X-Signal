from datetime import datetime, timezone
import asyncio

from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    OrderIntent,
    OrderIntentStatus,
    OrderIntentType,
    PositionState,
)
from xsignal.strategies.volume_price_efficiency_v1.live.position_store import (
    LivePositionRecord,
    get_live_position,
    update_live_position,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore
from xsignal.strategies.volume_price_efficiency_v1.live.user_data import (
    UserDataEventType,
    apply_user_data_event,
    parse_user_data_event,
    run_user_data_stream,
    user_data_ws_url_for_mode,
)


NOW = datetime(2026, 5, 9, tzinfo=timezone.utc)


def _store_with_open_position(tmp_path) -> tuple[LiveStore, str]:
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
            quantity=0.001,
            highest_high=110.0,
            stop_price=95.0,
            atr_at_entry=5.0,
            next_add_trigger=115.0,
            add_count=0,
            active_stop_client_order_id="XV1TSBTCSTOP",
            last_decision_open_time=None,
            strategy_interval="4h",
        ),
    )
    return store, position_id


def _record_intent(
    store: LiveStore,
    *,
    position_id: str,
    intent_type: OrderIntentType,
    client_order_id: str,
    status: OrderIntentStatus = OrderIntentStatus.EXCHANGE_CONFIRMED,
) -> None:
    store.record_order_intent(
        OrderIntent(
            intent_id=client_order_id,
            position_id=position_id,
            symbol="BTCUSDT",
            intent_type=intent_type,
            client_order_id=client_order_id,
            side="SELL" if intent_type in {OrderIntentType.STOP_PLACE, OrderIntentType.STOP_REPLACE} else "BUY",
            quantity=0.001,
            notional=100.0,
            price=None,
            stop_price=95.0 if intent_type in {OrderIntentType.STOP_PLACE, OrderIntentType.STOP_REPLACE} else None,
            created_at=NOW,
            status=status,
            exchange_order_id="12345",
            exchange_status="NEW",
            submitted_at=NOW,
        )
    )


def test_user_data_ws_url_uses_private_stream_hosts():
    assert user_data_ws_url_for_mode("testnet", "listen-key") == (
        "wss://stream.binancefuture.com/ws/listen-key"
    )
    assert user_data_ws_url_for_mode("live", "listen-key") == (
        "wss://fstream.binance.com/private/ws?listenKey=listen-key"
    )


def test_parse_order_trade_update_maps_strategy_client_order_id():
    event = parse_user_data_event(
        {
            "e": "ORDER_TRADE_UPDATE",
            "E": 1778313600123,
            "T": 1778313600456,
            "o": {
                "s": "BTCUSDT",
                "c": "XV1TSBTCSTOP",
                "S": "SELL",
                "o": "STOP_MARKET",
                "x": "TRADE",
                "X": "FILLED",
                "i": 987654,
                "z": "0.001",
                "ap": "95.5",
                "sp": "95.0",
                "R": True,
                "cp": True,
            },
        }
    )

    assert event.event_type == UserDataEventType.ORDER_TRADE_UPDATE
    assert event.symbol == "BTCUSDT"
    assert event.client_order_id == "XV1TSBTCSTOP"
    assert event.order_status == "FILLED"
    assert event.reduce_only is True
    assert event.close_position is True
    assert event.transaction_time == datetime(2026, 5, 9, 8, 0, 0, 456000, tzinfo=timezone.utc)


def test_apply_stop_fill_closes_position_and_resolves_intent(tmp_path):
    store, position_id = _store_with_open_position(tmp_path)
    _record_intent(
        store,
        position_id=position_id,
        intent_type=OrderIntentType.STOP_PLACE,
        client_order_id="XV1TSBTCSTOP",
    )

    result = apply_user_data_event(
        store,
        parse_user_data_event(
            {
                "e": "ORDER_TRADE_UPDATE",
                "E": 1778313600123,
                "T": 1778313600456,
                "o": {
                    "s": "BTCUSDT",
                    "c": "XV1TSBTCSTOP",
                    "S": "SELL",
                    "o": "STOP_MARKET",
                    "x": "TRADE",
                    "X": "FILLED",
                    "i": 987654,
                    "z": "0.001",
                    "ap": "95.5",
                    "R": True,
                    "cp": True,
                },
            }
        ),
    )

    assert result.intent_updated is True
    assert result.position_closed is True
    assert store.get_position_state(position_id) == PositionState.CLOSED
    intent = store.get_order_intent_by_client_order_id("XV1TSBTCSTOP")
    assert intent.status == OrderIntentStatus.RESOLVED
    assert intent.exchange_order_id == "987654"
    assert intent.exchange_status == "FILLED"
    assert intent.resolved_at == datetime(2026, 5, 9, 8, 0, 0, 456000, tzinfo=timezone.utc)
    record = get_live_position(store, position_id)
    assert record.quantity == 0.0
    assert record.active_stop_client_order_id is None


def test_apply_stop_cancel_resolves_replaced_stop_without_error_locking_position(tmp_path):
    store, position_id = _store_with_open_position(tmp_path)
    _record_intent(
        store,
        position_id=position_id,
        intent_type=OrderIntentType.STOP_PLACE,
        client_order_id="XV1TSBTCOLD",
    )

    result = apply_user_data_event(
        store,
        parse_user_data_event(
            {
                "e": "ORDER_TRADE_UPDATE",
                "E": 1778313600000,
                "T": 1778313600000,
                "o": {
                    "s": "BTCUSDT",
                    "c": "XV1TSBTCOLD",
                    "S": "SELL",
                    "o": "STOP_MARKET",
                    "x": "CANCELED",
                    "X": "CANCELED",
                    "i": 123,
                },
            }
        ),
    )

    assert result.intent_updated is True
    assert result.position_closed is False
    assert store.get_position_state(position_id) == PositionState.OPEN
    assert store.get_order_intent_by_client_order_id("XV1TSBTCOLD").status == (
        OrderIntentStatus.RESOLVED
    )


def test_apply_current_stop_cancel_error_locks_unprotected_position(tmp_path):
    store, position_id = _store_with_open_position(tmp_path)
    _record_intent(
        store,
        position_id=position_id,
        intent_type=OrderIntentType.STOP_PLACE,
        client_order_id="XV1TSBTCSTOP",
    )

    result = apply_user_data_event(
        store,
        parse_user_data_event(
            {
                "e": "ORDER_TRADE_UPDATE",
                "E": 1778313600000,
                "T": 1778313600000,
                "o": {
                    "s": "BTCUSDT",
                    "c": "XV1TSBTCSTOP",
                    "S": "SELL",
                    "o": "STOP_MARKET",
                    "x": "CANCELED",
                    "X": "CANCELED",
                    "i": 123,
                },
            }
        ),
    )

    assert result.intent_updated is True
    assert result.position_closed is False
    assert store.get_position_state(position_id) == PositionState.ERROR_LOCKED
    assert store.get_order_intent_by_client_order_id("XV1TSBTCSTOP").status == (
        OrderIntentStatus.RESOLVED
    )


def test_apply_account_update_syncs_active_position_quantity_and_entry_price(tmp_path):
    store, position_id = _store_with_open_position(tmp_path)

    result = apply_user_data_event(
        store,
        parse_user_data_event(
            {
                "e": "ACCOUNT_UPDATE",
                "E": 1778313600000,
                "T": 1778313600000,
                "a": {
                    "m": "ORDER",
                    "P": [
                        {
                            "s": "BTCUSDT",
                            "pa": "0.003",
                            "ep": "101.5",
                            "ps": "BOTH",
                        }
                    ],
                },
            }
        ),
    )

    assert result.positions_synced == 1
    record = get_live_position(store, position_id)
    assert record.state == PositionState.OPEN
    assert record.quantity == 0.003
    assert record.entry_price == 101.5


def test_apply_account_update_closes_active_position_when_exchange_amount_is_zero(tmp_path):
    store, position_id = _store_with_open_position(tmp_path)

    result = apply_user_data_event(
        store,
        parse_user_data_event(
            {
                "e": "ACCOUNT_UPDATE",
                "E": 1778313600000,
                "T": 1778313600000,
                "a": {
                    "m": "ORDER",
                    "P": [
                        {
                            "s": "BTCUSDT",
                            "pa": "0",
                            "ep": "0",
                            "ps": "BOTH",
                        }
                    ],
                },
            }
        ),
    )

    assert result.position_closed is True
    assert store.get_position_state(position_id) == PositionState.CLOSED


def test_run_user_data_stream_retries_listen_key_start_errors(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    sleeps = []

    class FailingStartBroker:
        starts = 0

        def start_user_data_stream(self):
            self.starts += 1
            raise RuntimeError("listen key denied")

    async def sleep(seconds):
        sleeps.append(seconds)
        stop_event.set()

    async def run() -> FailingStartBroker:
        broker = FailingStartBroker()
        await run_user_data_stream(
            store=store,
            broker=broker,
            mode="testnet",
            stop_event=stop_event,
            reconnect_backoff_seconds=7,
            sleep_func=sleep,
        )
        return broker

    stop_event = asyncio.Event()
    broker = asyncio.run(run())

    assert broker.starts == 1
    assert sleeps == [7]
