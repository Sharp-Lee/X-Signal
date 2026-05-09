from datetime import datetime, timedelta, timezone
import json

from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    OrderIntent,
    OrderIntentStatus,
    OrderIntentType,
    PositionState,
)
from xsignal.strategies.volume_price_efficiency_v1.live.status import (
    build_status_snapshot,
    collect_system_snapshot,
    parse_journal_summary,
    parse_socket_rows,
    render_status_text,
)
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore


NOW = datetime(2026, 5, 9, 12, tzinfo=timezone.utc)


def test_build_status_snapshot_summarizes_bars_cursors_and_open_risk(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    store.upsert_market_bar(
        {
            "symbol": "BTCUSDT",
            "interval": "1m",
            "open_time": NOW - timedelta(minutes=2),
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "quote_volume": 10.0,
            "is_complete": True,
        }
    )
    store.upsert_market_bar(
        {
            "symbol": "BTCUSDT",
            "interval": "1h",
            "open_time": NOW - timedelta(hours=1),
            "open": 100.0,
            "high": 110.0,
            "low": 90.0,
            "close": 106.0,
            "quote_volume": 1000.0,
            "is_complete": True,
        }
    )
    store.advance_market_cursor(
        symbol="BTCUSDT",
        interval="1m",
        open_time=NOW - timedelta(minutes=2),
    )
    position_id = store.create_position(symbol="BTCUSDT", state=PositionState.OPEN)
    store.record_order_intent(
        OrderIntent(
            intent_id="intent-1",
            position_id=position_id,
            symbol="BTCUSDT",
            intent_type=OrderIntentType.ENTRY,
            client_order_id="XV1TEBTC123",
            side="BUY",
            quantity=0.001,
            notional=20.0,
            price=None,
            stop_price=None,
            created_at=NOW,
            status=OrderIntentStatus.RESOLVED,
            exchange_status="FILLED",
            resolved_at=NOW,
        )
    )

    snapshot = build_status_snapshot(
        db_path=tmp_path / "live.sqlite",
        now=NOW,
        system_snapshot={
            "service_active": True,
            "live_service_active": False,
            "live_guard_present": False,
            "revision": "abc123",
            "sockets": [{"recv_q": 0, "send_q": 0, "peer": "1.2.3.4:443"}],
            "journal": {"reconcile_clean": 1, "stream_errors": 0, "rest_429": 0},
        },
    )

    assert snapshot["overall"] == "OK"
    assert snapshot["revision"] == "abc123"
    assert snapshot["market_bars"]["1m"]["rows"] == 1
    assert snapshot["market_bars"]["1h"]["latest_open_time"] == (NOW - timedelta(hours=1)).isoformat()
    assert snapshot["cursors"]["1m"]["symbols"] == 1
    assert snapshot["cursors"]["1m"]["max_lag_seconds"] == 120
    assert snapshot["positions"]["active"] == 1
    assert snapshot["orders"]["unresolved"] == 0


def test_status_snapshot_warns_on_unresolved_order_intents(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    position_id = store.create_position(symbol="BTCUSDT", state=PositionState.OPEN)
    store.record_order_intent(
        OrderIntent(
            intent_id="intent-1",
            position_id=position_id,
            symbol="BTCUSDT",
            intent_type=OrderIntentType.ENTRY,
            client_order_id="XV1TEBTC123",
            side="BUY",
            quantity=0.001,
            notional=20.0,
            price=None,
            stop_price=None,
            created_at=NOW,
            status=OrderIntentStatus.PENDING_SUBMIT,
        )
    )

    snapshot = build_status_snapshot(
        db_path=tmp_path / "live.sqlite",
        now=NOW,
        system_snapshot={
            "service_active": True,
            "live_service_active": False,
            "live_guard_present": False,
            "revision": "abc123",
            "sockets": [],
            "journal": {"reconcile_clean": 1, "stream_errors": 0, "rest_429": 0},
        },
    )

    assert snapshot["overall"] == "WARN"
    assert snapshot["orders"]["unresolved"] == 1
    assert "unresolved_order_intents" in snapshot["warnings"]


def test_exchange_confirmed_order_intents_are_reported_without_warning(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    position_id = store.create_position(symbol="BTCUSDT", state=PositionState.CLOSED)
    store.record_order_intent(
        OrderIntent(
            intent_id="intent-1",
            position_id=position_id,
            symbol="BTCUSDT",
            intent_type=OrderIntentType.ENTRY,
            client_order_id="XV1TEBTC123",
            side="BUY",
            quantity=0.001,
            notional=20.0,
            price=None,
            stop_price=None,
            created_at=NOW,
            status=OrderIntentStatus.EXCHANGE_CONFIRMED,
            exchange_status="FILLED",
            submitted_at=NOW,
        )
    )

    snapshot = build_status_snapshot(
        db_path=tmp_path / "live.sqlite",
        now=NOW,
        system_snapshot={
            "service_active": True,
            "live_service_active": False,
            "live_guard_present": False,
            "revision": "abc123",
            "sockets": [],
            "journal": {"reconcile_clean": 1, "stream_errors": 0, "rest_429": 0},
        },
    )

    assert snapshot["overall"] == "OK"
    assert snapshot["orders"]["exchange_confirmed"] == 1
    assert snapshot["orders"]["unresolved"] == 0


def test_status_snapshot_warns_on_socket_queue_and_reconcile_errors(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()

    snapshot = build_status_snapshot(
        db_path=tmp_path / "live.sqlite",
        now=NOW,
        system_snapshot={
            "service_active": True,
            "live_service_active": False,
            "live_guard_present": False,
            "revision": "abc123",
            "sockets": [{"recv_q": 15, "send_q": 0, "peer": "1.2.3.4:443"}],
            "journal": {"reconcile_clean": 0, "stream_errors": 2, "rest_429": 1},
        },
    )

    assert snapshot["overall"] == "WARN"
    assert "socket_queue_nonzero" in snapshot["warnings"]
    assert "recent_rest_429" in snapshot["warnings"]
    assert "no_recent_clean_reconcile" in snapshot["warnings"]


def test_parse_socket_rows_extracts_recv_send_and_peer():
    rows = parse_socket_rows(
        """
ESTAB 0      0           10.8.0.3:47022   35.73.243.130:443  users:(("xsignal",pid=1,fd=9))
ESTAB 47216  3           10.8.0.3:52206   35.73.243.130:443  users:(("xsignal",pid=1,fd=7))
        """
    )

    assert rows == [
        {
            "state": "ESTAB",
            "recv_q": 0,
            "send_q": 0,
            "local": "10.8.0.3:47022",
            "peer": "35.73.243.130:443",
        },
        {
            "state": "ESTAB",
            "recv_q": 47216,
            "send_q": 3,
            "local": "10.8.0.3:52206",
            "peer": "35.73.243.130:443",
        },
    ]


def test_parse_journal_summary_uses_latest_service_start_window():
    summary = parse_journal_summary(
        """
May 09 21:05:46 host xsignal-vpe-live[1]: {"event": "stream_error", "error": "old"}
May 09 21:06:39 host systemd[1]: Started xsignal-vpe-testnet-stream-daemon.service - X-Signal VPE testnet realtime WebSocket trading daemon.
May 09 21:06:40 host xsignal-vpe-live[2]: {"entry_gate": {"allow_entries": true, "reasons": []}, "errors": 0, "event": "reconcile_pass", "status": "clean"}
May 09 21:06:41 host xsignal-vpe-live[2]: {"event": "stream_connected", "url": "wss://stream.binancefuture.com/stream"}
        """
    )

    assert summary["stream_errors"] == 0
    assert summary["reconcile_clean"] == 1
    assert summary["stream_connected"] == 1


def test_collect_system_snapshot_degrades_when_system_commands_are_unavailable(tmp_path):
    def missing_runner(*args, **kwargs):
        raise FileNotFoundError("systemctl")

    snapshot = collect_system_snapshot(
        revision_file=tmp_path / "DEPLOY_REVISION",
        live_guard_file=tmp_path / "enable-live-trading",
        runner=missing_runner,
    )

    assert snapshot["system_available"] is False
    assert snapshot["service_active"] is False
    assert snapshot["sockets"] == []


def test_render_status_text_contains_operator_sections(tmp_path):
    store = LiveStore.open(tmp_path / "live.sqlite")
    store.initialize()
    snapshot = build_status_snapshot(
        db_path=tmp_path / "live.sqlite",
        now=NOW,
        system_snapshot={
            "service_active": True,
            "live_service_active": False,
            "live_guard_present": False,
            "revision": "abc123",
            "sockets": [],
            "journal": {"reconcile_clean": 1, "stream_errors": 0, "rest_429": 0},
        },
    )

    text = render_status_text(snapshot)

    assert "OVERALL OK" in text
    assert "SERVICE active=True" in text
    assert "ACTIVE_POSITIONS 0" in text
    assert json.dumps(snapshot, sort_keys=True) not in text
