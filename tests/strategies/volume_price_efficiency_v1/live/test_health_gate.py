from xsignal.strategies.volume_price_efficiency_v1.live.health_gate import EntryHealthGate
from xsignal.strategies.volume_price_efficiency_v1.live.reconcile import (
    ReconcileFinding,
    ReconcileStatus,
    ReconcileSummary,
)


def _summary(status: ReconcileStatus) -> ReconcileSummary:
    return ReconcileSummary(
        environment="testnet",
        allow_repair=False,
        findings=(
            ReconcileFinding(
                symbol="BTCUSDT",
                position_id=None,
                status=status,
                reason="test",
            ),
        ),
    )


def test_entry_health_gate_starts_closed_until_clean_reconcile():
    gate = EntryHealthGate()

    assert gate.allow_entries is False
    assert gate.snapshot() == {
        "allow_entries": False,
        "reasons": ["startup_reconcile_pending"],
    }

    gate.mark_reconcile(_summary(ReconcileStatus.CLEAN))

    assert gate.allow_entries is True
    assert gate.reasons == ()


def test_entry_health_gate_stays_closed_after_errors_until_clean_reconcile():
    gate = EntryHealthGate()
    gate.mark_reconcile(_summary(ReconcileStatus.CLEAN))

    gate.mark_stream_error("websocket disconnected")
    gate.mark_reconcile(_summary(ReconcileStatus.ERROR_LOCKED))

    assert gate.allow_entries is False
    assert gate.reasons == ("reconcile_error", "stream_error_since_reconcile")

    gate.mark_reconcile(_summary(ReconcileStatus.CLEAN))

    assert gate.allow_entries is True
    assert gate.reasons == ()


def test_entry_health_gate_flags_rate_limit_errors():
    gate = EntryHealthGate()
    gate.mark_reconcile(_summary(ReconcileStatus.CLEAN))

    gate.mark_stream_error("Binance API error 429 -1003: too many requests")

    assert gate.allow_entries is False
    assert gate.reasons == ("rest_rate_limited", "stream_error_since_reconcile")
