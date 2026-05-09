from __future__ import annotations

from xsignal.strategies.volume_price_efficiency_v1.live.reconcile import ReconcileSummary


STARTUP_RECONCILE_PENDING = "startup_reconcile_pending"
RECONCILE_ERROR = "reconcile_error"
STREAM_ERROR_SINCE_RECONCILE = "stream_error_since_reconcile"
REST_RATE_LIMITED = "rest_rate_limited"


class EntryHealthGate:
    def __init__(self) -> None:
        self._reasons: set[str] = {STARTUP_RECONCILE_PENDING}

    @property
    def allow_entries(self) -> bool:
        return not self._reasons

    @property
    def reasons(self) -> tuple[str, ...]:
        return tuple(sorted(self._reasons))

    def snapshot(self) -> dict[str, object]:
        return {
            "allow_entries": self.allow_entries,
            "reasons": list(self.reasons),
        }

    def mark_reconcile(self, summary: ReconcileSummary) -> None:
        if summary.error_count:
            self._reasons.discard(STARTUP_RECONCILE_PENDING)
            self._reasons.add(RECONCILE_ERROR)
            return
        self._reasons.clear()

    def mark_stream_error(self, error: str) -> None:
        self._reasons.add(STREAM_ERROR_SINCE_RECONCILE)
        if _is_rate_limit_error(error):
            self._reasons.add(REST_RATE_LIMITED)


def _is_rate_limit_error(error: str) -> bool:
    lowered = error.lower()
    return " 429" in f" {lowered}" or "-1003" in lowered or "too many requests" in lowered
