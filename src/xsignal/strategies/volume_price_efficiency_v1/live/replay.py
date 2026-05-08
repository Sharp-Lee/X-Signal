from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ReplaySummary:
    processed_bars: int
    accepted_signals: int
    submitted_orders: int
