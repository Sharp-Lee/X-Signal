from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


SUPPORTED_TIMEFRAMES = {"1h", "4h", "1d"}
EXPECTED_1M_COUNTS = {
    "1h": 60,
    "4h": 240,
    "1d": 1440,
}


def validate_timeframe(timeframe: str) -> str:
    if timeframe not in SUPPORTED_TIMEFRAMES:
        supported = ", ".join(sorted(SUPPORTED_TIMEFRAMES))
        raise ValueError(f"Unsupported timeframe {timeframe!r}; supported: {supported}")
    return timeframe


def expected_1m_count(timeframe: str) -> int:
    return EXPECTED_1M_COUNTS[validate_timeframe(timeframe)]


@dataclass(frozen=True)
class CanonicalRequest:
    timeframe: str
    universe: str = "all"
    range_name: str = "full_history"
    dataset_version: str = "v1"

    def __post_init__(self) -> None:
        validate_timeframe(self.timeframe)
        if self.universe != "all":
            raise ValueError("Only universe='all' is supported for canonical exports at project start")
        if self.range_name != "full_history":
            raise ValueError("Only range_name='full_history' is supported for canonical exports at project start")


@dataclass(frozen=True)
class Partition:
    timeframe: str
    year: int
    month: int | None = None

    @classmethod
    def from_datetime(cls, timeframe: str, value: datetime) -> "Partition":
        validate_timeframe(timeframe)
        if timeframe == "1d":
            return cls(timeframe=timeframe, year=value.year)
        return cls(timeframe=timeframe, year=value.year, month=value.month)

    @property
    def key(self) -> str:
        if self.month is None:
            return f"timeframe={self.timeframe}/year={self.year:04d}"
        return f"timeframe={self.timeframe}/year={self.year:04d}/month={self.month:02d}"
