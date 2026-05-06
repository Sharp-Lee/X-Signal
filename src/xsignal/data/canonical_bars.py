from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True)
class TimeframeSpec:
    name: str
    minutes: int
    clickhouse_interval: str
    partition_grain: str


FIXED_TIMEFRAME_SPECS = {
    "1m": TimeframeSpec("1m", 1, "INTERVAL 1 minute", "month"),
    "3m": TimeframeSpec("3m", 3, "INTERVAL 3 minute", "month"),
    "5m": TimeframeSpec("5m", 5, "INTERVAL 5 minute", "month"),
    "15m": TimeframeSpec("15m", 15, "INTERVAL 15 minute", "month"),
    "30m": TimeframeSpec("30m", 30, "INTERVAL 30 minute", "month"),
    "1h": TimeframeSpec("1h", 60, "INTERVAL 1 hour", "month"),
    "2h": TimeframeSpec("2h", 120, "INTERVAL 2 hour", "month"),
    "4h": TimeframeSpec("4h", 240, "INTERVAL 4 hour", "month"),
    "6h": TimeframeSpec("6h", 360, "INTERVAL 6 hour", "month"),
    "8h": TimeframeSpec("8h", 480, "INTERVAL 8 hour", "month"),
    "12h": TimeframeSpec("12h", 720, "INTERVAL 12 hour", "month"),
    "1d": TimeframeSpec("1d", 1440, "INTERVAL 1 day", "year"),
    "3d": TimeframeSpec("3d", 4320, "INTERVAL 3 day", "year"),
}
SUPPORTED_TIMEFRAMES = set(FIXED_TIMEFRAME_SPECS)
EXPECTED_1M_COUNTS = {name: spec.minutes for name, spec in FIXED_TIMEFRAME_SPECS.items()}
FILL_POLICIES = {"raw", "prev_close_zero_volume"}
FillPolicy = str
CANONICAL_BAR_COLUMNS = (
    "symbol",
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "quote_volume",
    "trade_count",
    "taker_buy_volume",
    "taker_buy_quote_volume",
    "bar_count",
    "is_complete",
)


def validate_timeframe(timeframe: str) -> str:
    if timeframe not in SUPPORTED_TIMEFRAMES:
        supported = ", ".join(sorted(SUPPORTED_TIMEFRAMES))
        raise ValueError(f"Unsupported timeframe {timeframe!r}; supported: {supported}")
    return timeframe


def timeframe_spec(timeframe: str) -> TimeframeSpec:
    validate_timeframe(timeframe)
    return FIXED_TIMEFRAME_SPECS[timeframe]


def expected_1m_count(timeframe: str) -> int:
    return EXPECTED_1M_COUNTS[validate_timeframe(timeframe)]


def validate_fill_policy(fill_policy: str) -> str:
    if fill_policy not in FILL_POLICIES:
        supported = ", ".join(sorted(FILL_POLICIES))
        raise ValueError(f"Unsupported fill_policy {fill_policy!r}; supported: {supported}")
    return fill_policy


@dataclass(frozen=True)
class CanonicalRequest:
    timeframe: str
    universe: str = "all"
    range_name: str = "full_history"
    dataset_version: str = "v1"
    fill_policy: str = "raw"

    def __post_init__(self) -> None:
        validate_timeframe(self.timeframe)
        validate_fill_policy(self.fill_policy)
        if self.universe != "all":
            raise ValueError("Only universe='all' is supported for canonical exports at project start")
        if self.range_name != "full_history":
            raise ValueError("Only range_name='full_history' is supported for canonical exports at project start")


@dataclass(frozen=True)
class Partition:
    timeframe: str
    year: int
    month: int | None = None

    def __post_init__(self) -> None:
        spec = timeframe_spec(self.timeframe)
        if self.year <= 0:
            raise ValueError("Partition year must be positive")
        if spec.partition_grain == "year":
            if self.month is not None:
                raise ValueError("Yearly partitions must not include a month")
            return
        if self.month is None:
            raise ValueError("Monthly partitions require a month")
        if not 1 <= self.month <= 12:
            raise ValueError("Partition month must be between 1 and 12")

    @classmethod
    def from_datetime(cls, timeframe: str, value: datetime) -> "Partition":
        spec = timeframe_spec(timeframe)
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("Partition datetime must be timezone-aware")
        value = value.astimezone(timezone.utc)
        if spec.partition_grain == "year":
            return cls(timeframe=timeframe, year=value.year)
        return cls(timeframe=timeframe, year=value.year, month=value.month)

    @property
    def key(self) -> str:
        if self.month is None:
            return f"timeframe={self.timeframe}/year={self.year:04d}"
        return f"timeframe={self.timeframe}/year={self.year:04d}/month={self.month:02d}"


def partition_bounds(partition: Partition) -> tuple[datetime, datetime]:
    if partition.month is None:
        start = datetime(partition.year, 1, 1, tzinfo=timezone.utc)
        end = datetime(partition.year + 1, 1, 1, tzinfo=timezone.utc)
        return start, end

    start = datetime(partition.year, partition.month, 1, tzinfo=timezone.utc)
    if partition.month == 12:
        end = datetime(partition.year + 1, 1, 1, tzinfo=timezone.utc)
    else:
        end = datetime(partition.year, partition.month + 1, 1, tzinfo=timezone.utc)
    return start, end
