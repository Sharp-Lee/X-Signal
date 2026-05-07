from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays


def _as_utc_datetime(value: object) -> datetime:
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)
    if isinstance(value, date):
        return datetime(value.year, value.month, value.day, tzinfo=timezone.utc)
    if isinstance(value, np.datetime64):
        seconds = value.astype("datetime64[s]").astype("int64")
        return datetime.fromtimestamp(int(seconds), tz=timezone.utc)
    if isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    raise TypeError(f"unsupported open_time type: {type(value)!r}")


def _json_time(value: object) -> str:
    return _as_utc_datetime(value).isoformat().replace("+00:00", "Z")


def _slice_arrays(arrays: OhlcvArrays, mask: np.ndarray) -> OhlcvArrays:
    return replace(
        arrays,
        open_times=arrays.open_times[mask],
        open=arrays.open[mask],
        high=arrays.high[mask],
        low=arrays.low[mask],
        close=arrays.close[mask],
        quote_volume=arrays.quote_volume[mask],
        quality=arrays.quality[mask],
    )


def holdout_mask_for_open_times(open_times: np.ndarray, *, holdout_days: int) -> np.ndarray:
    if holdout_days <= 0:
        raise ValueError("holdout_days must be positive")
    if len(open_times) == 0:
        raise ValueError("cannot split empty OHLCV arrays")
    last_time = _as_utc_datetime(open_times[-1])
    holdout_start = last_time - timedelta(days=holdout_days)
    times = np.array([_as_utc_datetime(value) for value in open_times], dtype=object)
    return np.array([value >= holdout_start for value in times], dtype=bool)


def split_research_and_holdout(
    arrays: OhlcvArrays,
    *,
    holdout_days: int,
) -> tuple[OhlcvArrays, OhlcvArrays | None, dict[str, str | int | None]]:
    if holdout_days < 0:
        raise ValueError("holdout_days must be non-negative")
    if holdout_days == 0:
        metadata = {
            "holdout_days": 0,
            "research_start": _json_time(arrays.open_times[0]) if len(arrays.open_times) else None,
            "research_end": _json_time(arrays.open_times[-1]) if len(arrays.open_times) else None,
            "holdout_start": None,
            "holdout_end": None,
        }
        return arrays, None, metadata
    if len(arrays.open_times) == 0:
        raise ValueError("cannot split empty OHLCV arrays")

    holdout_mask = holdout_mask_for_open_times(arrays.open_times, holdout_days=holdout_days)
    research_mask = ~holdout_mask
    if not research_mask.any():
        raise ValueError("holdout window leaves no research rows")

    research = _slice_arrays(arrays, research_mask)
    holdout = _slice_arrays(arrays, holdout_mask)
    metadata = {
        "holdout_days": holdout_days,
        "research_start": _json_time(research.open_times[0]),
        "research_end": _json_time(research.open_times[-1]),
        "holdout_start": _json_time(holdout.open_times[0]),
        "holdout_end": _json_time(holdout.open_times[-1]),
    }
    return research, holdout, metadata
