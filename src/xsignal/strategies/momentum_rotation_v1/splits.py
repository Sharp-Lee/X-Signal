from __future__ import annotations

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone

import numpy as np

from xsignal.strategies.momentum_rotation_v1.prepare import PreparedArrays


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
        text = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None or parsed.utcoffset() is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    raise TypeError(f"unsupported rebalance time type: {type(value)!r}")


def _slice_arrays(arrays: PreparedArrays, mask: np.ndarray) -> PreparedArrays:
    return replace(
        arrays,
        rebalance_times=arrays.rebalance_times[mask],
        close_1h=arrays.close_1h[mask],
        close_4h=arrays.close_4h[mask],
        close_1d=arrays.close_1d[mask],
        quote_volume_1d=arrays.quote_volume_1d[mask],
        complete_1h=arrays.complete_1h[mask],
        complete_4h=arrays.complete_4h[mask],
        complete_1d=arrays.complete_1d[mask],
        quality_1h_24h=arrays.quality_1h_24h[mask],
        quality_4h_7d=arrays.quality_4h_7d[mask],
        quality_1d_30d=arrays.quality_1d_30d[mask],
    )


def filter_by_rebalance_date(
    arrays: PreparedArrays,
    *,
    start_date: date | None = None,
    end_date: date | None = None,
) -> PreparedArrays:
    times = [_as_utc_datetime(value) for value in arrays.rebalance_times]
    mask = np.ones(len(times), dtype=bool)
    if start_date is not None:
        start = datetime(start_date.year, start_date.month, start_date.day, tzinfo=timezone.utc)
        mask &= np.array([value >= start for value in times], dtype=bool)
    if end_date is not None:
        end = datetime(end_date.year, end_date.month, end_date.day, tzinfo=timezone.utc)
        mask &= np.array([value < end for value in times], dtype=bool)
    return _slice_arrays(arrays, mask)


def split_research_and_holdout(
    arrays: PreparedArrays,
    *,
    holdout_days: int,
) -> tuple[PreparedArrays, PreparedArrays | None, dict[str, str | int | None]]:
    if holdout_days < 0:
        raise ValueError("holdout_days must be non-negative")
    if holdout_days == 0:
        metadata = {
            "holdout_days": 0,
            "research_start": _json_time(arrays.rebalance_times[0]) if len(arrays.rebalance_times) else None,
            "research_end": _json_time(arrays.rebalance_times[-1]) if len(arrays.rebalance_times) else None,
            "holdout_start": None,
            "holdout_end": None,
        }
        return arrays, None, metadata
    if len(arrays.rebalance_times) == 0:
        raise ValueError("cannot split empty prepared arrays")
    last_time = _as_utc_datetime(arrays.rebalance_times[-1])
    holdout_start = last_time - timedelta(days=holdout_days)
    times = np.array([_as_utc_datetime(value) for value in arrays.rebalance_times], dtype=object)
    research_mask = np.array([value < holdout_start for value in times], dtype=bool)
    holdout_mask = ~research_mask
    if not research_mask.any():
        raise ValueError("holdout window leaves no research rows")
    research = _slice_arrays(arrays, research_mask)
    holdout = _slice_arrays(arrays, holdout_mask)
    metadata = {
        "holdout_days": holdout_days,
        "research_start": _json_time(research.rebalance_times[0]),
        "research_end": _json_time(research.rebalance_times[-1]),
        "holdout_start": _json_time(holdout.rebalance_times[0]),
        "holdout_end": _json_time(holdout.rebalance_times[-1]),
    }
    return research, holdout, metadata


def _json_time(value: object) -> str:
    return _as_utc_datetime(value).isoformat().replace("+00:00", "Z")
