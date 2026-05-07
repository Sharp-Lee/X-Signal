from __future__ import annotations

from datetime import timedelta
from typing import Any

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.features import FeatureArrays


def _json_time(value) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _as_float_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def _event_row(
    *,
    arrays: OhlcvArrays,
    features: FeatureArrays,
    config: VolumePriceEfficiencyConfig,
    t_index: int,
    s_index: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    entry_index = t_index + 1
    if entry_index >= arrays.open.shape[0]:
        return None
    entry_price = arrays.open[entry_index, s_index]
    if not np.isfinite(entry_price) or entry_price <= 0:
        return None

    signal_open_time = arrays.open_times[t_index]
    entry_open_time = arrays.open_times[entry_index]
    row: dict[str, Any] = {
        "symbol": arrays.symbols[s_index],
        "signal_open_time": _json_time(signal_open_time),
        "decision_time": _json_time(signal_open_time + timedelta(hours=4)),
        "entry_open_time": _json_time(entry_open_time),
        "entry_price": float(entry_price),
        "move_unit": _as_float_or_none(features.move_unit[t_index, s_index]),
        "volume_unit": _as_float_or_none(features.volume_unit[t_index, s_index]),
        "efficiency": _as_float_or_none(features.efficiency[t_index, s_index]),
        "efficiency_threshold": _as_float_or_none(
            features.efficiency_threshold[t_index, s_index]
        ),
        "close_position": _as_float_or_none(features.close_position[t_index, s_index]),
        "body_ratio": _as_float_or_none(features.body_ratio[t_index, s_index]),
        "quote_volume": _as_float_or_none(arrays.quote_volume[t_index, s_index]),
        "volume_baseline": _as_float_or_none(features.volume_baseline[t_index, s_index]),
        "atr": _as_float_or_none(features.atr[t_index, s_index]),
    }
    for horizon in config.horizons:
        forward_index = t_index + horizon
        gross_key = f"forward_return_{horizon}"
        net_key = f"net_forward_return_{horizon}"
        if forward_index >= arrays.close.shape[0]:
            row[gross_key] = None
            row[net_key] = None
            continue
        forward_close = arrays.close[forward_index, s_index]
        if not np.isfinite(forward_close) or forward_close <= 0:
            row[gross_key] = None
            row[net_key] = None
            continue
        forward_return = float(forward_close / entry_price - 1.0)
        row[gross_key] = forward_return
        row[net_key] = forward_return - config.round_trip_cost
    if extra:
        row.update(extra)
    return row


def build_event_rows(
    arrays: OhlcvArrays,
    features: FeatureArrays,
    config: VolumePriceEfficiencyConfig,
    *,
    signal_mask: np.ndarray | None = None,
    extra_by_index: dict[tuple[int, int], dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    mask = features.signal if signal_mask is None else signal_mask
    rows = []
    for t_index, s_index in zip(*np.nonzero(mask), strict=False):
        row = _event_row(
            arrays=arrays,
            features=features,
            config=config,
            t_index=int(t_index),
            s_index=int(s_index),
            extra=(extra_by_index or {}).get((int(t_index), int(s_index))),
        )
        if row is not None:
            rows.append(row)
    return rows
