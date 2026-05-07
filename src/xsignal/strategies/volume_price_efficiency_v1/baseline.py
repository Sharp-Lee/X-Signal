from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.config import (
    VolumePriceEfficiencyConfig,
)
from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.events import build_event_rows
from xsignal.strategies.volume_price_efficiency_v1.features import FeatureArrays


def _month_key(value) -> str:
    if hasattr(value, "strftime"):
        return value.strftime("%Y-%m")
    return str(value)[:7]


def _has_entry_and_forward_horizon(
    arrays: OhlcvArrays,
    *,
    t_index: int,
    s_index: int,
    horizons: tuple[int, ...],
) -> bool:
    entry_index = t_index + 1
    if entry_index >= arrays.open.shape[0]:
        return False
    entry_open = arrays.open[entry_index, s_index]
    if not np.isfinite(entry_open) or entry_open <= 0:
        return False
    for horizon in horizons:
        forward_index = t_index + horizon
        if forward_index >= arrays.close.shape[0]:
            continue
        forward_close = arrays.close[forward_index, s_index]
        if np.isfinite(forward_close) and forward_close > 0:
            return True
    return False


def build_baseline_events(
    arrays: OhlcvArrays,
    features: FeatureArrays,
    config: VolumePriceEfficiencyConfig,
) -> list[dict]:
    signal_counts: Counter[tuple[int, str]] = Counter()
    for t_index, s_index in zip(*np.nonzero(features.signal), strict=False):
        signal_counts[(int(s_index), _month_key(arrays.open_times[int(t_index)]))] += 1

    candidates: dict[tuple[int, str], list[int]] = defaultdict(list)
    for t_index in range(arrays.open.shape[0]):
        month = _month_key(arrays.open_times[t_index])
        for s_index in range(arrays.open.shape[1]):
            key = (s_index, month)
            if key not in signal_counts:
                continue
            if features.signal[t_index, s_index] or not arrays.quality[t_index, s_index]:
                continue
            if not _has_entry_and_forward_horizon(
                arrays,
                t_index=t_index,
                s_index=s_index,
                horizons=config.horizons,
            ):
                continue
            candidates[key].append(t_index)

    rng = np.random.default_rng(config.baseline_seed)
    baseline_mask = np.zeros_like(features.signal, dtype=bool)
    extra_by_index = {}
    for (s_index, month), signal_count in sorted(signal_counts.items()):
        group_candidates = candidates[(s_index, month)].copy()
        if not group_candidates:
            continue
        rng.shuffle(group_candidates)
        selected = group_candidates[:signal_count]
        for t_index in selected:
            baseline_mask[t_index, s_index] = True
            extra_by_index[(t_index, s_index)] = {
                "matched_signal_month": month,
                "matched_signal_count_for_symbol_month": signal_count,
            }

    return build_event_rows(
        arrays,
        features,
        config,
        signal_mask=baseline_mask,
        extra_by_index=extra_by_index,
    )
