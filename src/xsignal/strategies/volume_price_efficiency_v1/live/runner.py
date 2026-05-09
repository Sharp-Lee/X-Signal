from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.data import OhlcvArrays
from xsignal.strategies.volume_price_efficiency_v1.features import compute_features
from xsignal.strategies.volume_price_efficiency_v1.live.capital import size_entry_notional
from xsignal.strategies.volume_price_efficiency_v1.live.config import (
    LiveTradingConfig,
    build_vpe_live_strategy_config,
)
from xsignal.strategies.volume_price_efficiency_v1.live.execution import (
    enter_long_with_protection,
    replace_trailing_stop,
    submit_pyramid_add,
)
from xsignal.strategies.volume_price_efficiency_v1.live.models import (
    AccountSnapshot,
    SymbolMetadata,
)
from xsignal.strategies.volume_price_efficiency_v1.live.order_normalizer import SymbolRules
from xsignal.strategies.volume_price_efficiency_v1.live.position_store import (
    list_active_live_positions,
    update_live_position,
)
from xsignal.strategies.volume_price_efficiency_v1.live.reconcile import run_reconciliation_pass
from xsignal.strategies.volume_price_efficiency_v1.live.signal_engine import build_live_signal_mask
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore


@dataclass(frozen=True)
class LiveCycleResult:
    scanned_symbols: int
    signal_count: int
    entries: int
    stop_updates: int
    adds: int
    blocked: bool = False
    reason: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "scanned_symbols": self.scanned_symbols,
            "signal_count": self.signal_count,
            "entries": self.entries,
            "stop_updates": self.stop_updates,
            "adds": self.adds,
            "blocked": self.blocked,
            "reason": self.reason,
        }


def run_live_cycle(
    *,
    store: LiveStore,
    broker,
    config: LiveTradingConfig,
    environment: str,
    arrays: OhlcvArrays,
    account: AccountSnapshot,
    metadata_by_symbol: dict[str, SymbolMetadata],
    prices_by_symbol: dict[str, float],
    now: datetime,
    reconcile_runner=run_reconciliation_pass,
    signal_mask_builder=build_live_signal_mask,
    feature_builder=None,
) -> LiveCycleResult:
    symbols = list(arrays.symbols)
    reconcile_summary = reconcile_runner(
        store=store,
        broker=broker,
        symbols=symbols,
        environment=environment,
        allow_repair=False,
        now=now,
    )
    if reconcile_summary.error_count:
        return LiveCycleResult(
            scanned_symbols=len(symbols),
            signal_count=0,
            entries=0,
            stop_updates=0,
            adds=0,
            blocked=True,
            reason="reconciliation_error",
        )

    features = (
        feature_builder(arrays)
        if feature_builder is not None
        else compute_features(arrays, build_vpe_live_strategy_config())
    )
    signal_mask = signal_mask_builder(arrays, config)
    latest_index = arrays.open.shape[0] - 1
    latest_signals = signal_mask[latest_index]
    active_by_symbol = {record.symbol: record for record in list_active_live_positions(store)}

    stop_updates = 0
    adds = 0
    for symbol, record in list(active_by_symbol.items()):
        if symbol not in arrays.symbols:
            continue
        s_index = symbols.index(symbol)
        bar_high = arrays.high[latest_index, s_index]
        atr = features.atr[latest_index, s_index]
        updated_record = record
        if np.isfinite(bar_high):
            updated_record = replace(
                updated_record,
                highest_high=max(record.highest_high or float(bar_high), float(bar_high)),
            )
        if np.isfinite(atr) and atr > 0 and updated_record.highest_high is not None:
            candidate_stop = updated_record.highest_high - config.atr_multiplier * float(atr)
            replaced = replace_trailing_stop(
                store=store,
                broker=broker,
                environment=environment,
                record=updated_record,
                candidate_stop_price=float(candidate_stop),
                now=now,
            )
            if replaced.stop_price != updated_record.stop_price:
                stop_updates += 1
            updated_record = replaced
        if (
            updated_record.next_add_trigger is not None
            and updated_record.add_count < config.pyramid_max_adds
            and np.isfinite(bar_high)
            and bar_high >= updated_record.next_add_trigger
            and prices_by_symbol.get(symbol, 0.0) >= updated_record.next_add_trigger
        ):
            updated_record = submit_pyramid_add(
                store=store,
                broker=broker,
                environment=environment,
                record=updated_record,
                quantity=updated_record.quantity,
                execution_price=prices_by_symbol[symbol],
                now=now,
            )
            adds += 1
        updated_record = replace(
            updated_record,
            last_decision_open_time=arrays.open_times[latest_index],
        )
        update_live_position(store, updated_record)

    entries = 0
    for s_index, symbol in enumerate(symbols):
        if not latest_signals[s_index] or symbol in active_by_symbol:
            continue
        metadata = metadata_by_symbol.get(symbol)
        price = prices_by_symbol.get(symbol)
        atr = features.atr[latest_index, s_index]
        if metadata is None or price is None or not np.isfinite(atr) or atr <= 0:
            continue
        notional = size_entry_notional(config, account)
        if notional <= 0:
            continue
        try:
            quantity = SymbolRules.from_metadata(metadata).market_quantity_from_notional(
                notional=notional,
                price=price,
            )
        except ValueError:
            continue
        enter_long_with_protection(
            store=store,
            broker=broker,
            config=config,
            environment=environment,
            symbol=symbol,
            quantity=float(quantity),
            entry_price=price,
            atr=float(atr),
            now=now,
        )
        entries += 1
        active_by_symbol[symbol] = list_active_live_positions(store)[-1]

    return LiveCycleResult(
        scanned_symbols=len(symbols),
        signal_count=int(np.count_nonzero(latest_signals)),
        entries=entries,
        stop_updates=stop_updates,
        adds=adds,
    )
