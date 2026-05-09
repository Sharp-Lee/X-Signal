from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime

import numpy as np

from xsignal.strategies.volume_price_efficiency_v1.features import compute_features
from xsignal.strategies.volume_price_efficiency_v1.live.bar_buffer import RollingBarBuffer
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
    OrderIntent,
    OrderIntentType,
    PositionState,
    SymbolMetadata,
)
from xsignal.strategies.volume_price_efficiency_v1.live.order_normalizer import SymbolRules
from xsignal.strategies.volume_price_efficiency_v1.live.position_store import (
    LivePositionRecord,
    list_active_live_positions,
    update_live_position,
)
from xsignal.strategies.volume_price_efficiency_v1.live.risk import evaluate_intent
from xsignal.strategies.volume_price_efficiency_v1.live.signal_engine import build_live_signal_mask
from xsignal.strategies.volume_price_efficiency_v1.live.store import LiveStore
from xsignal.strategies.volume_price_efficiency_v1.live.ws_market import KlineStreamEvent


@dataclass(frozen=True)
class RealtimeEventResult:
    closed_signal_checked: bool
    entries: int = 0
    stop_updates: int = 0
    adds: int = 0
    skipped_reason: str | None = None


class RealtimeStrategyService:
    def __init__(
        self,
        *,
        store: LiveStore,
        broker,
        config: LiveTradingConfig,
        environment: str,
        buffers: dict[str, RollingBarBuffer],
        metadata_by_symbol: dict[str, SymbolMetadata],
        account_provider,
        now_provider,
        feature_builder=None,
        signal_mask_builder=build_live_signal_mask,
    ) -> None:
        self.store = store
        self.broker = broker
        self.config = config
        self.environment = environment
        self.buffers = buffers
        self.metadata_by_symbol = metadata_by_symbol
        self.account_provider = account_provider
        self.now_provider = now_provider
        self.feature_builder = feature_builder
        self.signal_mask_builder = signal_mask_builder
        self._active_symbols = {record.symbol for record in list_active_live_positions(store)}

    def process_event(self, event: KlineStreamEvent) -> RealtimeEventResult:
        if event.is_closed:
            return self.process_closed_bar(
                event,
                allow_entry=True,
                allow_pyramid_add=True,
                allow_stop_replace=True,
            )
        return self.process_price_event(
            event,
            allow_pyramid_add=True,
            allow_stop_replace=True,
        )

    def process_price_event(
        self,
        event: KlineStreamEvent,
        *,
        allow_pyramid_add: bool = True,
        allow_stop_replace: bool = True,
    ) -> RealtimeEventResult:
        if not self._has_active_symbol_position(event.symbol):
            return RealtimeEventResult(False)
        metadata = self.metadata_by_symbol.get(event.symbol)
        stop_updates, adds = self._maintain_symbol_position_from_price(
            event=event,
            metadata=metadata,
            now=self.now_provider(),
            allow_pyramid_add=allow_pyramid_add,
            allow_stop_replace=allow_stop_replace,
        )
        return RealtimeEventResult(
            closed_signal_checked=False,
            stop_updates=stop_updates,
            adds=adds,
        )

    def process_closed_bar(
        self,
        event: KlineStreamEvent,
        *,
        allow_entry: bool = True,
        allow_pyramid_add: bool = True,
        allow_stop_replace: bool = True,
    ) -> RealtimeEventResult:
        buffer = self.buffers.get(event.interval)
        if buffer is None:
            return RealtimeEventResult(False, skipped_reason="interval_not_configured")
        if not event.is_closed:
            return self.process_price_event(
                event,
                allow_pyramid_add=allow_pyramid_add,
                allow_stop_replace=allow_stop_replace,
            )
        buffer.apply_event(event)
        if not allow_entry and not self._has_active_symbol_position(event.symbol):
            return RealtimeEventResult(True)
        arrays = buffer.to_arrays()
        if event.symbol not in arrays.symbols:
            return RealtimeEventResult(True, skipped_reason="event_not_in_buffer")
        symbol_index = arrays.symbols.index(event.symbol)
        if event.open_time not in set(arrays.open_times):
            return RealtimeEventResult(True, skipped_reason="event_not_in_buffer")
        time_index = list(arrays.open_times).index(event.open_time)
        features = (
            self.feature_builder(arrays)
            if self.feature_builder is not None
            else compute_features(arrays, build_vpe_live_strategy_config())
        )
        atr = features.atr[time_index, symbol_index]
        if not np.isfinite(atr) or atr <= 0:
            atr = np.nan

        metadata = self.metadata_by_symbol.get(event.symbol)
        stop_updates, adds = self._maintain_symbol_position(
            event=event,
            atr=float(atr) if np.isfinite(atr) else None,
            metadata=metadata,
            now=self.now_provider(),
            allow_pyramid_add=allow_pyramid_add,
            allow_stop_replace=allow_stop_replace,
            required_strategy_interval=event.interval,
        )

        if metadata is None:
            return RealtimeEventResult(True, stop_updates=stop_updates, adds=adds)
        if not allow_entry:
            return RealtimeEventResult(True, stop_updates=stop_updates, adds=adds)
        self._refresh_active_symbols()
        if self._has_active_symbol_position(event.symbol):
            return RealtimeEventResult(True, stop_updates=stop_updates, adds=adds)
        signal_mask = self.signal_mask_builder(arrays, self.config)
        if not signal_mask[time_index, symbol_index]:
            return RealtimeEventResult(True, stop_updates=stop_updates, adds=adds)
        if not np.isfinite(atr) or atr <= 0:
            return RealtimeEventResult(True, stop_updates=stop_updates, adds=adds)

        entry = self._try_enter(
            event=event,
            atr=float(atr),
            metadata=metadata,
            now=self.now_provider(),
            strategy_interval=event.interval,
        )
        return RealtimeEventResult(
            closed_signal_checked=True,
            entries=1 if entry else 0,
            stop_updates=stop_updates,
            adds=adds,
        )

    def _maintain_symbol_position(
        self,
        *,
        event: KlineStreamEvent,
        atr: float | None,
        metadata: SymbolMetadata | None,
        now: datetime,
        allow_pyramid_add: bool = True,
        allow_stop_replace: bool = True,
        required_strategy_interval: str | None = None,
    ) -> tuple[int, int]:
        stop_updates = 0
        adds = 0
        active = [record for record in list_active_live_positions(self.store) if record.symbol == event.symbol]
        if not active:
            self._active_symbols.discard(event.symbol)
        for record in active:
            if (
                required_strategy_interval is not None
                and record.strategy_interval is not None
                and record.strategy_interval != required_strategy_interval
            ):
                continue
            updated_record, stop_delta, add_delta = self._maintain_position_record(
                record=record,
                event=event,
                atr=atr,
                metadata=metadata,
                now=now,
                allow_pyramid_add=allow_pyramid_add,
                allow_stop_replace=allow_stop_replace,
            )
            update_live_position(self.store, updated_record)
            stop_updates += stop_delta
            adds += add_delta
        return stop_updates, adds

    def _maintain_symbol_position_from_price(
        self,
        *,
        event: KlineStreamEvent,
        metadata: SymbolMetadata | None,
        now: datetime,
        allow_pyramid_add: bool,
        allow_stop_replace: bool,
    ) -> tuple[int, int]:
        stop_updates = 0
        adds = 0
        active = [record for record in list_active_live_positions(self.store) if record.symbol == event.symbol]
        if not active:
            self._active_symbols.discard(event.symbol)
        for record in active:
            atr = self._latest_atr_for_symbol(
                interval=record.strategy_interval or event.interval,
                symbol=event.symbol,
            )
            if atr is None:
                atr = record.atr_at_entry
            updated_record, stop_delta, add_delta = self._maintain_position_record(
                record=record,
                event=event,
                atr=atr,
                metadata=metadata,
                now=now,
                allow_pyramid_add=allow_pyramid_add,
                allow_stop_replace=allow_stop_replace,
            )
            update_live_position(self.store, updated_record)
            stop_updates += stop_delta
            adds += add_delta
        return stop_updates, adds

    def _maintain_position_record(
        self,
        *,
        record: LivePositionRecord,
        event: KlineStreamEvent,
        atr: float | None,
        metadata: SymbolMetadata | None,
        now: datetime,
        allow_pyramid_add: bool,
        allow_stop_replace: bool,
    ) -> tuple[LivePositionRecord, int, int]:
        stop_updates = 0
        adds = 0
        updated_record = replace(
            record,
            highest_high=max(record.highest_high or event.high, event.high, event.close),
        )
        if allow_stop_replace and atr is not None and metadata is not None:
            candidate_stop = updated_record.highest_high - self.config.atr_multiplier * atr
            try:
                normalized_stop = float(SymbolRules.from_metadata(metadata).normalize_price(candidate_stop))
            except ValueError:
                normalized_stop = None
            if normalized_stop is not None:
                replaced = replace_trailing_stop(
                    store=self.store,
                    broker=self.broker,
                    environment=self.environment,
                    record=updated_record,
                    candidate_stop_price=normalized_stop,
                    now=now,
                )
                if replaced.stop_price != updated_record.stop_price:
                    stop_updates += 1
                updated_record = replaced
        if (
            allow_pyramid_add
            and metadata is not None
            and updated_record.next_add_trigger is not None
            and updated_record.add_count < self.config.pyramid_max_adds
            and event.high >= updated_record.next_add_trigger
            and event.close >= updated_record.next_add_trigger
        ):
            add_notional = updated_record.quantity * event.close
            account = self.account_provider()
            if _risk_accepted(
                config=self.config,
                environment=self.environment,
                intent_type=OrderIntentType.PYRAMID_ADD,
                symbol=event.symbol,
                side="BUY",
                quantity=updated_record.quantity,
                notional=add_notional,
                metadata=metadata,
                account=account,
                position_state=PositionState.OPEN,
                now=now,
            ):
                updated_record = submit_pyramid_add(
                    store=self.store,
                    broker=self.broker,
                    environment=self.environment,
                    record=updated_record,
                    quantity=updated_record.quantity,
                    execution_price=event.close,
                    now=now,
                )
                adds += 1
        return updated_record, stop_updates, adds

    def _latest_atr_for_symbol(self, *, interval: str, symbol: str) -> float | None:
        buffer = self.buffers.get(interval)
        if buffer is None:
            return None
        arrays = buffer.to_arrays()
        if symbol not in arrays.symbols or arrays.open.shape[0] == 0:
            return None
        symbol_index = arrays.symbols.index(symbol)
        features = (
            self.feature_builder(arrays)
            if self.feature_builder is not None
            else compute_features(arrays, build_vpe_live_strategy_config())
        )
        atr = features.atr[-1, symbol_index]
        if not np.isfinite(atr) or atr <= 0:
            return None
        return float(atr)

    def _try_enter(
        self,
        *,
        event: KlineStreamEvent,
        atr: float,
        metadata: SymbolMetadata,
        now: datetime,
        strategy_interval: str,
    ) -> bool:
        account = self.account_provider()
        notional = size_entry_notional(self.config, account)
        if notional <= 0:
            return False
        try:
            quantity = SymbolRules.from_metadata(metadata).market_quantity_from_notional(
                notional=notional,
                price=event.close,
            )
        except ValueError:
            return False
        entry_notional = float(quantity) * event.close
        if not _risk_accepted(
            config=self.config,
            environment=self.environment,
            intent_type=OrderIntentType.ENTRY,
            symbol=event.symbol,
            side="BUY",
            quantity=float(quantity),
            notional=entry_notional,
            metadata=metadata,
            account=account,
            position_state=PositionState.FLAT,
            now=now,
        ):
            return False
        enter_long_with_protection(
            store=self.store,
            broker=self.broker,
            config=self.config,
            environment=self.environment,
            symbol=event.symbol,
            quantity=float(quantity),
            entry_price=event.close,
            atr=atr,
            now=now,
            strategy_interval=strategy_interval,
        )
        self._active_symbols.add(event.symbol)
        return True

    def _has_active_symbol_position(self, symbol: str) -> bool:
        return symbol in self._active_symbols

    def has_active_symbol_position(self, symbol: str) -> bool:
        return self._has_active_symbol_position(symbol)

    def active_symbols(self) -> tuple[str, ...]:
        return tuple(sorted(self._active_symbols))

    def refresh_active_symbols(self) -> None:
        self._refresh_active_symbols()

    def _refresh_active_symbols(self) -> None:
        self._active_symbols = {record.symbol for record in list_active_live_positions(self.store)}


def _risk_accepted(
    *,
    config: LiveTradingConfig,
    environment: str,
    intent_type: OrderIntentType,
    symbol: str,
    side: str,
    quantity: float,
    notional: float,
    metadata: SymbolMetadata,
    account: AccountSnapshot,
    position_state: PositionState,
    now: datetime,
) -> bool:
    risk = evaluate_intent(
        config=config,
        intent=OrderIntent(
            intent_id=f"risk-check-{intent_type.value}-{environment}-{symbol}",
            position_id=f"{symbol}-risk-check",
            symbol=symbol,
            intent_type=intent_type,
            client_order_id=f"risk-check-{intent_type.value}-{environment}-{symbol}",
            side=side,
            quantity=quantity,
            notional=notional,
            price=None,
            stop_price=None,
            created_at=now,
        ),
        metadata=metadata,
        account=account,
        position_state=position_state,
        now=now,
    )
    return risk.accepted
