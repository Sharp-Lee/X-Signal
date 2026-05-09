from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, ROUND_DOWN

from xsignal.strategies.volume_price_efficiency_v1.live.models import SymbolMetadata


@dataclass(frozen=True)
class SymbolRules:
    symbol: str
    min_notional: Decimal
    price_tick: Decimal
    quantity_step: Decimal
    min_quantity: Decimal
    max_quantity: Decimal | None
    market_quantity_step: Decimal
    market_min_quantity: Decimal
    market_max_quantity: Decimal | None

    @classmethod
    def from_metadata(cls, metadata: SymbolMetadata) -> "SymbolRules":
        return cls(
            symbol=metadata.symbol,
            min_notional=_to_decimal(metadata.min_notional),
            price_tick=_to_decimal(metadata.price_tick),
            quantity_step=_to_decimal(metadata.quantity_step),
            min_quantity=_to_decimal(metadata.min_quantity),
            max_quantity=(
                _to_decimal(metadata.max_quantity) if metadata.max_quantity is not None else None
            ),
            market_quantity_step=_to_decimal(
                metadata.market_quantity_step or metadata.quantity_step
            ),
            market_min_quantity=_to_decimal(metadata.market_min_quantity or metadata.min_quantity),
            market_max_quantity=(
                _to_decimal(metadata.market_max_quantity)
                if metadata.market_max_quantity is not None
                else None
            ),
        )

    def normalize_price(self, price: Decimal | float | str) -> Decimal:
        normalized = floor_to_step(_to_decimal(price), self.price_tick)
        if normalized <= 0:
            raise ValueError("price must be positive after tick normalization")
        return normalized

    def normalize_market_quantity(self, quantity: Decimal | float | str) -> Decimal:
        normalized = floor_to_step(_to_decimal(quantity), self.market_quantity_step)
        if normalized < self.market_min_quantity:
            raise ValueError("below market min quantity")
        if self.market_max_quantity is not None and normalized > self.market_max_quantity:
            raise ValueError("above market max quantity")
        return normalized

    def market_quantity_from_notional(
        self,
        *,
        notional: Decimal | float | str,
        price: Decimal | float | str,
    ) -> Decimal:
        notional_decimal = _to_decimal(notional)
        price_decimal = _to_decimal(price)
        if notional_decimal < self.min_notional:
            raise ValueError("below min notional")
        if price_decimal <= 0:
            raise ValueError("price must be positive")
        quantity = self.normalize_market_quantity(notional_decimal / price_decimal)
        if quantity * price_decimal < self.min_notional:
            raise ValueError("below min notional after market quantity normalization")
        return quantity


def floor_to_step(value: Decimal | float | str, step: Decimal | float | str) -> Decimal:
    value_decimal = _to_decimal(value)
    step_decimal = _to_decimal(step)
    if step_decimal <= 0:
        raise ValueError("step must be positive")
    units = (value_decimal / step_decimal).to_integral_value(rounding=ROUND_DOWN)
    return units * step_decimal


def format_decimal(value: Decimal | float | str) -> str:
    return format(_to_decimal(value).normalize(), "f")


def _to_decimal(value: Decimal | float | str | None) -> Decimal:
    if value is None:
        raise ValueError("decimal value is required")
    if isinstance(value, Decimal):
        return value
    if isinstance(value, float):
        return Decimal(str(value)).quantize(Decimal("0.00000001"), rounding=ROUND_DOWN)
    return Decimal(str(value))
