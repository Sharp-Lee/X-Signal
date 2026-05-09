from decimal import Decimal

import pytest

from xsignal.strategies.volume_price_efficiency_v1.live.models import SymbolMetadata
from xsignal.strategies.volume_price_efficiency_v1.live.order_normalizer import (
    SymbolRules,
    format_decimal,
)


def _metadata(**overrides) -> SymbolMetadata:
    payload = {
        "symbol": "RAVEUSDT",
        "status": "TRADING",
        "min_notional": 5.0,
        "quantity_step": 1.0,
        "price_tick": 0.0001,
        "supports_stop_market": True,
        "trigger_protect": 0.05,
        "updated_at": None,
        "min_quantity": 1.0,
        "max_quantity": 1_000_000.0,
        "market_min_quantity": 1.0,
        "market_max_quantity": 100_000.0,
        "market_quantity_step": 1.0,
    }
    payload.update(overrides)
    return SymbolMetadata(**payload)


def test_symbol_rules_floor_low_price_market_quantity_from_notional():
    rules = SymbolRules.from_metadata(_metadata())

    quantity = rules.market_quantity_from_notional(notional=Decimal("20"), price=Decimal("0.03045"))

    assert quantity == Decimal("656")
    assert format_decimal(quantity) == "656"


def test_symbol_rules_rejects_quantity_below_market_lot_size():
    rules = SymbolRules.from_metadata(_metadata(symbol="RAVEUSDT"))

    with pytest.raises(ValueError, match="below market min quantity"):
        rules.normalize_market_quantity(Decimal("0.001"))


def test_symbol_rules_rejects_notional_below_exchange_min_notional_after_flooring():
    rules = SymbolRules.from_metadata(
        _metadata(
            symbol="BTCUSDT",
            min_notional=50.0,
            quantity_step=0.001,
            price_tick=0.1,
            min_quantity=0.001,
            market_min_quantity=0.001,
            market_quantity_step=0.001,
        )
    )

    with pytest.raises(ValueError, match="below min notional"):
        rules.market_quantity_from_notional(notional=Decimal("20"), price=Decimal("80000"))


def test_symbol_rules_floor_sub_cent_stop_price_to_tick_without_float_noise():
    rules = SymbolRules.from_metadata(
        _metadata(
            symbol="1000PEPEUSDT",
            price_tick=0.0000001,
            min_quantity=1.0,
            market_min_quantity=1.0,
            market_quantity_step=1.0,
        )
    )

    stop_price = rules.normalize_price(Decimal("0.006789876"))

    assert stop_price == Decimal("0.0067898")
    assert format_decimal(stop_price) == "0.0067898"
