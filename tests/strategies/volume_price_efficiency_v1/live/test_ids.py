import re

from xsignal.strategies.volume_price_efficiency_v1.live.ids import build_client_order_id


def test_client_order_id_is_deterministic_and_short():
    first = build_client_order_id(
        env="testnet",
        intent="PYRAMID_ADD",
        symbol="1000RATSUSDT",
        position_id="position-with-a-very-long-id",
        sequence=12,
    )
    second = build_client_order_id(
        env="testnet",
        intent="PYRAMID_ADD",
        symbol="1000RATSUSDT",
        position_id="position-with-a-very-long-id",
        sequence=12,
    )
    assert first == second
    assert len(first) <= 36
    assert first.startswith("XV1T")


def test_client_order_id_changes_for_different_sequence():
    first = build_client_order_id(
        env="testnet",
        intent="ENTRY",
        symbol="BTCUSDT",
        position_id="abc",
        sequence=1,
    )
    second = build_client_order_id(
        env="testnet",
        intent="ENTRY",
        symbol="BTCUSDT",
        position_id="abc",
        sequence=2,
    )
    assert first != second


def test_client_order_id_uses_only_binance_allowed_ascii_characters_for_chinese_symbols():
    client_order_id = build_client_order_id(
        env="testnet",
        intent="ENTRY",
        symbol="币安人生USDT",
        position_id="abc",
        sequence=1,
    )

    assert re.fullmatch(r"[\.\w\:/-]{1,36}", client_order_id, flags=re.ASCII)
    assert "币" not in client_order_id
