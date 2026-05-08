from __future__ import annotations

import hashlib


_INTENT_CODES = {
    "ENTRY": "E",
    "PYRAMID_ADD": "A",
    "STOP_PLACE": "S",
    "STOP_REPLACE": "R",
    "STOP_EXIT_OBSERVED": "X",
    "MANUAL_RECONCILE": "M",
}


def _env_code(env: str) -> str:
    if env == "testnet":
        return "T"
    if env == "live":
        return "L"
    if env == "reconcile-only":
        return "R"
    raise ValueError(f"unsupported env: {env}")


def build_client_order_id(
    *,
    env: str,
    intent: str,
    symbol: str,
    position_id: str,
    sequence: int,
) -> str:
    if sequence < 0:
        raise ValueError("sequence must be non-negative")
    intent_code = _INTENT_CODES.get(intent)
    if intent_code is None:
        raise ValueError(f"unsupported intent: {intent}")
    digest = hashlib.sha1(f"{symbol}|{position_id}|{sequence}".encode()).hexdigest()[:14]
    symbol_code = "".join(ch for ch in symbol.upper() if ch.isalnum())[:10]
    client_id = f"XV1{_env_code(env)}{intent_code}{symbol_code}{digest}{sequence:02d}"
    if len(client_id) > 36:
        client_id = f"XV1{_env_code(env)}{intent_code}{digest}{sequence:02d}"
    return client_id
