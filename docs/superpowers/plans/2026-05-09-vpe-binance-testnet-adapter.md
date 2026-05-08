# VPE Binance Testnet Adapter Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Binance USD-M Futures testnet broker adapter for the VPE live core, with signed REST requests, exchange/account parsers, order methods, and a guarded CLI smoke command.

**Architecture:** Keep all Binance I/O inside the live broker adapter boundary. The strategy, signal engine, state machine, capital allocator, and risk gate still do not know about Binance REST. The adapter uses a small standard-library REST client with injectable transport so unit tests never call Binance.

**Tech Stack:** Python 3.12 stdlib `hmac`, `hashlib`, `urllib`, `json`, `dataclasses`, existing live-core models, `pytest`, `ruff`.

---

## Source References

Implementation must follow the current official Binance USD-M Futures docs:

- General info: testnet REST base URL, API-key header, HMAC SHA256 signatures, timestamp/recvWindow rules.
- New Order: `POST /fapi/v1/order`, `MARKET` requires `quantity`, `newClientOrderId` max 36 chars, One-way Mode default `positionSide=BOTH`.
- Exchange Information: `GET /fapi/v1/exchangeInfo`, use `PRICE_FILTER.tickSize`, `LOT_SIZE.stepSize`, `MIN_NOTIONAL.notional`, `orderTypes`, and `triggerProtect`.
- Account Information V3: `GET /fapi/v3/account`, use `totalMarginBalance` and `availableBalance`.
- Position mode and multi-assets mode: `GET /fapi/v1/positionSide/dual`, `GET /fapi/v1/multiAssetsMargin`.
- Margin/leverage setup: `POST /fapi/v1/marginType`, `POST /fapi/v1/leverage`.
- Cancel/open order: `DELETE /fapi/v1/order`, `GET /fapi/v1/openOrder`.

## File Structure

- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/binance_rest.py`
  - Credentials, HTTP transport protocol, signed query builder, JSON request method, Binance error class.
- Create `src/xsignal/strategies/volume_price_efficiency_v1/live/binance_adapter.py`
  - Testnet broker adapter, exchange/account parsers, setup and order methods.
- Modify `src/xsignal/strategies/volume_price_efficiency_v1/live/cli.py`
  - Add `testnet-smoke` command that can run read-only checks and optional `/order/test` validation.
- Modify `README.md`
  - Document env vars and safe smoke command.
- Create tests:
  - `tests/strategies/volume_price_efficiency_v1/live/test_binance_rest.py`
  - `tests/strategies/volume_price_efficiency_v1/live/test_binance_adapter.py`
  - `tests/strategies/volume_price_efficiency_v1/live/test_binance_cli.py`

---

### Task 1: Signed REST Client

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/binance_rest.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_binance_rest.py`

- [ ] Write tests for HMAC signing, timestamp/recvWindow inclusion, API-key header, unsigned exchangeInfo request, and Binance JSON error handling.
- [ ] Implement `BinanceCredentials`, `HttpRequest`, `HttpResponse`, `Transport`, `UrlLibTransport`, `BinanceRestClient`, and `BinanceApiError`.
- [ ] Verify with:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/live/test_binance_rest.py -q
```

- [ ] Commit:

```bash
git add src/xsignal/strategies/volume_price_efficiency_v1/live/binance_rest.py tests/strategies/volume_price_efficiency_v1/live/test_binance_rest.py
git commit -m "feat: add binance signed rest client"
```

### Task 2: Exchange And Account Parsers

**Files:**
- Create: `src/xsignal/strategies/volume_price_efficiency_v1/live/binance_adapter.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_binance_adapter.py`

- [ ] Write tests that parse one `exchangeInfo` symbol into `SymbolMetadata`.
- [ ] Write tests that parse account V3 single-asset response into `AccountSnapshot`.
- [ ] Write tests that reject symbols missing `STOP_MARKET`, `LOT_SIZE`, `PRICE_FILTER`, or `MIN_NOTIONAL`.
- [ ] Implement `parse_symbol_metadata()` and `parse_account_snapshot()`.
- [ ] Verify with:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/live/test_binance_adapter.py -q
```

- [ ] Commit:

```bash
git add src/xsignal/strategies/volume_price_efficiency_v1/live/binance_adapter.py tests/strategies/volume_price_efficiency_v1/live/test_binance_adapter.py
git commit -m "feat: parse binance futures metadata"
```

### Task 3: Binance Testnet Broker Methods

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/binance_adapter.py`
- Modify: `tests/strategies/volume_price_efficiency_v1/live/test_binance_adapter.py`

- [ ] Write tests with a fake REST client for:
  - `get_position_mode()` maps `dualSidePosition=false` to `one_way`.
  - `get_multi_assets_mode()` maps `multiAssetsMargin=false` to `single_asset_usdt`.
  - `change_margin_type()` sends `ISOLATED`.
  - `change_leverage()` sends leverage `1`.
  - `market_buy()` sends `BUY`, `MARKET`, quantity, and compact client id.
  - `place_stop_market_close()` sends `SELL`, `STOP_MARKET`, `stopPrice`, `closePosition=true`, `workingType=CONTRACT_PRICE`, and no explicit quantity.
  - `cancel_order()` cancels by `origClientOrderId`.
  - `test_order()` hits `/fapi/v1/order/test`.
- [ ] Implement `BinanceUsdFuturesTestnetBroker`.
- [ ] Verify with:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/live/test_binance_adapter.py -q
```

- [ ] Commit:

```bash
git add src/xsignal/strategies/volume_price_efficiency_v1/live/binance_adapter.py tests/strategies/volume_price_efficiency_v1/live/test_binance_adapter.py
git commit -m "feat: add binance testnet broker adapter"
```

### Task 4: Guarded Testnet Smoke CLI

**Files:**
- Modify: `src/xsignal/strategies/volume_price_efficiency_v1/live/cli.py`
- Test: `tests/strategies/volume_price_efficiency_v1/live/test_binance_cli.py`

- [ ] Write parser tests for `testnet-smoke --symbol BTCUSDT`.
- [ ] Write tests that missing env keys produce a clear non-zero result.
- [ ] Implement read-only smoke:
  - create testnet REST client from `BINANCE_API_KEY` and `BINANCE_SECRET_KEY`
  - fetch server time through `GET /fapi/v1/time`
  - fetch `exchangeInfo`
  - fetch account position mode and multi-assets mode
  - fetch account snapshot
  - print concise JSON-safe status
- [ ] Implement optional `--submit-test-order` using `/fapi/v1/order/test`, never real matching-engine order submission.
- [ ] Verify with:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/live/test_binance_cli.py -q
```

- [ ] Commit:

```bash
git add src/xsignal/strategies/volume_price_efficiency_v1/live/cli.py tests/strategies/volume_price_efficiency_v1/live/test_binance_cli.py
git commit -m "feat: add binance testnet smoke cli"
```

### Task 5: Documentation And Full Verification

**Files:**
- Modify: `README.md`

- [ ] Document:
  - `BINANCE_API_KEY`
  - `BINANCE_SECRET_KEY`
  - testnet-only smoke command
  - `--submit-test-order` uses `/order/test` and does not enter the matching engine
  - real testnet lifecycle trading is the next phase
- [ ] Run:

```bash
.venv/bin/python -m pytest tests/strategies/volume_price_efficiency_v1/live -q
.venv/bin/python -m pytest -q
.venv/bin/ruff check .
```

- [ ] Commit:

```bash
git add README.md
git commit -m "docs: document binance testnet smoke workflow"
```

## Plan Self-Review

- Production order submission is still not enabled.
- CLI order placement is limited to Binance `/fapi/v1/order/test`.
- All network code has injectable transport and unit tests.
- Real matching-engine testnet lifecycle trading remains a separate follow-up.
