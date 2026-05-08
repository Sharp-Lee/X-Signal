import hashlib
import hmac
import json

import pytest

from xsignal.strategies.volume_price_efficiency_v1.live.binance_rest import (
    BinanceApiError,
    BinanceCredentials,
    BinanceRestClient,
    HttpResponse,
)


class RecordingTransport:
    def __init__(self, response: HttpResponse | None = None) -> None:
        self.response = response or HttpResponse(status=200, body=b"{}", headers={})
        self.requests = []

    def send(self, request):
        self.requests.append(request)
        return self.response


def test_signed_request_adds_timestamp_recv_window_signature_and_api_key():
    transport = RecordingTransport(HttpResponse(status=200, body=b'{"ok": true}', headers={}))
    credentials = BinanceCredentials(api_key="api-key", secret_key="secret")
    client = BinanceRestClient(
        base_url="https://testnet.binancefuture.com",
        credentials=credentials,
        transport=transport,
        now_ms=lambda: 1234567890000,
    )

    assert client.request("GET", "/fapi/v3/account", signed=True, params={"symbol": "BTCUSDT"})

    request = transport.requests[0]
    assert request.headers["X-MBX-APIKEY"] == "api-key"
    assert request.method == "GET"
    assert request.body is None
    query = request.url.split("?", 1)[1]
    unsigned = "symbol=BTCUSDT&timestamp=1234567890000&recvWindow=5000"
    signature = hmac.new(b"secret", unsigned.encode(), hashlib.sha256).hexdigest()
    assert query == f"{unsigned}&signature={signature}"


def test_unsigned_request_does_not_require_credentials():
    transport = RecordingTransport(HttpResponse(status=200, body=b'{"serverTime": 123}', headers={}))
    client = BinanceRestClient(
        base_url="https://testnet.binancefuture.com",
        credentials=None,
        transport=transport,
        now_ms=lambda: 1,
    )

    payload = client.request("GET", "/fapi/v1/time")

    assert payload == {"serverTime": 123}
    request = transport.requests[0]
    assert request.url == "https://testnet.binancefuture.com/fapi/v1/time"
    assert "X-MBX-APIKEY" not in request.headers


def test_post_request_sends_query_body():
    transport = RecordingTransport(HttpResponse(status=200, body=b'{"orderId": 1}', headers={}))
    client = BinanceRestClient(
        base_url="https://testnet.binancefuture.com",
        credentials=BinanceCredentials(api_key="api", secret_key="secret"),
        transport=transport,
        now_ms=lambda: 10,
    )

    client.request("POST", "/fapi/v1/order/test", signed=True, params={"symbol": "BTCUSDT"})

    request = transport.requests[0]
    assert request.method == "POST"
    assert request.url == "https://testnet.binancefuture.com/fapi/v1/order/test"
    assert request.body is not None
    assert b"symbol=BTCUSDT" in request.body
    assert b"signature=" in request.body


def test_binance_error_response_raises_stable_error():
    response = HttpResponse(
        status=400,
        body=json.dumps({"code": -2021, "msg": "Order would immediately trigger."}).encode(),
        headers={},
    )
    client = BinanceRestClient(
        base_url="https://testnet.binancefuture.com",
        credentials=None,
        transport=RecordingTransport(response),
        now_ms=lambda: 1,
    )

    with pytest.raises(BinanceApiError) as exc:
        client.request("GET", "/fapi/v1/time")

    assert exc.value.status == 400
    assert exc.value.code == -2021
    assert "immediately trigger" in exc.value.message
