from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import json
import time
from typing import Protocol
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class BinanceCredentials:
    api_key: str
    secret_key: str


@dataclass(frozen=True)
class HttpRequest:
    method: str
    url: str
    headers: dict[str, str]
    body: bytes | None


@dataclass(frozen=True)
class HttpResponse:
    status: int
    body: bytes
    headers: dict[str, str]


class Transport(Protocol):
    def send(self, request: HttpRequest) -> HttpResponse: ...


class BinanceApiError(RuntimeError):
    def __init__(self, *, status: int, code: int | None, message: str) -> None:
        super().__init__(f"Binance API error {status} {code}: {message}")
        self.status = status
        self.code = code
        self.message = message


class UrlLibTransport:
    def send(self, request: HttpRequest) -> HttpResponse:
        urllib_request = Request(
            request.url,
            data=request.body,
            headers=request.headers,
            method=request.method,
        )
        try:
            with urlopen(urllib_request, timeout=10) as response:  # noqa: S310
                return HttpResponse(
                    status=response.status,
                    body=response.read(),
                    headers=dict(response.headers.items()),
                )
        except HTTPError as exc:
            return HttpResponse(
                status=exc.code,
                body=exc.read(),
                headers=dict(exc.headers.items()),
            )


class BinanceRestClient:
    def __init__(
        self,
        *,
        base_url: str,
        credentials: BinanceCredentials | None,
        transport: Transport | None = None,
        now_ms=None,
        recv_window: int = 5000,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.credentials = credentials
        self.transport = transport or UrlLibTransport()
        self.now_ms = now_ms or (lambda: int(time.time() * 1000))
        self.recv_window = recv_window

    def request(
        self,
        method: str,
        path: str,
        *,
        signed: bool = False,
        params: dict[str, object] | None = None,
    ):
        method = method.upper()
        query_params = dict(params or {})
        headers: dict[str, str] = {}
        if signed:
            if self.credentials is None:
                raise ValueError("signed Binance request requires credentials")
            query_params["timestamp"] = self.now_ms()
            query_params["recvWindow"] = self.recv_window
            unsigned_query = urlencode(query_params)
            signature = hmac.new(
                self.credentials.secret_key.encode(),
                unsigned_query.encode(),
                hashlib.sha256,
            ).hexdigest()
            query_params["signature"] = signature
            headers["X-MBX-APIKEY"] = self.credentials.api_key
        elif self.credentials is not None:
            headers["X-MBX-APIKEY"] = self.credentials.api_key

        query = urlencode(query_params)
        url = f"{self.base_url}{path}"
        body = None
        if method in {"GET", "DELETE"}:
            if query:
                url = f"{url}?{query}"
        else:
            body = query.encode()
            headers["Content-Type"] = "application/x-www-form-urlencoded"

        response = self.transport.send(
            HttpRequest(method=method, url=url, headers=headers, body=body)
        )
        payload = self._decode_payload(response)
        if response.status >= 400:
            code = payload.get("code") if isinstance(payload, dict) else None
            message = payload.get("msg") if isinstance(payload, dict) else str(payload)
            raise BinanceApiError(status=response.status, code=code, message=message)
        return payload

    @staticmethod
    def _decode_payload(response: HttpResponse):
        if not response.body:
            return {}
        try:
            return json.loads(response.body.decode())
        except json.JSONDecodeError as exc:
            raise BinanceApiError(
                status=response.status,
                code=None,
                message=f"invalid JSON response: {exc}",
            ) from exc
