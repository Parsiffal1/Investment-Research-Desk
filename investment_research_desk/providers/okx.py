from __future__ import annotations

import base64
import hashlib
import hmac
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlencode

import httpx

from investment_research_desk.schemas import OHLCVBar, RunRequest


@dataclass(frozen=True)
class OkxAuth:
    api_key: str
    secret_key: str
    passphrase: str
    demo: bool = True


class OkxMarketDataProvider:
    name = "okx"

    def __init__(
        self,
        base_url: str = "https://www.okx.com",
        timeout: float = 10.0,
        api_key: str | None = None,
        secret_key: str | None = None,
        passphrase: str | None = None,
        demo: bool = True,
        read_only: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.auth = OkxAuth(api_key, secret_key, passphrase, demo) if api_key and secret_key and passphrase else None
        self.read_only = read_only

    def fetch_ohlcv(self, request: RunRequest) -> list[OHLCVBar]:
        params = {"instId": self.resolve_inst_id(request), "bar": self._bar_for_horizon(request.horizon), "limit": "100"}
        url = f"{self.base_url}/api/v5/market/candles"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
        payload = response.json()
        bars: list[OHLCVBar] = []
        for row in payload.get("data", []):
            timestamp_ms, open_, high, low, close, volume = row[:6]
            bars.append(
                OHLCVBar(
                    timestamp=datetime.fromtimestamp(int(timestamp_ms) / 1000, tz=timezone.utc),
                    open=float(open_),
                    high=float(high),
                    low=float(low),
                    close=float(close),
                    volume=float(volume),
                )
            )
        return list(reversed(bars))

    def resolve_inst_id(self, request: RunRequest) -> str:
        symbol = request.symbol.strip().upper()
        if "-" in symbol:
            return symbol
        if request.asset_class == "crypto":
            candidates = [f"{symbol}-USDT-SWAP", f"{symbol}-USD-SWAP", f"{symbol}-USDT"]
            for inst_id in candidates:
                if self._instrument_exists(inst_id):
                    return inst_id
            return candidates[0]
        return symbol

    def account_config(self) -> dict[str, Any]:
        return self._private_get("/api/v5/account/config")

    def account_balance(self, ccy: str | None = None) -> dict[str, Any]:
        params = {"ccy": ccy} if ccy else None
        return self._private_get("/api/v5/account/balance", params=params)

    def positions(self, inst_type: str | None = None, inst_id: str | None = None) -> dict[str, Any]:
        params = {key: value for key, value in {"instType": inst_type, "instId": inst_id}.items() if value}
        return self._private_get("/api/v5/account/positions", params=params or None)

    def account_position_risk(self, inst_type: str | None = None) -> dict[str, Any]:
        params = {"instType": inst_type} if inst_type else None
        return self._private_get("/api/v5/account/account-position-risk", params=params)

    def private_available(self) -> bool:
        return self.auth is not None

    def _instrument_exists(self, inst_id: str) -> bool:
        inst_type = "SWAP" if inst_id.endswith("-SWAP") else "SPOT"
        try:
            payload = self._public_get("/api/v5/public/instruments", {"instType": inst_type, "instId": inst_id})
            return bool(payload.get("data"))
        except Exception:
            return False

    def _public_get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}{path}", params=params)
            response.raise_for_status()
        return response.json()

    def _private_get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        if not self.auth:
            raise ValueError("OKX private API credentials are not configured")
        query = f"?{urlencode(params)}" if params else ""
        request_path = f"{path}{query}"
        headers = self._auth_headers("GET", request_path, "")
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}{path}", params=params, headers=headers)
            response.raise_for_status()
        payload = response.json()
        if payload.get("code") not in {None, "0"}:
            raise ValueError(f"OKX API error {payload.get('code')}: {payload.get('msg')}")
        return payload

    def _auth_headers(self, method: str, request_path: str, body: str) -> dict[str, str]:
        if not self.auth:
            raise ValueError("OKX private API credentials are not configured")
        timestamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        prehash = f"{timestamp}{method.upper()}{request_path}{body}"
        digest = hmac.new(self.auth.secret_key.encode(), prehash.encode(), hashlib.sha256).digest()
        headers = {
            "Content-Type": "application/json",
            "OK-ACCESS-KEY": self.auth.api_key,
            "OK-ACCESS-SIGN": base64.b64encode(digest).decode(),
            "OK-ACCESS-TIMESTAMP": timestamp,
            "OK-ACCESS-PASSPHRASE": self.auth.passphrase,
        }
        if self.auth.demo:
            headers["x-simulated-trading"] = "1"
        return headers

    @staticmethod
    def _bar_for_horizon(horizon: str) -> str:
        if horizon == "intraday":
            return "15m"
        if horizon == "short_term":
            return "1H"
        if horizon == "swing":
            return "4H"
        return "1D"
