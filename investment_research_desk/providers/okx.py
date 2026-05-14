from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import httpx

from investment_research_desk.schemas import OHLCVBar, RunRequest


class OkxMarketDataProvider:
    name = "okx"

    def __init__(self, base_url: str = "https://www.okx.com", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch_ohlcv(self, request: RunRequest) -> list[OHLCVBar]:
        params = {"instId": self.resolve_inst_id(request), "bar": self._bar_for_horizon(request.horizon), "limit": "100"}
        payload = self._public_get("/api/v5/market/candles", params)
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

    def fetch_swap_market_context(self, request: RunRequest) -> dict[str, Any]:
        inst_id = self.resolve_inst_id(request)
        if not inst_id.endswith("-SWAP") or not self._instrument_exists(inst_id):
            return {}
        index_inst_id = inst_id.removesuffix("-SWAP")
        context: dict[str, Any] = {
            "provider": self.name,
            "scope": "public_swap_market_only",
            "requested_symbol": request.symbol,
            "inst_id": inst_id,
            "warnings": [],
        }
        context["instrument"] = self._first_data(
            "/api/v5/public/instruments",
            {"instType": "SWAP", "instId": inst_id},
            context["warnings"],
        )
        context["ticker"] = self._first_data("/api/v5/market/ticker", {"instId": inst_id}, context["warnings"])
        context["mark_price"] = self._first_data(
            "/api/v5/public/mark-price",
            {"instType": "SWAP", "instId": inst_id},
            context["warnings"],
        )
        context["index_ticker"] = self._first_data(
            "/api/v5/market/index-ticker",
            {"instId": index_inst_id},
            context["warnings"],
        )
        context["funding_rate"] = self._first_data("/api/v5/public/funding-rate", {"instId": inst_id}, context["warnings"])
        context["funding_rate_history"] = self._data(
            "/api/v5/public/funding-rate-history",
            {"instId": inst_id, "limit": "20"},
            context["warnings"],
        )
        context["open_interest"] = self._first_data(
            "/api/v5/public/open-interest",
            {"instType": "SWAP", "instId": inst_id},
            context["warnings"],
        )
        context["price_limit"] = self._first_data("/api/v5/public/price-limit", {"instId": inst_id}, context["warnings"])
        orderbook = self._first_data("/api/v5/market/books", {"instId": inst_id, "sz": "25"}, context["warnings"])
        context["orderbook"] = orderbook
        context["orderbook_imbalance"] = _orderbook_imbalance(orderbook)
        context["recent_trades"] = self._data("/api/v5/market/trades", {"instId": inst_id, "limit": "50"}, context["warnings"])
        context["mark_index_spread"] = _mark_index_spread(context.get("mark_price"), context.get("index_ticker"))
        return context

    def resolve_inst_id(self, request: RunRequest) -> str:
        symbol = request.symbol.strip().upper()
        if symbol.endswith("-SWAP"):
            return symbol
        if "-" in symbol:
            candidate = f"{symbol}-SWAP"
            if self._instrument_exists(candidate):
                return candidate
            return symbol
        if request.asset_class == "crypto":
            candidates = [f"{symbol}-USDT-SWAP", f"{symbol}-USD-SWAP"]
            for inst_id in candidates:
                if self._instrument_exists(inst_id):
                    return inst_id
            return candidates[0]
        return symbol

    def _instrument_exists(self, inst_id: str) -> bool:
        try:
            payload = self._public_get("/api/v5/public/instruments", {"instType": "SWAP", "instId": inst_id})
            return bool(payload.get("data"))
        except Exception:
            return False

    def _public_get(self, path: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        with httpx.Client(timeout=self.timeout, headers={"User-Agent": "investment-research-desk/0.1"}) as client:
            response = client.get(f"{self.base_url}{path}", params=params)
            response.raise_for_status()
        payload = response.json()
        if payload.get("code") not in {None, "0"}:
            raise ValueError(f"OKX API error {payload.get('code')}: {payload.get('msg')}")
        return payload

    def _safe_public_get(self, path: str, params: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
        try:
            return self._public_get(path, params)
        except Exception as exc:
            warnings.append(f"{path} failed: {exc}")
            return {"data": []}

    def _data(self, path: str, params: dict[str, Any], warnings: list[str]) -> list[dict[str, Any]]:
        payload = self._safe_public_get(path, params, warnings)
        data = payload.get("data")
        return data if isinstance(data, list) else []

    def _first_data(self, path: str, params: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
        data = self._data(path, params, warnings)
        return data[0] if data and isinstance(data[0], dict) else {}

    @staticmethod
    def _bar_for_horizon(horizon: str) -> str:
        if horizon == "intraday":
            return "15m"
        if horizon == "short_term":
            return "1H"
        if horizon == "swing":
            return "4H"
        return "1D"


def _orderbook_imbalance(orderbook: dict[str, Any]) -> float | None:
    bids = orderbook.get("bids") or []
    asks = orderbook.get("asks") or []
    try:
        bid_size = sum(float(row[1]) for row in bids)
        ask_size = sum(float(row[1]) for row in asks)
    except Exception:
        return None
    total = bid_size + ask_size
    if total == 0:
        return None
    return round((bid_size - ask_size) / total, 4)


def _mark_index_spread(mark_price: dict[str, Any] | None, index_ticker: dict[str, Any] | None) -> float | None:
    try:
        mark = float((mark_price or {}).get("markPx"))
        index = float((index_ticker or {}).get("idxPx"))
    except Exception:
        return None
    if index == 0:
        return None
    return round((mark - index) / index, 6)
