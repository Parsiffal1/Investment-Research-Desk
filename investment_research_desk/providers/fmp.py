from __future__ import annotations

from datetime import datetime, timezone

import httpx

from investment_research_desk.schemas import NewsEvent, OHLCVBar, RunRequest


class FmpProvider:
    name = "fmp"

    def __init__(self, api_key: str | None, base_url: str = "https://financialmodelingprep.com/stable", timeout: float = 15.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def quote(self, symbol: str) -> dict | None:
        if not self.api_key:
            return None
        data = self._get("quote", {"symbol": symbol})
        if isinstance(data, list) and data:
            return data[0]
        return None

    def profile(self, symbol: str) -> dict | None:
        if not self.api_key:
            return None
        data = self._get("profile", {"symbol": symbol})
        if isinstance(data, list) and data:
            return data[0]
        return None

    def fetch_news(self, request: RunRequest) -> list[NewsEvent]:
        if not self.api_key:
            return []
        symbol = _news_symbol(request)
        endpoint = _news_endpoint(request.asset_class)
        if not symbol or not endpoint:
            return []
        data = self._get(endpoint, {"symbols": symbol})
        if not isinstance(data, list):
            return []
        events: list[NewsEvent] = []
        for item in data[:10]:
            title = item.get("title") or item.get("headline")
            if not title:
                continue
            published = item.get("publishedDate") or item.get("date") or item.get("published_at")
            try:
                published_at = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
                if published_at.tzinfo is None:
                    published_at = published_at.replace(tzinfo=timezone.utc)
            except Exception:
                published_at = datetime.now(timezone.utc)
            events.append(
                NewsEvent(
                    title=title,
                    summary=item.get("text") or item.get("site") or item.get("summary"),
                    source=f"fmp:{item.get('site') or 'news'}",
                    published_at=published_at,
                    url=item.get("url"),
                    event_type=_news_event_type(request.asset_class),
                    related_assets=[symbol],
                )
            )
        return events

    def fetch_ohlcv(self, request: RunRequest) -> list[OHLCVBar]:
        if not self.api_key:
            return []
        symbol = _normalize_equity_symbol(request.symbol)
        data = self._get("historical-price-eod/light", {"symbol": symbol})
        if not isinstance(data, list):
            return []
        bars: list[OHLCVBar] = []
        for row in data[:100]:
            try:
                raw_date = row.get("date")
                timestamp = datetime.fromisoformat(str(raw_date).replace("Z", "+00:00"))
                if timestamp.tzinfo is None:
                    timestamp = timestamp.replace(tzinfo=timezone.utc)
                close = _row_float(row, "close", "price")
                bars.append(
                    OHLCVBar(
                        timestamp=timestamp,
                        open=_row_float(row, "open", default=close),
                        high=_row_float(row, "high", default=close),
                        low=_row_float(row, "low", default=close),
                        close=close,
                        volume=float(row.get("volume") or 0),
                    )
                )
            except Exception:
                continue
        return list(reversed(bars))

    def _get(self, endpoint: str, params: dict) -> object:
        merged = {**params, "apikey": self.api_key}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/{endpoint.lstrip('/')}", params=merged)
            if response.status_code >= 400:
                raise RuntimeError(f"FMP endpoint '{endpoint}' failed with HTTP {response.status_code}")
        return response.json()


def _normalize_equity_symbol(symbol: str) -> str:
    return symbol.split(":")[-1].split("-")[0] if "-USDT" in symbol else symbol.upper()


def _news_symbol(request: RunRequest) -> str:
    if request.asset_class == "crypto":
        return request.symbol.upper().replace("-USDT-SWAP", "USD").replace("-USD-SWAP", "USD").replace("-", "")
    if request.asset_class == "fx":
        return request.symbol.upper().replace("/", "").replace("-", "")
    return _normalize_equity_symbol(request.symbol)


def _news_endpoint(asset_class: str) -> str | None:
    if asset_class == "crypto":
        return "news/crypto"
    if asset_class == "fx":
        return "news/forex"
    if asset_class in {"equity", "equity_index", "other"}:
        return "news/stock"
    return None


def _news_event_type(asset_class: str) -> str:
    if asset_class == "crypto":
        return "crypto_news"
    if asset_class == "fx":
        return "forex_news"
    return "market_news"


def _row_float(row: dict, *keys: str, default: float | None = None) -> float:
    for key in keys:
        value = row.get(key)
        if value is not None:
            return float(value)
    if default is not None:
        return default
    raise KeyError(keys[0] if keys else "value")
