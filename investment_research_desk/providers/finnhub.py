from __future__ import annotations

from datetime import datetime, timedelta, timezone

import httpx

from investment_research_desk.schemas import NewsEvent, RunRequest


class FinnhubProvider:
    name = "finnhub"

    def __init__(self, api_key: str | None, base_url: str = "https://finnhub.io/api/v1", timeout: float = 15.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def available(self) -> bool:
        return bool(self.api_key)

    def quote(self, symbol: str) -> dict | None:
        if not self.api_key:
            return None
        data = self._get("quote", {"symbol": _normalize_symbol(symbol)})
        return data if isinstance(data, dict) and data else None

    def fetch_news(self, request: RunRequest) -> list[NewsEvent]:
        if not self.api_key:
            return []
        symbol = _normalize_symbol(request.symbol)
        end = datetime.now(timezone.utc).date()
        start = end - timedelta(days=14)
        data = self._get("company-news", {"symbol": symbol, "from": start.isoformat(), "to": end.isoformat()})
        if not isinstance(data, list):
            return []
        events: list[NewsEvent] = []
        for item in data[:10]:
            headline = item.get("headline")
            if not headline:
                continue
            published_at = datetime.fromtimestamp(int(item.get("datetime", 0)), tz=timezone.utc)
            events.append(
                NewsEvent(
                    title=headline,
                    summary=item.get("summary"),
                    source=f"finnhub:{item.get('source') or 'news'}",
                    published_at=published_at,
                    url=item.get("url"),
                    event_type=item.get("category") or "company_news",
                    related_assets=[symbol],
                )
            )
        return events

    def _get(self, endpoint: str, params: dict) -> object:
        merged = {**params, "token": self.api_key}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(f"{self.base_url}/{endpoint.lstrip('/')}", params=merged)
            if response.status_code >= 400:
                raise RuntimeError(f"Finnhub endpoint '{endpoint}' failed with HTTP {response.status_code}")
        return response.json()


def _normalize_symbol(symbol: str) -> str:
    return symbol.upper().split(":")[-1]
