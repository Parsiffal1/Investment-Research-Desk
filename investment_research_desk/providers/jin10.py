from __future__ import annotations

from datetime import datetime, timezone

import httpx

from investment_research_desk.schemas import NewsEvent, RunRequest


class Jin10NewsProvider:
    name = "jin10"

    def __init__(self, api_url: str | None, api_key: str | None = None, timeout: float = 10.0):
        self.api_url = api_url
        self.api_key = api_key
        self.timeout = timeout

    def fetch_news(self, request: RunRequest) -> list[NewsEvent]:
        if not self.api_url:
            return []
        return self._fetch(request, query=request.tool_query or request.symbol, event_type=None)

    def fetch_global_news(self, request: RunRequest) -> list[NewsEvent]:
        if not self.api_url:
            return []
        query = request.tool_query or request.symbol or "global macro markets"
        return self._fetch(request, query=query, event_type="global_market_news")

    def _fetch(self, request: RunRequest, query: str, event_type: str | None) -> list[NewsEvent]:
        headers = {"x-api-key": self.api_key} if self.api_key else {}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(self.api_url, params={"q": query}, headers=headers)
            response.raise_for_status()
        payload = response.json()
        items = payload.get("data", payload if isinstance(payload, list) else [])
        events: list[NewsEvent] = []
        for item in items[:10]:
            title = item.get("title") or item.get("content") or item.get("headline")
            if not title:
                continue
            published = item.get("published_at") or item.get("time")
            try:
                published_at = datetime.fromisoformat(str(published).replace("Z", "+00:00"))
            except Exception:
                published_at = datetime.now(timezone.utc)
            events.append(
                NewsEvent(
                    title=title,
                    summary=item.get("summary") or item.get("content"),
                    source="jin10",
                    published_at=published_at,
                    url=item.get("url"),
                    event_type=event_type or item.get("event_type") or item.get("type"),
                    related_assets=[request.symbol],
                )
            )
        return events
