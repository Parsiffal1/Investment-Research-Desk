from __future__ import annotations

from datetime import datetime, timezone

import httpx

from investment_research_desk.schemas import NewsEvent, RunRequest, SentimentInput


class TavilySearchProvider:
    name = "tavily"

    def __init__(self, api_key: str | None, base_url: str = "https://api.tavily.com", timeout: float = 15.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch_sentiment_inputs(self, request: RunRequest) -> list[SentimentInput]:
        if not self.api_key:
            return []
        query = f"{request.symbol} market news macro sentiment"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/search",
                headers=headers,
                json={"query": query, "max_results": 5, "topic": "news"},
            )
            if response.status_code >= 400:
                raise RuntimeError(f"Tavily search failed with HTTP {response.status_code}")
        results = response.json().get("results", [])
        return [
            SentimentInput(
                text=(item.get("content") or item.get("title") or "").strip(),
                source="tavily",
                timestamp=datetime.now(timezone.utc),
                url=item.get("url"),
            )
            for item in results
            if (item.get("content") or item.get("title"))
        ]

    def fetch_news_events(self, request: RunRequest) -> list[NewsEvent]:
        if not self.api_key:
            return []
        query = request.symbol or "global markets macro news"
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        with httpx.Client(timeout=self.timeout) as client:
            response = client.post(
                f"{self.base_url}/search",
                headers=headers,
                json={"query": query, "max_results": 5, "topic": "news"},
            )
            if response.status_code >= 400:
                raise RuntimeError(f"Tavily news search failed with HTTP {response.status_code}")
        results = response.json().get("results", [])
        events: list[NewsEvent] = []
        for item in results:
            title = (item.get("title") or item.get("content") or "").strip()
            if not title:
                continue
            events.append(
                NewsEvent(
                    title=title,
                    summary=item.get("content"),
                    source="tavily",
                    published_at=datetime.now(timezone.utc),
                    url=item.get("url"),
                    event_type="global_market_news",
                    related_assets=[request.symbol],
                )
            )
        return events
