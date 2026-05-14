from __future__ import annotations

from typing import Protocol

from investment_research_desk.schemas import NewsEvent, OHLCVBar, RunRequest, SentimentInput


class MarketDataProvider(Protocol):
    name: str

    def fetch_ohlcv(self, request: RunRequest) -> list[OHLCVBar]:
        ...


class NewsProvider(Protocol):
    name: str

    def fetch_news(self, request: RunRequest) -> list[NewsEvent]:
        ...


class SearchProvider(Protocol):
    name: str

    def fetch_sentiment_inputs(self, request: RunRequest) -> list[SentimentInput]:
        ...

