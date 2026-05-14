from __future__ import annotations

import json
from pathlib import Path

from investment_research_desk.schemas import NewsEvent, NormalizedData, OHLCVBar, RunRequest, SentimentInput


class FixtureProvider:
    name = "fixture"

    def __init__(self, fixtures_dir: Path | None = None):
        self.fixtures_dir = fixtures_dir or Path("data/fixtures")

    def load(self, fixture: str) -> NormalizedData:
        path = self.fixtures_dir / f"{fixture}.json"
        if not path.exists():
            raise FileNotFoundError(f"Fixture not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        return NormalizedData.model_validate(data["normalized_data"])

    def request(self, fixture: str) -> RunRequest:
        path = self.fixtures_dir / f"{fixture}.json"
        if not path.exists():
            raise FileNotFoundError(f"Fixture not found: {path}")
        data = json.loads(path.read_text(encoding="utf-8"))
        request = RunRequest.model_validate(data["request"])
        request.fixture = fixture
        return request

    def expected_key_points(self, fixture: str) -> list[str]:
        path = self.fixtures_dir / f"{fixture}.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        return data.get("expected_key_points", [])

    def fetch_ohlcv(self, request: RunRequest) -> list[OHLCVBar]:
        if not request.fixture:
            return []
        return self.load(request.fixture).ohlcv

    def fetch_news(self, request: RunRequest) -> list[NewsEvent]:
        if not request.fixture:
            return []
        return self.load(request.fixture).news_events

    def fetch_sentiment_inputs(self, request: RunRequest) -> list[SentimentInput]:
        if not request.fixture:
            return []
        return self.load(request.fixture).sentiment_inputs

