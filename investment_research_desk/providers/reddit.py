from __future__ import annotations

import time
from datetime import datetime, timezone
from urllib.parse import urlencode

import httpx

from investment_research_desk.schemas import RunRequest, SentimentInput


DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")


class RedditProvider:
    name = "reddit"

    def __init__(self, timeout: float = 10.0, subreddits: tuple[str, ...] = DEFAULT_SUBREDDITS):
        self.timeout = timeout
        self.subreddits = subreddits

    def fetch_sentiment_inputs(self, request: RunRequest) -> list[SentimentInput]:
        symbol = _normalize_symbol(request.symbol)
        all_inputs: list[SentimentInput] = []
        with httpx.Client(timeout=self.timeout, headers={"User-Agent": "investment-research-desk/0.1"}) as client:
            for index, subreddit in enumerate(self.subreddits):
                if index:
                    time.sleep(0.4)
                query = urlencode({"q": symbol, "restrict_sr": "on", "sort": "new", "t": "week", "limit": 5})
                response = client.get(f"https://www.reddit.com/r/{subreddit}/search.json?{query}")
                if response.status_code >= 400:
                    continue
                children = ((response.json().get("data") or {}).get("children") or [])
                for child in children:
                    item = child.get("data") or {}
                    title = (item.get("title") or "").replace("\n", " ").strip()
                    body = (item.get("selftext") or "").replace("\n", " ").strip()
                    if not title and not body:
                        continue
                    created = item.get("created_utc")
                    timestamp = (
                        datetime.fromtimestamp(float(created), tz=timezone.utc)
                        if created
                        else datetime.now(timezone.utc)
                    )
                    text = (
                        f"[Reddit r/{subreddit} score={item.get('score', 0)} comments={item.get('num_comments', 0)}] "
                        f"{title}"
                    )
                    if body:
                        text += f" | {body[:240]}"
                    all_inputs.append(SentimentInput(text=text, source=f"reddit:{subreddit}", timestamp=timestamp))
        return all_inputs


def _normalize_symbol(symbol: str) -> str:
    return symbol.upper().split(":")[-1].split("-")[0]

