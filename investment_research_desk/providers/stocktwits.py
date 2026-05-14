from __future__ import annotations

from datetime import datetime, timezone

import httpx

from investment_research_desk.schemas import RunRequest, SentimentInput


class StockTwitsProvider:
    name = "stocktwits"

    def __init__(self, timeout: float = 10.0):
        self.timeout = timeout

    def fetch_sentiment_inputs(self, request: RunRequest) -> list[SentimentInput]:
        symbol = _normalize_symbol(request.symbol)
        url = f"https://api.stocktwits.com/api/2/streams/symbol/{symbol}.json"
        with httpx.Client(timeout=self.timeout, headers={"User-Agent": "investment-research-desk/0.1"}) as client:
            response = client.get(url)
            if response.status_code >= 400:
                raise RuntimeError(f"StockTwits stream failed with HTTP {response.status_code}")
        messages = response.json().get("messages") or []
        inputs: list[SentimentInput] = []
        for message in messages[:30]:
            body = (message.get("body") or "").replace("\n", " ").strip()
            if not body:
                continue
            sentiment_obj = ((message.get("entities") or {}).get("sentiment") or {})
            label = sentiment_obj.get("basic") if isinstance(sentiment_obj, dict) else None
            created = message.get("created_at")
            try:
                timestamp = datetime.fromisoformat(str(created).replace("Z", "+00:00"))
            except Exception:
                timestamp = datetime.now(timezone.utc)
            user = (message.get("user") or {}).get("username", "unknown")
            text = f"[StockTwits {label or 'unlabeled'} @{user}] {body}"
            inputs.append(SentimentInput(text=text, source="stocktwits", timestamp=timestamp))
        return inputs


def _normalize_symbol(symbol: str) -> str:
    return symbol.upper().split(":")[-1].split("-")[0]

