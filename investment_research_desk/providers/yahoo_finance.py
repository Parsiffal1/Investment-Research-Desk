from __future__ import annotations

from datetime import datetime, timezone

import httpx

from investment_research_desk.schemas import NewsEvent, OHLCVBar, RunRequest


class YahooFinanceProvider:
    name = "yahoo_finance"

    def __init__(self, timeout: float = 15.0):
        self.timeout = timeout

    def fetch_ohlcv(self, request: RunRequest) -> list[OHLCVBar]:
        symbol = _normalize_symbol(request.symbol)
        params = _chart_params(request.horizon)
        with httpx.Client(timeout=self.timeout, headers={"User-Agent": _user_agent()}) as client:
            response = client.get(f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}", params=params)
            if response.status_code >= 400:
                raise RuntimeError(f"Yahoo Finance chart failed with HTTP {response.status_code}")
        result = (((response.json().get("chart") or {}).get("result") or [])[:1] or [None])[0]
        if not result:
            return []
        timestamps = result.get("timestamp") or []
        quote = (((result.get("indicators") or {}).get("quote") or [])[:1] or [{}])[0]
        opens = quote.get("open") or []
        highs = quote.get("high") or []
        lows = quote.get("low") or []
        closes = quote.get("close") or []
        volumes = quote.get("volume") or []
        bars: list[OHLCVBar] = []
        for idx, timestamp in enumerate(timestamps):
            try:
                values = (opens[idx], highs[idx], lows[idx], closes[idx])
                if any(value is None for value in values):
                    continue
                bars.append(
                    OHLCVBar(
                        timestamp=datetime.fromtimestamp(int(timestamp), tz=timezone.utc),
                        open=float(opens[idx]),
                        high=float(highs[idx]),
                        low=float(lows[idx]),
                        close=float(closes[idx]),
                        volume=float(volumes[idx] or 0),
                    )
                )
            except Exception:
                continue
        return bars[-100:]

    def fetch_news(self, request: RunRequest) -> list[NewsEvent]:
        symbol = _normalize_symbol(request.symbol)
        return self._search_news(symbol, related_asset=symbol)

    def fetch_global_news(self, request: RunRequest) -> list[NewsEvent]:
        query = request.tool_query or request.symbol or "global markets macro crypto"
        return self._search_news(query, related_asset=request.symbol, event_type="global_market_news")

    def _search_news(self, query: str, related_asset: str, event_type: str = "market_news") -> list[NewsEvent]:
        params = {"q": query, "quotesCount": 0, "newsCount": 10}
        with httpx.Client(timeout=self.timeout, headers={"User-Agent": _user_agent()}) as client:
            response = client.get("https://query1.finance.yahoo.com/v1/finance/search", params=params)
            if response.status_code >= 400:
                raise RuntimeError(f"Yahoo Finance search failed with HTTP {response.status_code}")
        items = response.json().get("news") or []
        events: list[NewsEvent] = []
        for item in items:
            title = item.get("title")
            if not title:
                continue
            published = item.get("providerPublishTime")
            published_at = (
                datetime.fromtimestamp(int(published), tz=timezone.utc)
                if published
                else datetime.now(timezone.utc)
            )
            events.append(
                NewsEvent(
                    title=title,
                    summary=item.get("summary"),
                    source=f"yahoo_finance:{item.get('publisher') or 'news'}",
                    published_at=published_at,
                    url=item.get("link"),
                    event_type=event_type,
                    related_assets=[related_asset],
                )
            )
        return events


def _chart_params(horizon: str) -> dict[str, str]:
    if horizon == "intraday":
        return {"range": "5d", "interval": "30m"}
    if horizon == "short_term":
        return {"range": "1mo", "interval": "1d"}
    if horizon == "swing":
        return {"range": "3mo", "interval": "1d"}
    return {"range": "1y", "interval": "1d"}


def _normalize_symbol(symbol: str) -> str:
    return symbol.upper().split(":")[-1]


def _user_agent() -> str:
    return "investment-research-desk/0.1 (+local CLI)"
