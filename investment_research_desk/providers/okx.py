from __future__ import annotations

from datetime import datetime, timezone

import httpx

from investment_research_desk.schemas import OHLCVBar, RunRequest


class OkxMarketDataProvider:
    name = "okx"

    def __init__(self, base_url: str = "https://www.okx.com", timeout: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def fetch_ohlcv(self, request: RunRequest) -> list[OHLCVBar]:
        params = {"instId": request.symbol, "bar": self._bar_for_horizon(request.horizon), "limit": "100"}
        url = f"{self.base_url}/api/v5/market/candles"
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(url, params=params)
            response.raise_for_status()
        payload = response.json()
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

    @staticmethod
    def _bar_for_horizon(horizon: str) -> str:
        if horizon == "intraday":
            return "15m"
        if horizon == "short_term":
            return "1H"
        if horizon == "swing":
            return "4H"
        return "1D"

