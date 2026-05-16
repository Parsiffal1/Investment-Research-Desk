from datetime import datetime, timezone

from investment_research_desk.providers.fmp import FmpProvider
from investment_research_desk.providers.okx import OkxMarketDataProvider
from investment_research_desk.schemas import RunRequest


def test_fmp_market_data_uses_free_light_endpoint(monkeypatch):
    provider = FmpProvider("test-key")
    calls: list[str] = []

    def fake_get(endpoint, params):
        calls.append(endpoint)
        return [{"date": "2026-05-14", "price": 123.45, "volume": 1000}]

    monkeypatch.setattr(provider, "_get", fake_get)

    bars = provider.fetch_ohlcv(RunRequest(symbol="AAPL", asset_class="equity"))

    assert calls == ["historical-price-eod/light"]
    assert bars[0].timestamp == datetime(2026, 5, 14, tzinfo=timezone.utc)
    assert bars[0].open == 123.45
    assert bars[0].high == 123.45
    assert bars[0].low == 123.45
    assert bars[0].close == 123.45
    assert bars[0].volume == 1000


def test_fmp_news_uses_symbol_scoped_stock_endpoint(monkeypatch):
    provider = FmpProvider("test-key")
    calls: list[tuple[str, dict]] = []

    def fake_get(endpoint, params):
        calls.append((endpoint, params))
        return [
            {
                "title": "SPDR S&P 500 ETF Trust tracks equity rally",
                "text": "Market-focused article",
                "site": "example",
                "publishedDate": "2026-05-14T12:00:00Z",
                "url": "https://example.com/spy",
            }
        ]

    monkeypatch.setattr(provider, "_get", fake_get)

    events = provider.fetch_news(RunRequest(symbol="SPY", asset_class="equity"))

    assert calls == [("news/stock", {"symbols": "SPY"})]
    assert events[0].title == "SPDR S&P 500 ETF Trust tracks equity rally"
    assert events[0].related_assets == ["SPY"]


def test_fmp_news_treats_paid_endpoint_402_as_empty(monkeypatch):
    provider = FmpProvider("test-key")

    def fake_get(endpoint, params):
        raise RuntimeError(f"FMP endpoint '{endpoint}' failed with HTTP 402")

    monkeypatch.setattr(provider, "_get", fake_get)

    events = provider.fetch_news(RunRequest(symbol="ETH-USDT-SWAP", asset_class="crypto"))

    assert events == []


def test_okx_market_data_skips_non_swap_equity(monkeypatch):
    provider = OkxMarketDataProvider()

    def fail_public_get(path, params=None):
        raise AssertionError("OKX should not be called for non-SWAP equity market data")

    monkeypatch.setattr(provider, "_public_get", fail_public_get)

    request = RunRequest(symbol="NVDA", asset_class="equity")

    assert provider.fetch_ohlcv(request) == []
    assert provider.fetch_swap_market_context(request) == {}
