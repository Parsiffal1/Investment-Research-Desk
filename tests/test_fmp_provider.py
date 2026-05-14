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


def test_okx_market_data_skips_non_swap_equity(monkeypatch):
    provider = OkxMarketDataProvider()

    def fail_public_get(path, params=None):
        raise AssertionError("OKX should not be called for non-SWAP equity market data")

    monkeypatch.setattr(provider, "_public_get", fail_public_get)

    request = RunRequest(symbol="NVDA", asset_class="equity")

    assert provider.fetch_ohlcv(request) == []
    assert provider.fetch_swap_market_context(request) == {}
