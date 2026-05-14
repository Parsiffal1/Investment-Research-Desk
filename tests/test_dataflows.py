from investment_research_desk.config import load_settings
from investment_research_desk.dataflows import route_to_vendor
from investment_research_desk.schemas import RunRequest


def test_market_data_routes_to_configured_vendors():
    settings = load_settings()
    request = RunRequest(symbol="BTC-USDT-SWAP", asset_class="crypto", horizon="short_term")

    result = route_to_vendor("get_market_data", settings, request)

    assert isinstance(result.status, dict)
    assert "okx" in result.status


def test_fundamentals_route_returns_mapping_for_equity():
    settings = load_settings()
    request = RunRequest(symbol="AAPL", asset_class="equity", horizon="short_term")

    result = route_to_vendor("get_fundamentals", settings, request)

    assert isinstance(result.data, dict)
    assert "fmp" in result.status or "finnhub" in result.status

