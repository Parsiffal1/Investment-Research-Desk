from __future__ import annotations

from langchain_core.tools import tool

from investment_research_desk.config import load_settings
from investment_research_desk.dataflows.interface import route_to_vendor
from investment_research_desk.schemas import RunRequest


@tool
def get_market_data(symbol: str, asset_class: str = "crypto", horizon: str = "short_term") -> str:
    """Retrieve OHLCV market data through configured vendor routing."""
    request = RunRequest(symbol=symbol, asset_class=asset_class, horizon=horizon)  # type: ignore[arg-type]
    return str(route_to_vendor("get_market_data", load_settings(), request).data)


@tool
def get_news(symbol: str, asset_class: str = "equity", horizon: str = "short_term") -> str:
    """Retrieve news through configured vendor routing."""
    request = RunRequest(symbol=symbol, asset_class=asset_class, horizon=horizon)  # type: ignore[arg-type]
    return str(route_to_vendor("get_news", load_settings(), request).data)


@tool
def get_sentiment_inputs(symbol: str, asset_class: str = "equity", horizon: str = "short_term") -> str:
    """Retrieve search/social/commentary inputs through configured vendor routing."""
    request = RunRequest(symbol=symbol, asset_class=asset_class, horizon=horizon)  # type: ignore[arg-type]
    return str(route_to_vendor("get_sentiment_inputs", load_settings(), request).data)


@tool
def get_fundamentals(symbol: str, asset_class: str = "equity", horizon: str = "short_term") -> str:
    """Retrieve quote/profile fundamentals through configured vendor routing."""
    request = RunRequest(symbol=symbol, asset_class=asset_class, horizon=horizon)  # type: ignore[arg-type]
    return str(route_to_vendor("get_fundamentals", load_settings(), request).data)
