from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from investment_research_desk.config import Settings
from investment_research_desk.providers import (
    FinnhubProvider,
    FmpProvider,
    Jin10NewsProvider,
    OkxMarketDataProvider,
    RedditProvider,
    StockTwitsProvider,
    TavilySearchProvider,
    YahooFinanceProvider,
)
from investment_research_desk.schemas import RunRequest
from investment_research_desk.security import redact_secrets


TOOLS_CATEGORIES = {
    "market_data": {"description": "OHLCV market data", "tools": ["get_market_data"]},
    "news_data": {"description": "Ticker and macro news/events", "tools": ["get_news"]},
    "sentiment_data": {"description": "Search/social/commentary sentiment inputs", "tools": ["get_sentiment_inputs"]},
    "fundamental_data": {"description": "Quote, profile, and company context", "tools": ["get_fundamentals"]},
}

VENDOR_LIST = ["okx", "fmp", "finnhub", "tavily", "jin10", "yahoo_finance", "stocktwits", "reddit"]


@dataclass
class VendorRouteResult:
    data: Any
    status: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


def route_to_vendor(method: str, settings: Settings, request: RunRequest) -> VendorRouteResult:
    """Route a generic data method to configured vendors with fallback support."""
    category = _category_for_method(method)
    configured = _configured_vendors(category, settings)
    methods = _vendor_methods(settings)
    if method not in methods:
        raise ValueError(f"Unsupported dataflow method: {method}")

    available = methods[method]
    vendor_order = _fallback_chain(configured, list(available.keys()))
    combined = _initial_data(method)
    status: dict[str, str] = {}
    warnings: list[str] = []

    for vendor in vendor_order:
        impl = available.get(vendor)
        if impl is None:
            continue
        try:
            value = impl(request)
            status[vendor] = "success" if _has_data(value) else "empty"
            combined = _merge_data(method, combined, value)
            if method == "get_market_data" and _has_data(combined):
                break
        except Exception as exc:
            safe = _redact(str(exc), settings)
            status[vendor] = f"failed: {safe}"
            warnings.append(f"{vendor} {method} failed: {safe}")

    return VendorRouteResult(data=combined, status=status, warnings=warnings)


def _category_for_method(method: str) -> str:
    for category, info in TOOLS_CATEGORIES.items():
        if method in info["tools"]:
            return category
    raise ValueError(f"Method '{method}' not found in any dataflow category")


def _configured_vendors(category: str, settings: Settings) -> list[str]:
    raw = {
        "market_data": settings.market_data_vendors,
        "news_data": settings.news_data_vendors,
        "sentiment_data": settings.sentiment_data_vendors,
        "fundamental_data": settings.fundamental_data_vendors,
    }[category]
    return [item.strip() for item in raw.split(",") if item.strip()]


def _fallback_chain(primary: list[str], available: list[str]) -> list[str]:
    chain = list(primary)
    for vendor in available:
        if vendor not in chain:
            chain.append(vendor)
    return chain


def _vendor_methods(settings: Settings) -> dict[str, dict[str, Callable[[RunRequest], Any]]]:
    okx = OkxMarketDataProvider(settings.okx_base_url)
    fmp = FmpProvider(settings.fmp_api_key, settings.fmp_base_url)
    finnhub = FinnhubProvider(settings.finnhub_api_key, settings.finnhub_base_url)
    tavily = TavilySearchProvider(settings.tavily_api_key, settings.tavily_base_url)
    jin10 = Jin10NewsProvider(settings.jin10_api_url, settings.jin10_api_key)
    yahoo = YahooFinanceProvider()
    stocktwits = StockTwitsProvider()
    reddit = RedditProvider()
    return {
        "get_market_data": {"okx": okx.fetch_ohlcv, "fmp": fmp.fetch_ohlcv, "yahoo_finance": yahoo.fetch_ohlcv},
        "get_news": {"jin10": jin10.fetch_news, "finnhub": finnhub.fetch_news, "yahoo_finance": yahoo.fetch_news},
        "get_sentiment_inputs": {
            "tavily": tavily.fetch_sentiment_inputs,
            "stocktwits": stocktwits.fetch_sentiment_inputs,
            "reddit": reddit.fetch_sentiment_inputs,
        },
        "get_fundamentals": {
            "fmp": lambda request: _fmp_fundamentals(fmp, request),
            "finnhub": lambda request: _finnhub_fundamentals(finnhub, request),
        },
    }


def _fmp_fundamentals(provider: FmpProvider, request: RunRequest) -> dict[str, Any]:
    if not provider.available() or request.asset_class not in {"equity", "equity_index", "other"}:
        return {}
    symbol = request.symbol.split(":")[-1].split("-")[0] if "-USDT" in request.symbol else request.symbol.upper()
    return {"fmp_quote": provider.quote(symbol), "fmp_profile": provider.profile(symbol)}


def _finnhub_fundamentals(provider: FinnhubProvider, request: RunRequest) -> dict[str, Any]:
    if not provider.available() or request.asset_class not in {"equity", "other"}:
        return {}
    return {"finnhub_quote": provider.quote(request.symbol)}


def _initial_data(method: str) -> Any:
    return {} if method == "get_fundamentals" else []


def _merge_data(method: str, current: Any, value: Any) -> Any:
    if method == "get_fundamentals":
        merged = dict(current)
        if isinstance(value, dict):
            for key, item in value.items():
                if item is not None:
                    merged[key] = item
        return merged
    if isinstance(current, list):
        return current + (value or [])
    return value


def _has_data(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, dict):
        return any(item not in (None, [], {}) for item in value.values())
    return bool(value)


def _redact(text: str, settings: Settings) -> str:
    return redact_secrets(
        text,
        [settings.fmp_api_key, settings.finnhub_api_key, settings.tavily_api_key, settings.jin10_api_key],
    )
