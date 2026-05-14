from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    ollama_base_url: str = "http://localhost:11434/v1"
    ollama_model: str = "qwen3:8b"
    default_llm_provider: str = "auto"
    okx_base_url: str = "https://www.okx.com"
    okx_api_key: str | None = None
    okx_secret_key: str | None = None
    okx_passphrase: str | None = None
    okx_profile: str = "demo"
    okx_demo: bool = True
    okx_site: str = "global"
    okx_read_only: bool = True
    okx_tradekit_modules: str = "market"
    okx_config_path: str | None = None
    tavily_api_key: str | None = None
    tavily_base_url: str = "https://api.tavily.com"
    fmp_api_key: str | None = None
    fmp_base_url: str = "https://financialmodelingprep.com/stable"
    finnhub_api_key: str | None = None
    finnhub_base_url: str = "https://finnhub.io/api/v1"
    jin10_api_url: str | None = None
    jin10_api_key: str | None = None
    runs_dir: Path = Path("runs")
    market_data_vendors: str = "okx,fmp,yahoo_finance"
    news_data_vendors: str = "jin10,finnhub,yahoo_finance"
    sentiment_data_vendors: str = "tavily,stocktwits,reddit"
    fundamental_data_vendors: str = "fmp,finnhub"


def load_settings() -> Settings:
    load_dotenv()
    load_dotenv("notepad.env", override=False)
    return Settings(
        ollama_base_url=os.getenv("IRD_OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/"),
        ollama_model=os.getenv("IRD_OLLAMA_MODEL", "qwen3:8b"),
        default_llm_provider=os.getenv("IRD_DEFAULT_LLM_PROVIDER", "auto"),
        okx_base_url=os.getenv("OKX_BASE_URL", "https://www.okx.com").rstrip("/"),
        okx_api_key=os.getenv("OKX_API_KEY") or None,
        okx_secret_key=os.getenv("OKX_SECRET_KEY") or None,
        okx_passphrase=os.getenv("OKX_PASSPHRASE") or None,
        okx_profile=os.getenv("OKX_PROFILE", "demo"),
        okx_demo=_env_bool("OKX_DEMO", default=True),
        okx_site=os.getenv("OKX_SITE", "global"),
        okx_read_only=_env_bool("OKX_READ_ONLY", default=True),
        okx_tradekit_modules=os.getenv("OKX_TRADEKIT_MODULES", "market"),
        okx_config_path=os.getenv("OKX_CONFIG_PATH") or None,
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        tavily_base_url=os.getenv("TAVILY_BASE_URL", "https://api.tavily.com").rstrip("/"),
        fmp_api_key=os.getenv("FMP_API_KEY") or None,
        fmp_base_url=os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/stable").rstrip("/"),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY") or None,
        finnhub_base_url=os.getenv("FINNHUB_BASE_URL", "https://finnhub.io/api/v1").rstrip("/"),
        jin10_api_url=os.getenv("JIN10_API_URL") or None,
        jin10_api_key=os.getenv("JIN10_API_KEY") or None,
        runs_dir=Path(os.getenv("IRD_RUNS_DIR", "runs")),
        market_data_vendors=os.getenv("IRD_MARKET_DATA_VENDORS", "okx,fmp,yahoo_finance"),
        news_data_vendors=os.getenv("IRD_NEWS_DATA_VENDORS", "jin10,finnhub,yahoo_finance"),
        sentiment_data_vendors=os.getenv("IRD_SENTIMENT_DATA_VENDORS", "tavily,stocktwits,reddit"),
        fundamental_data_vendors=os.getenv("IRD_FUNDAMENTAL_DATA_VENDORS", "fmp,finnhub"),
    )


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}
