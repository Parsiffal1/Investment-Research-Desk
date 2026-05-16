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
    tavily_api_key: str | None = None
    tavily_base_url: str = "https://api.tavily.com"
    fmp_api_key: str | None = None
    fmp_base_url: str = "https://financialmodelingprep.com/stable"
    finnhub_api_key: str | None = None
    finnhub_base_url: str = "https://finnhub.io/api/v1"
    jin10_api_url: str | None = None
    jin10_api_key: str | None = None
    runs_dir: Path = Path("runs")
    market_data_vendors: str = "okx,yahoo_finance,fmp"
    news_data_vendors: str = "fmp,finnhub,yahoo_finance,jin10,tavily"
    sentiment_data_vendors: str = "tavily,stocktwits,reddit"
    fundamental_data_vendors: str = "fmp,finnhub"
    sentiment_provider: str = "main"
    sentiment_base_model: str = "Qwen/Qwen3-8B"
    sentiment_adapter_path: Path | None = None
    sentiment_score_batch_size: int = 4
    agent_execution_mode: str = "sequential"
    llm_timeout_sec: float = 180.0
    agent_tool_loop_timeout_sec: float = 240.0
    agent_max_tool_calls: int = 8
    report_language: str = "en"


def load_settings() -> Settings:
    load_dotenv()
    load_dotenv("notepad.env", override=False)
    return Settings(
        ollama_base_url=os.getenv("IRD_OLLAMA_BASE_URL", "http://localhost:11434/v1").rstrip("/"),
        ollama_model=os.getenv("IRD_OLLAMA_MODEL", "qwen3:8b"),
        default_llm_provider=os.getenv("IRD_DEFAULT_LLM_PROVIDER", "auto"),
        okx_base_url=os.getenv("OKX_BASE_URL", "https://www.okx.com").rstrip("/"),
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        tavily_base_url=os.getenv("TAVILY_BASE_URL", "https://api.tavily.com").rstrip("/"),
        fmp_api_key=os.getenv("FMP_API_KEY") or None,
        fmp_base_url=os.getenv("FMP_BASE_URL", "https://financialmodelingprep.com/stable").rstrip("/"),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY") or None,
        finnhub_base_url=os.getenv("FINNHUB_BASE_URL", "https://finnhub.io/api/v1").rstrip("/"),
        jin10_api_url=os.getenv("JIN10_API_URL") or None,
        jin10_api_key=os.getenv("JIN10_API_KEY") or None,
        runs_dir=Path(os.getenv("IRD_RUNS_DIR", "runs")),
        market_data_vendors=os.getenv("IRD_MARKET_DATA_VENDORS", "okx,yahoo_finance,fmp"),
        news_data_vendors=os.getenv("IRD_NEWS_DATA_VENDORS", "fmp,finnhub,yahoo_finance,jin10,tavily"),
        sentiment_data_vendors=os.getenv("IRD_SENTIMENT_DATA_VENDORS", "tavily,stocktwits,reddit"),
        fundamental_data_vendors=os.getenv("IRD_FUNDAMENTAL_DATA_VENDORS", "fmp,finnhub"),
        sentiment_provider=os.getenv("IRD_SENTIMENT_PROVIDER", "main"),
        sentiment_base_model=os.getenv("IRD_SENTIMENT_BASE_MODEL", "Qwen/Qwen3-8B"),
        sentiment_adapter_path=Path(path) if (path := os.getenv("IRD_SENTIMENT_ADAPTER_PATH")) else None,
        sentiment_score_batch_size=int(os.getenv("IRD_SENTIMENT_SCORE_BATCH_SIZE", "4")),
        agent_execution_mode=os.getenv("IRD_AGENT_EXECUTION_MODE", "sequential").strip().lower(),
        llm_timeout_sec=float(os.getenv("IRD_LLM_TIMEOUT_SEC", "180")),
        agent_tool_loop_timeout_sec=float(os.getenv("IRD_AGENT_TOOL_LOOP_TIMEOUT_SEC", "240")),
        agent_max_tool_calls=int(os.getenv("IRD_AGENT_MAX_TOOL_CALLS", "8")),
        report_language=os.getenv("IRD_REPORT_LANGUAGE", "en").strip().lower(),
    )
