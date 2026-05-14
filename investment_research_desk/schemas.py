from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


AssetClass = Literal["crypto", "precious_metal", "equity_index", "commodity", "fx", "equity", "other"]
Horizon = Literal["intraday", "short_term", "swing", "medium_term"]
ResearchDepth = Literal["quick", "standard", "deep"]
ViewLabel = Literal[
    "bullish",
    "bearish",
    "neutral",
    "mixed",
    "mixed_to_bullish",
    "mixed_to_bearish",
    "neutral_to_bullish",
    "neutral_to_bearish",
]
RiskLevel = Literal["low", "medium", "high", "unknown"]


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class RunRequest(BaseModel):
    symbol: str
    asset_class: AssetClass = "crypto"
    horizon: Horizon = "short_term"
    research_depth: ResearchDepth = "standard"
    run_mode: Literal["snapshot", "batch"] = "snapshot"
    fixture: str | None = None
    llm_provider: Literal["auto", "fake", "ollama"] = "auto"
    model: str | None = None


class OHLCVBar(BaseModel):
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class NewsEvent(BaseModel):
    title: str
    summary: str | None = None
    source: str
    published_at: datetime
    url: str | None = None
    event_type: str | None = None
    related_assets: list[str] = Field(default_factory=list)
    sentiment_hint: Literal["bullish", "bearish", "neutral", "mixed"] | None = None


class SentimentInput(BaseModel):
    text: str
    source: str
    timestamp: datetime
    url: str | None = None


class NormalizedData(BaseModel):
    symbol: str
    asset_class: AssetClass
    horizon: Horizon
    ohlcv: list[OHLCVBar] = Field(default_factory=list)
    news_events: list[NewsEvent] = Field(default_factory=list)
    sentiment_inputs: list[SentimentInput] = Field(default_factory=list)
    market_context: dict[str, Any] = Field(default_factory=dict)
    source_metadata: dict[str, Any] = Field(default_factory=dict)


class FundamentalMacroResult(BaseModel):
    fundamental_view: ViewLabel
    key_drivers: list[str]
    concerns: list[str]
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)


class NewsImpactResult(BaseModel):
    dominant_events: list[str]
    event_type_summary: dict[str, str]
    asset_impact: dict[str, str]
    impact_logic: str
    confidence: float = Field(ge=0, le=1)
    evidence: list[str] = Field(default_factory=list)


class SentimentResult(BaseModel):
    crowd_mood: str
    sentiment_label: Literal["bullish", "bearish", "neutral", "mixed"]
    sentiment_score: float = Field(ge=-1, le=1)
    evidence: list[str]
    confidence: float = Field(ge=0, le=1)


class TechnicalState(BaseModel):
    technical_view: ViewLabel
    trend: str
    momentum: str
    volatility_regime: str
    rsi_14: float | None = None
    macd_state: str
    atr_14: float | None = None
    bollinger_state: str
    realized_volatility: float | None = None
    max_drawdown: float | None = None
    mark_price: float | None = None
    index_price: float | None = None
    funding_rate: float | None = None
    open_interest: float | None = None
    orderbook_imbalance: float | None = None
    swap_context_summary: str | None = None
    support_zones: list[float] = Field(default_factory=list)
    resistance_zones: list[float] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class ResearchCase(BaseModel):
    thesis: str
    evidence: list[str]
    conditions: list[str]
    confidence: float = Field(ge=0, le=1)


class FinalResearchContext(BaseModel):
    symbol: str
    asset_class: AssetClass
    timestamp: datetime = Field(default_factory=utc_now)
    horizon: Horizon
    market_regime: str
    balanced_view: ViewLabel
    risk_level: RiskLevel
    confidence: float = Field(ge=0, le=1)
    fundamental_summary: str | None = None
    news_impact_summary: str
    sentiment_summary: str
    technical_summary: str
    constructive_case: ResearchCase
    risk_case: ResearchCase
    key_drivers: list[str]
    key_risks: list[str]
    uncertainty_factors: list[str]
    downstream_agent_context: str
    usage_constraints: list[str]
    source_metadata: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)


class AgentTrace(BaseModel):
    name: str
    status: Literal["success", "warning", "failed", "skipped"]
    latency_sec: float
    warnings: list[str] = Field(default_factory=list)


class RunTrace(BaseModel):
    run_id: str
    symbol: str
    model: str
    llm_provider: str
    started_at: datetime
    completed_at: datetime | None = None
    agents: list[AgentTrace] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    completed_steps: list[str] = Field(default_factory=list)


class RunMetrics(BaseModel):
    total_latency_sec: float
    raw_input_tokens: int
    final_context_tokens: int
    compression_ratio: float
    json_valid: bool = True
    schema_valid: bool = True
    guardrail_violations: list[str] = Field(default_factory=list)
