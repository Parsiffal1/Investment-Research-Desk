from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from investment_research_desk.agents.prompts import (
    ANALYST_TEAM_PROMPT,
    BEAR_RESEARCHER_PROMPT,
    BULL_RESEARCHER_PROMPT,
    CACHE_WRITER_PROMPT,
    DEBATE_MODERATOR_PROMPT,
    FUNDAMENTAL_MACRO_PROMPT,
    NEWS_IMPACT_PROMPT,
    PERSISTENCE_PROMPT,
    RESEARCH_REPORTER_PROMPT,
    RUN_CONTROLLER_PROMPT,
    SENTIMENT_PROMPT,
    TECHNICAL_PROMPT,
)

AgentTeam = Literal["controller", "analyst", "research", "reporting", "cache"]


class AgentContract(BaseModel):
    name: str
    team: AgentTeam
    role: str
    allowed_inputs: list[str]
    allowed_tools: list[str]
    forbidden_actions: list[str] = Field(default_factory=list)
    output_schema: str
    system_prompt: str


COMMON_FORBIDDEN_ACTIONS = [
    "do not issue direct buy, sell, short, or hold instructions",
    "do not provide order placement language",
    "do not provide position sizing",
    "do not guarantee profitability or returns",
    "do not treat research context as financial advice",
]


AGENT_CONTRACTS: dict[str, AgentContract] = {
    "run_controller": AgentContract(
        name="run_controller",
        team="controller",
        role="Initialize the run, preserve request metadata, and coordinate checkpoint-safe execution.",
        allowed_inputs=["RunRequest"],
        allowed_tools=["run_metadata"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="WorkflowState",
        system_prompt=RUN_CONTROLLER_PROMPT,
    ),
    "fundamental_macro": AgentContract(
        name="fundamental_macro",
        team="analyst",
        role="Assess fundamental and macro context without producing trading instructions.",
        allowed_inputs=["RunRequest", "source_metadata.fmp_profile", "source_metadata.fmp_quote", "source_metadata.finnhub_quote", "news_events"],
        allowed_tools=["route_to_vendor.get_fundamentals", "route_to_vendor.get_news", "fundamental_metadata_reader", "macro_event_classifier"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="FundamentalMacroResult",
        system_prompt=FUNDAMENTAL_MACRO_PROMPT,
    ),
    "news_impact": AgentContract(
        name="news_impact",
        team="analyst",
        role="Decide which news tools to call, classify candidate news/macro events, and summarize possible asset impact.",
        allowed_inputs=["RunRequest", "tool results from get_news/get_global_news"],
        allowed_tools=[
            "route_to_vendor.get_news",
            "route_to_vendor.get_global_news",
            "llm_query_planner",
            "news_event_classifier",
            "llm_json_refinement",
        ],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="NewsImpactResult",
        system_prompt=NEWS_IMPACT_PROMPT,
    ),
    "sentiment": AgentContract(
        name="sentiment",
        team="analyst",
        role="Aggregate social and search sentiment inputs into a short-horizon market mood read.",
        allowed_inputs=["RunRequest", "sentiment_inputs"],
        allowed_tools=["route_to_vendor.get_sentiment_inputs", "sentiment_term_scanner"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="SentimentResult",
        system_prompt=SENTIMENT_PROMPT,
    ),
    "technical": AgentContract(
        name="technical",
        team="analyst",
        role="Read OHLCV, deterministic indicators, and public OKX SWAP microstructure context.",
        allowed_inputs=["RunRequest", "ohlcv", "indicator_results", "market_context.okx_swap"],
        allowed_tools=[
            "route_to_vendor.get_market_data",
            "route_to_vendor.get_swap_market_context",
            "RSI",
            "MACD",
            "ATR",
            "Bollinger Bands",
            "realized volatility",
            "max drawdown",
            "support/resistance",
            "funding_rate_reader",
            "open_interest_reader",
            "orderbook_imbalance_reader",
        ],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="TechnicalState",
        system_prompt=TECHNICAL_PROMPT,
    ),
    "analyst_team": AgentContract(
        name="analyst_team",
        team="analyst",
        role="Aggregate analyst outputs into a handoff package for researchers.",
        allowed_inputs=["FundamentalMacroResult", "NewsImpactResult", "SentimentResult", "TechnicalState"],
        allowed_tools=["schema_validator", "analyst_synthesis"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="analyst_team_outputs.json",
        system_prompt=ANALYST_TEAM_PROMPT,
    ),
    "bull_researcher": AgentContract(
        name="bull_researcher",
        team="research",
        role="Build the constructive research case from analyst evidence.",
        allowed_inputs=["analyst_team", "FundamentalMacroResult", "NewsImpactResult", "SentimentResult", "TechnicalState"],
        allowed_tools=["evidence_selector", "case_builder"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="ResearchCase",
        system_prompt=BULL_RESEARCHER_PROMPT,
    ),
    "bear_researcher": AgentContract(
        name="bear_researcher",
        team="research",
        role="Build the risk case and challenge the constructive case.",
        allowed_inputs=["analyst_team", "bull_researcher output", "FundamentalMacroResult", "NewsImpactResult", "SentimentResult", "TechnicalState"],
        allowed_tools=["risk_evidence_selector", "case_challenger"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="ResearchCase",
        system_prompt=BEAR_RESEARCHER_PROMPT,
    ),
    "bull_bear_research_debate": AgentContract(
        name="bull_bear_research_debate",
        team="research",
        role="Moderate the bull and bear cases into explicit agreement points, tensions, and reporter handoff.",
        allowed_inputs=["bull_researcher output", "bear_researcher output"],
        allowed_tools=["debate_summarizer"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="research_debate.json",
        system_prompt=DEBATE_MODERATOR_PROMPT,
    ),
    "research_reporter": AgentContract(
        name="research_reporter",
        team="reporting",
        role="Convert analyst and debate outputs into the final structured research context and Markdown brief.",
        allowed_inputs=["NormalizedData", "analyst_team", "research_debate", "warnings"],
        allowed_tools=["schema_validator", "guardrail_checker", "markdown_renderer"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="FinalResearchContext",
        system_prompt=RESEARCH_REPORTER_PROMPT,
    ),
    "final_market_context_cache": AgentContract(
        name="final_market_context_cache",
        team="cache",
        role="Persist final market context for downstream research consumers.",
        allowed_inputs=["FinalResearchContext", "analyst_team", "research_debate"],
        allowed_tools=["cache_writer"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="final_market_context_cache.json",
        system_prompt=CACHE_WRITER_PROMPT,
    ),
    "persist": AgentContract(
        name="persist",
        team="cache",
        role="Persist final artifacts, trace, and metrics for the completed run.",
        allowed_inputs=["FinalResearchContext", "RunTrace", "RunMetrics", "rendered markdown"],
        allowed_tools=["json_writer", "markdown_writer", "metrics_writer"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="runs/{run_id}/artifact files",
        system_prompt=PERSISTENCE_PROMPT,
    ),
}


def get_agent_contract(name: str) -> AgentContract:
    try:
        return AGENT_CONTRACTS[name]
    except KeyError as exc:
        raise KeyError(f"No agent contract registered for {name}") from exc


def contract_manifest() -> dict[str, dict]:
    return {name: contract.model_dump(mode="json") for name, contract in AGENT_CONTRACTS.items()}
