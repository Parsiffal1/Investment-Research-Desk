from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


AgentTeam = Literal["controller", "data", "analyst", "research", "reporting", "cache"]


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
        system_prompt=(
            "You are the run controller for Investment Research Desk. Initialize state, preserve user request "
            "metadata, and never produce market recommendations."
        ),
    ),
    "data_ingestion": AgentContract(
        name="data_ingestion",
        team="data",
        role="Prepare fixture-backed data or live-run data context before analyst agents call their own tools.",
        allowed_inputs=["RunRequest", "provider settings", "fixture data"],
        allowed_tools=[
            "FixtureProvider",
            "NormalizedData shell builder",
        ],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="NormalizedData",
        system_prompt=(
            "You are the data ingestion node. Use only configured providers and fixtures, normalize evidence, "
            "and record provider warnings without leaking API keys."
        ),
    ),
    "fundamental_macro": AgentContract(
        name="fundamental_macro",
        team="analyst",
        role="Assess fundamental and macro context without producing trading instructions.",
        allowed_inputs=["RunRequest", "source_metadata.fmp_profile", "source_metadata.fmp_quote", "source_metadata.finnhub_quote", "news_events"],
        allowed_tools=["route_to_vendor.get_fundamentals", "route_to_vendor.get_news", "fundamental_metadata_reader", "macro_event_classifier"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="FundamentalMacroResult",
        system_prompt=(
            "You are a Fundamental/Macro Analyst. Use company profile, quote metadata, and macro/news context "
            "to identify drivers, concerns, evidence, and confidence. Return structured JSON only. "
            "Do not recommend trades, orders, or position sizes."
        ),
    ),
    "news_impact": AgentContract(
        name="news_impact",
        team="analyst",
        role="Classify recent news and macro events and summarize their possible asset impact.",
        allowed_inputs=["RunRequest", "news_events"],
        allowed_tools=["route_to_vendor.get_news", "news_event_classifier", "llm_json_refinement"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="NewsImpactResult",
        system_prompt=(
            "You are a News/Macro Impact Analyst. Analyze only supplied news events and macro headlines. "
            "Return one JSON object with impact_logic and confidence when asked to refine output. "
            "Avoid buy/sell/order/position-size language."
        ),
    ),
    "sentiment": AgentContract(
        name="sentiment",
        team="analyst",
        role="Aggregate social and search sentiment inputs into a short-horizon market mood read.",
        allowed_inputs=["RunRequest", "sentiment_inputs"],
        allowed_tools=["route_to_vendor.get_sentiment_inputs", "sentiment_term_scanner"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="SentimentResult",
        system_prompt=(
            "You are a Sentiment Analyst. Use only supplied sentiment inputs from routed sources such as "
            "Tavily, StockTwits, and Reddit. Produce sentiment label, score, evidence, and confidence. "
            "Do not infer execution decisions."
        ),
    ),
    "technical": AgentContract(
        name="technical",
        team="analyst",
        role="Compute deterministic technical state from OHLCV only.",
        allowed_inputs=["RunRequest", "ohlcv"],
        allowed_tools=[
            "route_to_vendor.get_market_data",
            "RSI",
            "MACD",
            "ATR",
            "Bollinger Bands",
            "realized volatility",
            "max drawdown",
            "support/resistance",
        ],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="TechnicalState",
        system_prompt=(
            "You are a Technical Analyst. Use only normalized OHLCV and deterministic indicator tools. "
            "Describe trend, momentum, volatility, and levels as research context only."
        ),
    ),
    "analyst_team": AgentContract(
        name="analyst_team",
        team="analyst",
        role="Aggregate analyst outputs into a handoff package for researchers.",
        allowed_inputs=["FundamentalMacroResult", "NewsImpactResult", "SentimentResult", "TechnicalState"],
        allowed_tools=["schema_validator", "analyst_synthesis"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="analyst_team_outputs.json",
        system_prompt=(
            "You are the Analyst Team coordinator. Combine analyst outputs into a structured handoff. "
            "Preserve disagreements and uncertainty."
        ),
    ),
    "bull_researcher": AgentContract(
        name="bull_researcher",
        team="research",
        role="Build the constructive research case from analyst evidence.",
        allowed_inputs=["analyst_team", "FundamentalMacroResult", "NewsImpactResult", "SentimentResult", "TechnicalState"],
        allowed_tools=["evidence_selector", "case_builder"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="ResearchCase",
        system_prompt=(
            "You are the Bull Researcher. Construct the strongest evidence-based supportive case from analyst outputs. "
            "Frame it as scenario research, not a trading recommendation."
        ),
    ),
    "bear_researcher": AgentContract(
        name="bear_researcher",
        team="research",
        role="Build the risk case and challenge the constructive case.",
        allowed_inputs=["analyst_team", "bull_researcher output", "FundamentalMacroResult", "NewsImpactResult", "SentimentResult", "TechnicalState"],
        allowed_tools=["risk_evidence_selector", "case_challenger"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="ResearchCase",
        system_prompt=(
            "You are the Bear Researcher. Construct the strongest evidence-based risk case from analyst outputs "
            "and challenge the constructive case. Stay within research context."
        ),
    ),
    "bull_bear_research_debate": AgentContract(
        name="bull_bear_research_debate",
        team="research",
        role="Moderate the bull and bear cases into explicit agreement points, tensions, and reporter handoff.",
        allowed_inputs=["bull_researcher output", "bear_researcher output"],
        allowed_tools=["debate_summarizer"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="research_debate.json",
        system_prompt=(
            "You are the Bull/Bear Research Debate moderator. Compare the two research cases, extract tensions, "
            "and hand off balanced context to the reporter. Do not choose a trade."
        ),
    ),
    "research_reporter": AgentContract(
        name="research_reporter",
        team="reporting",
        role="Convert analyst and debate outputs into the final structured research context and Markdown brief.",
        allowed_inputs=["NormalizedData", "analyst_team", "research_debate", "warnings"],
        allowed_tools=["schema_validator", "guardrail_checker", "markdown_renderer"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="FinalResearchContext",
        system_prompt=(
            "You are the Research Reporter. Produce a balanced final research context from analyst and debate outputs. "
            "Use clear uncertainty language and avoid execution instructions."
        ),
    ),
    "final_market_context_cache": AgentContract(
        name="final_market_context_cache",
        team="cache",
        role="Persist final market context for downstream research consumers.",
        allowed_inputs=["FinalResearchContext", "analyst_team", "research_debate"],
        allowed_tools=["cache_writer"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="final_market_context_cache.json",
        system_prompt=(
            "You are the final market context cache writer. Persist structured context and usage boundaries only."
        ),
    ),
    "persist": AgentContract(
        name="persist",
        team="cache",
        role="Persist final artifacts, trace, and metrics for the completed run.",
        allowed_inputs=["FinalResearchContext", "RunTrace", "RunMetrics", "rendered markdown"],
        allowed_tools=["json_writer", "markdown_writer", "metrics_writer"],
        forbidden_actions=COMMON_FORBIDDEN_ACTIONS,
        output_schema="runs/{run_id}/artifact files",
        system_prompt=(
            "You are the persistence node. Write final artifacts exactly as structured outputs and preserve "
            "guardrail warnings."
        ),
    ),
}


def get_agent_contract(name: str) -> AgentContract:
    try:
        return AGENT_CONTRACTS[name]
    except KeyError as exc:
        raise KeyError(f"No agent contract registered for {name}") from exc


def contract_manifest() -> dict[str, dict]:
    return {name: contract.model_dump(mode="json") for name, contract in AGENT_CONTRACTS.items()}
