from datetime import datetime, timezone

from investment_research_desk.agents import (
    ConstructiveCaseAnalyst,
    DebateModerator,
    FundamentalMacroAnalyst,
    NewsImpactAnalyst,
    ResearchReporter,
    RiskCaseAnalyst,
    SentimentAnalyst,
    TechnicalAnalyst,
)
from investment_research_desk.dataflows.interface import VendorRouteResult
from investment_research_desk.llm import FakeLLMClient
from investment_research_desk.llm.clients import OllamaLLMClient
from investment_research_desk.providers.fixtures import FixtureProvider
from investment_research_desk.schemas import NewsEvent, RunRequest


def test_ollama_json_repair_path():
    client = OllamaLLMClient("http://localhost:11434/v1", "qwen3:8b")
    calls = iter(["not json", '{"ok": true}'])
    client._chat_content = lambda system, user: next(calls)  # type: ignore[method-assign]

    assert client.chat_json("system", "user") == {"ok": True}


def test_analysis_and_research_agents_call_llm():
    data = FixtureProvider().load("gold_cpi")
    llm = FakeLLMClient()

    fundamental = FundamentalMacroAnalyst().run(data, llm)
    news = NewsImpactAnalyst().run(data, llm)
    sentiment = SentimentAnalyst().run(data, llm)
    technical = TechnicalAnalyst().run(data, llm)
    bull = ConstructiveCaseAnalyst().run(fundamental, news, sentiment, technical, llm)
    bear = RiskCaseAnalyst().run(fundamental, news, sentiment, technical, bull, llm)
    debate = DebateModerator().run(bull, bear, llm)
    ResearchReporter().run(
        data,
        fundamental,
        news,
        sentiment,
        technical,
        bull,
        bear,
        debate.model_dump(mode="json"),
        [],
        llm,
    )

    assert len(llm.calls) == 8
    called_agents = "\n".join(call["user"] for call in llm.calls)
    assert "Agent: fundamental_macro" in called_agents
    assert "Agent: news_impact" in called_agents
    assert "Agent: sentiment" in called_agents
    assert "Agent: technical" in called_agents
    assert "Agent: bull_researcher" in called_agents
    assert "Agent: bear_researcher" in called_agents
    assert "Agent: bull_bear_research_debate" in called_agents
    assert "Agent: research_reporter" in called_agents
    assert "indicator_results" in called_agents


def test_news_analyst_uses_llm_driven_tool_loop():
    llm = FakeLLMClient()
    request = RunRequest(symbol="BTC-USDT-SWAP", asset_class="crypto", horizon="short_term", llm_provider="fake")
    calls: list[tuple[str, str]] = []
    now = datetime.now(timezone.utc)

    def route_tool(method: str, tool_request: RunRequest):
        calls.append((method, tool_request.symbol))
        return VendorRouteResult(
            data=[
                NewsEvent(
                    title=f"{tool_request.symbol} relevant macro event",
                    summary="Tool loop candidate",
                    source="fake_news",
                    published_at=now,
                )
            ],
            status={"fake_news": "success"},
        )

    result, data = NewsImpactAnalyst().run_with_tools(request, llm, route_tool)

    assert result.dominant_events
    assert {method for method, _ in calls} == {"get_news", "get_global_news"}
    assert ("get_news", "BTC-USDT-SWAP") in calls
    assert data.source_metadata["tool_call_policy"] == "llm_planned_tool_calls_with_targeted_search_minimum"
    assert data.source_metadata["minimum_targeted_search_enforced"] is False
    assert data.source_metadata["llm_query_plan"]["calls"]
    assert data.source_metadata["llm_tool_calls"]


def test_news_analyst_forces_targeted_search_when_llm_skips_tools():
    class NoToolLLM(FakeLLMClient):
        def chat_json(self, system, user):
            self.calls.append({"system": system, "user": user})
            if "NewsToolPlan" in user:
                return {"calls": [], "stop_reason": "skip"}
            return super().chat_json(system, user)

    llm = NoToolLLM()
    request = RunRequest(symbol="NVDA", asset_class="equity", horizon="short_term", llm_provider="fake")
    calls: list[tuple[str, str]] = []
    now = datetime.now(timezone.utc)

    def route_tool(method: str, tool_request: RunRequest):
        calls.append((method, tool_request.symbol))
        return VendorRouteResult(
            data=[
                NewsEvent(
                    title=f"{tool_request.symbol} company news",
                    summary="Forced contract candidate",
                    source="fake_news",
                    published_at=now,
                )
            ],
            status={"fake_news": "success", "tavily": "success"},
        )

    result, data = NewsImpactAnalyst().run_with_tools(request, llm, route_tool)

    assert ("get_news", "NVDA") in calls
    assert result.dominant_events == ["NVDA company news"]
    assert data.source_metadata["minimum_targeted_search_enforced"] is True
    assert data.source_metadata["forced_contract_calls"][0]["forced_by_contract"] is True


def test_news_forced_search_fallback_filters_unrelated_provider_metadata():
    class NoToolLLM(FakeLLMClient):
        def chat_json(self, system, user):
            self.calls.append({"system": system, "user": user})
            if "NewsToolPlan" in user:
                return {"calls": [], "stop_reason": "skip"}
            return super().chat_json(system, user)

    llm = NoToolLLM()
    request = RunRequest(symbol="NVDA", asset_class="equity", horizon="short_term", llm_provider="fake")
    now = datetime.now(timezone.utc)

    def route_tool(method: str, tool_request: RunRequest):
        return VendorRouteResult(
            data=[
                NewsEvent(
                    title="Prediction: XRP Will Hit $5 in 2026",
                    summary="Crypto-only item with provider metadata noise",
                    source="fake_news",
                    published_at=now,
                    related_assets=["NVDA"],
                ),
                NewsEvent(
                    title="Nvidia data center demand lifts chip sector",
                    summary="NVDA-linked AI infrastructure demand",
                    source="fake_news",
                    published_at=now,
                    related_assets=["NVDA"],
                ),
            ],
            status={"fake_news": "success", "tavily": "success"},
        )

    result, data = NewsImpactAnalyst().run_with_tools(request, llm, route_tool)

    assert result.dominant_events == ["Nvidia data center demand lifts chip sector"]
    assert data.source_metadata["filtered_candidate_count"] == 1
