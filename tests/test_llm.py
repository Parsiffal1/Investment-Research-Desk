from datetime import datetime, timezone

from investment_research_desk.agents import (
    ConstructiveCaseAnalyst,
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


def test_all_seven_analysis_agents_call_llm():
    data = FixtureProvider().load("gold_cpi")
    llm = FakeLLMClient()

    fundamental = FundamentalMacroAnalyst().run(data, llm)
    news = NewsImpactAnalyst().run(data, llm)
    sentiment = SentimentAnalyst().run(data, llm)
    technical = TechnicalAnalyst().run(data, llm)
    bull = ConstructiveCaseAnalyst().run(fundamental, news, sentiment, technical, llm)
    bear = RiskCaseAnalyst().run(fundamental, news, sentiment, technical, bull, llm)
    ResearchReporter().run(data, fundamental, news, sentiment, technical, bull, bear, [], llm)

    assert len(llm.calls) == 7
    called_agents = "\n".join(call["user"] for call in llm.calls)
    assert "Agent: fundamental_macro" in called_agents
    assert "Agent: news_impact" in called_agents
    assert "Agent: sentiment" in called_agents
    assert "Agent: technical" in called_agents
    assert "Agent: bull_researcher" in called_agents
    assert "Agent: bear_researcher" in called_agents
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
    assert data.source_metadata["tool_call_policy"] == "llm_driven_tool_loop"
    assert data.source_metadata["llm_tool_calls"]
