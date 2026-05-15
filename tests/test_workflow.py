from pathlib import Path
from datetime import datetime, timezone

from investment_research_desk.config import load_settings
from investment_research_desk.dataflows.interface import VendorRouteResult
from investment_research_desk.graph import ResearchWorkflow
from investment_research_desk.sentiment_runtime import FakeSentimentClassifier
from investment_research_desk.schemas import FinalResearchContext, NewsEvent, OHLCVBar, RunRequest, SentimentInput


def test_fixture_workflow_creates_artifacts(tmp_path: Path):
    settings = load_settings()
    workflow = ResearchWorkflow(settings=settings, runs_dir=tmp_path)
    request = RunRequest(symbol="XAU-USDT-SWAP", asset_class="precious_metal", fixture="gold_cpi", llm_provider="fake")

    state = workflow.run(request, checkpoint=True)

    context = FinalResearchContext.model_validate(state["final_context"])
    assert context.symbol == "XAU-USDT-SWAP"
    assert context.directional_view in {"bullish", "bearish"}
    assert context.directional_rationale
    assert context.key_drivers
    run_dir = tmp_path / state["run_id"]
    assert (run_dir / "agent_contracts.json").exists()
    assert (run_dir / "final_research_context.json").exists()
    assert (run_dir / "research_brief.md").exists()
    assert (run_dir / "trace.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "checkpoint.json").exists()
    assert (run_dir / "analyst_team_outputs.json").exists()
    assert (run_dir / "research_debate.json").exists()
    assert (run_dir / "final_market_context_cache.json").exists()
    trace_names = [agent["name"] for agent in state["trace"]["agents"]]
    assert "fundamental_macro" in trace_names
    assert "news_impact" in trace_names
    assert "sentiment" in trace_names
    assert "technical" in trace_names
    assert "analyst_team" in trace_names
    assert "bull_researcher" in trace_names
    assert "bear_researcher" in trace_names
    assert "bull_bear_research_debate" in trace_names
    assert "research_reporter" in trace_names
    assert "final_market_context_cache" in trace_names
    assert "data_ingestion" not in trace_names
    assert "analyst_layer" not in trace_names
    assert "research_layer" not in trace_names
    assert state["analyst_team"]["execution_mode"] == "parallel_thread_pool"
    assert state["research_debate"]["round_count"] == 2
    assert len(state["research_debate"]["rounds"]) == 4


def test_resume_from_checkpoint_completes(tmp_path: Path):
    settings = load_settings()
    workflow = ResearchWorkflow(settings=settings, runs_dir=tmp_path)
    request = RunRequest(symbol="XAU-USDT-SWAP", asset_class="precious_metal", fixture="gold_cpi", llm_provider="fake")

    first = workflow.run(request, checkpoint=True)
    resumed = workflow.run(request, checkpoint=True, resume_run_id=first["run_id"])

    assert resumed["run_id"] == first["run_id"]
    assert "persist" in resumed["completed_steps"]


def test_resume_from_mid_graph_checkpoint_continues_remaining_agents(tmp_path: Path):
    settings = load_settings()
    workflow = ResearchWorkflow(settings=settings, runs_dir=tmp_path)
    request = RunRequest(symbol="XAU-USDT-SWAP", asset_class="precious_metal", fixture="gold_cpi", llm_provider="fake")

    first = workflow.run(request, checkpoint=True)
    checkpoint = workflow.store.load_checkpoint(first["run_id"])
    kept_steps = [
        "run_controller",
        "analyst_team",
    ]
    kept_trace_names = set(kept_steps + ["fundamental_macro", "news_impact", "sentiment", "technical"])
    checkpoint["completed_steps"] = kept_steps
    checkpoint["trace"]["completed_steps"] = kept_steps
    checkpoint["trace"]["agents"] = [agent for agent in checkpoint["trace"]["agents"] if agent["name"] in kept_trace_names]
    for key in ["constructive", "risk", "research_debate", "final_context", "final_market_context_cache", "metrics", "output_paths"]:
        checkpoint.pop(key, None)
    workflow.store.save_checkpoint(first["run_id"], checkpoint)

    resumed = workflow.run(request, checkpoint=True, resume_run_id=first["run_id"])

    assert resumed["completed_steps"][-6:] == [
        "bull_researcher",
        "bear_researcher",
        "bull_bear_research_debate",
        "research_reporter",
        "final_market_context_cache",
        "persist",
    ]
    assert (tmp_path / first["run_id"] / "final_research_context.json").exists()
    assert (tmp_path / first["run_id"] / "final_market_context_cache.json").exists()


def test_live_analysts_call_their_own_dataflow_tools(tmp_path: Path, monkeypatch):
    calls: list[str] = []
    now = datetime.now(timezone.utc)

    def fake_route_to_vendor(method, settings, request):
        calls.append(method)
        if method == "get_market_data":
            bars = [
                OHLCVBar(timestamp=now, open=100 + i, high=102 + i, low=99 + i, close=101 + i, volume=1000 + i)
                for i in range(30)
            ]
            return VendorRouteResult(data=bars, status={"fake_market": "success"})
        if method == "get_swap_market_context":
            return VendorRouteResult(
                data={
                    "provider": "okx",
                    "scope": "public_swap_market_only",
                    "inst_id": "FAKE-USDT-SWAP",
                    "mark_price": {"markPx": "130.5"},
                    "index_ticker": {"idxPx": "130.0"},
                    "funding_rate": {"fundingRate": "0.0001"},
                    "open_interest": {"oi": "1000"},
                    "orderbook_imbalance": 0.2,
                },
                status={"fake_okx": "success"},
            )
        if method == "get_news":
            return VendorRouteResult(
                data=[
                    NewsEvent(
                        title="Macro event supports test asset",
                        summary="Fixture-like live news",
                        source="fake_news",
                        published_at=now,
                    )
                ],
                status={"fake_news": "success"},
            )
        if method == "get_global_news":
            return VendorRouteResult(
                data=[
                    NewsEvent(
                        title="Global macro liquidity event affects test asset",
                        summary="Global macro test news",
                        source="fake_global_news",
                        published_at=now,
                        event_type="global_market_news",
                    )
                ],
                status={"fake_global_news": "success"},
            )
        if method == "get_sentiment_inputs":
            return VendorRouteResult(
                data=[SentimentInput(text="Market discussion is mixed but constructive", source="fake_social", timestamp=now)],
                status={"fake_social": "success"},
            )
        if method == "get_fundamentals":
            return VendorRouteResult(
                data={"fmp_quote": {"changePercentage": 1.2}, "fmp_profile": {"companyName": "Fake Asset"}},
                status={"fake_fundamentals": "success"},
            )
        raise AssertionError(f"unexpected method {method}")

    monkeypatch.setattr("investment_research_desk.graph.workflow.route_to_vendor", fake_route_to_vendor)
    workflow = ResearchWorkflow(settings=load_settings(), runs_dir=tmp_path)
    request = RunRequest(symbol="FAKE", asset_class="equity", horizon="short_term", llm_provider="fake")

    state = workflow.run(request, checkpoint=True)

    assert "get_market_data" in calls
    assert "get_swap_market_context" in calls
    assert "get_sentiment_inputs" in calls
    assert "get_fundamentals" in calls
    assert calls.count("get_news") >= 2
    assert "get_global_news" in calls
    assert state["data"]["source_metadata"]["tool_call_policy"] == "tradingagents_style_llm_tool_loop"
    assert "fundamental_macro" in state["data"]["source_metadata"]["llm_tool_calls"]
    assert "news_impact" in state["data"]["source_metadata"]["llm_tool_calls"]
    assert "sentiment" in state["data"]["source_metadata"]["llm_tool_calls"]
    assert "technical" in state["data"]["source_metadata"]["llm_tool_calls"]
    assert state["data"]["source_metadata"]["tool_call_budget"]["technical"]["executed"] <= state["data"]["source_metadata"]["tool_call_budget"]["technical"]["max"]
    assert state["data"]["source_metadata"]["agent_tool_status"]["technical"]["get_market_data"] == {"fake_market": "success"}
    assert state["data"]["source_metadata"]["agent_tool_status"]["technical"]["get_swap_market_context"] == {"fake_okx": "success"}
    assert state["technical"]["funding_rate"] == 0.0001
    assert state["data"]["source_metadata"]["agent_tool_status"]["fundamental_macro"]["get_fundamentals"] == {
        "fake_fundamentals": "success"
    }
    assert "get_global_news" in state["data"]["source_metadata"]["agent_tool_status"]["news_impact"]


def test_sentiment_adapter_runtime_is_used_only_by_sentiment_agent(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        "investment_research_desk.graph.workflow.make_sentiment_classifier",
        lambda settings, request: FakeSentimentClassifier(),
    )
    workflow = ResearchWorkflow(settings=load_settings(), runs_dir=tmp_path)
    request = RunRequest(
        symbol="XAU-USDT-SWAP",
        asset_class="precious_metal",
        fixture="gold_cpi",
        llm_provider="fake",
        sentiment_provider="fake",
    )

    state = workflow.run(request, checkpoint=False)

    assert state["sentiment"]["sentiment_label"] in {"bullish", "bearish", "neutral", "mixed"}
    assert "sentiment_runtime" in state["data"]["source_metadata"]
    assert state["data"]["source_metadata"]["sentiment_runtime"]["provider"] == "fake"
    assert state["data"]["source_metadata"]["agent_tool_status"]["sentiment"] == {"fixture": "success"}


def test_news_tool_call_preserves_symbol_and_passes_refined_query(tmp_path: Path, monkeypatch):
    seen: list[RunRequest] = []

    def fake_route_to_vendor(method, settings, request):
        seen.append(request)
        return VendorRouteResult(data=[], status={"fake": "empty"})

    monkeypatch.setattr("investment_research_desk.graph.workflow.route_to_vendor", fake_route_to_vendor)
    workflow = ResearchWorkflow(settings=load_settings(), runs_dir=tmp_path)
    request = RunRequest(symbol="SPY", asset_class="equity", horizon="short_term", llm_provider="fake")
    collected = {
        "ohlcv": [],
        "news_events": [],
        "sentiment_inputs": [],
        "market_context": {},
        "source_metadata": {},
        "tool_status": {},
        "warnings": [],
        "tool_calls": [],
        "contract_floor_calls": [],
        "executed_tool_count": 0,
        "max_tool_calls": 4,
        "executed_tool_counts": {},
    }

    workflow._execute_agent_tool(
        "news_impact",
        request,
        "get_news",
        {"symbol": "SPY", "query": "SPDR S&P 500 ETF Trust market news"},
        collected,
    )

    assert seen[0].symbol == "SPY"
    assert seen[0].tool_query == "SPDR S&P 500 ETF Trust market news"


def test_default_news_tool_arguments_expand_ambiguous_spy_query():
    args = ResearchWorkflow._default_tool_arguments("get_news", RunRequest(symbol="SPY", asset_class="equity"))

    assert args["symbol"] == "SPY"
    assert "SPDR S&P 500 ETF Trust" in args["query"]
    assert args["query"] != "SPY"
