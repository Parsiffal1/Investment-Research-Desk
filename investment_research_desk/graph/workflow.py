from __future__ import annotations

import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from investment_research_desk.agents import (
    ConstructiveCaseAnalyst,
    FundamentalMacroAnalyst,
    NewsImpactAnalyst,
    ResearchReporter,
    RiskCaseAnalyst,
    SentimentAnalyst,
    TechnicalAnalyst,
    contract_manifest,
    get_agent_contract,
)
from investment_research_desk.config import Settings, load_settings
from investment_research_desk.dataflows import route_to_vendor
from investment_research_desk.llm import make_llm_client
from investment_research_desk.persistence import RunStore
from investment_research_desk.providers import FixtureProvider
from investment_research_desk.schemas import (
    AgentTrace,
    FinalResearchContext,
    FundamentalMacroResult,
    NewsEvent,
    NewsImpactResult,
    NormalizedData,
    ResearchCase,
    RunMetrics,
    RunRequest,
    RunTrace,
    SentimentResult,
    TechnicalState,
)
from investment_research_desk.security import redact_secrets
from investment_research_desk.tools.guardrails import find_guardrail_violations
from investment_research_desk.tools.metrics import approximate_tokens, compression_ratio


class WorkflowState(TypedDict, total=False):
    run_id: str
    request: dict[str, Any]
    data: dict[str, Any]
    fundamental: dict[str, Any]
    news: dict[str, Any]
    sentiment: dict[str, Any]
    technical: dict[str, Any]
    analyst_team: dict[str, Any]
    constructive: dict[str, Any]
    risk: dict[str, Any]
    research_debate: dict[str, Any]
    final_context: dict[str, Any]
    final_market_context_cache: dict[str, Any]
    agent_contracts: dict[str, Any]
    trace: dict[str, Any]
    metrics: dict[str, Any]
    warnings: list[str]
    completed_steps: list[str]
    output_paths: dict[str, str]
    checkpoint_enabled: bool


class ParallelAgentError(RuntimeError):
    def __init__(self, message: str, trace: AgentTrace):
        super().__init__(message)
        self.trace = trace


class ResearchWorkflow:
    def __init__(
        self,
        settings: Settings | None = None,
        runs_dir: Path | None = None,
        progress_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.settings = settings or load_settings()
        self.store = RunStore(runs_dir or self.settings.runs_dir)
        self.fixture_provider = FixtureProvider()
        self.progress_callback = progress_callback
        self.graph = self._build_graph()

    def run(self, request: RunRequest, checkpoint: bool = False, resume_run_id: str | None = None) -> WorkflowState:
        if resume_run_id:
            state = self.store.load_checkpoint(resume_run_id)
            state["warnings"] = state.get("warnings", []) + [f"resumed from checkpoint for run_id={resume_run_id}"]
        else:
            run_id = self._new_run_id(request)
            trace = RunTrace(
                run_id=run_id,
                symbol=request.symbol,
                model=request.model or self.settings.ollama_model,
                llm_provider=request.llm_provider,
                started_at=datetime.now(timezone.utc),
            )
            state = {
                "run_id": run_id,
                "request": request.model_dump(mode="json"),
                "agent_contracts": contract_manifest(),
                "trace": trace.model_dump(mode="json"),
                "warnings": [],
                "completed_steps": [],
                "output_paths": {},
            }
            self.store.ensure_run_dir(run_id)
            self.store.write_json(run_id, "input.json", request)
            self.store.write_json(run_id, "agent_contracts.json", state["agent_contracts"])
        state["checkpoint_enabled"] = checkpoint
        return self.graph.invoke(state)

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("run_controller", self._run_controller)
        graph.add_node("analyst_team", self._analyst_team)
        graph.add_node("bull_researcher", self._bull_researcher)
        graph.add_node("bear_researcher", self._bear_researcher)
        graph.add_node("bull_bear_research_debate", self._bull_bear_research_debate)
        graph.add_node("research_reporter", self._research_reporter)
        graph.add_node("final_market_context_cache", self._final_market_context_cache)
        graph.add_node("persist", self._persist)
        graph.add_edge(START, "run_controller")
        graph.add_edge("run_controller", "analyst_team")
        graph.add_edge("analyst_team", "bull_researcher")
        graph.add_edge("bull_researcher", "bear_researcher")
        graph.add_edge("bear_researcher", "bull_bear_research_debate")
        graph.add_edge("bull_bear_research_debate", "research_reporter")
        graph.add_edge("research_reporter", "final_market_context_cache")
        graph.add_edge("final_market_context_cache", "persist")
        graph.add_edge("persist", END)
        return graph.compile()

    def _run_controller(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            request = RunRequest.model_validate(s["request"])
            if request.fixture:
                data = self.fixture_provider.load(request.fixture)
                data.source_metadata["provider_mode"] = "fixture"
            else:
                data = NormalizedData(
                    symbol=request.symbol,
                    asset_class=request.asset_class,
                    horizon=request.horizon,
                    source_metadata={
                        "provider_mode": "live",
                        "tool_call_policy": "analyst_agents_call_allowed_tools",
                        "agent_tool_status": {},
                    },
                )
            s["data"] = data.model_dump(mode="json")
            self.store.write_json(s["run_id"], "normalized_data.json", data)
            return s

        return self._run_step(state, "run_controller", work)

    def _fundamental_macro(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            request = RunRequest.model_validate(s["request"])
            data = self._scope_data(NormalizedData.model_validate(s["data"]), "fundamental_macro")
            llm = self._make_llm(request, s)
            fundamental = FundamentalMacroAnalyst().run(data, llm)
            s["fundamental"] = fundamental.model_dump(mode="json")
            self._write_analyst_outputs(s)
            return s

        return self._run_step(state, "fundamental_macro", work)

    def _news_impact(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            request = RunRequest.model_validate(s["request"])
            data = self._scope_data(NormalizedData.model_validate(s["data"]), "news_impact")
            llm = self._make_llm(request, s)
            news = NewsImpactAnalyst().run(data, llm)
            s["news"] = news.model_dump(mode="json")
            self._write_analyst_outputs(s)
            return s

        return self._run_step(state, "news_impact", work)

    def _sentiment(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            request = RunRequest.model_validate(s["request"])
            data = self._scope_data(NormalizedData.model_validate(s["data"]), "sentiment")
            llm = self._make_llm(request, s)
            sentiment = SentimentAnalyst().run(data, llm)
            s["sentiment"] = sentiment.model_dump(mode="json")
            self._write_analyst_outputs(s)
            return s

        return self._run_step(state, "sentiment", work)

    def _technical(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            request = RunRequest.model_validate(s["request"])
            data = self._scope_data(NormalizedData.model_validate(s["data"]), "technical")
            llm = self._make_llm(request, s)
            technical = TechnicalAnalyst().run(data, llm)
            s["technical"] = technical.model_dump(mode="json")
            self._write_analyst_outputs(s)
            return s

        return self._run_step(state, "technical", work)

    def _analyst_team(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            request = RunRequest.model_validate(s["request"])
            seed_data = NormalizedData.model_validate(s["data"])
            outputs, data_slices, traces, warnings = self._run_analysts_parallel(request, seed_data)
            s["fundamental"] = outputs["fundamental_macro"]
            s["news"] = outputs["news_impact"]
            s["sentiment"] = outputs["sentiment"]
            s["technical"] = outputs["technical"]
            data = self._merge_agent_data(request, seed_data, data_slices)
            s["data"] = data.model_dump(mode="json")
            self.store.write_json(s["run_id"], "normalized_data.json", data)
            tool_warnings = [
                f"{agent_name}: {warning}"
                for agent_name, agent_warnings in data.source_metadata.get("agent_tool_warnings", {}).items()
                for warning in agent_warnings
            ]
            if tool_warnings:
                s["warnings"] = list(s.get("warnings", [])) + tool_warnings
            for agent_trace in traces:
                self._append_trace(s, agent_trace)
            if warnings:
                s["warnings"] = list(s.get("warnings", [])) + warnings
            self._write_analyst_outputs(s)
            fundamental = FundamentalMacroResult.model_validate(s["fundamental"])
            news = NewsImpactResult.model_validate(s["news"])
            sentiment = SentimentResult.model_validate(s["sentiment"])
            technical = TechnicalState.model_validate(s["technical"])
            analyst_team = {
                "team": "Analyst Team",
                "contract": get_agent_contract("analyst_team").model_dump(mode="json"),
                "execution_mode": "parallel_thread_pool",
                "symbol": data.symbol,
                "asset_class": data.asset_class,
                "horizon": data.horizon,
                "members": {
                    "fundamentals_analyst": fundamental.model_dump(mode="json"),
                    "news_analyst": news.model_dump(mode="json"),
                    "sentiment_analyst": sentiment.model_dump(mode="json"),
                    "technical_analyst": technical.model_dump(mode="json"),
                },
                "synthesis": {
                    "fundamental_view": fundamental.fundamental_view,
                    "news_view": news.asset_impact.get(data.symbol, "mixed"),
                    "sentiment_label": sentiment.sentiment_label,
                    "technical_view": technical.technical_view,
                    "research_handoff": (
                        "Use these analyst outputs as evidence for bull and bear researchers. "
                        "Do not convert them into order instructions or position sizing."
                    ),
                },
            }
            s["analyst_team"] = analyst_team
            self.store.write_json(s["run_id"], "analyst_team_outputs.json", analyst_team)
            return s

        return self._run_step(state, "analyst_team", work)

    def _run_analysts_parallel(
        self, request: RunRequest, data: NormalizedData
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[AgentTrace], list[str]]:
        jobs = {
            "fundamental_macro": lambda: self._run_fundamental_agent(request, data),
            "news_impact": lambda: self._run_news_agent(request, data),
            "sentiment": lambda: self._run_sentiment_agent(request, data),
            "technical": lambda: self._run_technical_agent(request, data),
        }
        outputs: dict[str, dict[str, Any]] = {}
        data_slices: dict[str, dict[str, Any]] = {}
        traces: list[AgentTrace] = []
        warnings: list[str] = []
        with ThreadPoolExecutor(max_workers=len(jobs), thread_name_prefix="ird-analyst") as executor:
            for name in jobs:
                self._emit_progress("agent_status", name, "in_progress")
            futures = {executor.submit(self._run_parallel_agent, name, call): name for name, call in jobs.items()}
            for future in as_completed(futures):
                name = futures[future]
                try:
                    output, data_slice, trace = future.result()
                    outputs[name] = output
                    data_slices[name] = data_slice
                    traces.append(trace)
                    self._emit_progress(
                        "agent_result",
                        name,
                        "completed",
                        payload={"output": output, "data": data_slice, "trace": trace.model_dump(mode="json")},
                    )
                except ParallelAgentError as exc:
                    warnings.append(f"{name} failed in parallel analyst layer: {exc}")
                    traces.append(exc.trace)
                    self._emit_progress(
                        "agent_status",
                        name,
                        "failed",
                        payload={"warnings": exc.trace.warnings},
                    )
                    raise
        order = ["fundamental_macro", "news_impact", "sentiment", "technical"]
        traces.sort(key=lambda trace: order.index(trace.name) if trace.name in order else len(order))
        return outputs, data_slices, traces, warnings

    @staticmethod
    def _run_parallel_agent(name: str, call) -> tuple[dict[str, Any], dict[str, Any], AgentTrace]:
        get_agent_contract(name)
        started = time.perf_counter()
        try:
            result, data = call()
            trace = AgentTrace(name=name, status="success", latency_sec=round(time.perf_counter() - started, 4))
            return result.model_dump(mode="json"), data.model_dump(mode="json"), trace
        except Exception as exc:
            trace = AgentTrace(name=name, status="failed", latency_sec=round(time.perf_counter() - started, 4), warnings=[str(exc)])
            raise ParallelAgentError(f"{name} failed: {exc}", trace) from exc

    def _run_fundamental_agent(self, request: RunRequest, seed_data: NormalizedData):
        data = self._agent_data("fundamental_macro", request, seed_data)
        result = FundamentalMacroAnalyst().run(data, self._make_llm_for_request(request))
        return result, data

    def _run_news_agent(self, request: RunRequest, seed_data: NormalizedData):
        if request.fixture:
            data = self._agent_data("news_impact", request, seed_data)
            result = NewsImpactAnalyst().run(data, self._make_llm_for_request(request))
            return result, data
        result, data = NewsImpactAnalyst().run_with_tools(
            request,
            self._make_llm_for_request(request),
            lambda method, tool_request: route_to_vendor(method, self.settings, tool_request),
        )
        return result, data

    def _run_sentiment_agent(self, request: RunRequest, seed_data: NormalizedData):
        data = self._agent_data("sentiment", request, seed_data)
        result = SentimentAnalyst().run(data, self._make_llm_for_request(request))
        return result, data

    def _run_technical_agent(self, request: RunRequest, seed_data: NormalizedData):
        data = self._agent_data("technical", request, seed_data)
        result = TechnicalAnalyst().run(data, self._make_llm_for_request(request))
        return result, data

    def _agent_data(self, agent_name: str, request: RunRequest, seed_data: NormalizedData) -> NormalizedData:
        if request.fixture:
            data = self._scope_data(seed_data, agent_name)
            data.source_metadata["tool_call_policy"] = "fixture_data_scoped_to_agent_contract"
            data.source_metadata["agent_tool_status"] = {agent_name: {"fixture": "success"}}
            return data
        if agent_name == "technical":
            market_result = route_to_vendor("get_market_data", self.settings, request)
            swap_context_result = route_to_vendor("get_swap_market_context", self.settings, request)
            return NormalizedData(
                symbol=request.symbol,
                asset_class=request.asset_class,
                horizon=request.horizon,
                ohlcv=market_result.data,
                market_context={"okx_swap": swap_context_result.data},
                source_metadata={
                    "provider_mode": "live",
                    "tool_call_policy": "agent_called_allowed_tools",
                    "agent_tool_status": {
                        agent_name: {
                            "get_market_data": market_result.status,
                            "get_swap_market_context": swap_context_result.status,
                        }
                    },
                    "warnings": market_result.warnings + swap_context_result.warnings,
                },
            )
        if agent_name == "news_impact":
            news_result = route_to_vendor("get_news", self.settings, request)
            return NormalizedData(
                symbol=request.symbol,
                asset_class=request.asset_class,
                horizon=request.horizon,
                news_events=news_result.data,
                source_metadata={
                    "provider_mode": "live",
                    "tool_call_policy": "agent_called_allowed_tools",
                    "agent_tool_status": {agent_name: {"get_news": news_result.status}},
                    "warnings": news_result.warnings,
                },
            )
        if agent_name == "sentiment":
            sentiment_result = route_to_vendor("get_sentiment_inputs", self.settings, request)
            return NormalizedData(
                symbol=request.symbol,
                asset_class=request.asset_class,
                horizon=request.horizon,
                sentiment_inputs=sentiment_result.data,
                source_metadata={
                    "provider_mode": "live",
                    "tool_call_policy": "agent_called_allowed_tools",
                    "agent_tool_status": {agent_name: {"get_sentiment_inputs": sentiment_result.status}},
                    "warnings": sentiment_result.warnings,
                },
            )
        if agent_name == "fundamental_macro":
            fundamentals_result = route_to_vendor("get_fundamentals", self.settings, request)
            news_result = route_to_vendor("get_news", self.settings, request)
            fundamentals = fundamentals_result.data if isinstance(fundamentals_result.data, dict) else {}
            return NormalizedData(
                symbol=request.symbol,
                asset_class=request.asset_class,
                horizon=request.horizon,
                news_events=news_result.data,
                source_metadata={
                    "provider_mode": "live",
                    "tool_call_policy": "agent_called_allowed_tools",
                    "agent_tool_status": {
                        agent_name: {
                            "get_fundamentals": fundamentals_result.status,
                            "get_news": news_result.status,
                        }
                    },
                    "warnings": fundamentals_result.warnings + news_result.warnings,
                    **fundamentals,
                },
            )
        raise ValueError(f"No agent data tool plan registered for {agent_name}")

    def _bull_researcher(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            request = RunRequest.model_validate(s["request"])
            llm = self._make_llm(request, s)
            fundamental = FundamentalMacroResult.model_validate(s["fundamental"])
            news = NewsImpactResult.model_validate(s["news"])
            sentiment = SentimentResult.model_validate(s["sentiment"])
            technical = TechnicalState.model_validate(s["technical"])
            constructive = ConstructiveCaseAnalyst().run(fundamental, news, sentiment, technical, llm)
            s["constructive"] = constructive.model_dump(mode="json")
            self._write_bull_risk_outputs(s)
            return s

        return self._run_step(state, "bull_researcher", work)

    def _bear_researcher(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            request = RunRequest.model_validate(s["request"])
            llm = self._make_llm(request, s)
            fundamental = FundamentalMacroResult.model_validate(s["fundamental"])
            news = NewsImpactResult.model_validate(s["news"])
            sentiment = SentimentResult.model_validate(s["sentiment"])
            technical = TechnicalState.model_validate(s["technical"])
            constructive = ResearchCase.model_validate(s["constructive"])
            risk = RiskCaseAnalyst().run(fundamental, news, sentiment, technical, constructive, llm)
            s["risk"] = risk.model_dump(mode="json")
            self._write_bull_risk_outputs(s)
            return s

        return self._run_step(state, "bear_researcher", work)

    def _bull_bear_research_debate(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            constructive = ResearchCase.model_validate(s["constructive"])
            risk = ResearchCase.model_validate(s["risk"])
            shared_evidence = sorted(set(constructive.evidence).intersection(risk.evidence))
            debate = {
                "team": "Bull/Bear Research Debate",
                "contract": get_agent_contract("bull_bear_research_debate").model_dump(mode="json"),
                "bull_researcher": {
                    "role": "constructive researcher",
                    "thesis": constructive.thesis,
                    "evidence": constructive.evidence,
                    "conditions": constructive.conditions,
                    "confidence": constructive.confidence,
                },
                "bear_researcher": {
                    "role": "risk researcher",
                    "thesis": risk.thesis,
                    "evidence": risk.evidence,
                    "conditions": risk.conditions,
                    "confidence": risk.confidence,
                },
                "points_of_agreement": shared_evidence,
                "key_tensions": [
                    "constructive case requires confirmation from support and macro conditions",
                    "risk case emphasizes repricing, volatility, and data coverage uncertainty",
                ],
                "reporter_handoff": (
                    "Produce balanced research context only. Avoid buy/sell wording, order language, "
                    "position sizing, and profitability claims."
                ),
            }
            s["research_debate"] = debate
            self.store.write_json(s["run_id"], "research_debate.json", debate)
            return s

        return self._run_step(state, "bull_bear_research_debate", work)

    def _research_reporter(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            request = RunRequest.model_validate(s["request"])
            llm = self._make_llm(request, s)
            data = NormalizedData.model_validate(s["data"])
            fundamental = FundamentalMacroResult.model_validate(s["fundamental"])
            news = NewsImpactResult.model_validate(s["news"])
            sentiment = SentimentResult.model_validate(s["sentiment"])
            technical = TechnicalState.model_validate(s["technical"])
            constructive = ResearchCase.model_validate(s["constructive"])
            risk = ResearchCase.model_validate(s["risk"])
            final = ResearchReporter().run(
                data,
                fundamental,
                news,
                sentiment,
                technical,
                constructive,
                risk,
                list(s.get("warnings", [])),
                llm,
            )
            s["final_context"] = final.model_dump(mode="json")
            markdown = render_markdown_brief(final)
            violations = find_guardrail_violations(final.model_dump_json() + "\n" + markdown)
            if violations:
                s["warnings"] = list(s.get("warnings", [])) + [f"guardrail warnings: {', '.join(violations)}"]
            return s

        return self._run_step(state, "research_reporter", work)

    def _final_market_context_cache(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            final = FinalResearchContext.model_validate(s["final_context"])
            cache = {
                "cache_name": "final_market_context_cache",
                "contract": get_agent_contract("final_market_context_cache").model_dump(mode="json"),
                "run_id": s["run_id"],
                "cache_key": f"{final.symbol}:{final.asset_class}:{final.horizon}",
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "final_market_context": final.model_dump(mode="json"),
                "analyst_team_summary": s.get("analyst_team", {}).get("synthesis", {}),
                "research_debate_summary": {
                    "bull_thesis": s.get("research_debate", {}).get("bull_researcher", {}).get("thesis"),
                    "bear_thesis": s.get("research_debate", {}).get("bear_researcher", {}).get("thesis"),
                    "key_tensions": s.get("research_debate", {}).get("key_tensions", []),
                },
                "usage_boundary": {
                    "purpose": "research context cache for downstream strategy research",
                    "not_for": ["order execution", "position sizing", "profit guarantee"],
                },
            }
            s["final_market_context_cache"] = cache
            path = self.store.write_json(s["run_id"], "final_market_context_cache.json", cache)
            output_paths = dict(s.get("output_paths", {}))
            output_paths["final_market_context_cache"] = str(path)
            s["output_paths"] = output_paths
            return s

        return self._run_step(state, "final_market_context_cache", work)

    def _persist(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            final = FinalResearchContext.model_validate(s["final_context"])
            trace = RunTrace.model_validate(s["trace"])
            trace.completed_at = datetime.now(timezone.utc)
            trace.warnings = list(dict.fromkeys(trace.warnings + s.get("warnings", [])))
            trace.completed_steps = s.get("completed_steps", [])
            raw_tokens = approximate_tokens({"data": s.get("data"), "analyst_outputs": s.get("fundamental")})
            final_tokens = approximate_tokens(final.model_dump(mode="json"))
            violations = find_guardrail_violations(final.model_dump_json() + "\n" + render_markdown_brief(final))
            metrics = RunMetrics(
                total_latency_sec=round((trace.completed_at - trace.started_at).total_seconds(), 3),
                raw_input_tokens=raw_tokens,
                final_context_tokens=final_tokens,
                compression_ratio=compression_ratio(raw_tokens, final_tokens),
                guardrail_violations=violations,
            )
            s["metrics"] = metrics.model_dump(mode="json")
            markdown = render_markdown_report(s)
            paths = dict(s.get("output_paths", {}))
            paths.update(
                {
                    "final_research_context": str(self.store.write_json(s["run_id"], "final_research_context.json", final)),
                    "research_brief": str(self.store.write_text(s["run_id"], "research_brief.md", markdown)),
                    "trace": str(self.store.write_json(s["run_id"], "trace.json", trace)),
                    "metrics": str(self.store.write_json(s["run_id"], "metrics.json", metrics)),
                }
            )
            s["trace"] = trace.model_dump(mode="json")
            s["output_paths"] = paths
            return s

        return self._run_step(state, "persist", work, checkpoint_after=False)

    def _run_step(self, state: WorkflowState, step_name: str, work, checkpoint_after: bool = True) -> WorkflowState:
        completed = list(state.get("completed_steps", []))
        if step_name in completed:
            self._emit_progress("agent_status", step_name, "completed", state=state, payload={"skipped": True})
            return state
        get_agent_contract(step_name)
        started = time.perf_counter()
        self._emit_progress("agent_status", step_name, "in_progress", state=state)
        try:
            state = work(state)
            status = "success"
            warnings: list[str] = []
        except Exception as exc:
            status = "failed"
            warnings = [str(exc)]
            state["warnings"] = list(state.get("warnings", [])) + [f"{step_name} failed: {exc}"]
            self._emit_progress("agent_status", step_name, "failed", state=state, payload={"warnings": warnings})
            raise
        finally:
            latency = round(time.perf_counter() - started, 4)
            if step_name != "persist":
                self._append_trace(state, AgentTrace(name=step_name, status=status, latency_sec=latency, warnings=warnings))
        completed.append(step_name)
        state["completed_steps"] = completed
        self._emit_progress(
            "agent_result",
            step_name,
            "completed",
            state=state,
            payload={"latency_sec": latency, "warnings": warnings},
        )
        if checkpoint_after and state.get("checkpoint_enabled"):
            self.store.save_checkpoint(state["run_id"], self._checkpoint_state(state))
        return state

    def _emit_progress(
        self,
        event_type: str,
        name: str,
        status: str,
        state: WorkflowState | None = None,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(
            {
                "type": event_type,
                "name": name,
                "status": status,
                "state": state,
                "payload": payload or {},
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    def _make_llm(self, request: RunRequest, state: WorkflowState):
        llm = self._make_llm_for_request(request)
        if llm.provider == "fake" and request.llm_provider == "auto":
            warning = "Ollama unavailable; fixture run used deterministic fake LLM fallback."
            if warning not in state.get("warnings", []):
                state["warnings"] = list(state.get("warnings", [])) + [warning]
        return llm

    def _make_llm_for_request(self, request: RunRequest):
        return make_llm_client(
            self.settings,
            request.llm_provider,
            request.model,
            allow_fake_fallback=bool(request.fixture),
        )

    def _write_analyst_outputs(self, state: WorkflowState) -> None:
        outputs = {
            "fundamental_macro": state.get("fundamental"),
            "news_impact": state.get("news"),
            "sentiment": state.get("sentiment"),
            "technical": state.get("technical"),
        }
        self.store.write_json(
            state["run_id"],
            "analyst_outputs.json",
            {key: value for key, value in outputs.items() if value is not None},
        )

    def _write_bull_risk_outputs(self, state: WorkflowState) -> None:
        outputs = {
            "constructive_case": state.get("constructive"),
            "risk_case": state.get("risk"),
        }
        self.store.write_json(
            state["run_id"],
            "bull_risk_outputs.json",
            {key: value for key, value in outputs.items() if value is not None},
        )

    @staticmethod
    def _merge_agent_data(
        request: RunRequest, seed_data: NormalizedData, data_slices: dict[str, dict[str, Any]]
    ) -> NormalizedData:
        slices = {name: NormalizedData.model_validate(value) for name, value in data_slices.items()}
        news_events = _dedupe_news_events(
            [
                event
                for name in ("fundamental_macro", "news_impact")
                for event in slices.get(name, NormalizedData(symbol=request.symbol, asset_class=request.asset_class, horizon=request.horizon)).news_events
            ]
        )
        sentiment_inputs = slices.get(
            "sentiment", NormalizedData(symbol=request.symbol, asset_class=request.asset_class, horizon=request.horizon)
        ).sentiment_inputs
        ohlcv = slices.get("technical", NormalizedData(symbol=request.symbol, asset_class=request.asset_class, horizon=request.horizon)).ohlcv
        market_context = slices.get(
            "technical", NormalizedData(symbol=request.symbol, asset_class=request.asset_class, horizon=request.horizon)
        ).market_context
        source_metadata: dict[str, Any] = {
            "provider_mode": seed_data.source_metadata.get("provider_mode", "live"),
            "tool_call_policy": "analyst_agents_called_allowed_tools",
            "agent_tool_status": {},
            "agent_tool_warnings": {},
        }
        for name, data in slices.items():
            source_metadata["agent_tool_status"].update(data.source_metadata.get("agent_tool_status", {}))
            warnings = data.source_metadata.get("warnings") or []
            if warnings:
                source_metadata["agent_tool_warnings"][name] = warnings
            for key, value in data.source_metadata.items():
                if key not in {"provider_mode", "tool_call_policy", "agent_tool_status", "warnings"}:
                    source_metadata[key] = value
        return NormalizedData(
            symbol=request.symbol,
            asset_class=request.asset_class,
            horizon=request.horizon,
            ohlcv=ohlcv,
            news_events=news_events,
            sentiment_inputs=sentiment_inputs,
            market_context=market_context,
            source_metadata=source_metadata,
        )

    @staticmethod
    def _scope_data(data: NormalizedData, agent_name: str) -> NormalizedData:
        if agent_name == "fundamental_macro":
            metadata_keys = {"fmp_profile", "fmp_quote", "finnhub_quote", "source_status", "provider_mode"}
            return NormalizedData(
                symbol=data.symbol,
                asset_class=data.asset_class,
                horizon=data.horizon,
                news_events=data.news_events,
                source_metadata={key: value for key, value in data.source_metadata.items() if key in metadata_keys},
            )
        if agent_name == "news_impact":
            return NormalizedData(
                symbol=data.symbol,
                asset_class=data.asset_class,
                horizon=data.horizon,
                news_events=data.news_events,
            )
        if agent_name == "sentiment":
            return NormalizedData(
                symbol=data.symbol,
                asset_class=data.asset_class,
                horizon=data.horizon,
                sentiment_inputs=data.sentiment_inputs,
            )
        if agent_name == "technical":
            return NormalizedData(
                symbol=data.symbol,
                asset_class=data.asset_class,
                horizon=data.horizon,
                ohlcv=data.ohlcv,
                market_context=data.market_context,
            )
        raise ValueError(f"No data scope registered for agent {agent_name}")

    @staticmethod
    def _append_trace(state: WorkflowState, agent_trace: AgentTrace) -> None:
        trace = RunTrace.model_validate(state["trace"])
        trace.agents.append(agent_trace)
        state["trace"] = trace.model_dump(mode="json")

    @staticmethod
    def _checkpoint_state(state: WorkflowState) -> dict[str, Any]:
        return dict(state)

    @staticmethod
    def _new_run_id(request: RunRequest) -> str:
        safe_symbol = request.symbol.lower().replace("/", "-").replace(":", "-")
        return f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{safe_symbol}_{uuid.uuid4().hex[:8]}"

    def _safe_warning(self, text: str) -> str:
        return redact_secrets(
            text,
            [
                self.settings.fmp_api_key,
                self.settings.finnhub_api_key,
                self.settings.tavily_api_key,
                self.settings.jin10_api_key,
            ],
        )


def render_markdown_brief(final: FinalResearchContext) -> str:
    drivers = "\n".join(f"- {item}" for item in final.key_drivers) or "- None"
    risks = "\n".join(f"- {item}" for item in final.key_risks) or "- None"
    return (
        f"# Investment Research Brief: {final.symbol}\n\n"
        f"Use as research context only. This is not financial advice, an order instruction, or position sizing guidance.\n\n"
        f"- Horizon: {final.horizon}\n"
        f"- Market regime: {final.market_regime}\n"
        f"- Balanced view: {final.balanced_view}\n"
        f"- Risk level: {final.risk_level}\n"
        f"- Confidence: {final.confidence}\n\n"
        f"## Context\n\n{final.news_impact_summary}\n\n"
        f"## Technical State\n\n{final.technical_summary}\n\n"
        f"## Constructive Case\n\n{final.constructive_case.thesis}\n\n"
        f"## Risk Case\n\n{final.risk_case.thesis}\n\n"
        f"## Key Drivers\n\n{drivers}\n\n"
        f"## Key Risks\n\n{risks}\n"
    )


def render_markdown_report(state: dict[str, Any]) -> str:
    final = FinalResearchContext.model_validate(state["final_context"])
    fundamental = FundamentalMacroResult.model_validate(state["fundamental"])
    news = NewsImpactResult.model_validate(state["news"])
    sentiment = SentimentResult.model_validate(state["sentiment"])
    technical = TechnicalState.model_validate(state["technical"])
    constructive = ResearchCase.model_validate(state["constructive"])
    risk = ResearchCase.model_validate(state["risk"])
    data = NormalizedData.model_validate(state["data"])
    metrics = state.get("metrics") or {}

    return "\n\n".join(
        [
            f"# Investment Research Desk Report: {final.symbol}",
            (
                "Use as research context only. This is not financial advice, an order instruction, "
                "position sizing guidance, or a profitability claim."
            ),
            "## Executive Context\n"
            f"- Asset class: {final.asset_class}\n"
            f"- Horizon: {final.horizon}\n"
            f"- Market regime: {final.market_regime}\n"
            f"- Balanced view: {final.balanced_view}\n"
            f"- Risk level: {final.risk_level}\n"
            f"- Confidence: {final.confidence}",
            "## Fundamental / Macro Analyst\n"
            f"- View: {fundamental.fundamental_view}\n"
            f"- Confidence: {fundamental.confidence}\n\n"
            f"### Key Drivers\n{_md_list(fundamental.key_drivers)}\n\n"
            f"### Concerns\n{_md_list(fundamental.concerns)}\n\n"
            f"### Evidence\n{_md_list(fundamental.evidence)}",
            "## News / Macro Impact Analyst\n"
            f"- Impact logic: {news.impact_logic}\n"
            f"- Confidence: {news.confidence}\n"
            f"- Asset impact: {news.asset_impact.get(final.symbol, 'mixed')}\n\n"
            f"### Dominant Events\n{_md_list(news.dominant_events)}\n\n"
            f"### Evidence\n{_md_list(news.evidence)}",
            "## Sentiment Analyst\n"
            f"- Crowd mood: {sentiment.crowd_mood}\n"
            f"- Label: {sentiment.sentiment_label}\n"
            f"- Score: {sentiment.sentiment_score}\n"
            f"- Confidence: {sentiment.confidence}\n\n"
            f"### Evidence\n{_md_list(sentiment.evidence)}",
            "## Technical Analyst\n"
            f"- View: {technical.technical_view}\n"
            f"- Trend: {technical.trend}\n"
            f"- Momentum: {technical.momentum}\n"
            f"- Volatility regime: {technical.volatility_regime}\n"
            f"- RSI 14: {technical.rsi_14}\n"
            f"- MACD state: {technical.macd_state}\n"
            f"- ATR 14: {technical.atr_14}\n"
            f"- Realized volatility: {technical.realized_volatility}\n"
            f"- Max drawdown: {technical.max_drawdown}\n"
            f"- OKX mark price: {technical.mark_price}\n"
            f"- OKX index price: {technical.index_price}\n"
            f"- OKX funding rate: {technical.funding_rate}\n"
            f"- OKX open interest: {technical.open_interest}\n"
            f"- OKX orderbook imbalance: {technical.orderbook_imbalance}\n"
            f"- SWAP context: {technical.swap_context_summary or 'None'}\n"
            f"- Support zones: {', '.join(map(str, technical.support_zones)) or 'None'}\n"
            f"- Resistance zones: {', '.join(map(str, technical.resistance_zones)) or 'None'}\n"
            f"- Confidence: {technical.confidence}",
            "## Bull / Constructive Researcher\n"
            f"### Thesis\n{constructive.thesis}\n\n"
            f"### Evidence\n{_md_list(constructive.evidence)}\n\n"
            f"### Conditions\n{_md_list(constructive.conditions)}\n\n"
            f"- Confidence: {constructive.confidence}",
            "## Bear / Risk Researcher\n"
            f"### Thesis\n{risk.thesis}\n\n"
            f"### Evidence\n{_md_list(risk.evidence)}\n\n"
            f"### Conditions\n{_md_list(risk.conditions)}\n\n"
            f"- Confidence: {risk.confidence}",
            "## Final Research Reporter\n"
            f"### Fundamental Summary\n{final.fundamental_summary}\n\n"
            f"### News Impact Summary\n{final.news_impact_summary}\n\n"
            f"### Sentiment Summary\n{final.sentiment_summary}\n\n"
            f"### Technical Summary\n{final.technical_summary}\n\n"
            f"### Key Drivers\n{_md_list(final.key_drivers)}\n\n"
            f"### Key Risks\n{_md_list(final.key_risks)}\n\n"
            f"### Uncertainty Factors\n{_md_list(final.uncertainty_factors)}",
            "## Data And Run Metadata\n"
            f"- OHLCV bars: {len(data.ohlcv)}\n"
            f"- Market context sections: {', '.join(data.market_context.keys()) or 'None'}\n"
            f"- News events: {len(data.news_events)}\n"
            f"- Sentiment inputs: {len(data.sentiment_inputs)}\n"
            f"- Provider mode: {data.source_metadata.get('provider_mode', 'unknown')}\n"
            f"- Tool policy: {data.source_metadata.get('tool_call_policy', 'unknown')}\n"
            f"- Guardrail violations: {_guardrail_summary(metrics)}",
            f"## Usage Constraints\n{_md_list(final.usage_constraints)}",
            f"## Downstream Context\n{final.downstream_agent_context}",
        ]
    )


def _md_list(items: list[Any]) -> str:
    return "\n".join(f"- {item}" for item in items) if items else "- None"


def _guardrail_summary(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "pending during report render"
    violations = metrics.get("guardrail_violations", [])
    return ", ".join(violations) if violations else "None"


def _dedupe_news_events(events: list[NewsEvent]) -> list[NewsEvent]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[NewsEvent] = []
    for event in events:
        key = (event.title.strip().lower(), event.source.strip().lower(), event.published_at.isoformat())
        if key not in seen:
            deduped.append(event)
            seen.add(key)
    return deduped
