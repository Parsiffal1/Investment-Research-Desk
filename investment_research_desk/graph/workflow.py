from __future__ import annotations

import time
import uuid
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

from investment_research_desk.agents import (
    ConstructiveCaseAnalyst,
    DebateModerator,
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
from investment_research_desk.sentiment_runtime import make_sentiment_classifier
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
                data.source_metadata["language"] = request.language
            else:
                data = NormalizedData(
                    symbol=request.symbol,
                    asset_class=request.asset_class,
                    horizon=request.horizon,
                    source_metadata={
                        "provider_mode": "live",
                        "tool_call_policy": "analyst_agents_call_allowed_tools",
                        "agent_tool_status": {},
                        "language": request.language,
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
            classifier = self._make_sentiment_classifier_for_request(request)
            sentiment = SentimentAnalyst().run(data, llm, classifier)
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
            data.source_metadata["agent_execution_mode"] = self._analyst_execution_mode(request)
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
                "execution_mode": "parallel_thread_pool" if self._analyst_execution_mode(request) == "parallel" else "sequential",
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
        mode = self._analyst_execution_mode(request)
        outputs: dict[str, dict[str, Any]] = {}
        data_slices: dict[str, dict[str, Any]] = {}
        traces: list[AgentTrace] = []
        warnings: list[str] = []
        if mode == "sequential":
            for name, call in jobs.items():
                self._emit_progress("agent_status", name, "in_progress")
                try:
                    output, data_slice, trace = self._run_parallel_agent(name, call)
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
                    warnings.append(f"{name} failed in sequential analyst layer: {exc}")
                    traces.append(exc.trace)
                    self._emit_progress("agent_status", name, "failed", payload={"warnings": exc.trace.warnings})
                    raise
            return outputs, data_slices, traces, warnings
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

    def _analyst_execution_mode(self, request: RunRequest) -> str:
        configured = (self.settings.agent_execution_mode or "sequential").strip().lower()
        if configured not in {"sequential", "parallel"}:
            return "sequential"
        if configured == "parallel":
            return "parallel"
        if request.fixture or request.llm_provider == "fake":
            return "parallel"
        return "sequential"

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
        data = (
            self._agent_data("fundamental_macro", request, seed_data)
            if request.fixture
            else self._run_agent_tool_loop(
                "fundamental_macro",
                request,
                ["get_fundamentals", "get_news"],
                required_tools=["get_fundamentals"],
                max_rounds=4,
            )
        )
        result = FundamentalMacroAnalyst().run(data, self._make_llm_for_request(request))
        return result, data

    def _run_news_agent(self, request: RunRequest, seed_data: NormalizedData):
        if request.fixture:
            data = self._agent_data("news_impact", request, seed_data)
            result = NewsImpactAnalyst().run(data, self._make_llm_for_request(request))
            return result, data
        data = self._run_agent_tool_loop(
            "news_impact",
            request,
            ["get_news", "get_global_news"],
            required_tools=["get_news"],
            max_rounds=5,
        )
        result = NewsImpactAnalyst().run(data, self._make_llm_for_request(request))
        return result, data

    def _run_sentiment_agent(self, request: RunRequest, seed_data: NormalizedData):
        data = (
            self._agent_data("sentiment", request, seed_data)
            if request.fixture
            else self._run_agent_tool_loop(
                "sentiment",
                request,
                ["get_sentiment_inputs"],
                required_tools=["get_sentiment_inputs"],
                max_rounds=3,
            )
        )
        classifier = self._make_sentiment_classifier_for_request(request)
        if classifier is not None:
            data.source_metadata["sentiment_runtime"] = classifier.runtime_metadata()
        result = SentimentAnalyst().run(data, self._make_llm_for_request(request), classifier)
        return result, data

    def _run_technical_agent(self, request: RunRequest, seed_data: NormalizedData):
        data = (
            self._agent_data("technical", request, seed_data)
            if request.fixture
            else self._run_agent_tool_loop(
                "technical",
                request,
                ["get_market_data", "get_swap_market_context"],
                required_tools=["get_market_data"],
                max_rounds=4,
            )
        )
        result = TechnicalAnalyst().run(data, self._make_llm_for_request(request))
        return result, data

    def _run_agent_tool_loop(
        self,
        agent_name: str,
        request: RunRequest,
        tool_names: list[str],
        required_tools: list[str],
        max_rounds: int,
    ) -> NormalizedData:
        contract = get_agent_contract(agent_name)
        llm = self._make_llm_for_request(request)
        collected: dict[str, Any] = {
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
            "max_tool_calls": max(1, min(self.settings.agent_max_tool_calls, max_rounds * max(1, len(tool_names)))),
            "executed_tool_counts": {},
            "timeout": False,
            "timeout_detail": None,
        }

        def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            payload = self._execute_agent_tool(agent_name, request, name, arguments, collected)
            collected["tool_calls"].append({"name": name, "arguments": arguments, "result": payload})
            return payload

        prompt = self._agent_tool_loop_prompt(agent_name, request, tool_names)
        try:
            raw = llm.chat_tools_json(contract.system_prompt, prompt, self._tool_specs(tool_names), execute_tool, max_rounds=max_rounds)
        except Exception as exc:
            collected["timeout"] = _looks_like_timeout(exc)
            collected["timeout_detail"] = self._safe_warning(str(exc))
            retained = _retained_evidence_count(collected)
            reason = "timed out" if collected["timeout"] else "failed"
            collected["warnings"].append(
                self._safe_warning(
                    f"{agent_name} tool loop {reason}; retained_partial_evidence={retained}; "
                    f"executed_tool_calls={collected['executed_tool_count']}; detail={exc}"
                )
            )

        called = {call.get("name") for call in collected["tool_calls"] if isinstance(call, dict)}
        for required in required_tools:
            if required not in called:
                payload = execute_tool(required, self._default_tool_arguments(required, request))
                collected["contract_floor_calls"].append({"name": required, "arguments": self._default_tool_arguments(required, request), "result": payload})

        return self._collected_tool_data(agent_name, request, collected)

    def _execute_agent_tool(
        self,
        agent_name: str,
        request: RunRequest,
        name: str,
        arguments: dict[str, Any],
        collected: dict[str, Any],
    ) -> dict[str, Any]:
        if collected["executed_tool_count"] >= collected["max_tool_calls"]:
            return {"error": f"tool call budget exceeded for {agent_name}: max={collected['max_tool_calls']}"}
        per_tool_counts = collected["executed_tool_counts"]
        if per_tool_counts.get(name, 0) >= 4:
            return {"error": f"tool call budget exceeded for {agent_name}.{name}: max=4"}
        collected["executed_tool_count"] += 1
        per_tool_counts[name] = per_tool_counts.get(name, 0) + 1
        query = self._tool_query(name, request, arguments)
        symbol = str(arguments.get("symbol") or request.symbol).strip() or request.symbol
        local_request = request.model_copy(update={"symbol": symbol, "tool_query": query})
        result = route_to_vendor(name, self.settings, local_request)
        collected["tool_status"][name] = result.status
        collected["warnings"].extend(result.warnings)
        if name in {"get_news", "get_global_news"}:
            events = [event for event in result.data if isinstance(event, NewsEvent)]
            collected["news_events"].extend(events)
            return {"status": result.status, "warnings": result.warnings, "events": [event.model_dump(mode="json") for event in events[:12]]}
        if name == "get_sentiment_inputs":
            raw_inputs = result.data if isinstance(result.data, list) else []
            inputs, rejected = _filter_relevant_sentiment_inputs(raw_inputs, request)
            collected["sentiment_inputs"].extend(inputs)
            collected["source_metadata"]["sentiment_filter"] = {
                "raw_count": len(raw_inputs),
                "kept_count": len(inputs),
                "rejected_count": len(rejected),
                "query": query,
            }
            if raw_inputs and not inputs:
                collected["warnings"].append("sentiment relevance filter rejected all retrieved inputs")
            return {
                "status": result.status,
                "warnings": result.warnings,
                "sentiment_inputs": [item.model_dump(mode="json") for item in inputs[:12]],
                "rejected_count": len(rejected),
            }
        if name == "get_market_data":
            collected["ohlcv"] = result.data if isinstance(result.data, list) else []
            return {"status": result.status, "warnings": result.warnings, "bar_count": len(collected["ohlcv"])}
        if name == "get_swap_market_context":
            if result.data:
                collected["market_context"]["okx_swap"] = result.data
            return {"status": result.status, "warnings": result.warnings, "market_context": result.data}
        if name == "get_fundamentals":
            if isinstance(result.data, dict):
                collected["source_metadata"].update(result.data)
            return {"status": result.status, "warnings": result.warnings, "fundamentals": result.data}
        return {"error": f"unsupported tool: {name}"}

    @staticmethod
    def _tool_query(name: str, request: RunRequest, arguments: dict[str, Any]) -> str:
        query = str(arguments.get("query") or "").strip()
        if name == "get_sentiment_inputs":
            if not query or query.upper() == request.symbol.upper() or "SWAP" in query.upper():
                return _default_sentiment_query(request)
            return query
        return query or request.symbol

    def _collected_tool_data(self, agent_name: str, request: RunRequest, collected: dict[str, Any]) -> NormalizedData:
        source_metadata = {
            "provider_mode": "live",
            "tool_call_policy": "tradingagents_style_llm_tool_loop",
            "agent_tool_status": {agent_name: collected["tool_status"]},
            "warnings": collected["warnings"],
            "llm_tool_calls": collected["tool_calls"],
            "contract_floor_calls": collected["contract_floor_calls"],
            "tool_call_budget": {
                "executed": collected["executed_tool_count"],
                "max": collected["max_tool_calls"],
                "per_tool": collected["executed_tool_counts"],
            },
            "tool_loop_timeout": collected["timeout"],
            "tool_loop_timeout_detail": collected["timeout_detail"],
            "retained_partial_evidence": _retained_evidence_count(collected),
            "language": request.language,
            **collected["source_metadata"],
        }
        return NormalizedData(
            symbol=request.symbol,
            asset_class=request.asset_class,
            horizon=request.horizon,
            ohlcv=collected["ohlcv"],
            news_events=_dedupe_news_events(collected["news_events"]),
            sentiment_inputs=collected["sentiment_inputs"],
            market_context=collected["market_context"],
            source_metadata=source_metadata,
        )

    @staticmethod
    def _agent_tool_loop_prompt(agent_name: str, request: RunRequest, tool_names: list[str]) -> str:
        tool_lines = "\n".join(f"- {name}" for name in tool_names)
        financial_scope = _financial_tool_query_instruction(request)
        return (
            f"Agent: {agent_name}\n"
            f"Instrument: {request.symbol}\n"
            f"Asset class: {request.asset_class}\n"
            f"Horizon: {request.horizon}\n\n"
            "Use the available tools to collect only the evidence this agent needs. "
            "Before calling a tool, refine the query or symbol argument for this instrument. "
            f"{financial_scope} "
            "Call tools only while they add useful evidence, then stop and return a compact JSON object with a summary. "
            "Do not make financial advice, order, or position-size statements.\n\n"
            f"Available tools:\n{tool_lines}"
        )

    @staticmethod
    def _default_tool_arguments(name: str, request: RunRequest) -> dict[str, Any]:
        if name in {"get_news", "get_global_news"}:
            return {"symbol": request.symbol, "query": _default_financial_query(request), "limit": 5}
        if name == "get_sentiment_inputs":
            return {"symbol": request.symbol, "query": _default_sentiment_query(request)}
        return {"symbol": request.symbol}

    @staticmethod
    def _tool_specs(tool_names: list[str]) -> list[dict[str, Any]]:
        specs = {
            "get_market_data": {
                "description": "Retrieve OHLCV bars for the exact market instrument.",
                "properties": {"symbol": {"type": "string"}, "horizon": {"type": "string"}},
                "required": ["symbol"],
            },
            "get_swap_market_context": {
                "description": "Retrieve OKX public SWAP mark/index, funding, open interest, and orderbook context.",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
            "get_news": {
                "description": (
                    "Retrieve targeted financial news. Keep symbol as the exact ticker/instrument for ticker-scoped "
                    "APIs and put the expanded finance-specific search phrase in query."
                ),
                "properties": {
                    "symbol": {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
            "get_global_news": {
                "description": (
                    "Retrieve broader macro, policy, liquidity, and cross-asset market news with a finance-specific "
                    "query. Include symbol when the macro query is anchored to the instrument."
                ),
                "properties": {
                    "symbol": {"type": "string"},
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 10},
                },
                "required": ["query"],
            },
            "get_sentiment_inputs": {
                "description": (
                    "Retrieve search/social/commentary sentiment inputs for the financial instrument. Keep symbol "
                    "exact for ticker-scoped social APIs and put expanded finance-specific terms in query."
                ),
                "properties": {"symbol": {"type": "string"}, "query": {"type": "string"}},
                "required": ["symbol"],
            },
            "get_fundamentals": {
                "description": "Retrieve quote, profile, and company context for the instrument where available.",
                "properties": {"symbol": {"type": "string"}},
                "required": ["symbol"],
            },
        }
        return [
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": specs[name]["description"],
                    "parameters": {
                        "type": "object",
                        "properties": specs[name]["properties"],
                        "required": specs[name]["required"],
                    },
                },
            }
            for name in tool_names
        ]

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
                market_context={"okx_swap": swap_context_result.data} if swap_context_result.data else {},
                source_metadata={
                    "provider_mode": "live",
                    "language": request.language,
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
                    "language": request.language,
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
                    "language": request.language,
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
                    "language": request.language,
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
            constructive = ConstructiveCaseAnalyst().run(fundamental, news, sentiment, technical, llm, language=request.language)
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
            risk = RiskCaseAnalyst().run(fundamental, news, sentiment, technical, constructive, llm, language=request.language)
            s["risk"] = risk.model_dump(mode="json")
            self._write_bull_risk_outputs(s)
            return s

        return self._run_step(state, "bear_researcher", work)

    def _bull_bear_research_debate(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            request = RunRequest.model_validate(s["request"])
            llm = self._make_llm(request, s)
            fundamental = FundamentalMacroResult.model_validate(s["fundamental"])
            news = NewsImpactResult.model_validate(s["news"])
            sentiment = SentimentResult.model_validate(s["sentiment"])
            technical = TechnicalState.model_validate(s["technical"])
            constructive = ResearchCase.model_validate(s["constructive"])
            risk = ResearchCase.model_validate(s["risk"])
            debate_rounds: list[dict[str, Any]] = [
                {"round": 1, "speaker": "bull_researcher", "case": constructive.model_dump(mode="json")},
                {"round": 1, "speaker": "bear_researcher", "case": risk.model_dump(mode="json")},
            ]
            max_rounds = self._debate_rounds(request)
            for round_no in range(2, max_rounds + 1):
                bull_started = time.perf_counter()
                self._emit_progress("agent_status", f"bull_researcher_round_{round_no}", "in_progress", state=s)
                constructive = ConstructiveCaseAnalyst().run(
                    fundamental,
                    news,
                    sentiment,
                    technical,
                    llm,
                    debate_history=debate_rounds,
                    opponent_case=risk,
                    language=request.language,
                )
                self._append_trace(
                    s,
                    AgentTrace(
                        name=f"bull_researcher_round_{round_no}",
                        status="success",
                        latency_sec=round(time.perf_counter() - bull_started, 4),
                    ),
                )
                self._emit_progress(
                    "agent_result",
                    f"bull_researcher_round_{round_no}",
                    "completed",
                    state=s,
                    payload={"output": constructive.model_dump(mode="json")},
                )
                debate_rounds.append({"round": round_no, "speaker": "bull_researcher", "case": constructive.model_dump(mode="json")})

                bear_started = time.perf_counter()
                self._emit_progress("agent_status", f"bear_researcher_round_{round_no}", "in_progress", state=s)
                risk = RiskCaseAnalyst().run(
                    fundamental,
                    news,
                    sentiment,
                    technical,
                    constructive,
                    llm,
                    debate_history=debate_rounds,
                    language=request.language,
                )
                self._append_trace(
                    s,
                    AgentTrace(
                        name=f"bear_researcher_round_{round_no}",
                        status="success",
                        latency_sec=round(time.perf_counter() - bear_started, 4),
                    ),
                )
                self._emit_progress(
                    "agent_result",
                    f"bear_researcher_round_{round_no}",
                    "completed",
                    state=s,
                    payload={"output": risk.model_dump(mode="json")},
                )
                debate_rounds.append({"round": round_no, "speaker": "bear_researcher", "case": risk.model_dump(mode="json")})
            s["constructive"] = constructive.model_dump(mode="json")
            s["risk"] = risk.model_dump(mode="json")
            self._write_bull_risk_outputs(s)
            moderation = DebateModerator().run(constructive, risk, llm, language=request.language)
            debate = {
                "team": "Bull/Bear Research Debate",
                "contract": get_agent_contract("bull_bear_research_debate").model_dump(mode="json"),
                "round_count": max_rounds,
                "rounds": debate_rounds,
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
                "points_of_agreement": moderation.points_of_agreement,
                "key_tensions": moderation.key_tensions,
                "evidence_quality_notes": moderation.evidence_quality_notes,
                "reporter_handoff": moderation.reporter_handoff,
                "confidence": moderation.confidence,
            }
            s["research_debate"] = debate
            self.store.write_json(s["run_id"], "research_debate.json", debate)
            return s

        return self._run_step(state, "bull_bear_research_debate", work)

    @staticmethod
    def _debate_rounds(request: RunRequest) -> int:
        return {"quick": 1, "standard": 2, "deep": 3}.get(request.research_depth, 2)

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
                s.get("research_debate", {}),
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
            markdown = render_markdown_report(s)
            violations = find_guardrail_violations(final.model_dump_json() + "\n" + markdown)
            metrics = RunMetrics(
                total_latency_sec=round((trace.completed_at - trace.started_at).total_seconds(), 3),
                raw_input_tokens=raw_tokens,
                final_context_tokens=final_tokens,
                compression_ratio=compression_ratio(raw_tokens, final_tokens),
                guardrail_violations=violations,
                runtime={
                    "agent_execution_mode": NormalizedData.model_validate(s["data"]).source_metadata.get("agent_execution_mode"),
                    "llm_timeout_sec": self.settings.llm_timeout_sec,
                    "agent_tool_loop_timeout_sec": self.settings.agent_tool_loop_timeout_sec,
                    "agent_max_tool_calls": self.settings.agent_max_tool_calls,
                    "sentiment_runtime": NormalizedData.model_validate(s["data"]).source_metadata.get("sentiment_runtime"),
                },
            )
            s["metrics"] = metrics.model_dump(mode="json")
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

    def _make_sentiment_classifier_for_request(self, request: RunRequest):
        return make_sentiment_classifier(self.settings, request)

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
            "agent_execution_mode": "parallel" if seed_data.source_metadata.get("provider_mode") == "fixture" else "tradingagents_configured",
            "language": seed_data.source_metadata.get("language", request.language),
            "tool_call_policy": (
                "fixture_data_scoped_to_agent_contract"
                if seed_data.source_metadata.get("provider_mode") == "fixture"
                else "tradingagents_style_llm_tool_loop"
            ),
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
                    if key in {"llm_tool_calls", "contract_floor_calls", "tool_call_budget"}:
                        source_metadata.setdefault(key, {})[name] = value
                    else:
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
            metadata_keys = {"fmp_profile", "fmp_quote", "finnhub_quote", "source_status", "provider_mode", "language"}
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
                source_metadata={"language": data.source_metadata.get("language", "en")},
            )
        if agent_name == "sentiment":
            return NormalizedData(
                symbol=data.symbol,
                asset_class=data.asset_class,
                horizon=data.horizon,
                sentiment_inputs=data.sentiment_inputs,
                source_metadata={"language": data.source_metadata.get("language", "en")},
            )
        if agent_name == "technical":
            return NormalizedData(
                symbol=data.symbol,
                asset_class=data.asset_class,
                horizon=data.horizon,
                ohlcv=data.ohlcv,
                market_context=data.market_context,
                source_metadata={"language": data.source_metadata.get("language", "en")},
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
    language = str(final.source_metadata.get("language", "en"))
    h = _markdown_labels(language)
    drivers = "\n".join(f"- {item}" for item in final.key_drivers) or "- None"
    risks = "\n".join(f"- {item}" for item in final.key_risks) or "- None"
    return (
        f"# {h['brief_title']}: {final.symbol}\n\n"
        f"{h['boundary']}\n\n"
        f"- {h['horizon']}: {_report_value(final.horizon, language)}\n"
        f"- {h['market_regime']}: {_report_value(final.market_regime, language)}\n"
        f"- {h['directional_view']}: {_report_value(final.directional_view, language)}\n"
        f"- {h['directional_rationale']}: {final.directional_rationale}\n"
        f"- {h['balanced_view']}: {_report_value(final.balanced_view, language)}\n"
        f"- {h['risk_level']}: {_report_value(final.risk_level, language)}\n"
        f"- {h['confidence']}: {final.confidence}\n\n"
        f"## {h['context']}\n\n{final.news_impact_summary}\n\n"
        f"## {h['technical_state']}\n\n{final.technical_summary}\n\n"
        f"## {h['constructive_case']}\n\n{final.constructive_case.thesis}\n\n"
        f"## {h['risk_case']}\n\n{final.risk_case.thesis}\n\n"
        f"## {h['key_drivers']}\n\n{drivers}\n\n"
        f"## {h['key_risks']}\n\n{risks}\n"
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
    debate = state.get("research_debate") or {}
    metrics = state.get("metrics") or {}
    language = str(data.source_metadata.get("language", "en"))
    h = _markdown_labels(language)

    return "\n\n".join(
        [
            f"# {h['title']}: {final.symbol}",
            h["boundary"],
            f"## {h['executive']}\n"
            f"- {h['asset_class']}: {_report_value(final.asset_class, language)}\n"
            f"- {h['horizon']}: {_report_value(final.horizon, language)}\n"
            f"- {h['market_regime']}: {_report_value(final.market_regime, language)}\n"
            f"- {h['directional_view']}: {_report_value(final.directional_view, language)}\n"
            f"- {h['directional_rationale']}: {final.directional_rationale}\n"
            f"- {h['balanced_view']}: {_report_value(final.balanced_view, language)}\n"
            f"- {h['risk_level']}: {_report_value(final.risk_level, language)}\n"
            f"- {h['confidence']}: {final.confidence}",
            f"## {h['fundamental_title']}\n"
            f"- {h['view']}: {_report_value(fundamental.fundamental_view, language)}\n"
            f"- {h['confidence']}: {fundamental.confidence}\n\n"
            f"### {h['key_drivers']}\n{_md_list(fundamental.key_drivers, language)}\n\n"
            f"### {h['concerns']}\n{_md_list(fundamental.concerns, language)}\n\n"
            f"### {h['evidence']}\n{_md_list(fundamental.evidence, language)}",
            f"## {h['news_title']}\n"
            f"- {h['impact_logic']}: {news.impact_logic}\n"
            f"- {h['confidence']}: {news.confidence}\n"
            f"- {h['asset_impact']}: {_report_value(news.asset_impact.get(final.symbol, 'mixed'), language)}\n\n"
            f"### {h['dominant_events']}\n{_md_list(news.dominant_events, language)}\n\n"
            f"### {h['evidence']}\n{_md_list(news.evidence, language)}",
            f"## {h['sentiment_title']}\n"
            f"- {h['crowd_mood']}: {sentiment.crowd_mood}\n"
            f"- {h['label']}: {_report_value(sentiment.sentiment_label, language)}\n"
            f"- {h['score']}: {sentiment.sentiment_score}\n"
            f"- {h['confidence']}: {sentiment.confidence}\n\n"
            f"### {h['evidence']}\n{_md_list(sentiment.evidence, language)}",
            f"## {h['technical_title']}\n"
            f"- {h['view']}: {_report_value(technical.technical_view, language)}\n"
            f"- {h['trend']}: {_report_value(technical.trend, language)}\n"
            f"- {h['momentum']}: {_report_value(technical.momentum, language)}\n"
            f"- {h['volatility_regime']}: {_report_value(technical.volatility_regime, language)}\n"
            f"- RSI 14: {technical.rsi_14}\n"
            f"- {h['macd_state']}: {_report_value(technical.macd_state, language)}\n"
            f"- ATR 14: {technical.atr_14}\n"
            f"- {h['realized_volatility']}: {technical.realized_volatility}\n"
            f"- {h['max_drawdown']}: {technical.max_drawdown}\n"
            f"- {h['okx_mark_price']}: {_report_value(technical.mark_price, language)}\n"
            f"- {h['okx_index_price']}: {_report_value(technical.index_price, language)}\n"
            f"- {h['okx_funding_rate']}: {_report_value(technical.funding_rate, language)}\n"
            f"- {h['okx_open_interest']}: {_report_value(technical.open_interest, language)}\n"
            f"- {h['okx_orderbook_imbalance']}: {_report_value(technical.orderbook_imbalance, language)}\n"
            f"- {h['swap_context']}: {technical.swap_context_summary or _none_text(language)}\n"
            f"- {h['support_zones']}: {', '.join(map(str, technical.support_zones)) or _none_text(language)}\n"
            f"- {h['resistance_zones']}: {', '.join(map(str, technical.resistance_zones)) or _none_text(language)}\n"
            f"- {h['confidence']}: {technical.confidence}",
            f"## {h['bull_title']}\n"
            f"### {h['thesis']}\n{constructive.thesis}\n\n"
            f"### {h['evidence']}\n{_md_list(constructive.evidence, language)}\n\n"
            f"### {h['conditions']}\n{_md_list(constructive.conditions, language)}\n\n"
            f"- {h['confidence']}: {constructive.confidence}",
            f"## {h['bear_title']}\n"
            f"### {h['thesis']}\n{risk.thesis}\n\n"
            f"### {h['evidence']}\n{_md_list(risk.evidence, language)}\n\n"
            f"### {h['conditions']}\n{_md_list(risk.conditions, language)}\n\n"
            f"- {h['confidence']}: {risk.confidence}",
            f"## {h['debate']}\n"
            f"- {h['rounds']}: {debate.get('round_count', 1)}\n"
            f"- {h['points_agreement']}: {_inline_list(debate.get('points_of_agreement', []), language)}\n"
            f"- {h['key_tensions']}: {_inline_list(debate.get('key_tensions', []), language)}\n"
            f"- {h['evidence_quality']}: {_inline_list(debate.get('evidence_quality_notes', []), language)}\n\n"
            f"### {h['debate_conclusion']}\n{debate.get('reporter_handoff', _none_text(language))}\n\n"
            f"### {h['round_log']}\n{_debate_round_log(debate.get('rounds', []), language)}",
            f"## {h['final_reporter']}\n"
            f"### {h['fundamental_summary']}\n{final.fundamental_summary}\n\n"
            f"### {h['news_summary']}\n{final.news_impact_summary}\n\n"
            f"### {h['sentiment_summary']}\n{final.sentiment_summary}\n\n"
            f"### {h['technical_summary']}\n{final.technical_summary}\n\n"
            f"### {h['key_drivers']}\n{_md_list(final.key_drivers, language)}\n\n"
            f"### {h['key_risks']}\n{_md_list(final.key_risks, language)}\n\n"
            f"### {h['uncertainty']}\n{_md_list(final.uncertainty_factors, language)}",
            f"## {h['data_metadata']}\n"
            f"- {h['ohlcv_bars']}: {len(data.ohlcv)}\n"
            f"- {h['market_context_sections']}: {', '.join(data.market_context.keys()) or _none_text(language)}\n"
            f"- {h['news_events']}: {len(data.news_events)}\n"
            f"- {h['sentiment_inputs']}: {len(data.sentiment_inputs)}\n"
            f"- {h['provider_mode']}: {_report_value(data.source_metadata.get('provider_mode', 'unknown'), language)}\n"
            f"- {h['tool_policy']}: {_report_value(data.source_metadata.get('tool_call_policy', 'unknown'), language)}\n"
            f"- {h['agent_execution_mode']}: {_report_value(data.source_metadata.get('agent_execution_mode', 'unknown'), language)}\n"
            f"- {h['sentiment_runtime']}: {_report_value(data.source_metadata.get('sentiment_runtime', 'main'), language)}\n"
            f"- {h['provider_warnings']}: {_inline_list(_flatten_warnings(data.source_metadata.get('agent_tool_warnings')))}\n"
            f"- {h['guardrail_violations']}: {_report_value(_guardrail_summary(metrics), language)}",
            f"## {h['usage_constraints']}\n{_md_list(final.usage_constraints)}",
            f"## {h['downstream']}\n{final.downstream_agent_context}",
        ]
    )


def _markdown_labels(language: str) -> dict[str, str]:
    if language == "zh":
        return {
            "title": "Investment Research Desk 投研策略台报告",
            "brief_title": "Investment Research Desk 投研摘要",
            "boundary": "仅作投研上下文使用，不是投资建议、下单指令、仓位建议或收益承诺。",
            "executive": "执行摘要",
            "asset_class": "资产类别",
            "horizon": "研究周期",
            "market_regime": "市场状态",
            "directional_view": "方向判断",
            "directional_rationale": "判断依据",
            "balanced_view": "综合观点",
            "risk_level": "风险等级",
            "confidence": "置信度",
            "context": "上下文",
            "technical_state": "技术状态",
            "constructive_case": "建设性情景",
            "risk_case": "风险情景",
            "fundamental_title": "基本面/宏观分析师",
            "news_title": "新闻/宏观影响分析师",
            "sentiment_title": "情绪分析师",
            "technical_title": "技术分析师",
            "bull_title": "建设性研究员",
            "bear_title": "风险研究员",
            "final_reporter": "最终研究报告",
            "view": "观点",
            "key_drivers": "关键驱动",
            "concerns": "担忧事项",
            "evidence": "证据",
            "impact_logic": "影响逻辑",
            "asset_impact": "资产影响",
            "dominant_events": "主要事件",
            "crowd_mood": "市场情绪",
            "label": "标签",
            "score": "分数",
            "trend": "趋势",
            "momentum": "动量",
            "volatility_regime": "波动状态",
            "macd_state": "MACD状态",
            "realized_volatility": "实现波动率",
            "max_drawdown": "最大回撤",
            "okx_mark_price": "OKX标记价格",
            "okx_index_price": "OKX指数价格",
            "okx_funding_rate": "OKX资金费率",
            "okx_open_interest": "OKX未平仓量",
            "okx_orderbook_imbalance": "OKX订单簿不平衡",
            "swap_context": "SWAP上下文",
            "support_zones": "支撑区",
            "resistance_zones": "阻力区",
            "thesis": "核心论点",
            "conditions": "成立条件",
            "debate": "多空辩论",
            "rounds": "轮次",
            "points_agreement": "共识",
            "key_tensions": "主要分歧",
            "evidence_quality": "证据质量说明",
            "round_log": "辩论记录",
            "debate_conclusion": "辩论结论",
            "final_reporter": "最终研究报告",
            "fundamental_summary": "基本面/宏观摘要",
            "news_summary": "新闻影响摘要",
            "sentiment_summary": "情绪摘要",
            "technical_summary": "技术摘要",
            "key_risks": "关键风险",
            "uncertainty": "不确定因素",
            "data_metadata": "数据与运行元信息",
            "ohlcv_bars": "OHLCV K线数量",
            "market_context_sections": "市场上下文模块",
            "news_events": "新闻事件数量",
            "sentiment_inputs": "情绪输入数量",
            "provider_mode": "数据源模式",
            "tool_policy": "工具策略",
            "agent_execution_mode": "Agent执行模式",
            "sentiment_runtime": "情绪运行时",
            "provider_warnings": "数据源警告",
            "guardrail_violations": "护栏违规",
            "usage_constraints": "使用约束",
            "downstream": "下游上下文",
        }
    return {
        "title": "Investment Research Desk Report",
        "brief_title": "Investment Research Brief",
        "boundary": (
            "Use as research context only. This is not financial advice, an order instruction, "
            "position sizing guidance, or a profitability claim."
        ),
        "executive": "Executive Context",
        "asset_class": "Asset class",
        "horizon": "Horizon",
        "market_regime": "Market regime",
        "directional_view": "Directional view",
        "directional_rationale": "Directional rationale",
        "balanced_view": "Balanced view",
        "risk_level": "Risk level",
        "confidence": "Confidence",
        "context": "Context",
        "technical_state": "Technical State",
        "constructive_case": "Constructive Case",
        "risk_case": "Risk Case",
        "fundamental_title": "Fundamental / Macro Analyst",
        "news_title": "News / Macro Impact Analyst",
        "sentiment_title": "Sentiment Analyst",
        "technical_title": "Technical Analyst",
        "bull_title": "Bull / Constructive Researcher",
        "bear_title": "Bear / Risk Researcher",
        "final_reporter": "Final Research Reporter",
        "view": "View",
        "key_drivers": "Key Drivers",
        "concerns": "Concerns",
        "evidence": "Evidence",
        "impact_logic": "Impact logic",
        "asset_impact": "Asset impact",
        "dominant_events": "Dominant Events",
        "crowd_mood": "Crowd mood",
        "label": "Label",
        "score": "Score",
        "trend": "Trend",
            "momentum": "Momentum",
            "volatility_regime": "Volatility regime",
            "macd_state": "MACD state",
            "realized_volatility": "Realized volatility",
            "max_drawdown": "Max drawdown",
            "okx_mark_price": "OKX mark price",
            "okx_index_price": "OKX index price",
            "okx_funding_rate": "OKX funding rate",
            "okx_open_interest": "OKX open interest",
            "okx_orderbook_imbalance": "OKX orderbook imbalance",
            "swap_context": "SWAP context",
            "support_zones": "Support zones",
            "resistance_zones": "Resistance zones",
            "thesis": "Thesis",
            "conditions": "Conditions",
            "rounds": "Rounds",
            "points_agreement": "Points of agreement",
            "key_tensions": "Key tensions",
            "evidence_quality": "Evidence quality notes",
            "round_log": "Round Log",
        "debate": "Bull / Bear Debate",
        "debate_conclusion": "Debate Conclusion",
        "fundamental_summary": "Fundamental Summary",
        "news_summary": "News Impact Summary",
        "sentiment_summary": "Sentiment Summary",
        "technical_summary": "Technical Summary",
        "key_risks": "Key Risks",
        "uncertainty": "Uncertainty Factors",
        "data_metadata": "Data And Run Metadata",
        "ohlcv_bars": "OHLCV bars",
        "market_context_sections": "Market context sections",
        "news_events": "News events",
        "sentiment_inputs": "Sentiment inputs",
        "provider_mode": "Provider mode",
        "tool_policy": "Tool policy",
        "agent_execution_mode": "Agent execution mode",
        "sentiment_runtime": "Sentiment runtime",
        "provider_warnings": "Provider warnings",
        "guardrail_violations": "Guardrail violations",
        "usage_constraints": "Usage Constraints",
        "downstream": "Downstream Context",
    }


def _md_list(items: list[Any], language: str = "en") -> str:
    return "\n".join(f"- {item}" for item in items) if items else f"- {_none_text(language)}"


def _none_text(language: str) -> str:
    return "无" if language == "zh" else "None"


def _report_value(value: Any, language: str) -> str:
    text = str(value)
    if language != "zh":
        return text
    mapping = {
        "bullish": "看多",
        "bearish": "看空",
        "neutral": "中性",
        "mixed": "分歧",
        "mixed_to_bullish": "分歧偏多",
        "mixed_to_bearish": "分歧偏空",
        "neutral_to_bullish": "中性偏多",
        "neutral_to_bearish": "中性偏空",
        "low": "低",
        "medium": "中",
        "high": "高",
        "unknown": "未知",
        "short_term": "短期",
        "medium_term": "中期",
        "intraday": "日内",
        "swing": "波段",
        "crypto": "加密资产",
        "equity": "股票",
        "equity_index": "股票指数",
        "precious_metal": "贵金属",
        "commodity": "大宗商品",
        "fx": "外汇",
        "other": "其他",
        "uptrend": "上行趋势",
        "downtrend": "下行趋势",
        "sideways": "震荡",
        "positive": "正向",
        "negative": "负向",
        "normal": "正常",
        "None": "无",
        "none": "无",
        "fixture": "固定样例",
        "live": "实时",
        "parallel": "并行",
        "sequential": "序列",
        "main": "主模型",
        "pending during report render": "报告渲染中待检查",
        "fixture_data_scoped_to_agent_contract": "固定样例数据按Agent契约隔离",
        "tradingagents_style_llm_tool_loop": "TradingAgents风格LLM工具循环",
        "analyst_agents_call_allowed_tools": "分析师Agent调用授权工具",
    }
    return mapping.get(text, text)


def _inline_list(items: list[Any], language: str = "en") -> str:
    return "; ".join(str(item) for item in items) if items else _none_text(language)


def _debate_round_log(rounds: list[dict[str, Any]], language: str = "en") -> str:
    rows: list[str] = []
    for item in rounds:
        case = item.get("case") if isinstance(item, dict) else {}
        thesis = case.get("thesis") if isinstance(case, dict) else None
        if language == "zh":
            speaker = {"bull_researcher": "建设性研究员", "bear_researcher": "风险研究员"}.get(str(item.get("speaker")), item.get("speaker"))
            rows.append(f"- 第 {item.get('round')} 轮：{speaker} - {thesis or '未记录论点'}")
        else:
            rows.append(f"- Round {item.get('round')}: {item.get('speaker')} - {thesis or 'No thesis recorded'}")
    return "\n".join(rows) if rows else f"- {_none_text(language)}"


def _guardrail_summary(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "pending during report render"
    violations = metrics.get("guardrail_violations", [])
    return ", ".join(violations) if violations else "None"


def _flatten_warnings(value: Any) -> list[str]:
    if not value:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, dict):
        rows: list[str] = []
        for key, items in value.items():
            if isinstance(items, list):
                rows.extend(f"{key}: {item}" for item in items)
            else:
                rows.append(f"{key}: {items}")
        return rows
    return [str(value)]


def _financial_tool_query_instruction(request: RunRequest) -> str:
    default_query = _default_financial_query(request)
    return (
        "Interpret the instrument as a financial market symbol. Do not use a naturally ambiguous bare ticker as the "
        f"only web/search query. Keep symbol='{request.symbol}' for ticker-scoped APIs; use finance-specific query "
        f"phrases such as '{default_query}' for search APIs. Reject non-financial meanings before admitting evidence."
    )


def _default_financial_query(request: RunRequest) -> str:
    symbol = request.symbol.upper()
    aliases = {
        "SPY": "SPDR S&P 500 ETF Trust S&P 500 ETF market news flows macro rates earnings",
        "QQQ": "Invesco QQQ Trust Nasdaq 100 ETF market news mega-cap technology flows",
        "DIA": "SPDR Dow Jones Industrial Average ETF market news Dow blue chip equities",
        "IWM": "iShares Russell 2000 ETF small-cap equities market news",
        "GLD": "SPDR Gold Shares gold ETF bullion market news real rates dollar",
        "SLV": "iShares Silver Trust silver ETF metals market news dollar real rates",
    }
    if symbol in aliases:
        return aliases[symbol]
    if request.asset_class == "crypto":
        base = symbol.replace("-USDT-SWAP", "").replace("-USD-SWAP", "").replace("-USDT", "").replace("-USD", "")
        return f"{base} crypto perpetual swap market news funding ETF regulation liquidity"
    if request.asset_class == "equity_index":
        return f"{symbol} equity index ETF market news flows macro rates earnings"
    if request.asset_class == "equity":
        return f"{symbol} stock company news earnings guidance analyst rating sector market"
    if request.asset_class == "precious_metal":
        return f"{symbol} precious metals futures ETF market news real rates dollar inflation"
    if request.asset_class == "commodity":
        return f"{symbol} commodity futures market news supply demand macro"
    if request.asset_class == "fx":
        return f"{symbol} foreign exchange market news central bank rates macro"
    return f"{symbol} financial market news macro sector issuer"


def _default_sentiment_query(request: RunRequest) -> str:
    symbol = request.symbol.upper()
    if request.asset_class == "crypto":
        base = symbol.replace("-USDT-SWAP", "").replace("-USD-SWAP", "").replace("-USDT", "").replace("-USD", "")
        crypto_names = {"BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana", "XRP": "XRP", "DOGE": "Dogecoin"}
        name = crypto_names.get(base, base)
        return f"{name} {base} crypto price sentiment ETF flows staking regulation liquidity"
    if request.asset_class == "equity":
        return f"{symbol} stock investor sentiment earnings analyst rating market discussion"
    if request.asset_class == "equity_index":
        return f"{symbol} ETF investor sentiment flows macro rates market discussion"
    return _default_financial_query(request)


def _filter_relevant_sentiment_inputs(inputs: list[SentimentInput], request: RunRequest) -> tuple[list[SentimentInput], list[SentimentInput]]:
    kept: list[SentimentInput] = []
    rejected: list[SentimentInput] = []
    for item in inputs:
        if _sentiment_input_is_relevant(item, request):
            kept.append(item)
        else:
            rejected.append(item)
    return kept, rejected


def _sentiment_input_is_relevant(item: SentimentInput, request: RunRequest) -> bool:
    haystack = " ".join([item.text, item.url or "", item.source]).upper()
    if _is_promotional_crypto_text(haystack):
        return False
    tokens = _workflow_instrument_query_tokens(request)
    if any(_contains_financial_token(haystack, token) for token in tokens):
        return True
    if request.asset_class == "crypto" and any(term in haystack for term in {"CRYPTO", "BLOCKCHAIN", "DEFI", "STAKING", "ETF"}):
        return any(term in haystack for term in {"ETHEREUM", "BITCOIN", "SOLANA", "BTC", "ETH", "SOL", "ALTCOIN"})
    return False


def _is_promotional_crypto_text(text: str) -> bool:
    promo_terms = {
        "PRESALE",
        "PEPETO",
        "NEXT 100X",
        "100X",
        "MILLIONAIRES",
        "PRICE PREDICTION",
        "BUY BEFORE",
        "LISTING CLOSES",
        "NEW CRYPTO",
        "TOKEN SALE",
        "AIRDROP",
    }
    return any(term in text for term in promo_terms)


def _workflow_instrument_query_tokens(request: RunRequest) -> list[str]:
    symbol = request.symbol.upper()
    ignored_parts = {"USD", "USDT", "USDC", "SWAP", "PERP", "PERPETUAL"}
    parts = [part for part in re.split(r"[^A-Z0-9]+", symbol) if part and part not in ignored_parts]
    tokens = [symbol, *parts]
    aliases = {
        "BTC": "BITCOIN",
        "ETH": "ETHEREUM",
        "SOL": "SOLANA",
        "SPY": "SPDR S&P 500 ETF TRUST",
        "QQQ": "INVESCO QQQ TRUST",
        "DIA": "SPDR DOW JONES INDUSTRIAL AVERAGE ETF",
        "IWM": "ISHARES RUSSELL 2000 ETF",
        "GLD": "SPDR GOLD SHARES",
        "SLV": "ISHARES SILVER TRUST",
        "XAU": "GOLD",
        "GC": "GOLD",
        "NVDA": "NVIDIA",
        "TSLA": "TESLA",
        "AAPL": "APPLE",
        "MSFT": "MICROSOFT",
        "GOOG": "GOOGLE",
        "GOOGL": "ALPHABET",
        "AMZN": "AMAZON",
        "META": "META",
        "NFLX": "NETFLIX",
        "AMD": "ADVANCED MICRO DEVICES",
    }
    for part in parts:
        alias = aliases.get(part)
        if alias:
            tokens.append(alias)
            if "S&P 500" in alias:
                tokens.append("S&P 500")
            if "NASDAQ 100" in alias:
                tokens.append("NASDAQ 100")
    return list(dict.fromkeys(tokens))


def _contains_financial_token(text: str, token: str) -> bool:
    normalized = token.upper().strip()
    if not normalized:
        return False
    if " " in normalized or "-" in normalized:
        return normalized in text
    return re.search(rf"(?<![A-Z0-9-]){re.escape(normalized)}(?![A-Z0-9-])", text) is not None


def _looks_like_timeout(exc: Exception) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "timeout" in text or "timed out" in text or "readtimeout" in text


def _retained_evidence_count(collected: dict[str, Any]) -> int:
    return (
        len(collected.get("ohlcv") or [])
        + len(collected.get("news_events") or [])
        + len(collected.get("sentiment_inputs") or [])
        + len(collected.get("market_context") or {})
        + len(collected.get("source_metadata") or {})
    )


def _dedupe_news_events(events: list[NewsEvent]) -> list[NewsEvent]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[NewsEvent] = []
    for event in events:
        key = (event.title.strip().lower(), event.source.strip().lower(), event.published_at.isoformat())
        if key not in seen:
            deduped.append(event)
            seen.add(key)
    return deduped
