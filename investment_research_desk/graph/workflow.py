from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TypedDict

from langgraph.graph import END, START, StateGraph

from investment_research_desk.agents import (
    ConstructiveCaseAnalyst,
    FundamentalMacroAnalyst,
    NewsImpactAnalyst,
    ResearchReporter,
    RiskCaseAnalyst,
    SentimentAnalyst,
    TechnicalAnalyst,
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
    constructive: dict[str, Any]
    risk: dict[str, Any]
    final_context: dict[str, Any]
    trace: dict[str, Any]
    metrics: dict[str, Any]
    warnings: list[str]
    completed_steps: list[str]
    output_paths: dict[str, str]
    checkpoint_enabled: bool


class ResearchWorkflow:
    def __init__(self, settings: Settings | None = None, runs_dir: Path | None = None):
        self.settings = settings or load_settings()
        self.store = RunStore(runs_dir or self.settings.runs_dir)
        self.fixture_provider = FixtureProvider()
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
                "trace": trace.model_dump(mode="json"),
                "warnings": [],
                "completed_steps": [],
                "output_paths": {},
            }
            self.store.ensure_run_dir(run_id)
            self.store.write_json(run_id, "input.json", request)
        state["checkpoint_enabled"] = checkpoint
        return self.graph.invoke(state)

    def _build_graph(self):
        graph = StateGraph(WorkflowState)
        graph.add_node("run_controller", self._run_controller)
        graph.add_node("data_ingestion", self._data_ingestion)
        graph.add_node("fundamental_macro", self._fundamental_macro)
        graph.add_node("news_impact", self._news_impact)
        graph.add_node("sentiment", self._sentiment)
        graph.add_node("technical", self._technical)
        graph.add_node("constructive_case", self._constructive_case)
        graph.add_node("risk_case", self._risk_case)
        graph.add_node("research_reporter", self._research_reporter)
        graph.add_node("persist", self._persist)
        graph.add_edge(START, "run_controller")
        graph.add_edge("run_controller", "data_ingestion")
        graph.add_edge("data_ingestion", "fundamental_macro")
        graph.add_edge("fundamental_macro", "news_impact")
        graph.add_edge("news_impact", "sentiment")
        graph.add_edge("sentiment", "technical")
        graph.add_edge("technical", "constructive_case")
        graph.add_edge("constructive_case", "risk_case")
        graph.add_edge("risk_case", "research_reporter")
        graph.add_edge("research_reporter", "persist")
        graph.add_edge("persist", END)
        return graph.compile()

    def _run_controller(self, state: WorkflowState) -> WorkflowState:
        return self._run_step(state, "run_controller", lambda s: s)

    def _data_ingestion(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            request = RunRequest.model_validate(s["request"])
            warnings = list(s.get("warnings", []))
            if request.fixture:
                data = self.fixture_provider.load(request.fixture)
                data.source_metadata["provider_mode"] = "fixture"
            else:
                market_result = route_to_vendor("get_market_data", self.settings, request)
                news_result = route_to_vendor("get_news", self.settings, request)
                sentiment_result = route_to_vendor("get_sentiment_inputs", self.settings, request)
                fundamentals_result = route_to_vendor("get_fundamentals", self.settings, request)
                warnings.extend(market_result.warnings)
                warnings.extend(news_result.warnings)
                warnings.extend(sentiment_result.warnings)
                warnings.extend(fundamentals_result.warnings)
                fundamentals = fundamentals_result.data if isinstance(fundamentals_result.data, dict) else {}
                data = NormalizedData(
                    symbol=request.symbol,
                    asset_class=request.asset_class,
                    horizon=request.horizon,
                    ohlcv=market_result.data,
                    news_events=news_result.data,
                    sentiment_inputs=sentiment_result.data,
                    source_metadata={
                        "provider_mode": "live",
                        "source_status": {
                            "market_data": market_result.status,
                            "news_data": news_result.status,
                            "sentiment_data": sentiment_result.status,
                            "fundamental_data": fundamentals_result.status,
                        },
                        **fundamentals,
                    },
                )
            s["data"] = data.model_dump(mode="json")
            s["warnings"] = warnings
            self.store.write_json(s["run_id"], "normalized_data.json", data)
            return s

        return self._run_step(state, "data_ingestion", work)

    def _fundamental_macro(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            request = RunRequest.model_validate(s["request"])
            data = NormalizedData.model_validate(s["data"])
            llm = self._make_llm(request, s)
            fundamental = FundamentalMacroAnalyst().run(data, llm)
            s["fundamental"] = fundamental.model_dump(mode="json")
            self._write_analyst_outputs(s)
            return s

        return self._run_step(state, "fundamental_macro", work)

    def _news_impact(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            request = RunRequest.model_validate(s["request"])
            data = NormalizedData.model_validate(s["data"])
            llm = self._make_llm(request, s)
            news = NewsImpactAnalyst().run(data, llm)
            s["news"] = news.model_dump(mode="json")
            self._write_analyst_outputs(s)
            return s

        return self._run_step(state, "news_impact", work)

    def _sentiment(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            request = RunRequest.model_validate(s["request"])
            data = NormalizedData.model_validate(s["data"])
            llm = self._make_llm(request, s)
            sentiment = SentimentAnalyst().run(data, llm)
            s["sentiment"] = sentiment.model_dump(mode="json")
            self._write_analyst_outputs(s)
            return s

        return self._run_step(state, "sentiment", work)

    def _technical(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            request = RunRequest.model_validate(s["request"])
            data = NormalizedData.model_validate(s["data"])
            llm = self._make_llm(request, s)
            technical = TechnicalAnalyst().run(data, llm)
            s["technical"] = technical.model_dump(mode="json")
            self._write_analyst_outputs(s)
            return s

        return self._run_step(state, "technical", work)

    def _constructive_case(self, state: WorkflowState) -> WorkflowState:
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

        return self._run_step(state, "constructive_case", work)

    def _risk_case(self, state: WorkflowState) -> WorkflowState:
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

        return self._run_step(state, "risk_case", work)

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

    def _persist(self, state: WorkflowState) -> WorkflowState:
        def work(s: WorkflowState) -> WorkflowState:
            final = FinalResearchContext.model_validate(s["final_context"])
            markdown = render_markdown_brief(final)
            trace = RunTrace.model_validate(s["trace"])
            trace.completed_at = datetime.now(timezone.utc)
            trace.warnings = list(dict.fromkeys(trace.warnings + s.get("warnings", [])))
            trace.completed_steps = s.get("completed_steps", [])
            raw_tokens = approximate_tokens({"data": s.get("data"), "analyst_outputs": s.get("fundamental")})
            final_tokens = approximate_tokens(final.model_dump(mode="json"))
            violations = find_guardrail_violations(final.model_dump_json() + "\n" + markdown)
            metrics = RunMetrics(
                total_latency_sec=round((trace.completed_at - trace.started_at).total_seconds(), 3),
                raw_input_tokens=raw_tokens,
                final_context_tokens=final_tokens,
                compression_ratio=compression_ratio(raw_tokens, final_tokens),
                guardrail_violations=violations,
            )
            paths = {
                "final_research_context": str(self.store.write_json(s["run_id"], "final_research_context.json", final)),
                "research_brief": str(self.store.write_text(s["run_id"], "research_brief.md", markdown)),
                "trace": str(self.store.write_json(s["run_id"], "trace.json", trace)),
                "metrics": str(self.store.write_json(s["run_id"], "metrics.json", metrics)),
            }
            s["trace"] = trace.model_dump(mode="json")
            s["metrics"] = metrics.model_dump(mode="json")
            s["output_paths"] = paths
            return s

        return self._run_step(state, "persist", work, checkpoint_after=False)

    def _run_step(self, state: WorkflowState, step_name: str, work, checkpoint_after: bool = True) -> WorkflowState:
        completed = list(state.get("completed_steps", []))
        if step_name in completed:
            return state
        started = time.perf_counter()
        try:
            state = work(state)
            status = "success"
            warnings: list[str] = []
        except Exception as exc:
            status = "failed"
            warnings = [str(exc)]
            state["warnings"] = list(state.get("warnings", [])) + [f"{step_name} failed: {exc}"]
            raise
        finally:
            latency = round(time.perf_counter() - started, 4)
            if step_name != "persist":
                self._append_trace(state, AgentTrace(name=step_name, status=status, latency_sec=latency, warnings=warnings))
        completed.append(step_name)
        state["completed_steps"] = completed
        if checkpoint_after and state.get("checkpoint_enabled"):
            self.store.save_checkpoint(state["run_id"], self._checkpoint_state(state))
        return state

    def _make_llm(self, request: RunRequest, state: WorkflowState):
        llm = make_llm_client(
            self.settings,
            request.llm_provider,
            request.model,
            allow_fake_fallback=bool(request.fixture),
        )
        if llm.provider == "fake" and request.llm_provider == "auto":
            warning = "Ollama unavailable; fixture run used deterministic fake LLM fallback."
            if warning not in state.get("warnings", []):
                state["warnings"] = list(state.get("warnings", [])) + [warning]
        return llm

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
