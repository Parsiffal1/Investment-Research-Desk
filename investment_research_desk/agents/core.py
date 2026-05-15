from __future__ import annotations

import json
import re
from typing import Any, Callable, Literal, TypeVar

from pydantic import BaseModel, Field

from investment_research_desk.llm import LLMClient
from investment_research_desk.schemas import (
    FinalResearchContext,
    FundamentalMacroResult,
    NewsEvent,
    NewsImpactResult,
    NormalizedData,
    ResearchCase,
    ResearchDebateResult,
    RunRequest,
    SentimentResult,
    TechnicalState,
    ViewLabel,
)
from investment_research_desk.tools.indicators import (
    atr,
    bollinger_state,
    macd,
    max_drawdown,
    realized_volatility,
    rsi,
    support_resistance,
    trend_label,
)
from investment_research_desk.agents.contracts import get_agent_contract
from investment_research_desk.agents.prompts import STRUCTURED_JSON_RULES
from investment_research_desk.sentiment_runtime import SentimentClassifier, aggregate_predictions

TModel = TypeVar("TModel", bound=BaseModel)


class NewsToolCallPlan(BaseModel):
    name: Literal["get_news", "get_global_news"]
    query: str
    limit: int = Field(default=5, ge=1, le=10)
    rationale: str = ""


class NewsToolPlan(BaseModel):
    calls: list[NewsToolCallPlan] = Field(default_factory=list, max_length=6)
    stop_reason: str = ""


BULLISH_TERMS = {
    "safe-haven",
    "demand",
    "rate cut",
    "dovish",
    "supportive",
    "rebound",
    "uptrend",
    "bullish",
    "weaker dollar",
}
BEARISH_TERMS = {
    "hawkish",
    "stronger dollar",
    "real yields",
    "selloff",
    "pressure",
    "risk",
    "bearish",
    "inflation surprise",
    "delayed rate cuts",
}


class FundamentalMacroAnalyst:
    name = "fundamental_macro"

    def run(self, data: NormalizedData, llm: LLMClient) -> FundamentalMacroResult:
        request = RunRequest(symbol=data.symbol, asset_class=data.asset_class, horizon=data.horizon)
        relevant_events, ranked_events = _filter_relevant_news_events(data.news_events, request)
        evidence_events = relevant_events or data.news_events
        titles = " ".join(event.title for event in evidence_events).lower()
        drivers = []
        concerns = []
        profile = data.source_metadata.get("fmp_profile") or {}
        quote = data.source_metadata.get("fmp_quote") or data.source_metadata.get("finnhub_quote") or {}
        if profile:
            company = profile.get("companyName") or profile.get("companyName") or data.symbol
            sector = profile.get("sector") or profile.get("industry")
            drivers.append(f"{company} profile data is available" + (f" in {sector}" if sector else ""))
        if quote:
            change_pct = quote.get("changePercentage") or quote.get("dp")
            if change_pct is not None:
                try:
                    change_pct = float(change_pct)
                    if change_pct > 0:
                        drivers.append(f"latest quote shows positive price change of {round(change_pct, 2)}%")
                    elif change_pct < 0:
                        concerns.append(f"latest quote shows negative price change of {round(change_pct, 2)}%")
                except Exception:
                    pass
        if "safe" in titles or "geopolitical" in titles:
            drivers.append("safe-haven demand remains a relevant macro driver")
        if "cpi" in titles or "inflation" in titles:
            concerns.append("inflation data can reprice rate expectations")
        if "dollar" in titles:
            concerns.append("dollar strength can pressure risk-sensitive or dollar-priced assets")
        if not drivers:
            drivers.append("macro context is present but not strongly one-sided")
        if not concerns:
            concerns.append("limited macro evidence increases uncertainty")
        fallback = FundamentalMacroResult(
            fundamental_view=_view_from_counts(len(drivers), len(concerns)),
            key_drivers=drivers,
            concerns=concerns,
            confidence=0.66,
            evidence=[event.title for event in evidence_events[:3]],
        )
        result = _llm_structured(
            self.name,
            llm,
            {
                "symbol": data.symbol,
                "asset_class": data.asset_class,
                "source_metadata": data.source_metadata,
                "news_events": [event.model_dump(mode="json") for event in evidence_events[:8]],
                "ranked_candidate_news_events": ranked_events[:16],
                "instruction": (
                    "For crypto and SWAP instruments, treat equity-style fundamentals as unavailable unless supplied. "
                    "Do not use unrelated company headlines as direct evidence. Use ranked_candidate_news_events to "
                    "separate direct, macro, indirect, and low-relevance evidence."
                ),
            },
            FundamentalMacroResult,
            fallback,
        )
        return result


class NewsImpactAnalyst:
    name = "news_impact"

    def run(self, data: NormalizedData, llm: LLMClient) -> NewsImpactResult:
        request = RunRequest(symbol=data.symbol, asset_class=data.asset_class, horizon=data.horizon)
        relevant_events, ranked_events = _filter_relevant_news_events(data.news_events, request)
        scoped_data = data.model_copy(update={"news_events": relevant_events or data.news_events})
        fallback = self._fallback(scoped_data)
        result = _llm_structured(
            self.name,
            llm,
            {
                "symbol": data.symbol,
                "asset_class": data.asset_class,
                "news_events": [event.model_dump(mode="json") for event in scoped_data.news_events[:12]],
                "ranked_candidate_news_events": ranked_events[:16],
            },
            NewsImpactResult,
            fallback,
        )
        return result

    def run_with_tools(
        self,
        request: RunRequest,
        llm: LLMClient,
        route_tool: Callable[[str, RunRequest], Any],
        max_rounds: int = 4,
    ) -> tuple[NewsImpactResult, NormalizedData]:
        collected_events: list[NewsEvent] = []
        tool_status: dict[str, Any] = {}
        tool_warnings: list[str] = []
        tool_calls: list[dict[str, Any]] = []
        forced_contract_calls: list[dict[str, Any]] = []
        filtered_candidate_count = 0
        max_total_tool_calls = 6
        max_tool_calls_by_name = {"get_news": 4, "get_global_news": 2}

        def execute_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            if name not in {"get_news", "get_global_news"}:
                return {"error": f"unsupported tool: {name}"}
            if len(tool_calls) >= max_total_tool_calls:
                return {"error": f"tool call budget exceeded: max_total={max_total_tool_calls}"}
            if _tool_call_count(tool_calls, name) >= max_tool_calls_by_name[name]:
                return {"error": f"tool call budget exceeded for {name}: max={max_tool_calls_by_name[name]}"}
            query = str(arguments.get("query") or request.symbol).strip() or request.symbol
            local_request = request.model_copy(update={"symbol": query})
            result = route_tool(name, local_request)
            events = [event for event in result.data if isinstance(event, NewsEvent)]
            collected_events.extend(events)
            tool_status.setdefault(name, []).append({"query": query, "status": result.status})
            tool_warnings.extend(result.warnings)
            payload = {
                "query": query,
                "status": result.status,
                "warnings": result.warnings,
                "events": [event.model_dump(mode="json") for event in events[:8]],
            }
            tool_calls.append({"name": name, "arguments": arguments, "event_count": len(events), "status": result.status})
            return payload

        empty_data = NormalizedData(
            symbol=request.symbol,
            asset_class=request.asset_class,
            horizon=request.horizon,
            source_metadata={"provider_mode": "live", "tool_call_policy": "llm_planned_tool_calls"},
        )
        fallback = self._fallback(empty_data)
        plan = self._plan_tool_queries(request, llm, max_total_tool_calls, max_tool_calls_by_name)
        for planned in plan.calls[:max_total_tool_calls]:
            execute_tool(
                planned.name,
                {
                    "query": planned.query,
                    "limit": planned.limit,
                    "rationale": planned.rationale,
                    "planned_by_llm": True,
                },
            )
        relevant_events, ranked_events = _filter_relevant_news_events(collected_events, request)
        filtered_candidate_count = len(_dedupe_news_events(collected_events)) - len(relevant_events)
        planned_data = NormalizedData(
            symbol=request.symbol,
            asset_class=request.asset_class,
            horizon=request.horizon,
            news_events=relevant_events or _dedupe_news_events(collected_events),
        )
        result = _llm_structured(
            self.name,
            llm,
            {
                "symbol": request.symbol,
                "asset_class": request.asset_class,
                "horizon": request.horizon,
                "llm_query_plan": plan.model_dump(mode="json"),
                "candidate_news_events": [event.model_dump(mode="json") for event in planned_data.news_events[:16]],
                "ranked_candidate_news_events": ranked_events[:24],
                "filtered_candidate_count": filtered_candidate_count,
                "instruction": (
                    "The LLM first optimized/refined the news queries in llm_query_plan. "
                    "Use ranked_candidate_news_events to reject unrelated items. Admit evidence only when relevance is "
                    "direct, macro, or indirect with an explicit transmission channel. Do not cite low-relevance items."
                ),
            },
            NewsImpactResult,
            self._fallback(planned_data),
        )
        if not _has_targeted_news_call(tool_calls, request):
            for query in _targeted_news_queries(request):
                before_count = len(collected_events)
                payload = execute_tool("get_news", {"query": query, "limit": 5, "forced_by_contract": True})
                forced_contract_calls.append(
                    {
                        "name": "get_news",
                        "arguments": {"query": query, "limit": 5},
                        "forced_by_contract": True,
                        "error": payload.get("error"),
                    }
                )
                if len(collected_events) > before_count or not payload.get("error"):
                    break
        if forced_contract_calls:
            all_events = _dedupe_news_events(collected_events)
            related_events, ranked_events = _filter_relevant_news_events(all_events, request)
            relevant_events = related_events
            filtered_candidate_count = len(all_events) - len(related_events or all_events)
            refinement_data = NormalizedData(
                symbol=request.symbol,
                asset_class=request.asset_class,
                horizon=request.horizon,
                news_events=related_events or all_events,
            )
            result = _llm_structured(
                self.name,
                llm,
                {
                    "symbol": request.symbol,
                    "asset_class": request.asset_class,
                    "horizon": request.horizon,
                    "news_events": [event.model_dump(mode="json") for event in refinement_data.news_events[:16]],
                    "ranked_candidate_news_events": ranked_events[:24],
                    "filtered_candidate_count": filtered_candidate_count,
                    "llm_tool_calls": tool_calls,
                    "forced_contract_calls": forced_contract_calls,
                    "instruction": (
                        "The system enforced the minimum targeted news-search contract because the first tool loop "
                        "did not include a direct instrument-specific get_news call. Evaluate these candidate events, "
                        "reject unrelated items, and only admit evidence with a clear link to the instrument."
                    ),
                },
                NewsImpactResult,
                self._fallback(refinement_data),
            )
        data = NormalizedData(
            symbol=request.symbol,
            asset_class=request.asset_class,
            horizon=request.horizon,
            news_events=relevant_events or _dedupe_news_events(collected_events),
            source_metadata={
                "provider_mode": "live",
                "tool_call_policy": "llm_planned_tool_calls_with_targeted_search_minimum",
                "minimum_targeted_search_enforced": bool(forced_contract_calls),
                "filtered_candidate_count": filtered_candidate_count,
                "llm_query_plan": plan.model_dump(mode="json"),
                "agent_tool_status": {self.name: tool_status},
                "warnings": tool_warnings,
                "llm_tool_calls": tool_calls,
                "forced_contract_calls": forced_contract_calls,
            },
        )
        return result, data

    def _plan_tool_queries(
        self,
        request: RunRequest,
        llm: LLMClient,
        max_total_tool_calls: int,
        max_tool_calls_by_name: dict[str, int],
    ) -> NewsToolPlan:
        contract = get_agent_contract(self.name)
        default_plan = _default_news_tool_plan(request)
        prompt = (
            f"Agent: {self.name}\n"
            f"Instrument: {request.symbol}\n"
            f"Asset class: {request.asset_class}\n"
            f"Horizon: {request.horizon}\n\n"
            "Before any news tool is executed, optimize and refine the exact tool queries. "
            "Decide whether to call get_news, whether to call get_global_news, how many calls are useful, "
            "and when enough evidence should be collected. Return only a query plan; do not write the final report yet.\n\n"
            "Available tools:\n"
            "- get_news(query, limit): targeted symbol, asset, issuer, sector, or instrument-specific news.\n"
            "- get_global_news(query, limit): broader macro, liquidity, policy, rates, cross-asset, or market-structure news.\n\n"
            f"Hard tool budget: at most {max_total_tool_calls} calls total, "
            f"at most {max_tool_calls_by_name['get_news']} get_news calls, "
            f"and at most {max_tool_calls_by_name['get_global_news']} get_global_news calls.\n"
            "Include a direct instrument-specific get_news query unless the instrument is invalid. "
            "For crypto SWAP instruments, include both exact instrument context and underlying asset news when useful. "
            "Prefer queries that can surface direct instrument, underlying asset, ETF/flow, regulatory, rates, dollar, "
            "liquidity, derivatives, exchange, and market-structure evidence. Avoid broad generic corporate-news queries "
            "unless the transmission channel to the instrument is explicit.\n\n"
            f"Pydantic JSON schema:\n{json.dumps(NewsToolPlan.model_json_schema(), ensure_ascii=False, default=str)}\n\n"
            f"Candidate output JSON:\n{json.dumps(default_plan.model_dump(mode='json'), ensure_ascii=False, default=str)}"
        )
        try:
            raw = llm.chat_json(contract.system_prompt, prompt)
            if isinstance(raw.get("result"), dict):
                raw = raw["result"]
            plan = NewsToolPlan.model_validate(raw)
        except Exception:
            plan = default_plan
        bounded: list[NewsToolCallPlan] = []
        counts = {"get_news": 0, "get_global_news": 0}
        for call in plan.calls:
            if len(bounded) >= max_total_tool_calls:
                break
            if call.name not in counts:
                continue
            if counts[call.name] >= max_tool_calls_by_name[call.name]:
                continue
            query = call.query.strip()
            if not query:
                continue
            bounded.append(call.model_copy(update={"query": query, "limit": min(max(call.limit, 1), 10)}))
            counts[call.name] += 1
        return NewsToolPlan(calls=bounded, stop_reason=plan.stop_reason)

    def _fallback(self, data: NormalizedData) -> NewsImpactResult:
        dominant = [event.title for event in data.news_events[:5]]
        event_types: dict[str, str] = {}
        bullish = bearish = 0
        evidence: list[str] = []
        for event in data.news_events:
            event_type = event.event_type or "general"
            event_types[event_type] = "high_importance" if event_type in {"inflation", "central_bank"} else "medium_importance"
            text = f"{event.title} {event.summary or ''}".lower()
            bullish += sum(1 for term in BULLISH_TERMS if term in text)
            bearish += sum(1 for term in BEARISH_TERMS if term in text)
            evidence.append(event.title)
        impact = _view_from_counts(bullish, bearish)
        impact_logic = _impact_logic(impact)
        return NewsImpactResult(
            dominant_events=dominant or ["no material news events found"],
            event_type_summary=event_types or {"general": "low_importance"},
            asset_impact={data.symbol: impact},
            impact_logic=impact_logic,
            confidence=0.7 if dominant else 0.45,
            evidence=evidence[:5],
        )


class SentimentAnalyst:
    name = "sentiment"

    def run(self, data: NormalizedData, llm: LLMClient, sentiment_classifier: SentimentClassifier | None = None) -> SentimentResult:
        texts = [item.text for item in data.sentiment_inputs]
        joined = " ".join(texts).lower()
        bullish = sum(1 for term in BULLISH_TERMS if term in joined)
        bearish = sum(1 for term in BEARISH_TERMS if term in joined)
        total = max(1, bullish + bearish)
        score = round((bullish - bearish) / total, 2)
        if score > 0.2:
            label = "bullish"
            mood = "constructive"
        elif score < -0.2:
            label = "bearish"
            mood = "risk_off"
        elif bullish and bearish:
            label = "mixed"
            mood = "divided"
        else:
            label = "neutral"
            mood = "quiet"
        evidence = texts[:5] or ["no sentiment inputs available"]
        fallback = SentimentResult(
            crowd_mood=mood,
            sentiment_label=label,
            sentiment_score=score,
            evidence=evidence,
            confidence=0.68 if texts else 0.35,
        )
        adapter_payload: dict[str, Any] | None = None
        if sentiment_classifier is not None and data.sentiment_inputs:
            try:
                predictions = sentiment_classifier.classify(data.sentiment_inputs)
                fallback = aggregate_predictions(data.sentiment_inputs, predictions)
                adapter_payload = {
                    "runtime": sentiment_classifier.runtime_metadata(),
                    "aggregate": fallback.model_dump(mode="json"),
                    "classifications": [
                        {
                            "label": row.label,
                            "score_margin": row.score_margin,
                            "label_scores": row.label_scores,
                            "source": row.source,
                            "text": row.text,
                        }
                        for row in predictions[:20]
                    ],
                }
            except Exception as exc:
                fallback.evidence = [f"sentiment adapter unavailable: {exc}", *fallback.evidence]
        result = _llm_structured(
            self.name,
            llm,
            {
                "symbol": data.symbol,
                "asset_class": data.asset_class,
                "sentiment_inputs": [item.model_dump(mode="json") for item in data.sentiment_inputs[:20]],
                "sentiment_adapter": adapter_payload,
                "structured_sentiment_blocks": _sentiment_source_blocks(data.sentiment_inputs),
                "instruction": (
                    "Read sentiment as pre-collected source blocks. Distinguish institutional/news framing, retail "
                    "posts, community discussion, sponsored material, and generic market chatter. Weight sample size, "
                    "source quality, and direct relevance to the instrument before selecting evidence. If "
                    "sentiment_adapter is present, use its aggregate label and score as the classification authority "
                    "and use the main LLM only to summarize evidence and caveats."
                ),
            },
            SentimentResult,
            fallback,
        )
        if adapter_payload is not None:
            return result.model_copy(
                update={
                    "crowd_mood": fallback.crowd_mood,
                    "sentiment_label": fallback.sentiment_label,
                    "sentiment_score": fallback.sentiment_score,
                    "confidence": fallback.confidence,
                }
            )
        return result


class TechnicalAnalyst:
    name = "technical"

    def run(self, data: NormalizedData, llm: LLMClient) -> TechnicalState:
        bars = data.ohlcv
        trend = trend_label(bars)
        rsi_14 = rsi(bars)
        macd_line, macd_signal, macd_hist = macd(bars)
        atr_14 = atr(bars)
        rv = realized_volatility(bars)
        dd = max_drawdown(bars)
        supports, resistances = support_resistance(bars)
        macd_state = _macd_state(macd_hist)
        momentum = _momentum_state(rsi_14, macd_hist)
        view = _technical_view(trend, rsi_14, macd_hist)
        vol_regime = "elevated" if rv is not None and rv > 0.7 else "normal"
        swap_context = data.market_context.get("okx_swap") if isinstance(data.market_context, dict) else {}
        if not isinstance(swap_context, dict):
            swap_context = {}
        mark_price = _nested_float(swap_context, "mark_price", "markPx")
        index_price = _nested_float(swap_context, "index_ticker", "idxPx")
        funding_rate = _nested_float(swap_context, "funding_rate", "fundingRate")
        open_interest = _nested_float(swap_context, "open_interest", "oi")
        orderbook_imbalance = _float_or_none(swap_context.get("orderbook_imbalance"))
        fallback = TechnicalState(
            technical_view=view,
            trend=trend,
            momentum=momentum,
            volatility_regime=vol_regime,
            rsi_14=rsi_14,
            macd_state=macd_state,
            atr_14=atr_14,
            bollinger_state=bollinger_state(bars),
            realized_volatility=rv,
            max_drawdown=dd,
            mark_price=mark_price,
            index_price=index_price,
            funding_rate=funding_rate,
            open_interest=open_interest,
            orderbook_imbalance=orderbook_imbalance,
            swap_context_summary=_swap_context_summary(swap_context),
            support_zones=supports,
            resistance_zones=resistances,
            confidence=0.82 if len(bars) >= 26 and swap_context else 0.78 if len(bars) >= 26 else 0.45,
        )
        return _llm_structured(
            self.name,
            llm,
            {
                "symbol": data.symbol,
                "asset_class": data.asset_class,
                "indicator_results": fallback.model_dump(mode="json"),
                "recent_ohlcv": [bar.model_dump(mode="json") for bar in bars[-10:]],
                "swap_market_context": swap_context,
                "instruction": (
                    "Read the deterministic indicator results and OKX public SWAP market context. "
                    "Do not recalculate indicators. Interpret funding, open interest, mark/index spread, price limits, "
                    "recent trades, and orderbook imbalance as derivative market context only."
                ),
            },
            TechnicalState,
            fallback,
        )


class ConstructiveCaseAnalyst:
    name = "bull_researcher"

    def run(
        self,
        fundamental: FundamentalMacroResult,
        news: NewsImpactResult,
        sentiment: SentimentResult,
        technical: TechnicalState,
        llm: LLMClient,
        debate_history: list[dict[str, Any]] | None = None,
        opponent_case: ResearchCase | None = None,
    ) -> ResearchCase:
        evidence = []
        evidence.extend(fundamental.key_drivers[:2])
        evidence.extend(news.evidence[:2])
        if sentiment.sentiment_label in {"bullish", "mixed"}:
            evidence.append(f"sentiment is {sentiment.sentiment_label} with score {sentiment.sentiment_score}")
        if technical.technical_view in {"bullish", "mixed_to_bullish", "neutral_to_bullish"}:
            evidence.append(f"technical state is {technical.technical_view} with trend {technical.trend}")
        if not evidence:
            evidence.append("constructive evidence is limited in the current inputs")
        fallback = ResearchCase(
            thesis="Constructive case depends on supportive macro drivers and technical confirmation.",
            evidence=evidence[:6],
            conditions=[
                "price remains above identified support zones",
                "macro news does not reprice risk sharply against the asset",
            ],
            confidence=min(0.82, max(0.45, (fundamental.confidence + news.confidence + technical.confidence) / 3)),
        )
        return _llm_structured(
            self.name,
            llm,
            {
                "fundamental": fundamental.model_dump(mode="json"),
                "news": news.model_dump(mode="json"),
                "sentiment": sentiment.model_dump(mode="json"),
                "technical": technical.model_dump(mode="json"),
                "debate_history": debate_history or [],
                "bear_researcher": opponent_case.model_dump(mode="json") if opponent_case else None,
                "instruction": (
                    "Build the strongest constructive research case, but do not cite weak or unrelated evidence. "
                    "Address risk evidence directly and state conditions that would keep the constructive case valid."
                ),
            },
            ResearchCase,
            fallback,
        )


class RiskCaseAnalyst:
    name = "bear_researcher"

    def run(
        self,
        fundamental: FundamentalMacroResult,
        news: NewsImpactResult,
        sentiment: SentimentResult,
        technical: TechnicalState,
        constructive: ResearchCase,
        llm: LLMClient,
        debate_history: list[dict[str, Any]] | None = None,
    ) -> ResearchCase:
        evidence = []
        evidence.extend(fundamental.concerns[:3])
        if sentiment.sentiment_label in {"bearish", "mixed"}:
            evidence.append(f"sentiment is {sentiment.sentiment_label}, indicating disagreement or caution")
        if technical.momentum in {"negative", "positive_but_slowing"}:
            evidence.append(f"technical momentum is {technical.momentum}")
        if technical.volatility_regime == "elevated":
            evidence.append("realized volatility is elevated")
        if not evidence:
            evidence.append("risk evidence is limited but data coverage remains incomplete")
        fallback = ResearchCase(
            thesis="Risk case centers on macro repricing, volatility, and possible technical failure.",
            evidence=evidence[:6],
            conditions=[
                "break below support zones",
                "hawkish macro surprise or stronger dollar impulse",
                "news flow contradicts the constructive case",
            ],
            confidence=min(0.85, max(0.45, (fundamental.confidence + news.confidence + technical.confidence) / 3)),
        )
        return _llm_structured(
            self.name,
            llm,
            {
                "fundamental": fundamental.model_dump(mode="json"),
                "news": news.model_dump(mode="json"),
                "sentiment": sentiment.model_dump(mode="json"),
                "technical": technical.model_dump(mode="json"),
                "bull_researcher": constructive.model_dump(mode="json"),
                "debate_history": debate_history or [],
                "instruction": (
                    "Challenge the constructive case using direct risks, evidence quality issues, data gaps, and "
                    "technical invalidation conditions. Do not overstate unsupported downside claims."
                ),
            },
            ResearchCase,
            fallback,
        )


class DebateModerator:
    name = "bull_bear_research_debate"

    def run(self, constructive: ResearchCase, risk: ResearchCase, llm: LLMClient) -> ResearchDebateResult:
        shared_evidence = sorted(set(constructive.evidence).intersection(risk.evidence))
        fallback = ResearchDebateResult(
            points_of_agreement=shared_evidence,
            key_tensions=[
                "constructive case requires confirmation from support, catalyst quality, and macro conditions",
                "risk case emphasizes repricing, volatility, evidence gaps, and data coverage uncertainty",
            ],
            evidence_quality_notes=[
                "direct instrument evidence should be weighted above broad market or sector proxies",
                "low-relevance news should not be promoted into final key drivers or key risks",
            ],
            reporter_handoff=(
                "Produce balanced research context only. Weigh direct evidence, technical confirmation, data gaps, "
                "and news/sentiment noise before assigning directional_view."
            ),
            confidence=round((constructive.confidence + risk.confidence) / 2, 2),
        )
        return _llm_structured(
            self.name,
            llm,
            {
                "bull_researcher": constructive.model_dump(mode="json"),
                "bear_researcher": risk.model_dump(mode="json"),
                "instruction": (
                    "Moderate the debate. Identify agreements, tensions, evidence-quality concerns, and reporter "
                    "handoff. Do not make a trade recommendation or mention order execution."
                ),
            },
            ResearchDebateResult,
            fallback,
        )


class ResearchReporter:
    name = "research_reporter"

    def run(
        self,
        data: NormalizedData,
        fundamental: FundamentalMacroResult,
        news: NewsImpactResult,
        sentiment: SentimentResult,
        technical: TechnicalState,
        constructive: ResearchCase,
        risk: ResearchCase,
        research_debate: dict[str, Any],
        warnings: list[str],
        llm: LLMClient,
    ) -> FinalResearchContext:
        balanced = _balance_view(news.asset_impact.get(data.symbol, "mixed"), sentiment.sentiment_label, technical.technical_view)
        directional_view, directional_rationale = _directional_view(
            balanced,
            sentiment,
            technical,
            constructive,
            risk,
            news.asset_impact.get(data.symbol, "mixed"),
        )
        risk_level = _risk_level(risk, technical)
        key_drivers = _dedupe(constructive.evidence + fundamental.key_drivers + news.dominant_events)[:6]
        key_risks = _dedupe(risk.evidence + fundamental.concerns)[:6]
        fallback = FinalResearchContext(
            symbol=data.symbol,
            asset_class=data.asset_class,
            horizon=data.horizon,
            market_regime=f"{technical.trend}_{technical.volatility_regime}",
            directional_view=directional_view,
            directional_rationale=directional_rationale,
            balanced_view=balanced,
            risk_level=risk_level,
            confidence=round((fundamental.confidence + news.confidence + sentiment.confidence + technical.confidence) / 4, 2),
            fundamental_summary="; ".join(fundamental.key_drivers[:2] + fundamental.concerns[:2]),
            news_impact_summary=news.impact_logic,
            sentiment_summary=f"Market mood is {sentiment.crowd_mood}; label={sentiment.sentiment_label}; score={sentiment.sentiment_score}.",
            technical_summary=(
                f"Trend={technical.trend}; momentum={technical.momentum}; RSI={technical.rsi_14}; "
                f"MACD={technical.macd_state}; OKX_SWAP funding={technical.funding_rate}; "
                f"open_interest={technical.open_interest}; orderbook_imbalance={technical.orderbook_imbalance}."
            ),
            constructive_case=constructive,
            risk_case=risk,
            key_drivers=key_drivers,
            key_risks=key_risks,
            uncertainty_factors=[
                "live data coverage may vary by provider",
                "LLM summaries require evidence review before downstream strategy use",
            ],
            downstream_agent_context="Use as research context only. A separate decision, risk, and execution system is required before any trading action.",
            usage_constraints=[
                "not financial advice",
                "not an order instruction",
                "does not include position sizing",
                "does not claim profitability",
            ],
            source_metadata=_report_source_metadata(data.source_metadata),
            warnings=warnings,
        )
        return _llm_structured(
            self.name,
            llm,
            {
                "normalized_data_summary": {
                    "symbol": data.symbol,
                    "asset_class": data.asset_class,
                    "horizon": data.horizon,
                    "source_metadata": data.source_metadata,
                },
                "fundamental": fundamental.model_dump(mode="json"),
                "news": news.model_dump(mode="json"),
                "sentiment": sentiment.model_dump(mode="json"),
                "technical": technical.model_dump(mode="json"),
                "bull_researcher": constructive.model_dump(mode="json"),
                "bear_researcher": risk.model_dump(mode="json"),
                "research_debate": research_debate,
                "warnings": warnings,
                "instruction": (
                    "Use the debate handoff and evidence-quality notes when assigning directional_view and risk_level. "
                    "If direct evidence is thin or sentiment/news is noisy, explicitly discount confidence."
                ),
            },
            FinalResearchContext,
            fallback,
        )


def _llm_structured(
    agent_name: str,
    llm: LLMClient,
    input_payload: dict[str, Any],
    schema_model: type[TModel],
    fallback: TModel,
) -> TModel:
    contract = get_agent_contract(agent_name)
    candidate = fallback.model_dump(mode="json")
    user_prompt = (
        f"Agent: {agent_name}\n"
        f"Output schema name: {schema_model.__name__}\n"
        f"{STRUCTURED_JSON_RULES} "
        "Use the candidate output as the conservative baseline, but refine interpretation when the provided evidence supports it. "
        "For deterministic numeric fields, preserve the candidate values exactly unless the schema field is interpretive text. "
        "If evidence is weak, keep confidence lower and name the data gap in an existing summary/evidence field.\n\n"
        f"Pydantic JSON schema:\n{json.dumps(schema_model.model_json_schema(), ensure_ascii=False, default=str)}\n\n"
        f"Allowed inputs:\n{json.dumps(contract.allowed_inputs, ensure_ascii=False)}\n\n"
        f"Allowed tools:\n{json.dumps(contract.allowed_tools, ensure_ascii=False)}\n\n"
        f"Input payload:\n{json.dumps(input_payload, ensure_ascii=False, default=str)}\n\n"
        f"Candidate output JSON:\n{json.dumps(candidate, ensure_ascii=False, default=str)}"
    )
    try:
        raw = llm.chat_json(contract.system_prompt, user_prompt)
        if isinstance(raw.get("result"), dict):
            raw = raw["result"]
        result = schema_model.model_validate(raw)
        return _preserve_deterministic_fields(agent_name, result, fallback)
    except Exception:
        return fallback


def _preserve_deterministic_fields(agent_name: str, result: TModel, fallback: TModel) -> TModel:
    if isinstance(result, TechnicalState) and isinstance(fallback, TechnicalState):
        return result.model_copy(
            update={
                "rsi_14": fallback.rsi_14,
                "atr_14": fallback.atr_14,
                "bollinger_state": fallback.bollinger_state,
                "realized_volatility": fallback.realized_volatility,
                "max_drawdown": fallback.max_drawdown,
                "mark_price": fallback.mark_price,
                "index_price": fallback.index_price,
                "funding_rate": fallback.funding_rate,
                "open_interest": fallback.open_interest,
                "orderbook_imbalance": fallback.orderbook_imbalance,
                "support_zones": fallback.support_zones,
                "resistance_zones": fallback.resistance_zones,
            }
        )
    if isinstance(result, FinalResearchContext) and isinstance(fallback, FinalResearchContext):
        return result.model_copy(
            update={
                "symbol": fallback.symbol,
                "asset_class": fallback.asset_class,
                "horizon": fallback.horizon,
                "source_metadata": fallback.source_metadata,
                "warnings": fallback.warnings,
                "usage_constraints": fallback.usage_constraints,
            }
        )
    return result


def _tool_call_count(tool_calls: list[dict[str, Any]], name: str) -> int:
    return sum(1 for call in tool_calls if call.get("name") == name)


def _has_targeted_news_call(tool_calls: list[dict[str, Any]], request: RunRequest) -> bool:
    for call in tool_calls:
        if call.get("name") != "get_news":
            continue
        arguments = call.get("arguments") or {}
        query = str(arguments.get("query") or "")
        if _query_mentions_instrument(query, request):
            return True
    return False


def _query_mentions_instrument(query: str, request: RunRequest) -> bool:
    normalized_query = query.upper()
    return any(token in normalized_query for token in _instrument_query_tokens(request))


def _instrument_query_tokens(request: RunRequest) -> list[str]:
    symbol = request.symbol.upper()
    ignored_parts = {"USD", "USDT", "USDC", "SWAP", "PERP", "PERPETUAL"}
    parts = [part for part in re.split(r"[^A-Z0-9]+", symbol) if part and part not in ignored_parts]
    tokens = [symbol, *parts]
    aliases = {
        "BTC": "BITCOIN",
        "ETH": "ETHEREUM",
        "SOL": "SOLANA",
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
    return _dedupe(tokens)


def _targeted_news_queries(request: RunRequest) -> list[str]:
    symbol = request.symbol.upper()
    parts = [part for part in re.split(r"[^A-Z0-9]+", symbol) if part]
    queries = [request.symbol]
    if request.asset_class == "crypto" and parts:
        primary = parts[0]
        crypto_names = {"BTC": "Bitcoin", "ETH": "Ethereum", "SOL": "Solana"}
        queries.append(f"{primary} {crypto_names.get(primary, primary)} crypto news")
    elif request.asset_class in {"equity", "equity_index", "other"}:
        queries.append(f"{symbol} stock company news")
    else:
        queries.append(f"{request.symbol} market news")
    return _dedupe(queries)


def _default_news_tool_plan(request: RunRequest) -> NewsToolPlan:
    calls = [
        NewsToolCallPlan(
            name="get_news",
            query=query,
            limit=5,
            rationale="minimum direct instrument-specific evidence collection",
        )
        for query in _targeted_news_queries(request)[:2]
    ]
    calls.append(
        NewsToolCallPlan(
            name="get_global_news",
            query=f"{request.symbol} {request.asset_class} macro market liquidity risk",
            limit=5,
            rationale="broader macro and cross-asset context",
        )
    )
    return NewsToolPlan(calls=calls, stop_reason="default bounded plan")


def _instrument_related_news_events(events: list[NewsEvent], request: RunRequest) -> list[NewsEvent]:
    tokens = _instrument_query_tokens(request)
    related: list[NewsEvent] = []
    for event in events:
        haystack = " ".join(
            [
                event.title,
                event.summary or "",
                event.url or "",
            ]
        ).upper()
        if any(token in haystack for token in tokens):
            related.append(event)
    return related


def _filter_relevant_news_events(events: list[NewsEvent], request: RunRequest) -> tuple[list[NewsEvent], list[dict[str, Any]]]:
    deduped = _dedupe_news_events(events)
    ranked: list[dict[str, Any]] = []
    relevant: list[NewsEvent] = []
    for event in deduped:
        relevance, reason = _news_relevance(event, request)
        ranked.append(
            {
                "title": event.title,
                "source": event.source,
                "published_at": event.published_at.isoformat(),
                "relevance": relevance,
                "reason": reason,
                "url": event.url,
            }
        )
        if relevance != "low":
            relevant.append(event)
    order = {"direct": 0, "macro": 1, "indirect": 2, "low": 3}
    ranked.sort(key=lambda item: (order.get(str(item["relevance"]), 9), str(item["published_at"])), reverse=False)
    relevant.sort(key=lambda event: order.get(_news_relevance(event, request)[0], 9))
    return relevant, ranked


def _news_relevance(event: NewsEvent, request: RunRequest) -> tuple[str, str]:
    haystack = " ".join([event.title, event.summary or "", event.url or ""]).upper()
    tokens = _instrument_query_tokens(request)
    if any(token in haystack for token in tokens):
        return "direct", "mentions the instrument, underlying asset, or known alias in title, summary, or URL"
    macro_terms = {
        "FED",
        "FOMC",
        "CPI",
        "INFLATION",
        "RATE",
        "YIELD",
        "DOLLAR",
        "LIQUIDITY",
        "TREASURY",
        "ETF",
        "REGULATION",
        "SEC",
        "GEOPOLITICAL",
        "WAR",
        "RISK APPETITE",
        "FUNDING",
        "OPEN INTEREST",
    }
    if any(term in haystack for term in macro_terms):
        return "macro", "contains a macro, policy, liquidity, regulatory, or market-structure channel"
    if request.asset_class == "equity" and any(term in haystack for term in {"EARNINGS", "GUIDANCE", "REVENUE", "MARGIN"}):
        return "indirect", "contains equity fundamental context that may matter if tied to the issuer or sector"
    if request.asset_class == "crypto" and any(term in haystack for term in {"CRYPTO", "BLOCKCHAIN", "MINING", "STABLECOIN"}):
        return "indirect", "contains crypto-sector context without a direct instrument mention"
    return "low", "no clear direct, macro, sector, or instrument transmission channel"


def _sentiment_source_blocks(inputs: list[SentimentInput]) -> dict[str, Any]:
    blocks: dict[str, list[dict[str, Any]]] = {
        "news_or_search": [],
        "stocktwits": [],
        "reddit": [],
        "other": [],
    }
    for item in inputs[:40]:
        source = item.source.lower()
        if "stocktwits" in source:
            bucket = "stocktwits"
        elif "reddit" in source:
            bucket = "reddit"
        elif any(token in source for token in ("tavily", "news", "yahoo", "finnhub")):
            bucket = "news_or_search"
        else:
            bucket = "other"
        blocks[bucket].append(item.model_dump(mode="json"))
    return {key: {"count": len(value), "items": value[:12]} for key, value in blocks.items()}


def _news_tool_specs() -> list[dict[str, Any]]:
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query chosen by the analyst, such as BTC ETF flows, Bitcoin macro liquidity, or the exact instrument symbol.",
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of candidate events requested.",
                "minimum": 1,
                "maximum": 10,
            },
        },
        "required": ["query"],
    }
    return [
        {
            "type": "function",
            "function": {
                "name": "get_news",
                "description": "Retrieve targeted candidate news for a symbol, asset, or specific market topic.",
                "parameters": parameters,
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_global_news",
                "description": "Retrieve broader macro, policy, liquidity, and cross-asset market news candidates.",
                "parameters": parameters,
            },
        },
    ]


def _dedupe_news_events(events: list[NewsEvent]) -> list[NewsEvent]:
    seen: set[tuple[str, str, str]] = set()
    deduped: list[NewsEvent] = []
    for event in events:
        key = (event.title.strip().lower(), event.source.strip().lower(), event.published_at.isoformat())
        if key not in seen:
            deduped.append(event)
            seen.add(key)
    return deduped


def _view_from_counts(positive: int, negative: int) -> ViewLabel:
    if positive > negative + 1:
        return "mixed_to_bullish"
    if negative > positive + 1:
        return "mixed_to_bearish"
    if positive > negative:
        return "neutral_to_bullish"
    if negative > positive:
        return "neutral_to_bearish"
    return "mixed"


def _impact_logic(view: str) -> str:
    if "bullish" in view:
        return "Current events lean supportive, but the output remains research context rather than a trading signal."
    if "bearish" in view:
        return "Current events lean cautious due to macro or risk repricing pressure."
    return "Current events contain offsetting forces, so the impact is mixed and scenario-dependent."


def _macd_state(hist: float | None) -> str:
    if hist is None:
        return "insufficient_data"
    if hist > 0.2:
        return "positive"
    if hist > 0:
        return "positive_flattening"
    if hist < -0.2:
        return "negative"
    return "negative_flattening"


def _nested_float(payload: dict[str, Any], outer: str, inner: str) -> float | None:
    value = payload.get(outer)
    if not isinstance(value, dict):
        return None
    return _float_or_none(value.get(inner))


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except Exception:
        return None


def _swap_context_summary(context: dict[str, Any]) -> str | None:
    if not context:
        return None
    inst_id = context.get("inst_id", "unknown")
    funding = _nested_float(context, "funding_rate", "fundingRate")
    open_interest = _nested_float(context, "open_interest", "oi")
    imbalance = _float_or_none(context.get("orderbook_imbalance"))
    spread = _float_or_none(context.get("mark_index_spread"))
    parts = [f"OKX public SWAP context for {inst_id}"]
    if funding is not None:
        parts.append(f"funding_rate={funding}")
    if open_interest is not None:
        parts.append(f"open_interest={open_interest}")
    if imbalance is not None:
        parts.append(f"orderbook_imbalance={imbalance}")
    if spread is not None:
        parts.append(f"mark_index_spread={spread}")
    return "; ".join(parts)


def _momentum_state(rsi_14: float | None, macd_hist: float | None) -> str:
    if rsi_14 is None or macd_hist is None:
        return "insufficient_data"
    if rsi_14 > 70:
        return "positive_overextended"
    if rsi_14 > 55 and macd_hist > 0:
        return "positive"
    if rsi_14 < 45 and macd_hist < 0:
        return "negative"
    if macd_hist > 0:
        return "positive_but_slowing"
    return "mixed"


def _technical_view(trend: str, rsi_14: float | None, macd_hist: float | None) -> ViewLabel:
    if trend == "uptrend" and (macd_hist or 0) >= 0:
        return "mixed_to_bullish"
    if trend == "downtrend" and (macd_hist or 0) <= 0:
        return "mixed_to_bearish"
    if rsi_14 is not None and rsi_14 > 60:
        return "neutral_to_bullish"
    if rsi_14 is not None and rsi_14 < 40:
        return "neutral_to_bearish"
    return "mixed"


def _balance_view(news_view: str, sentiment_label: str, technical_view: str) -> ViewLabel:
    joined = f"{news_view} {sentiment_label} {technical_view}"
    bullish = joined.count("bullish")
    bearish = joined.count("bearish")
    return _view_from_counts(bullish, bearish)


def _directional_view(
    balanced_view: str,
    sentiment: SentimentResult,
    technical: TechnicalState,
    constructive: ResearchCase,
    risk: ResearchCase,
    news_view: str,
) -> tuple[str, str]:
    score = (
        _view_score(balanced_view) * 1.5
        + _view_score(news_view)
        + _view_score(sentiment.sentiment_label)
        + _view_score(technical.technical_view)
        + sentiment.sentiment_score
        + constructive.confidence
        - risk.confidence
    )
    direction = "bullish" if score >= 0 else "bearish"
    rationale = (
        f"Directional research judgment is {direction} because balanced_view={balanced_view}, "
        f"news_impact={news_view}, sentiment={sentiment.sentiment_label}/{sentiment.sentiment_score}, "
        f"technical_view={technical.technical_view}, constructive_confidence={constructive.confidence}, "
        f"risk_confidence={risk.confidence}."
    )
    return direction, rationale


def _view_score(view: str) -> float:
    if view in {"bullish", "mixed_to_bullish"}:
        return 1.0
    if view == "neutral_to_bullish":
        return 0.5
    if view in {"bearish", "mixed_to_bearish"}:
        return -1.0
    if view == "neutral_to_bearish":
        return -0.5
    return 0.0


def _report_source_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    safe_keys = {
        "provider_mode",
        "tool_call_policy",
        "agent_tool_status",
        "agent_tool_warnings",
        "minimum_targeted_search_enforced",
        "filtered_candidate_count",
        "tool_call_budget",
    }
    safe = {key: value for key, value in metadata.items() if key in safe_keys}
    if "contract_floor_calls" in metadata:
        safe["contract_floor_calls"] = _summarize_tool_calls(metadata["contract_floor_calls"])
    if "llm_tool_calls" in metadata:
        safe["llm_tool_calls"] = _summarize_tool_calls(metadata["llm_tool_calls"])
    return safe


def _summarize_tool_calls(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _summarize_tool_calls(item) for key, item in value.items()}
    if isinstance(value, list):
        summarized = []
        for item in value:
            if isinstance(item, dict):
                summarized.append(
                    {
                        "name": item.get("name"),
                        "arguments": item.get("arguments"),
                        "has_result": "result" in item,
                    }
                )
        return summarized
    return value


def _risk_level(risk: ResearchCase, technical: TechnicalState) -> str:
    if technical.volatility_regime == "elevated" or len(risk.evidence) >= 4:
        return "high"
    if len(risk.evidence) >= 2:
        return "medium"
    return "low"


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        normalized = item.strip()
        if normalized and normalized.lower() not in seen:
            out.append(normalized)
            seen.add(normalized.lower())
    return out
