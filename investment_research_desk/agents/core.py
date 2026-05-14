from __future__ import annotations

from investment_research_desk.llm import LLMClient
from investment_research_desk.schemas import (
    FinalResearchContext,
    FundamentalMacroResult,
    NewsImpactResult,
    NormalizedData,
    ResearchCase,
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
        titles = " ".join(event.title for event in data.news_events).lower()
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
        return FundamentalMacroResult(
            fundamental_view=_view_from_counts(len(drivers), len(concerns)),
            key_drivers=drivers,
            concerns=concerns,
            confidence=0.66,
            evidence=[event.title for event in data.news_events[:3]],
        )


class NewsImpactAnalyst:
    name = "news_impact"

    def run(self, data: NormalizedData, llm: LLMClient) -> NewsImpactResult:
        contract = get_agent_contract(self.name)
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
        if llm.provider != "fake" and dominant:
            try:
                llm_result = llm.chat_json(
                    contract.system_prompt,
                    (
                        "Analyze the market impact for the target asset. "
                        "Return JSON with keys impact_logic and confidence. "
                        f"Target: {data.symbol}. Events: {dominant}"
                    ),
                )
                impact_logic = str(llm_result.get("impact_logic") or impact_logic)
            except Exception:
                impact_logic = f"{impact_logic} LLM impact refinement failed; deterministic fallback was used."
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

    def run(self, data: NormalizedData, llm: LLMClient) -> SentimentResult:
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
        return SentimentResult(
            crowd_mood=mood,
            sentiment_label=label,
            sentiment_score=score,
            evidence=evidence,
            confidence=0.68 if texts else 0.35,
        )


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
        return TechnicalState(
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
            support_zones=supports,
            resistance_zones=resistances,
            confidence=0.78 if len(bars) >= 26 else 0.45,
        )


class ConstructiveCaseAnalyst:
    name = "constructive_case"

    def run(
        self,
        fundamental: FundamentalMacroResult,
        news: NewsImpactResult,
        sentiment: SentimentResult,
        technical: TechnicalState,
        llm: LLMClient,
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
        return ResearchCase(
            thesis="Constructive case depends on supportive macro drivers and technical confirmation.",
            evidence=evidence[:6],
            conditions=[
                "price remains above identified support zones",
                "macro news does not reprice risk sharply against the asset",
            ],
            confidence=min(0.82, max(0.45, (fundamental.confidence + news.confidence + technical.confidence) / 3)),
        )


class RiskCaseAnalyst:
    name = "risk_case"

    def run(
        self,
        fundamental: FundamentalMacroResult,
        news: NewsImpactResult,
        sentiment: SentimentResult,
        technical: TechnicalState,
        constructive: ResearchCase,
        llm: LLMClient,
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
        return ResearchCase(
            thesis="Risk case centers on macro repricing, volatility, and possible technical failure.",
            evidence=evidence[:6],
            conditions=[
                "break below support zones",
                "hawkish macro surprise or stronger dollar impulse",
                "news flow contradicts the constructive case",
            ],
            confidence=min(0.85, max(0.45, (fundamental.confidence + news.confidence + technical.confidence) / 3)),
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
        warnings: list[str],
        llm: LLMClient,
    ) -> FinalResearchContext:
        balanced = _balance_view(news.asset_impact.get(data.symbol, "mixed"), sentiment.sentiment_label, technical.technical_view)
        risk_level = _risk_level(risk, technical)
        key_drivers = _dedupe(constructive.evidence + fundamental.key_drivers + news.dominant_events)[:6]
        key_risks = _dedupe(risk.evidence + fundamental.concerns)[:6]
        return FinalResearchContext(
            symbol=data.symbol,
            asset_class=data.asset_class,
            horizon=data.horizon,
            market_regime=f"{technical.trend}_{technical.volatility_regime}",
            balanced_view=balanced,
            risk_level=risk_level,
            confidence=round((fundamental.confidence + news.confidence + sentiment.confidence + technical.confidence) / 4, 2),
            fundamental_summary="; ".join(fundamental.key_drivers[:2] + fundamental.concerns[:2]),
            news_impact_summary=news.impact_logic,
            sentiment_summary=f"Market mood is {sentiment.crowd_mood}; label={sentiment.sentiment_label}; score={sentiment.sentiment_score}.",
            technical_summary=f"Trend={technical.trend}; momentum={technical.momentum}; RSI={technical.rsi_14}; MACD={technical.macd_state}.",
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
            source_metadata=data.source_metadata,
            warnings=warnings,
        )


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
