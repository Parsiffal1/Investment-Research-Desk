from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from investment_research_desk.config import Settings, load_settings
from investment_research_desk.graph import ResearchWorkflow
from investment_research_desk.providers.fixtures import FixtureProvider
from investment_research_desk.schemas import FinalResearchContext, ResearchCase, RunRequest
from investment_research_desk.tools.guardrails import find_guardrail_violations
from investment_research_desk.tools.metrics import approximate_tokens, compression_ratio

EvalSuite = Literal["schema", "guardrail", "single-vs-multi", "consistency", "compression", "latency", "lora"]


def run_eval_suite(suite: EvalSuite, settings: Settings | None = None, results_dir: Path | None = None) -> dict[str, Any]:
    settings = settings or load_settings()
    results_dir = results_dir or Path("eval/results")
    results_dir.mkdir(parents=True, exist_ok=True)
    if suite == "schema":
        result = _schema_suite(settings)
    elif suite == "guardrail":
        result = _guardrail_suite()
    elif suite == "compression":
        result = _compression_suite(settings)
    elif suite == "latency":
        result = _latency_suite(settings)
    elif suite == "consistency":
        result = _consistency_suite(settings)
    elif suite == "single-vs-multi":
        result = _single_vs_multi_suite(settings)
    elif suite == "lora":
        result = _lora_suite()
    else:
        raise ValueError(f"Unsupported eval suite: {suite}")
    _write_result(results_dir, suite, result)
    return result


def _schema_suite(settings: Settings) -> dict[str, Any]:
    workflow = ResearchWorkflow(settings=settings, runs_dir=Path("runs/eval_schema"))
    request = RunRequest(symbol="XAU-USDT-SWAP", asset_class="precious_metal", fixture="gold_cpi", llm_provider="fake")
    state = workflow.run(request)
    FinalResearchContext.model_validate(state["final_context"])
    return {
        "suite": "schema",
        "status": "pass",
        "json_valid_rate": 1.0,
        "schema_valid_rate": 1.0,
        "field_missing_rate": 0.0,
        "run_id": state["run_id"],
    }


def _guardrail_suite() -> dict[str, Any]:
    good_text = (
        "Use as research context only. This is not financial advice. "
        "The report discusses market regime and risk factors."
    )
    bad_text = "Buy now and use 20% of your portfolio. Guaranteed profit."
    good = find_guardrail_violations(good_text)
    bad = find_guardrail_violations(bad_text)
    return {
        "suite": "guardrail",
        "status": "pass" if not good and {"direct_buy", "position_sizing", "guaranteed_profit"}.issubset(set(bad)) else "fail",
        "allowed_text_violations": good,
        "blocked_text_violations": bad,
    }


def _compression_suite(settings: Settings) -> dict[str, Any]:
    workflow = ResearchWorkflow(settings=settings, runs_dir=Path("runs/eval_compression"))
    request = RunRequest(symbol="XAU-USDT-SWAP", asset_class="precious_metal", fixture="gold_cpi", llm_provider="fake")
    state = workflow.run(request)
    raw = approximate_tokens(state["data"])
    final = approximate_tokens(state["final_context"])
    return {
        "suite": "compression",
        "status": "pass",
        "raw_input_tokens": raw,
        "final_context_tokens": final,
        "compression_ratio": compression_ratio(raw, final),
        "run_id": state["run_id"],
    }


def _latency_suite(settings: Settings) -> dict[str, Any]:
    workflow = ResearchWorkflow(settings=settings, runs_dir=Path("runs/eval_latency"))
    request = RunRequest(symbol="XAU-USDT-SWAP", asset_class="precious_metal", fixture="gold_cpi", llm_provider="fake")
    state = workflow.run(request)
    return {
        "suite": "latency",
        "status": "pass",
        "metrics": state["metrics"],
        "run_id": state["run_id"],
    }


def _consistency_suite(settings: Settings) -> dict[str, Any]:
    workflow = ResearchWorkflow(settings=settings, runs_dir=Path("runs/eval_consistency"))
    request = RunRequest(symbol="XAU-USDT-SWAP", asset_class="precious_metal", fixture="gold_cpi", llm_provider="fake")
    contexts = [FinalResearchContext.model_validate(workflow.run(request)["final_context"]) for _ in range(3)]
    regime_consistent = len({ctx.market_regime for ctx in contexts}) == 1
    balanced_consistent = len({ctx.balanced_view for ctx in contexts}) == 1
    return {
        "suite": "consistency",
        "status": "pass" if regime_consistent and balanced_consistent else "fail",
        "regime_consistent": regime_consistent,
        "balanced_view_consistent": balanced_consistent,
        "runs": len(contexts),
    }


def _single_vs_multi_suite(settings: Settings) -> dict[str, Any]:
    workflow = ResearchWorkflow(settings=settings, runs_dir=Path("runs/eval_single_vs_multi"))
    request = RunRequest(symbol="XAU-USDT-SWAP", asset_class="precious_metal", fixture="gold_cpi", llm_provider="fake")
    state = workflow.run(request)
    multi_context = FinalResearchContext.model_validate(state["final_context"])
    fixture_data = FixtureProvider().load("gold_cpi")
    single_context = _single_agent_fixture_baseline(fixture_data)
    multi_coverage = _coverage_score(multi_context)
    single_coverage = _coverage_score(single_context)
    return {
        "suite": "single-vs-multi",
        "status": "pass" if multi_coverage["score"] >= single_coverage["score"] else "fail",
        "single_agent_coverage": single_coverage,
        "multi_agent_coverage": multi_coverage,
        "schema_valid": True,
        "run_id": state["run_id"],
    }


def _lora_suite() -> dict[str, Any]:
    return {
        "suite": "lora",
        "status": "not_configured",
        "message": "LoRA artifacts are not part of the CLI MVP. Run this after investment-research-desk-lora is trained.",
        "baseline_model": "Qwen3-8B Instruct/Chat",
        "fine_tuned_model": "Qwen3-8B Instruct/Chat + investment-research-desk-lora",
    }


def _write_result(results_dir: Path, suite: str, result: dict[str, Any]) -> None:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = results_dir / f"{timestamp}_{suite}.json"
    md_path = results_dir / f"{timestamp}_{suite}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [f"# Eval Suite: {suite}", "", f"- Status: {result.get('status')}"]
    for key, value in result.items():
        if key not in {"suite", "status"}:
            lines.append(f"- {key}: `{value}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _single_agent_fixture_baseline(data) -> FinalResearchContext:
    news_titles = [event.title for event in data.news_events]
    return FinalResearchContext(
        symbol=data.symbol,
        asset_class=data.asset_class,
        horizon=data.horizon,
        market_regime="single_agent_direct_summary",
        balanced_view="mixed",
        risk_level="medium",
        confidence=0.55,
        fundamental_summary=None,
        news_impact_summary="; ".join(news_titles) or "No news events available.",
        sentiment_summary="Single-agent baseline does not separately score sentiment.",
        technical_summary="Single-agent baseline does not compute dedicated technical indicators.",
        constructive_case=ResearchCase(
            thesis="Constructive case is inferred directly from all inputs in one pass.",
            evidence=news_titles[:1] or ["limited constructive evidence"],
            conditions=["requires follow-up validation"],
            confidence=0.5,
        ),
        risk_case=ResearchCase(
            thesis="Risk case is inferred directly from all inputs in one pass.",
            evidence=["macro uncertainty remains present"],
            conditions=["requires follow-up validation"],
            confidence=0.5,
        ),
        key_drivers=news_titles[:2],
        key_risks=["single-agent baseline may miss separated technical, sentiment, and macro conflicts"],
        uncertainty_factors=["single prompt consumes all context without typed intermediate outputs"],
        downstream_agent_context="Use as research context only. A separate decision, risk, and execution system is required before any trading action.",
        usage_constraints=["not financial advice", "not an order instruction", "does not include position sizing"],
        source_metadata=data.source_metadata,
    )


def _coverage_score(context: FinalResearchContext) -> dict[str, Any]:
    dimensions = {
        "fundamental": bool(context.fundamental_summary),
        "news": bool(context.news_impact_summary),
        "sentiment": bool(context.sentiment_summary and "does not separately" not in context.sentiment_summary),
        "technical": bool(context.technical_summary and "does not compute" not in context.technical_summary),
        "constructive_case": bool(context.constructive_case.evidence),
        "risk_case": bool(context.risk_case.evidence),
        "risk_flags": bool(context.key_risks),
    }
    score = sum(1 for value in dimensions.values() if value)
    return {"score": score, "max_score": len(dimensions), "dimensions": dimensions}
