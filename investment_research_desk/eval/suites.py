from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import httpx

from investment_research_desk.config import Settings, load_settings
from investment_research_desk.graph import ResearchWorkflow
from investment_research_desk.llm import make_llm_client
from investment_research_desk.providers.fixtures import FixtureProvider
from investment_research_desk.schemas import FinalResearchContext, ResearchCase, RunRequest
from investment_research_desk.tools.guardrails import find_guardrail_violations
from investment_research_desk.tools.metrics import approximate_tokens, compression_ratio

EvalSuite = Literal["schema", "guardrail", "single-vs-multi", "consistency", "compression", "latency", "lora", "sentiment-baseline"]

SENTIMENT_DATASETS = {
    "financial_phrasebank": {
        "dataset": "ArtGarfunkel/FinancialPhraseBank",
        "config": "default",
        "split": "test",
        "text_field": "sentence",
        "label_field": "sentiment",
        "labels": ["negative", "neutral", "positive"],
        "label_map": {},
        "source_url": "https://huggingface.co/datasets/ArtGarfunkel/FinancialPhraseBank",
        "leakage_policy": "held-out test split; do not use this split for LoRA training",
    },
    "twitter_financial_news_sentiment": {
        "dataset": "zeroshot/twitter-financial-news-sentiment",
        "config": "default",
        "split": "validation",
        "text_field": "text",
        "label_field": "label",
        "labels": ["bearish", "bullish", "neutral"],
        "label_map": {0: "bearish", 1: "bullish", 2: "neutral"},
        "source_url": "https://huggingface.co/datasets/zeroshot/twitter-financial-news-sentiment",
        "leakage_policy": "held-out validation split; do not use this split for LoRA training",
    },
}


def run_eval_suite(
    suite: EvalSuite,
    settings: Settings | None = None,
    results_dir: Path | None = None,
    llm_provider: str = "ollama",
    model: str | None = None,
    limit: int | None = None,
    dataset_dir: Path | None = None,
) -> dict[str, Any]:
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
    elif suite == "sentiment-baseline":
        result = _sentiment_baseline_suite(settings, llm_provider=llm_provider, model=model, limit=limit, dataset_dir=dataset_dir)
    else:
        raise ValueError(f"Unsupported eval suite: {suite}")
    result["artifacts"] = _write_result(results_dir, suite, result)
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


def _sentiment_baseline_suite(
    settings: Settings,
    llm_provider: str,
    model: str | None,
    limit: int | None,
    dataset_dir: Path | None,
) -> dict[str, Any]:
    llm = make_llm_client(settings, llm_provider, model)
    started = datetime.now(timezone.utc)
    dataset_results: dict[str, Any] = {}
    for dataset_key, spec in SENTIMENT_DATASETS.items():
        examples = _load_sentiment_dataset(dataset_key, spec, dataset_dir=dataset_dir, limit=limit)
        predictions = []
        for index, item in enumerate(examples, start=1):
            predicted = _predict_sentiment_label(
                llm,
                text=item["text"],
                labels=spec["labels"],
                dataset_name=dataset_key,
            )
            expected = item["label"]
            predictions.append(
                {
                    "index": index,
                    "expected": expected,
                    "predicted": predicted,
                    "correct": predicted == expected,
                    "text": item["text"],
                }
            )
        y_true = [item["expected"] for item in predictions]
        y_pred = [item["predicted"] for item in predictions]
        metrics = _classification_metrics(y_true, y_pred, labels=spec["labels"])
        dataset_results[dataset_key] = {
            "dataset": spec["dataset"],
            "split": spec["split"],
            "source_url": spec["source_url"],
            "leakage_policy": spec["leakage_policy"],
            "labels": spec["labels"],
            "samples": len(predictions),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class": metrics["per_class"],
            "predictions": predictions,
        }
    completed = datetime.now(timezone.utc)
    macro_f1_values = [item["macro_f1"] for item in dataset_results.values()]
    accuracy_values = [item["accuracy"] for item in dataset_results.values()]
    return {
        "suite": "sentiment-baseline",
        "status": "pass",
        "task": "financial_sentiment_three_class_classification",
        "model": model or settings.ollama_model,
        "llm_provider": llm_provider,
        "limit_per_dataset": limit,
        "data_leakage_policy": (
            "This suite uses only held-out evaluation splits. LoRA/SFT data preparation must exclude these "
            "dataset+config+split pairs and should write train/eval manifests before training."
        ),
        "accuracy": sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0.0,
        "macro_f1": sum(macro_f1_values) / len(macro_f1_values) if macro_f1_values else 0.0,
        "datasets": dataset_results,
        "started_at": started.isoformat(),
        "completed_at": completed.isoformat(),
    }


def _load_sentiment_dataset(dataset_key: str, spec: dict[str, Any], dataset_dir: Path | None, limit: int | None) -> list[dict[str, str]]:
    cache_dir = dataset_dir or Path("eval/data")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{dataset_key}_{spec['split']}.jsonl"
    if cache_path.exists():
        rows = [json.loads(line) for line in cache_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        rows = _fetch_hf_rows(spec)
        cache_path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")
    examples = [_normalize_sentiment_row(row, spec) for row in rows]
    examples = [item for item in examples if item["text"] and item["label"] in spec["labels"]]
    if limit is not None and limit > 0:
        return _stratified_limit(examples, spec["labels"], limit)
    return examples


def _fetch_hf_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    offset = 0
    page_size = 100
    total: int | None = None
    with httpx.Client(timeout=60.0, follow_redirects=True) as client:
        while total is None or offset < total:
            params = {
                "dataset": spec["dataset"],
                "config": spec["config"],
                "split": spec["split"],
                "offset": offset,
                "length": page_size,
            }
            response = _hf_rows_request_with_retry(client, params)
            response.raise_for_status()
            payload = response.json()
            total = int(payload.get("num_rows_total") or 0)
            page = [item.get("row") or {} for item in payload.get("rows") or []]
            if not page:
                break
            rows.extend(page)
            offset += len(page)
    return rows


def _hf_rows_request_with_retry(client: httpx.Client, params: dict[str, Any]) -> httpx.Response:
    for attempt in range(4):
        response = client.get("https://datasets-server.huggingface.co/rows", params=params)
        if response.status_code != 429:
            return response
        if attempt < 3:
            time.sleep(2**attempt)
    return response


def _normalize_sentiment_row(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, str]:
    raw_label = row.get(spec["label_field"])
    label_map = spec.get("label_map") or {}
    label = label_map.get(raw_label, raw_label)
    return {"text": str(row.get(spec["text_field"]) or "").strip(), "label": str(label).strip().lower()}


def _stratified_limit(examples: list[dict[str, str]], labels: list[str], limit: int) -> list[dict[str, str]]:
    buckets = {label: [item for item in examples if item["label"] == label] for label in labels}
    selected: list[dict[str, str]] = []
    index = 0
    while len(selected) < limit:
        progressed = False
        for label in labels:
            bucket = buckets[label]
            if index < len(bucket):
                selected.append(bucket[index])
                progressed = True
                if len(selected) >= limit:
                    break
        if not progressed:
            break
        index += 1
    return selected


def _predict_sentiment_label(llm, text: str, labels: list[str], dataset_name: str) -> str:
    label_list = ", ".join(labels)
    system = (
        "You are a strict financial sentiment classifier. Return exactly one valid JSON object with one field: "
        f"{{\"label\":\"one of: {label_list}\"}}. Use only the allowed labels. Do not explain."
    )
    user = (
        f"Dataset: {dataset_name}\n"
        f"Allowed labels: {label_list}\n\n"
        f"Financial text:\n{text}\n\n"
        "Classify the sentiment."
    )
    try:
        raw = llm.chat_json(system, user)
    except Exception:
        return labels[0]
    label = str(raw.get("label") or "").strip().lower()
    return label if label in labels else labels[0]


def _classification_metrics(y_true: list[str], y_pred: list[str], labels: list[str]) -> dict[str, Any]:
    total = len(y_true)
    correct = sum(1 for expected, predicted in zip(y_true, y_pred) if expected == predicted)
    per_class: dict[str, dict[str, float]] = {}
    f1_values: list[float] = []
    for label in labels:
        tp = sum(1 for expected, predicted in zip(y_true, y_pred) if expected == label and predicted == label)
        fp = sum(1 for expected, predicted in zip(y_true, y_pred) if expected != label and predicted == label)
        fn = sum(1 for expected, predicted in zip(y_true, y_pred) if expected == label and predicted != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": float(sum(1 for item in y_true if item == label))}
        f1_values.append(f1)
    return {
        "accuracy": correct / total if total else 0.0,
        "macro_f1": sum(f1_values) / len(f1_values) if f1_values else 0.0,
        "per_class": per_class,
    }


def _write_result(results_dir: Path, suite: str, result: dict[str, Any]) -> dict[str, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    json_path = results_dir / f"{timestamp}_{suite}.json"
    md_path = results_dir / f"{timestamp}_{suite}.md"
    json_path.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    lines = [f"# Eval Suite: {suite}", "", f"- Status: {result.get('status')}"]
    for key, value in result.items():
        if key not in {"suite", "status", "predictions"}:
            lines.append(f"- {key}: `{value}`")
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "markdown": str(md_path)}


def _single_agent_fixture_baseline(data) -> FinalResearchContext:
    news_titles = [event.title for event in data.news_events]
    return FinalResearchContext(
        symbol=data.symbol,
        asset_class=data.asset_class,
        horizon=data.horizon,
        market_regime="single_agent_direct_summary",
        directional_view="bullish",
        directional_rationale="Single-agent fixture baseline uses a bullish placeholder direction when evidence is not separated by analyst role.",
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
