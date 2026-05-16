from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from investment_research_desk.config import Settings
from investment_research_desk.schemas import RunRequest, SentimentInput, SentimentResult

RUNTIME_LABELS = ["bearish", "bullish", "neutral"]
DEFAULT_SENTIMENT_BASE_MODEL = "Qwen/Qwen3-8B"
DEFAULT_ADAPTER_ROOT = Path("models/investment-research-desk-lora-sentiment")


@dataclass(frozen=True)
class SentimentPrediction:
    label: str
    score_margin: float
    label_scores: dict[str, float]
    text: str
    source: str | None = None


class SentimentClassifier(Protocol):
    provider: str

    def classify(self, inputs: list[SentimentInput]) -> list[SentimentPrediction]:
        ...

    def runtime_metadata(self) -> dict[str, Any]:
        ...


class FakeSentimentClassifier:
    provider = "fake"

    def classify(self, inputs: list[SentimentInput]) -> list[SentimentPrediction]:
        predictions = []
        for item in inputs:
            text = item.text.lower()
            if any(term in text for term in ["bearish", "risk", "selloff", "pressure"]):
                label = "bearish"
            elif any(term in text for term in ["bullish", "constructive", "rebound", "uptrend"]):
                label = "bullish"
            else:
                label = "neutral"
            predictions.append(
                SentimentPrediction(
                    label=label,
                    score_margin=1.0,
                    label_scores={candidate: 1.0 if candidate == label else 0.0 for candidate in RUNTIME_LABELS},
                    text=item.text,
                    source=item.source,
                )
            )
        return predictions

    def runtime_metadata(self) -> dict[str, Any]:
        return {"provider": self.provider, "labels": RUNTIME_LABELS, "method": "deterministic_test_classifier"}


class HfPeftSentimentClassifier:
    provider = "hf-peft"

    def __init__(self, base_model: str, adapter_path: Path, score_batch_size: int = 4) -> None:
        self.base_model = base_model
        self.adapter_path = adapter_path
        self.score_batch_size = max(1, score_batch_size)
        self._model: Any | None = None
        self._tokenizer: Any | None = None
        self._load_latency_sec: float | None = None

    def classify(self, inputs: list[SentimentInput]) -> list[SentimentPrediction]:
        if not inputs:
            return []
        model, tokenizer = self._load()
        examples = [{"text": item.text, "label": "neutral"} for item in inputs]
        scored = _score_runtime_labels(
            model,
            tokenizer,
            "twitter_financial_news_sentiment",
            RUNTIME_LABELS,
            examples,
            self.score_batch_size,
        )
        return [
            SentimentPrediction(
                label=str(row["label"]),
                score_margin=float(row.get("score_margin") or 0.0),
                label_scores={str(key): float(value) for key, value in (row.get("label_scores") or {}).items()},
                text=item.text,
                source=item.source,
            )
            for item, row in zip(inputs, scored, strict=True)
        ]

    def runtime_metadata(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "base_model": self.base_model,
            "adapter_path": str(self.adapter_path),
            "labels": RUNTIME_LABELS,
            "method": "forced_choice_label_scoring",
            "score_batch_size": self.score_batch_size,
            "load_latency_sec": self._load_latency_sec,
        }

    def _load(self) -> tuple[Any, Any]:
        if self._model is not None and self._tokenizer is not None:
            return self._model, self._tokenizer
        _require_runtime_packages()
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch

        if not self.adapter_path.exists():
            raise RuntimeError(f"Sentiment adapter path does not exist: {self.adapter_path}")
        config_path = self.adapter_path.parent / "training_config.json"
        base_model = self.base_model
        if config_path.exists():
            try:
                train_config = json.loads(config_path.read_text(encoding="utf-8"))
                base_model = str(train_config.get("base_model") or base_model)
            except json.JSONDecodeError:
                pass
        tokenizer = AutoTokenizer.from_pretrained(self.adapter_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        import time

        started = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=quant_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, self.adapter_path)
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self.base_model = base_model
        self._load_latency_sec = round(time.perf_counter() - started, 3)
        return model, tokenizer


def make_sentiment_classifier(settings: Settings, request: RunRequest) -> SentimentClassifier | None:
    configured_provider = "main" if request.llm_provider == "fake" and request.sentiment_provider is None else settings.sentiment_provider
    provider = (request.sentiment_provider or configured_provider or "main").strip().lower()
    if provider in {"", "main", "none", "disabled"}:
        return None
    if provider == "fake":
        return FakeSentimentClassifier()
    if provider != "hf-peft":
        raise RuntimeError(f"Unsupported sentiment provider: {provider}")
    adapter_path = Path(request.sentiment_adapter_path) if request.sentiment_adapter_path else settings.sentiment_adapter_path
    if adapter_path is None:
        adapter_path = discover_latest_adapter()
    if adapter_path is None:
        raise RuntimeError("IRD_SENTIMENT_ADAPTER_PATH or --sentiment-adapter-path is required for hf-peft sentiment.")
    missing = missing_runtime_packages()
    if missing:
        raise RuntimeError(f"Missing sentiment adapter runtime packages: {', '.join(missing)}")
    base_model = request.sentiment_base_model or settings.sentiment_base_model or DEFAULT_SENTIMENT_BASE_MODEL
    batch_size = request.sentiment_score_batch_size or settings.sentiment_score_batch_size
    return HfPeftSentimentClassifier(base_model=base_model, adapter_path=adapter_path, score_batch_size=batch_size)


def discover_latest_adapter(root: Path = DEFAULT_ADAPTER_ROOT) -> Path | None:
    if not root.exists():
        return None
    candidates = [path for path in root.glob("*/adapter") if path.is_dir() and (path / "adapter_config.json").exists()]
    if not candidates:
        return None
    return sorted(candidates, key=lambda path: path.parent.name, reverse=True)[0]


def aggregate_predictions(inputs: list[SentimentInput], predictions: list[SentimentPrediction]) -> SentimentResult:
    if not predictions:
        return SentimentResult(
            crowd_mood="quiet",
            sentiment_label="neutral",
            sentiment_score=0.0,
            evidence=["sentiment adapter returned no classifications"],
            confidence=0.35,
        )
    score_map = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}
    score = round(sum(score_map.get(row.label, 0.0) for row in predictions) / len(predictions), 2)
    labels = [row.label for row in predictions]
    bullish = labels.count("bullish")
    bearish = labels.count("bearish")
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
    avg_margin = sum(max(row.score_margin, 0.0) for row in predictions) / len(predictions)
    coverage = min(1.0, len(predictions) / max(1, len(inputs)))
    confidence = min(0.9, round(0.45 + 0.25 * coverage + min(avg_margin, 1.0) * 0.2, 2))
    evidence = [
        f"adapter_classification={row.label}; source={row.source or 'unknown'}; text={_truncate(row.text, 180)}"
        for row in predictions[:8]
    ]
    return SentimentResult(
        crowd_mood=mood,
        sentiment_label=label,
        sentiment_score=score,
        evidence=evidence,
        confidence=confidence,
    )


def _require_runtime_packages() -> None:
    missing = missing_runtime_packages()
    if missing:
        raise RuntimeError(f"Missing sentiment adapter runtime packages: {', '.join(missing)}")


def missing_runtime_packages() -> list[str]:
    missing = []
    for package in ["torch", "transformers", "peft", "bitsandbytes", "accelerate"]:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    return missing


def _score_runtime_labels(
    model: Any,
    tokenizer: Any,
    dataset_key: str,
    labels: list[str],
    examples: list[dict[str, Any]],
    batch_size: int,
) -> list[dict[str, Any]]:
    import torch
    import torch.nn.functional as F

    rows: list[tuple[int, str, str, int]] = []
    for example_index, item in enumerate(examples):
        label_list = ", ".join(labels)
        prompt = _chat_prompt(dataset_key, label_list, item["text"])
        prompt_len = len(tokenizer(prompt, add_special_tokens=False)["input_ids"])
        for label in labels:
            continuation = json.dumps({"label": label}, ensure_ascii=False) + "<|im_end|>"
            rows.append((example_index, label, prompt + continuation, prompt_len))

    scores_by_example: list[dict[str, float]] = [dict() for _item in examples]
    batch_size = max(batch_size, 1)
    for start in range(0, len(rows), batch_size):
        batch = rows[start : start + batch_size]
        encoded = tokenizer([row[2] for row in batch], return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
        with torch.no_grad():
            logits = model(**encoded).logits
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        shift_mask = attention_mask[:, 1:].contiguous()
        token_losses = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            reduction="none",
        ).view(shift_labels.shape)
        for row_index, (example_index, label, _text, prompt_len) in enumerate(batch):
            candidate_mask = torch.zeros_like(shift_mask[row_index], dtype=torch.bool)
            candidate_mask[max(prompt_len - 1, 0) :] = shift_mask[row_index, max(prompt_len - 1, 0) :].bool()
            loss = token_losses[row_index][candidate_mask].mean().item() if candidate_mask.any() else float("inf")
            scores_by_example[example_index][label] = -loss

    predictions: list[dict[str, Any]] = []
    for label_scores in scores_by_example:
        ordered = sorted(label_scores.items(), key=lambda item: item[1], reverse=True)
        best_label = ordered[0][0] if ordered else labels[0]
        margin = ordered[0][1] - ordered[1][1] if len(ordered) > 1 else 0.0
        predictions.append({"label": best_label, "score_margin": margin, "label_scores": label_scores})
    return predictions


def _chat_prompt(dataset_key: str, label_list: str, text: str) -> str:
    return (
        f"<|im_start|>system\n/no_think\nJSON only. Choose one label: {label_list}. Schema: {{\"label\":\"<label>\"}}<|im_end|>\n"
        f"<|im_start|>user\ndataset: {dataset_key}\nlabels: {label_list}\ntext: {text}\n/no_think<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _truncate(text: str, limit: int) -> str:
    normalized = " ".join(text.split())
    return normalized if len(normalized) <= limit else normalized[: limit - 3] + "..."
