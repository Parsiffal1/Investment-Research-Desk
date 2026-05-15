from __future__ import annotations

import json
import shutil
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from investment_research_desk.eval import suites


BASELINE_METRICS = {
    "accuracy": 0.7900227722635215,
    "macro_f1": 0.7771392512613777,
    "datasets": {
        "financial_phrasebank": {"accuracy": 0.8078512396694215, "macro_f1": 0.805105716432507},
        "twitter_financial_news_sentiment": {"accuracy": 0.7721943048576214, "macro_f1": 0.7491727860902486},
    },
}

LORA_DATASETS = {
    "financial_phrasebank": {
        **suites.SENTIMENT_DATASETS["financial_phrasebank"],
        "train_splits": ["train", "validation"],
        "eval_split": "test",
    },
    "twitter_financial_news_sentiment": {
        **suites.SENTIMENT_DATASETS["twitter_financial_news_sentiment"],
        "train_splits": ["train"],
        "eval_split": "validation",
    },
}


@dataclass(frozen=True)
class LoraTrainingConfig:
    base_model: str = "Qwen/Qwen3-8B"
    output_root: str = "models/investment-research-desk-lora-sentiment"
    method: str = "qlora_4bit_nf4"
    target_modules: str = "all-linear"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    num_train_epochs: int = 2
    learning_rate: float = 2e-4
    per_device_train_batch_size: int = 1
    gradient_accumulation_steps: int = 16
    max_seq_length: int = 256
    lr_scheduler_type: str = "cosine"
    warmup_ratio: float = 0.03
    gradient_checkpointing: bool = True
    seed: int = 42


def prepare_lora_data(
    output_dir: Path,
    dataset_dir: Path | None = None,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    train_examples: list[dict[str, Any]] = []
    dev_examples: list[dict[str, Any]] = []
    eval_examples: list[dict[str, Any]] = []
    train_manifest: list[dict[str, Any]] = []
    dev_manifest: list[dict[str, Any]] = []
    eval_manifest: list[dict[str, Any]] = []
    split_summary: dict[str, Any] = {}

    for dataset_key, spec in LORA_DATASETS.items():
        labels = spec["labels"]
        raw_training_examples: list[dict[str, Any]] = []
        raw_training_manifest: list[dict[str, Any]] = []
        for split in spec["train_splits"]:
            split_spec = {**spec, "split": split}
            examples = _load_examples(dataset_key, split_spec, dataset_dir, limit)
            raw_training_examples.extend(examples)
            raw_training_manifest.extend(suites._manifest_entries(dataset_key, split_spec, examples))
        eval_spec = {**spec, "split": spec["eval_split"]}
        heldout_examples = _load_examples(dataset_key, eval_spec, dataset_dir, limit)
        split_eval_manifest = suites._manifest_entries(dataset_key, eval_spec, heldout_examples)
        filtered_training_pairs, removed_overlap_count = _exclude_eval_hash_overlaps(
            list(zip(raw_training_examples, raw_training_manifest, strict=True)),
            split_eval_manifest,
        )
        split_dev_pairs = _dev_pair_slice(filtered_training_pairs)
        split_train_pairs = filtered_training_pairs[len(split_dev_pairs) :]
        split_dev_examples = [example for example, _manifest in split_dev_pairs]
        split_dev_manifest = [manifest for _example, manifest in split_dev_pairs]
        split_train_examples = [example for example, _manifest in split_train_pairs]
        split_train_manifest = [manifest for _example, manifest in split_train_pairs]
        train_examples.extend(_format_sft_examples(dataset_key, labels, split_train_examples))
        dev_examples.extend(_format_sft_examples(dataset_key, labels, split_dev_examples))
        train_manifest.extend(split_train_manifest)
        dev_manifest.extend(split_dev_manifest)
        eval_examples.extend(_format_sft_examples(dataset_key, labels, heldout_examples))
        eval_manifest.extend(split_eval_manifest)
        split_summary[dataset_key] = {
            "train_splits": spec["train_splits"],
            "eval_split": spec["eval_split"],
            "raw_train_samples": len(raw_training_examples),
            "removed_train_eval_hash_overlaps": removed_overlap_count,
            "train_samples": len(split_train_examples),
            "dev_samples": len(split_dev_examples),
            "eval_samples": len(heldout_examples),
        }

    train_eval_overlap = _manifest_overlap(train_manifest, eval_manifest)
    dev_eval_overlap = _manifest_overlap(dev_manifest, eval_manifest)
    train_dev_overlap = _manifest_overlap(train_manifest, dev_manifest)
    leakage_check = _in_memory_leakage_check(train_manifest, eval_manifest, train_eval_overlap, dev_eval_overlap)
    result = {
        "status": "dry_run" if dry_run else "written",
        "output_dir": str(output_dir),
        "datasets": split_summary,
        "train_samples": len(train_examples),
        "dev_samples": len(dev_examples),
        "eval_samples": len(eval_examples),
        "leakage_check": leakage_check,
        "train_eval_overlap": train_eval_overlap,
        "dev_eval_overlap": dev_eval_overlap,
        "train_dev_overlap": train_dev_overlap,
    }
    if dry_run:
        return result

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "train.jsonl", train_examples)
    _write_jsonl(output_dir / "dev.jsonl", dev_examples)
    _write_jsonl(output_dir / "eval.jsonl", eval_examples)
    _write_jsonl(output_dir / "train_manifest.jsonl", train_manifest)
    _write_jsonl(output_dir / "dev_manifest.jsonl", dev_manifest)
    _write_jsonl(output_dir / "eval_manifest.jsonl", eval_manifest)
    _write_json(output_dir / "data_summary.json", result)
    return result


def train_lora_sentiment(
    data_dir: Path,
    output_root: Path,
    config: LoraTrainingConfig,
    dry_run: bool = False,
) -> dict[str, Any]:
    run_dir = output_root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    training_config = asdict(config)
    training_config["output_dir"] = str(run_dir)
    training_config["data_dir"] = str(data_dir)
    if dry_run:
        return {
            "status": "dry_run",
            "training_config": training_config,
            "required_packages": ["torch", "transformers", "datasets", "trl", "peft", "bitsandbytes", "accelerate"],
        }

    _require_training_packages()
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer
    import torch

    run_dir.mkdir(parents=True, exist_ok=True)
    _write_json(run_dir / "training_config.json", training_config)
    _copy_manifest_artifacts(data_dir, run_dir)
    dataset = load_dataset("json", data_files={"train": str(data_dir / "train.jsonl"), "validation": str(data_dir / "dev.jsonl")})
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    args = SFTConfig(
        output_dir=str(run_dir / "trainer"),
        num_train_epochs=config.num_train_epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_ratio=config.warmup_ratio,
        gradient_checkpointing=config.gradient_checkpointing,
        bf16=True,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        seed=config.seed,
        report_to=[],
        dataset_text_field="text",
        max_length=config.max_seq_length,
        packing=False,
    )
    trainer = SFTTrainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    metrics = trainer.evaluate()
    adapter_dir = run_dir / "adapter"
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)
    _write_json(run_dir / "dev_metrics.json", {"train": train_result.metrics, "eval": metrics})
    return {"status": "trained", "run_dir": str(run_dir), "adapter": str(adapter_dir), "dev_metrics": metrics}


def eval_lora_sentiment(
    adapter_path: Path,
    output_dir: Path,
    dataset_dir: Path | None = None,
    limit: int | None = None,
    contract_limit: int = 6,
    score_batch_size: int = 4,
    dry_run: bool = False,
) -> dict[str, Any]:
    if dry_run:
        return {
            "status": "dry_run",
            "adapter_path": str(adapter_path),
            "limit": limit,
            "contract_limit": contract_limit,
            "score_batch_size": score_batch_size,
            "eval_method": "forced_choice_label_scoring",
            "baseline": BASELINE_METRICS,
        }
    _require_eval_packages()
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch

    config_path = adapter_path.parent / "training_config.json"
    train_config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else asdict(LoraTrainingConfig())
    base_model = train_config.get("base_model", LoraTrainingConfig().base_model)
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    dataset_results: dict[str, Any] = {}
    contract_results: dict[str, Any] = {}
    for dataset_key, spec in LORA_DATASETS.items():
        eval_spec = {**spec, "split": spec["eval_split"]}
        examples = _load_examples(dataset_key, eval_spec, dataset_dir, limit)
        scored_predictions = _score_adapter_labels(model, tokenizer, dataset_key, spec["labels"], examples, score_batch_size)
        predictions = []
        for index, (item, scored) in enumerate(zip(examples, scored_predictions, strict=True), start=1):
            expected = item["label"]
            predictions.append(
                {
                    "index": index,
                    "expected": expected,
                    "predicted": scored["label"],
                    "correct": scored["label"] == expected,
                    "text": item["text"],
                    "score_margin": scored["score_margin"],
                    "label_scores": scored["label_scores"],
                }
            )
        contract_predictions = []
        for index, item in enumerate(examples[: max(contract_limit, 0)], start=1):
            predicted, violations, raw_text = _generate_adapter_label(model, tokenizer, dataset_key, spec["labels"], item["text"])
            contract_predictions.append(
                {
                    "index": index,
                    "expected": item["label"],
                    "predicted": predicted,
                    "correct": predicted == item["label"],
                    "text": item["text"],
                    "raw_output": raw_text,
                    "output_violations": violations,
                }
            )
        metrics = suites._classification_metrics(
            [item["expected"] for item in predictions],
            [item["predicted"] for item in predictions],
            spec["labels"],
        )
        dataset_results[dataset_key] = {
            "dataset": spec["dataset"],
            "split": spec["eval_split"],
            "labels": spec["labels"],
            "samples": len(predictions),
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "per_class": metrics["per_class"],
            "predictions": predictions,
        }
        contract_results[dataset_key] = {
            "dataset": spec["dataset"],
            "split": spec["eval_split"],
            "labels": spec["labels"],
            "samples": len(contract_predictions),
            "predictions": contract_predictions,
        }
    accuracy_values = [item["accuracy"] for item in dataset_results.values()]
    macro_values = [item["macro_f1"] for item in dataset_results.values()]
    output_contract = suites._sentiment_output_contract(contract_results)
    violations = output_contract.get("violations", {})
    output_contract["status"] = "pass" if not any(violations.values()) else "fail"
    result = {
        "status": "pass" if output_contract["status"] == "pass" else "fail",
        "adapter_path": str(adapter_path),
        "eval_method": "forced_choice_label_scoring",
        "contract_check_method": "generative_json_sample",
        "contract_limit_per_dataset": max(contract_limit, 0),
        "score_batch_size": score_batch_size,
        "baseline": BASELINE_METRICS,
        "accuracy": sum(accuracy_values) / len(accuracy_values) if accuracy_values else 0.0,
        "macro_f1": sum(macro_values) / len(macro_values) if macro_values else 0.0,
        "baseline_delta": {},
        "output_contract": output_contract,
        "datasets": dataset_results,
        "contract_samples": contract_results,
    }
    result["baseline_delta"] = {
        "accuracy": result["accuracy"] - BASELINE_METRICS["accuracy"],
        "macro_f1": result["macro_f1"] - BASELINE_METRICS["macro_f1"],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "heldout_eval_results.json", result)
    (output_dir / "heldout_eval_results.md").write_text(_lora_eval_markdown(result), encoding="utf-8")
    return result


def _load_examples(dataset_key: str, spec: dict[str, Any], dataset_dir: Path | None, limit: int | None) -> list[dict[str, str]]:
    try:
        from datasets import load_dataset
    except ImportError:
        return suites._load_sentiment_dataset(dataset_key, spec, dataset_dir=dataset_dir, limit=limit)

    cache_dir = str(dataset_dir) if dataset_dir else None
    dataset = load_dataset(spec["dataset"], spec["config"], split=spec["split"], cache_dir=cache_dir)
    rows: list[dict[str, Any]] = []
    max_scan = len(dataset) if not limit or limit <= 0 else min(len(dataset), max(200, limit * 20))
    for index in range(max_scan):
        rows.append({"row_idx": index, "row": dataset[index]})
    examples = [suites._normalize_sentiment_row(row, spec) for row in rows]
    examples = [item for item in examples if item["text"] and item["label"] in spec["labels"]]
    if limit is not None and limit > 0:
        return suites._stratified_limit(examples, spec["labels"], limit)
    return examples


def _format_sft_examples(dataset_key: str, labels: list[str], examples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in examples:
        label_list = ", ".join(labels)
        rows.append(
            {
                "text": _chat_text(dataset_key, label_list, item["text"], item["label"]),
                "messages": [
                    {"role": "system", "content": f"/no_think\nJSON only. Choose one label: {label_list}. Schema: {{\"label\":\"<label>\"}}"},
                    {"role": "user", "content": f"dataset: {dataset_key}\nlabels: {label_list}\ntext: {item['text']}"},
                    {"role": "assistant", "content": json.dumps({"label": item["label"]}, ensure_ascii=False)},
                ],
                "label": item["label"],
                "dataset": dataset_key,
                "row_idx": item.get("row_idx"),
            }
        )
    return rows


def _chat_text(dataset_key: str, label_list: str, text: str, label: str) -> str:
    return _chat_prompt(dataset_key, label_list, text) + json.dumps({"label": label}, ensure_ascii=False) + "<|im_end|>"


def _chat_prompt(dataset_key: str, label_list: str, text: str) -> str:
    return (
        f"<|im_start|>system\n/no_think\nJSON only. Choose one label: {label_list}. Schema: {{\"label\":\"<label>\"}}<|im_end|>\n"
        f"<|im_start|>user\ndataset: {dataset_key}\nlabels: {label_list}\ntext: {text}\n/no_think<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def _exclude_eval_hash_overlaps(
    training_pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    eval_manifest: list[dict[str, Any]],
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], int]:
    eval_text_hashes = {item["text_sha256"] for item in eval_manifest}
    eval_norm_hashes = {item["normalized_text_sha256"] for item in eval_manifest}
    filtered = [
        (example, manifest)
        for example, manifest in training_pairs
        if manifest["text_sha256"] not in eval_text_hashes and manifest["normalized_text_sha256"] not in eval_norm_hashes
    ]
    return filtered, len(training_pairs) - len(filtered)


def _dev_pair_slice(
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    size: int = 64,
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    if len(pairs) <= 1:
        return []
    dev_size = min(size, max(1, len(pairs) // 10), len(pairs) - 1)
    return pairs[:dev_size]


def _manifest_overlap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> dict[str, Any]:
    left_rows = {suites._row_key(item) for item in left if item.get("row_idx") is not None}
    right_rows = {suites._row_key(item) for item in right if item.get("row_idx") is not None}
    left_hashes = {item["normalized_text_sha256"] for item in left}
    right_hashes = {item["normalized_text_sha256"] for item in right}
    return {
        "row_overlap_count": len(left_rows & right_rows),
        "normalized_text_hash_overlap_count": len(left_hashes & right_hashes),
    }


def _in_memory_leakage_check(
    train_manifest: list[dict[str, Any]],
    eval_manifest: list[dict[str, Any]],
    train_eval_overlap: dict[str, Any],
    dev_eval_overlap: dict[str, Any],
) -> dict[str, Any]:
    eval_split_keys = sorted({f"{item['dataset']}::{item['config']}::{item['split']}" for item in eval_manifest})
    split_overlaps = sorted(
        {
            f"{item['dataset']}::{item['config']}::{item['split']}"
            for item in train_manifest
            if f"{item.get('dataset')}::{item.get('config')}::{item.get('split')}" in eval_split_keys
        }
    )
    failed = bool(
        split_overlaps
        or train_eval_overlap["row_overlap_count"]
        or train_eval_overlap["normalized_text_hash_overlap_count"]
        or dev_eval_overlap["row_overlap_count"]
        or dev_eval_overlap["normalized_text_hash_overlap_count"]
    )
    return {
        "status": "fail" if failed else "pass",
        "heldout_eval_splits": eval_split_keys,
        "train_exclusion_keys": eval_split_keys,
        "train_samples": len(train_manifest),
        "eval_samples": len(eval_manifest),
        "split_overlaps": split_overlaps,
        "train_eval_overlap": train_eval_overlap,
        "dev_eval_overlap": dev_eval_overlap,
    }


def _score_adapter_labels(
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


def _generate_adapter_label(model: Any, tokenizer: Any, dataset_key: str, labels: list[str], text: str) -> tuple[str, list[str], str]:
    import torch

    label_list = ", ".join(labels)
    prompt = _chat_prompt(dataset_key, label_list, text)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=suites.SENTIMENT_EVAL_MAX_TOKENS, do_sample=False)
    decoded = tokenizer.decode(generated[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
    try:
        raw = json.loads(decoded.strip())
    except json.JSONDecodeError:
        raw = {}
    violations = suites._sentiment_output_violations(raw, decoded, "", labels)
    label = str(raw.get("label") or "").strip().lower()
    return (label if label in labels else labels[0]), violations, decoded


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False, default=str) for row in rows) + "\n", encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def _copy_manifest_artifacts(data_dir: Path, run_dir: Path) -> None:
    for name in ["train_manifest.jsonl", "dev_manifest.jsonl", "eval_manifest.jsonl", "data_summary.json"]:
        source = data_dir / name
        if source.exists():
            shutil.copyfile(source, run_dir / name)


def _lora_eval_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Investment Research Desk Sentiment LoRA Held-out Evaluation",
        "",
        f"- Adapter: `{result.get('adapter_path')}`",
        f"- Accuracy: {_metric(result.get('accuracy'))}",
        f"- Macro-F1: {_metric(result.get('macro_f1'))}",
        f"- Baseline accuracy delta: {_metric((result.get('baseline_delta') or {}).get('accuracy'), signed=True)}",
        f"- Baseline Macro-F1 delta: {_metric((result.get('baseline_delta') or {}).get('macro_f1'), signed=True)}",
        f"- Output contract: `{(result.get('output_contract') or {}).get('status', 'unknown')}`",
        "",
        "## Dataset Results",
        "",
    ]
    for dataset_key, dataset_result in (result.get("datasets") or {}).items():
        lines.extend(
            [
                f"### {dataset_key}",
                "",
                f"- Split: `{dataset_result.get('split')}`",
                f"- Samples: {dataset_result.get('samples')}",
                f"- Accuracy: {_metric(dataset_result.get('accuracy'))}",
                f"- Macro-F1: {_metric(dataset_result.get('macro_f1'))}",
                "",
                "| Class | Precision | Recall | F1 |",
                "|---|---:|---:|---:|",
            ]
        )
        for label, class_metrics in (dataset_result.get("per_class") or {}).items():
            lines.append(
                f"| {label} | {_metric(class_metrics.get('precision'))} | {_metric(class_metrics.get('recall'))} | {_metric(class_metrics.get('f1'))} |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _metric(value: Any, signed: bool = False) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    prefix = "+" if signed and value >= 0 else ""
    return f"{prefix}{value:.4f}"


def _require_training_packages() -> None:
    missing = []
    for package in ["torch", "transformers", "datasets", "peft", "trl", "bitsandbytes", "accelerate"]:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise RuntimeError(f"Missing WSL training packages: {', '.join(missing)}")


def _require_eval_packages() -> None:
    missing = []
    for package in ["torch", "transformers", "peft", "bitsandbytes", "accelerate"]:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    if missing:
        raise RuntimeError(f"Missing adapter eval packages: {', '.join(missing)}")
