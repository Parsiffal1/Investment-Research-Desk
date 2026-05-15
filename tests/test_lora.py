import json
from pathlib import Path

from investment_research_desk.lora import LoraTrainingConfig, prepare_lora_data, train_lora_sentiment
from investment_research_desk.lora import sentiment


def test_lora_prepare_data_keeps_heldout_splits_out_of_train(tmp_path: Path, monkeypatch):
    def fake_load_examples(dataset_key, spec, dataset_dir, limit):
        return [
            {"row_idx": index, "text": f"{dataset_key} {spec['split']} text {index}", "label": spec["labels"][index % len(spec["labels"])]}
            for index in range(12)
        ]

    monkeypatch.setattr(sentiment, "_load_examples", fake_load_examples)

    result = prepare_lora_data(tmp_path, dry_run=False)

    assert result["train_samples"] > 0
    assert result["eval_samples"] > 0
    assert result["leakage_check"]["status"] == "pass"
    assert result["train_eval_overlap"]["row_overlap_count"] == 0
    assert result["train_eval_overlap"]["normalized_text_hash_overlap_count"] == 0
    train_manifest = [
        json.loads(line)
        for line in (tmp_path / "train_manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert any("FinancialPhraseBank" in row["dataset"] for row in train_manifest)
    assert not any(row["dataset_key"] == "financial_phrasebank" and row["split"] == "test" for row in train_manifest)
    assert not any(row["dataset_key"] == "twitter_financial_news_sentiment" and row["split"] == "validation" for row in train_manifest)


def test_lora_sft_example_target_is_json_label_only():
    rows = sentiment._format_sft_examples(
        "financial_phrasebank",
        ["negative", "neutral", "positive"],
        [{"row_idx": 1, "text": "Profit rose.", "label": "positive"}],
    )

    assistant = rows[0]["messages"][-1]

    assert assistant == {"role": "assistant", "content": '{"label": "positive"}'}
    assert "<think" not in rows[0]["text"].lower()
    assert "because" not in assistant["content"].lower()


def test_lora_training_config_serializes_for_dry_run(tmp_path: Path):
    config = LoraTrainingConfig()

    result = train_lora_sentiment(tmp_path, tmp_path / "models", config, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["training_config"]["base_model"] == "Qwen/Qwen3-8B"
    assert result["training_config"]["lora_r"] == 16
    assert result["training_config"]["gradient_accumulation_steps"] == 16


def test_lora_eval_markdown_contains_metrics():
    report = sentiment._lora_eval_markdown(
        {
            "adapter_path": "models/example/adapter",
            "accuracy": 0.81,
            "macro_f1": 0.79,
            "baseline_delta": {"accuracy": 0.02, "macro_f1": -0.01},
            "output_contract": {"status": "pass"},
            "datasets": {
                "financial_phrasebank": {
                    "split": "test",
                    "samples": 3,
                    "accuracy": 1.0,
                    "macro_f1": 1.0,
                    "per_class": {"positive": {"precision": 1.0, "recall": 1.0, "f1": 1.0}},
                }
            },
        }
    )

    assert "Accuracy: 0.8100" in report
    assert "Baseline Macro-F1 delta: -0.0100" in report
    assert "| positive | 1.0000 | 1.0000 | 1.0000 |" in report
