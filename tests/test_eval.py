from pathlib import Path

from investment_research_desk.config import load_settings
from investment_research_desk.eval import run_eval_suite
from investment_research_desk.eval import suites


def test_schema_eval_writes_result(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path.cwd())
    result = run_eval_suite("schema", settings=load_settings(), results_dir=tmp_path)

    assert result["status"] == "pass"
    assert list(tmp_path.glob("*_schema.json"))
    assert list(tmp_path.glob("*_schema.md"))


def test_guardrail_eval_passes(tmp_path: Path):
    result = run_eval_suite("guardrail", results_dir=tmp_path)

    assert result["status"] == "pass"


def test_sentiment_baseline_uses_heldout_splits():
    assert suites.SENTIMENT_DATASETS["financial_phrasebank"]["split"] == "test"
    assert suites.SENTIMENT_DATASETS["twitter_financial_news_sentiment"]["split"] == "validation"


def test_sentiment_baseline_reports_accuracy_and_macro_f1(tmp_path: Path, monkeypatch):
    class RuleLLM:
        provider = "fake"
        model = "rule"
        calls: list[tuple[str, str]] = []

        def chat_json(self, system: str, user: str) -> dict:
            self.calls.append((system, user))
            lowered = user.lower()
            if "bearish" in system and "downgraded" in lowered:
                return {"label": "bearish"}
            if "bullish" in system and "beats estimates" in lowered:
                return {"label": "bullish"}
            if "positive" in system and "profit rose" in lowered:
                return {"label": "positive"}
            if "negative" in system and "loss widened" in lowered:
                return {"label": "negative"}
            return {"label": "neutral"}

        def healthcheck(self):
            return True, "ok"

    def fake_loader(dataset_key, spec, dataset_dir, limit):
        if dataset_key == "financial_phrasebank":
            return [
                {"text": "Company profit rose during the quarter.", "label": "positive"},
                {"text": "Company loss widened during the quarter.", "label": "negative"},
                {"text": "The company held its annual meeting.", "label": "neutral"},
            ]
        return [
            {"text": "$ABC downgraded by analysts", "label": "bearish"},
            {"text": "$XYZ beats estimates", "label": "bullish"},
            {"text": "$XYZ holds annual shareholder meeting", "label": "neutral"},
        ]

    rule_llm = RuleLLM()
    monkeypatch.setattr(suites, "make_llm_client", lambda *args, **kwargs: rule_llm)
    monkeypatch.setattr(suites, "_load_sentiment_dataset", fake_loader)

    result = run_eval_suite("sentiment-baseline", results_dir=tmp_path, llm_provider="fake", limit=3)

    assert result["status"] == "pass"
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["metric_backend"] == "huggingface-evaluate"
    assert result["inference_mode"] == "no_think"
    assert result["max_tokens"] == suites.SENTIMENT_EVAL_MAX_TOKENS
    assert result["output_contract"]["reasoning_effort"] == "none"
    assert result["output_contract"]["violations"]["thinking_output"] == 0
    assert rule_llm.calls
    assert all("/no_think" in system for system, _ in rule_llm.calls)
    assert all("Do not explain" not in system for system, _ in rule_llm.calls)
    assert result["datasets"]["financial_phrasebank"]["split"] == "test"
    assert result["datasets"]["twitter_financial_news_sentiment"]["split"] == "validation"
    assert result["leakage_check"]["status"] == "not_checked_no_train_manifest"
    assert Path(result["artifacts"]["manifest"]).exists()
    assert list(tmp_path.glob("*_sentiment-baseline.json"))


def test_sentiment_output_contract_detects_explanatory_and_thinking_output():
    violations = suites._sentiment_output_violations(
        {"label": "positive", "why": "because earnings rose"},
        '<think>hidden</think>\n{"label":"positive","why":"because earnings rose"}',
        "",
        ["negative", "neutral", "positive"],
    )

    assert "thinking_output" in violations
    assert "non_json_wrapper" in violations
    assert "extra_json_fields" in violations


def test_sentiment_limit_is_stratified():
    examples = [
        {"text": "neg 1", "label": "negative"},
        {"text": "neg 2", "label": "negative"},
        {"text": "neu 1", "label": "neutral"},
        {"text": "pos 1", "label": "positive"},
    ]

    selected = suites._stratified_limit(examples, ["negative", "neutral", "positive"], 3)

    assert [item["label"] for item in selected] == ["negative", "neutral", "positive"]


def test_leakage_check_detects_train_eval_overlap(tmp_path: Path):
    eval_entries = [
        {
            "dataset": "dataset/a",
            "config": "default",
            "split": "test",
            "row_idx": 1,
            "text_sha256": "same",
            "normalized_text_sha256": "same_norm",
        }
    ]
    train_manifest = tmp_path / "train_manifest.jsonl"
    train_manifest.write_text(
        '{"dataset":"dataset/a","config":"default","split":"test","row_idx":1,"text_sha256":"same","normalized_text_sha256":"same_norm"}\n',
        encoding="utf-8",
    )

    result = suites._leakage_check(eval_entries, train_manifest)

    assert result["status"] == "fail"
    assert result["split_overlaps"] == ["dataset/a::default::test"]
    assert result["text_hash_overlap_count"] == 1
    assert result["normalized_text_hash_overlap_count"] == 1
