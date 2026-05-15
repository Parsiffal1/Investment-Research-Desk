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

        def chat_json(self, system: str, user: str) -> dict:
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

    monkeypatch.setattr(suites, "make_llm_client", lambda *args, **kwargs: RuleLLM())
    monkeypatch.setattr(suites, "_load_sentiment_dataset", fake_loader)

    result = run_eval_suite("sentiment-baseline", results_dir=tmp_path, llm_provider="fake", limit=3)

    assert result["status"] == "pass"
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["datasets"]["financial_phrasebank"]["split"] == "test"
    assert result["datasets"]["twitter_financial_news_sentiment"]["split"] == "validation"
    assert list(tmp_path.glob("*_sentiment-baseline.json"))


def test_sentiment_limit_is_stratified():
    examples = [
        {"text": "neg 1", "label": "negative"},
        {"text": "neg 2", "label": "negative"},
        {"text": "neu 1", "label": "neutral"},
        {"text": "pos 1", "label": "positive"},
    ]

    selected = suites._stratified_limit(examples, ["negative", "neutral", "positive"], 3)

    assert [item["label"] for item in selected] == ["negative", "neutral", "positive"]
