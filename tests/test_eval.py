import json
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
            return self.chat_json_options(system, user)

        def chat_json_options(self, system: str, user: str, **kwargs) -> dict:
            if '"items"' in user:
                payload = json.loads(user)
                return {
                    "predictions": [
                        {"id": item["id"], "choice": self._choice(item["options"], self._label(system, item["text"]))}
                        for item in payload["items"]
                    ]
                }

            options_text = user.split("Options: ", 1)[1].split("\n", 1)[0]
            options = json.loads(options_text)
            return {"choice": self._choice(options, self._label(system, user))}

        @staticmethod
        def _label(system: str, text: str) -> str:
            lowered = text.lower()
            if "downgraded" in lowered:
                return "bearish"
            if "beats estimates" in lowered:
                return "bullish"
            if "profit rose" in lowered:
                return "positive"
            if "loss widened" in lowered:
                return "negative"
            return "neutral"

        @staticmethod
        def _choice(options: dict, label: str) -> str:
            for choice, option_label in options.items():
                if option_label == label:
                    return choice
            return "A"

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

    result = run_eval_suite("sentiment-baseline", results_dir=tmp_path, llm_provider="fake", limit=3, batch_size=3)

    assert result["status"] == "pass"
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["batch_size"] == 3
    assert result["datasets"]["financial_phrasebank"]["split"] == "test"
    assert result["datasets"]["twitter_financial_news_sentiment"]["split"] == "validation"
    first_prediction = result["datasets"]["financial_phrasebank"]["predictions"][0]
    assert set(first_prediction["label_options"]) == {"A", "B", "C"}
    assert first_prediction["predicted_choice"] in {"A", "B", "C"}
    assert result["leakage_check"]["status"] == "not_checked_no_train_manifest"
    assert Path(result["artifacts"]["manifest"]).exists()
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


def test_batch_prediction_parser_accepts_valid_labels_only():
    raw = {"predictions": [{"id": 1, "choice": "B"}, {"id": 2, "choice": "D"}]}

    assert suites._parse_batch_predictions(raw, {"A", "B", "C"}) == {1: "B"}


def test_stable_label_options_permute_labels_without_loss():
    labels = ["negative", "neutral", "positive"]

    options = suites._stable_label_options(labels, "sample-key")

    assert set(options) == {"A", "B", "C"}
    assert set(options.values()) == set(labels)


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
