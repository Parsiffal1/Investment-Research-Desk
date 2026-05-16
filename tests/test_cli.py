import json

from typer.testing import CliRunner

from investment_research_desk.cli import app
from investment_research_desk.cli import _format_eval_value
from investment_research_desk.cli_contract import build_run_request


runner = CliRunner()


def test_cli_help():
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "Investment Research Desk" in result.output


def test_cli_report_fixture(tmp_path):
    result = runner.invoke(
        app,
        [
            "report",
            "--fixture",
            "gold_cpi",
            "--llm-provider",
            "fake",
            "--runs-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert "Research Context" in result.output
    assert list(tmp_path.glob("*/final_research_context.json"))
    assert list(tmp_path.glob("*/final_market_context_cache.json"))


def test_cli_report_fixture_accepts_chinese_language(tmp_path):
    result = runner.invoke(
        app,
        [
            "report",
            "--fixture",
            "gold_cpi",
            "--llm-provider",
            "fake",
            "--language",
            "zh",
            "--runs-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    final_context = next(tmp_path.glob("*/final_research_context.json"))
    payload = json.loads(final_context.read_text(encoding="utf-8"))
    assert payload["source_metadata"]["language"] == "zh"


def test_cli_demo_uses_fixture_fake_path(tmp_path):
    result = runner.invoke(app, ["demo", "--runs-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "Research Context" in result.output
    assert list(tmp_path.glob("*/final_research_context.json"))


def test_cli_report_rejects_missing_symbol_for_live_run(tmp_path):
    result = runner.invoke(
        app,
        [
            "report",
            "--llm-provider",
            "fake",
            "--runs-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 2
    assert "--symbol is required" in result.output


def test_interactive_style_symbol_normalization_and_asset_inference():
    equity = build_run_request(
        symbol=" nvda ",
        asset_class=None,
        horizon="short_term",
        research_depth="standard",
        fixture=None,
        llm_provider="ollama",
        model="qwen3:8b",
    )
    crypto = build_run_request(
        symbol=" btc-usdt-swap ",
        asset_class="auto",
        horizon="short_term",
        research_depth="standard",
        fixture=None,
        llm_provider="ollama",
        model="qwen3:8b",
    )

    assert equity.symbol == "NVDA"
    assert equity.asset_class == "equity"
    assert crypto.symbol == "BTC-USDT-SWAP"
    assert crypto.asset_class == "crypto"


def test_build_run_request_accepts_sentiment_adapter_contract(tmp_path):
    request = build_run_request(
        symbol="eth-usdt-swap",
        asset_class="auto",
        horizon="short_term",
        research_depth="standard",
        fixture=None,
        llm_provider="ollama",
        model="qwen3:8b",
        sentiment_provider="hf-peft",
        sentiment_base_model="Qwen/Qwen3-8B",
        sentiment_adapter_path=tmp_path / "adapter",
        sentiment_score_batch_size=2,
    )

    assert request.sentiment_provider == "hf-peft"
    assert request.sentiment_base_model == "Qwen/Qwen3-8B"
    assert request.sentiment_adapter_path == str(tmp_path / "adapter")
    assert request.sentiment_score_batch_size == 2


def test_build_run_request_rejects_invalid_report_language():
    try:
        build_run_request(
            symbol="ETH-USDT-SWAP",
            asset_class="auto",
            horizon="short_term",
            research_depth="standard",
            fixture=None,
            llm_provider="ollama",
            model="qwen3:8b",
            language="fr",
        )
    except ValueError as exc:
        assert "language must be one of: en, zh" in str(exc)
    else:
        raise AssertionError("invalid report language should be rejected")


def test_cli_runs_lists_checkpoint(tmp_path):
    report_result = runner.invoke(
        app,
        [
            "report",
            "--fixture",
            "gold_cpi",
            "--llm-provider",
            "fake",
            "--checkpoint",
            "--runs-dir",
            str(tmp_path),
        ],
    )
    assert report_result.exit_code == 0

    runs_result = runner.invoke(app, ["runs", "--runs-dir", str(tmp_path), "--resumable-only"])

    assert runs_result.exit_code == 0
    assert "ird report --resume" in runs_result.output


def test_cli_eval_guardrail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["eval", "--suite", "guardrail"])

    assert result.exit_code == 0
    assert "guardrail" in result.output


def test_eval_console_formatter_summarizes_large_dataset_payload():
    value = {
        "financial_phrasebank": {
            "samples": 300,
            "accuracy": 0.8066666666666666,
            "macro_f1": 0.8243142589928828,
            "predictions": [{"text": "non-ascii \u01c6 text"}],
        }
    }

    formatted = _format_eval_value("datasets", value)

    assert "financial_phrasebank: samples=300" in formatted
    assert "0.8067" in formatted
    assert "predictions" not in formatted


def test_cli_lora_dry_run_commands(tmp_path, monkeypatch):
    calls = {}

    monkeypatch.setattr(
        "investment_research_desk.cli.prepare_lora_data",
        lambda **kwargs: {"status": "dry_run", "train_samples": 10, "eval_samples": 4},
    )
    def fake_train(**kwargs):
        calls["train"] = kwargs
        return {"status": "dry_run", "training_config": {"base_model": "Qwen/Qwen3-8B"}}

    def fake_eval(**kwargs):
        calls["eval"] = kwargs
        return {"status": "dry_run", "baseline": {"accuracy": 0.79}}

    monkeypatch.setattr("investment_research_desk.cli.train_lora_sentiment", fake_train)
    monkeypatch.setattr("investment_research_desk.cli.eval_lora_sentiment", fake_eval)

    prepare = runner.invoke(app, ["lora", "prepare-data", "--output-dir", str(tmp_path), "--dry-run"])
    train = runner.invoke(app, ["lora", "train", "--data-dir", str(tmp_path), "--output-root", str(tmp_path / "models"), "--epochs", "1", "--dry-run"])
    evaluate = runner.invoke(
        app,
        ["lora", "eval", "--adapter-path", str(tmp_path / "adapter"), "--contract-limit", "2", "--score-batch-size", "3", "--dry-run"],
    )

    assert prepare.exit_code == 0
    assert train.exit_code == 0
    assert evaluate.exit_code == 0
    assert "dry_run" in prepare.output
    assert "Qwen/Qwen3-8B" in train.output
    assert calls["train"]["config"].num_train_epochs == 1
    assert calls["eval"]["contract_limit"] == 2
    assert calls["eval"]["score_batch_size"] == 3
