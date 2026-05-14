from typer.testing import CliRunner

from investment_research_desk.cli import app


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


def test_cli_eval_guardrail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    result = runner.invoke(app, ["eval", "--suite", "guardrail"])

    assert result.exit_code == 0
    assert "guardrail" in result.output

