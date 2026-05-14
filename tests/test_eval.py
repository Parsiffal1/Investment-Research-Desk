from pathlib import Path

from investment_research_desk.config import load_settings
from investment_research_desk.eval import run_eval_suite


def test_schema_eval_writes_result(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(Path.cwd())
    result = run_eval_suite("schema", settings=load_settings(), results_dir=tmp_path)

    assert result["status"] == "pass"
    assert list(tmp_path.glob("*_schema.json"))
    assert list(tmp_path.glob("*_schema.md"))


def test_guardrail_eval_passes(tmp_path: Path):
    result = run_eval_suite("guardrail", results_dir=tmp_path)

    assert result["status"] == "pass"

