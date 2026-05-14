from pathlib import Path

from investment_research_desk.config import load_settings
from investment_research_desk.graph import ResearchWorkflow
from investment_research_desk.schemas import FinalResearchContext, RunRequest


def test_fixture_workflow_creates_artifacts(tmp_path: Path):
    settings = load_settings()
    workflow = ResearchWorkflow(settings=settings, runs_dir=tmp_path)
    request = RunRequest(symbol="XAU-USDT-SWAP", asset_class="precious_metal", fixture="gold_cpi", llm_provider="fake")

    state = workflow.run(request, checkpoint=True)

    context = FinalResearchContext.model_validate(state["final_context"])
    assert context.symbol == "XAU-USDT-SWAP"
    assert context.key_drivers
    run_dir = tmp_path / state["run_id"]
    assert (run_dir / "final_research_context.json").exists()
    assert (run_dir / "research_brief.md").exists()
    assert (run_dir / "trace.json").exists()
    assert (run_dir / "metrics.json").exists()
    assert (run_dir / "checkpoint.json").exists()
    trace_names = [agent["name"] for agent in state["trace"]["agents"]]
    assert "fundamental_macro" in trace_names
    assert "news_impact" in trace_names
    assert "sentiment" in trace_names
    assert "technical" in trace_names
    assert "constructive_case" in trace_names
    assert "risk_case" in trace_names
    assert "research_reporter" in trace_names
    assert "analyst_layer" not in trace_names
    assert "research_layer" not in trace_names


def test_resume_from_checkpoint_completes(tmp_path: Path):
    settings = load_settings()
    workflow = ResearchWorkflow(settings=settings, runs_dir=tmp_path)
    request = RunRequest(symbol="XAU-USDT-SWAP", asset_class="precious_metal", fixture="gold_cpi", llm_provider="fake")

    first = workflow.run(request, checkpoint=True)
    resumed = workflow.run(request, checkpoint=True, resume_run_id=first["run_id"])

    assert resumed["run_id"] == first["run_id"]
    assert "persist" in resumed["completed_steps"]


def test_resume_from_mid_graph_checkpoint_continues_remaining_agents(tmp_path: Path):
    settings = load_settings()
    workflow = ResearchWorkflow(settings=settings, runs_dir=tmp_path)
    request = RunRequest(symbol="XAU-USDT-SWAP", asset_class="precious_metal", fixture="gold_cpi", llm_provider="fake")

    first = workflow.run(request, checkpoint=True)
    checkpoint = workflow.store.load_checkpoint(first["run_id"])
    kept_steps = [
        "run_controller",
        "data_ingestion",
        "fundamental_macro",
        "news_impact",
        "sentiment",
        "technical",
    ]
    checkpoint["completed_steps"] = kept_steps
    checkpoint["trace"]["completed_steps"] = kept_steps
    checkpoint["trace"]["agents"] = [agent for agent in checkpoint["trace"]["agents"] if agent["name"] in kept_steps]
    for key in ["constructive", "risk", "final_context", "metrics", "output_paths"]:
        checkpoint.pop(key, None)
    workflow.store.save_checkpoint(first["run_id"], checkpoint)

    resumed = workflow.run(request, checkpoint=True, resume_run_id=first["run_id"])

    assert resumed["completed_steps"][-4:] == ["constructive_case", "risk_case", "research_reporter", "persist"]
    assert (tmp_path / first["run_id"] / "final_research_context.json").exists()
