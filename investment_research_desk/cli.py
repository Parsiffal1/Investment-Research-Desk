from __future__ import annotations

import json
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import httpx
import questionary
import typer
from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

from investment_research_desk import __version__
from investment_research_desk.cli_contract import (
    ALLOWED_LLM_PROVIDERS,
    AssetClassOption,
    CLIInteractionContract,
    HorizonOption,
    LLMProviderOption,
    REQUIRED_ARTIFACTS,
    ReportLanguageOption,
    ResearchDepthOption,
    SentimentProviderOption,
    TEAM_FLOW,
    build_run_request,
    discover_runs,
)
from investment_research_desk.config import load_settings
from investment_research_desk.eval import run_eval_suite
from investment_research_desk.graph import ResearchWorkflow
from investment_research_desk.llm import OllamaLLMClient
from investment_research_desk.lora import LoraTrainingConfig, eval_lora_sentiment, prepare_lora_data, train_lora_sentiment
from investment_research_desk.persistence import RunStore
from investment_research_desk.providers.okx import OkxMarketDataProvider
from investment_research_desk.schemas import FinalResearchContext
from investment_research_desk.sentiment_runtime import discover_latest_adapter, missing_runtime_packages

console = Console()
app = typer.Typer(
    name="ird",
    help="Investment Research Desk / 投研策略台 CLI",
    invoke_without_command=True,
    no_args_is_help=False,
)
config_app = typer.Typer(help="Configuration and runtime checks")
app.add_typer(config_app, name="config")
okx_app = typer.Typer(help="OKX public SWAP market checks")
app.add_typer(okx_app, name="okx")
lora_app = typer.Typer(help="LoRA sentiment fine-tuning workflows")
app.add_typer(lora_app, name="lora")


CLI_STYLE = questionary.Style(
    [
        ("qmark", "fg:#6ee7b7 bold"),
        ("question", "fg:#e5e7eb bold"),
        ("answer", "fg:#facc15 bold"),
        ("pointer", "fg:#a78bfa bold"),
        ("highlighted", "fg:#a78bfa bold"),
        ("selected", "fg:#6ee7b7"),
        ("separator", "fg:#64748b"),
        ("instruction", "fg:#94a3b8"),
        ("text", "fg:#e5e7eb"),
    ]
)

ASCII_LOGO = r"""
    ____                 __                      __     ____                 __
   /  _/___ _   _____  / /____  _________ ___  / /_   / __ \___  _________/ /__
   / // __ \ | / / _ \/ __/ _ \/ ___/ __ `__ \/ __/  / / / / _ \/ ___/ __  / _ \
 _/ // / / / |/ /  __/ /_/  __(__  ) / / / / / /_   / /_/ /  __(__  ) /_/ /  __/
/___/_/ /_/|___/\___/\__/\___/____/_/ /_/ /_/\__/  /_____/\___/____/\__,_/\___/
"""

AGENT_TEAMS = {
    "Run Control": ["run_controller"],
    "Analyst Team": ["fundamental_macro", "news_impact", "sentiment", "technical"],
    "Bull/Bear Research Debate": ["bull_researcher", "bear_researcher", "bull_bear_research_debate"],
    "Research Reporter": ["research_reporter", "final_market_context_cache", "persist"],
}

AGENT_LABELS = {
    "run_controller": "Run Controller",
    "fundamental_macro": "Fundamental / Macro Analyst",
    "news_impact": "News / Macro Impact Analyst",
    "sentiment": "Sentiment Analyst",
    "technical": "Technical Analyst",
    "analyst_team": "Analyst Team Synthesis",
    "bull_researcher": "Bull Researcher",
    "bear_researcher": "Bear Researcher",
    "bull_bear_research_debate": "Research Debate",
    "research_reporter": "Research Reporter",
    "final_market_context_cache": "Market Context Cache",
    "persist": "Persist Artifacts",
}


class CLIRunDashboard:
    def __init__(self, request) -> None:
        self.request = request
        self.started = time.perf_counter()
        self.statuses = {agent: "pending" for agents in AGENT_TEAMS.values() for agent in agents}
        self.messages: deque[tuple[str, str, str]] = deque(maxlen=80)
        self.tool_calls = 0
        self.llm_reports = 0
        self.current_report = "Waiting for analysis report..."

    def add_message(self, message_type: str, content: str) -> None:
        self.messages.append((datetime.now().strftime("%H:%M:%S"), message_type, content))

    def handle_event(self, event: dict[str, Any]) -> None:
        name = event["name"]
        status = event["status"]
        payload = event.get("payload") or {}
        if name in self.statuses:
            self.statuses[name] = "completed" if status == "completed" else status
        if event["type"] == "agent_status":
            self.add_message("System", f"{AGENT_LABELS.get(name, name)} {status}")
            return
        if event["type"] == "agent_result":
            self.add_message("Reasoning", f"{AGENT_LABELS.get(name, name)} completed")
            if name in {"fundamental_macro", "news_impact", "sentiment", "technical"}:
                self.llm_reports += 1
                self.current_report = _markdown_for_agent_result(name, payload.get("output") or {})
                self._count_tool_calls(payload.get("data") or {})
            elif event.get("state"):
                self.current_report = _markdown_for_state_progress(name, event["state"])

    def _count_tool_calls(self, data: dict[str, Any]) -> None:
        metadata = data.get("source_metadata") if isinstance(data, dict) else {}
        statuses = metadata.get("agent_tool_status") if isinstance(metadata, dict) else {}
        self.tool_calls += _count_status_entries(statuses)

    def render(self) -> Layout:
        layout = Layout()
        layout.split_column(Layout(name="header", size=5), Layout(name="main"), Layout(name="footer", size=3))
        layout["main"].split_column(Layout(name="upper", ratio=2), Layout(name="report", ratio=3))
        layout["upper"].split_row(Layout(name="progress", ratio=2), Layout(name="messages", ratio=3))
        layout["header"].update(_runtime_header(self.request))
        layout["progress"].update(_runtime_progress_panel(self.statuses))
        layout["messages"].update(_runtime_messages_panel(self.messages))
        layout["report"].update(Panel(_plain_report_text(self.current_report), title="Current Report", border_style="green"))
        elapsed = time.perf_counter() - self.started
        footer = f"Tool Calls: {self.tool_calls} | Generated Reports: {self.llm_reports} | Elapsed: {int(elapsed // 60):02d}:{int(elapsed % 60):02d}"
        layout["footer"].update(Panel(Align.center(footer), border_style="grey50"))
        return layout


@app.callback(invoke_without_command=True)
def root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", help="Show version and exit."),
) -> None:
    if version:
        console.print(f"investment-research-desk {__version__}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        interactive()


def interactive() -> None:
    settings = load_settings()
    contract = _collect_interactive_contract()
    if contract.mode == "config_check":
        config_check()
        return
    if contract.mode == "list_runs":
        list_runs(runs_dir=contract.runs_dir, resumable_only=False)
        return
    if contract.mode == "exit":
        console.print("[dim]Exited without running research.[/dim]")
        return
    if contract.mode == "resume":
        run_report(
            symbol=None,
            asset_class=AssetClassOption.crypto,
            horizon=HorizonOption.short_term,
            research_depth=ResearchDepthOption.standard,
            fixture=None,
            llm_provider=LLMProviderOption.auto,
            model=None,
            checkpoint=contract.checkpoint,
            resume=contract.resume_run_id,
            clear_checkpoints=contract.clear_checkpoints,
            runs_dir=contract.runs_dir,
            sentiment_provider=None,
            sentiment_base_model=None,
            sentiment_adapter_path=None,
            sentiment_score_batch_size=None,
            language=settings.report_language,
        )
        return
    if contract.request is None:
        _exit_with_error("No run request was created by the interactive contract.")
    _run_workflow(contract.request, checkpoint=contract.checkpoint, resume=None, clear_checkpoints=False, runs_dir=contract.runs_dir)


def _collect_interactive_contract() -> CLIInteractionContract:
    settings = load_settings()
    console.print(_welcome_panel())
    console.print(_workflow_panel())

    console.print(_step_panel(1, "Action", "Choose the next Investment Research Desk workflow."))
    action = _select(
        "Select action",
        [
            questionary.Choice("New research report", "new_report"),
            questionary.Choice("Resume previous run", "resume"),
            questionary.Choice("View run history", "list_runs"),
            questionary.Choice("System check", "config_check"),
            questionary.Choice("Exit", "exit"),
        ],
    )
    runs_dir = settings.runs_dir

    if action in {"config_check", "list_runs", "exit"}:
        return CLIInteractionContract(mode=action, request=None, checkpoint=False, resume_run_id=None, runs_dir=runs_dir)

    if action == "resume":
        console.print(_step_panel(2, "Checkpoint", "Select a resumable run and continue from the last saved graph step."))
        run_id = _select_resume_run(runs_dir)
        return CLIInteractionContract(mode="resume", request=None, checkpoint=True, resume_run_id=run_id, runs_dir=runs_dir)

    symbol: str | None = None
    horizon: str | HorizonOption = HorizonOption.short_term

    console.print(_step_panel(2, "Instrument", "Enter the exact symbol to analyze. Examples: NVDA, AAPL, BTC-USDT-SWAP.", "BTC-USDT-SWAP"))
    symbol = questionary.text(
        "Symbol",
        default="BTC-USDT-SWAP",
        validate=lambda value: bool(value.strip()) or "Symbol is required.",
        style=CLI_STYLE,
    ).ask()
    console.print(_step_panel(3, "Research Horizon", "Select the time horizon for analysis framing and prompt context.", HorizonOption.short_term.value))
    horizon = _enum_select("Horizon", HorizonOption, HorizonOption.short_term)
    console.print(_step_panel(4, "Research Depth", "Select how much reasoning/debate depth to request from the workflow.", ResearchDepthOption.standard.value))
    research_depth = _enum_select("Research depth", ResearchDepthOption, ResearchDepthOption.standard)
    default_language = _default_report_language_option(settings.report_language)
    console.print(_step_panel(5, "Report Language", "Select the language used in the console report and Markdown brief.", default_language.value))
    language = _enum_select("Report language", ReportLanguageOption, default_language)

    try:
        request = build_run_request(
            symbol=symbol,
            asset_class="auto",
            horizon=horizon,
            research_depth=research_depth,
            fixture=None,
            llm_provider=LLMProviderOption.ollama,
            model=settings.ollama_model,
            sentiment_provider=None,
            sentiment_base_model=None,
            sentiment_adapter_path=None,
            sentiment_score_batch_size=None,
            language=language,
        )
    except ValueError as exc:
        _exit_with_error(str(exc))

    _print_request_review(request, checkpoint=True, runs_dir=runs_dir, mode="live")
    if not _confirm("Start this research run?", default=True):
        raise typer.Exit()
    return CLIInteractionContract(
        mode="new_report",
        request=request,
        checkpoint=True,
        resume_run_id=None,
        runs_dir=runs_dir,
    )


@app.command()
def demo(
    fixture: str = typer.Option("gold_cpi", "--fixture", help="Fixture scenario for local demo."),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir", help="Override runs output directory."),
) -> None:
    run_report(
        symbol=None,
        asset_class=AssetClassOption.crypto,
        horizon=HorizonOption.short_term,
        research_depth=ResearchDepthOption.standard,
        fixture=fixture,
        llm_provider=LLMProviderOption.fake,
        model=None,
        checkpoint=False,
        resume=None,
        clear_checkpoints=False,
        runs_dir=runs_dir,
        sentiment_provider=None,
        sentiment_base_model=None,
        sentiment_adapter_path=None,
        sentiment_score_batch_size=None,
    )


@app.command()
def report(
    symbol: Optional[str] = typer.Option(None, "--symbol", help="Instrument symbol, e.g. BTC-USDT-SWAP."),
    asset_class: Optional[AssetClassOption] = typer.Option(None, "--asset-class", help="Optional provider route override."),
    horizon: HorizonOption = typer.Option(HorizonOption.short_term, "--horizon", help="Research horizon."),
    research_depth: ResearchDepthOption = typer.Option(ResearchDepthOption.standard, "--research-depth", help="quick, standard, or deep."),
    fixture: Optional[str] = typer.Option(None, "--fixture", help="Run from frozen fixture data."),
    llm_provider: LLMProviderOption = typer.Option(LLMProviderOption.auto, "--llm-provider", help="auto, fake, or ollama."),
    model: Optional[str] = typer.Option(None, "--model", help="Override model name."),
    sentiment_provider: Optional[SentimentProviderOption] = typer.Option(None, "--sentiment-provider", help="main, fake, or hf-peft."),
    sentiment_base_model: Optional[str] = typer.Option(None, "--sentiment-base-model", help="HF base model for sentiment adapter runtime."),
    sentiment_adapter_path: Optional[Path] = typer.Option(None, "--sentiment-adapter-path", help="PEFT adapter path for Sentiment Analyst."),
    sentiment_score_batch_size: Optional[int] = typer.Option(None, "--sentiment-score-batch-size", min=1, max=16, help="Forced-choice scoring batch size."),
    language: Optional[str] = typer.Option(None, "--language", help="Report language: en or zh. Defaults to IRD_REPORT_LANGUAGE."),
    checkpoint: bool = typer.Option(False, "--checkpoint", help="Save resumable checkpoints after graph steps."),
    resume: Optional[str] = typer.Option(None, "--resume", help="Resume from checkpoint by run_id."),
    clear_checkpoints: bool = typer.Option(False, "--clear-checkpoints", help="Clear saved checkpoints before running."),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir", help="Override runs output directory."),
) -> None:
    run_report(
        symbol=symbol,
        asset_class=asset_class,
        horizon=horizon,
        research_depth=research_depth,
        fixture=fixture,
        llm_provider=llm_provider,
        model=model,
        sentiment_provider=sentiment_provider,
        sentiment_base_model=sentiment_base_model,
        sentiment_adapter_path=sentiment_adapter_path,
        sentiment_score_batch_size=sentiment_score_batch_size,
        language=language,
        checkpoint=checkpoint,
        resume=resume,
        clear_checkpoints=clear_checkpoints,
        runs_dir=runs_dir,
    )


def run_report(
    symbol: str | None,
    asset_class: str | AssetClassOption | None,
    horizon: str | HorizonOption,
    research_depth: str | ResearchDepthOption,
    fixture: str | None,
    llm_provider: str | LLMProviderOption,
    model: str | None,
    sentiment_provider: str | SentimentProviderOption | None,
    sentiment_base_model: str | None,
    sentiment_adapter_path: Path | str | None,
    sentiment_score_batch_size: int | None,
    checkpoint: bool,
    resume: str | None,
    clear_checkpoints: bool,
    runs_dir: Path | None,
    language: str | None = None,
) -> None:
    if resume:
        _run_resume(resume, checkpoint=checkpoint, clear_checkpoints=clear_checkpoints, runs_dir=runs_dir)
        return
    try:
        request = build_run_request(
            symbol=symbol,
            asset_class=asset_class,
            horizon=horizon,
            research_depth=research_depth,
            fixture=fixture,
            llm_provider=llm_provider,
            model=model,
            sentiment_provider=sentiment_provider,
            sentiment_base_model=sentiment_base_model,
            sentiment_adapter_path=sentiment_adapter_path,
            sentiment_score_batch_size=sentiment_score_batch_size,
            language=language or load_settings().report_language,
        )
    except ValueError as exc:
        _exit_with_error(str(exc))
    _run_workflow(request, checkpoint=checkpoint, resume=None, clear_checkpoints=clear_checkpoints, runs_dir=runs_dir)


def _run_resume(resume_run_id: str, checkpoint: bool, clear_checkpoints: bool, runs_dir: Path | None) -> None:
    settings = load_settings()
    store = RunStore(runs_dir or settings.runs_dir)
    if clear_checkpoints:
        _clear_checkpoints(runs_dir)
    checkpoint_path = store.run_dir(resume_run_id) / "checkpoint.json"
    if not checkpoint_path.exists():
        _exit_with_error(
            f"No checkpoint found for run_id={resume_run_id}",
            hints=[
                f"Run `ird runs --runs-dir {store.base_dir}` to list available runs.",
                "Use `ird report --checkpoint ...` to create resumable checkpoints.",
            ],
        )
    placeholder = build_run_request(
        symbol="RESUME",
        asset_class=AssetClassOption.crypto,
        horizon=HorizonOption.short_term,
        research_depth=ResearchDepthOption.standard,
        fixture=None,
        llm_provider=LLMProviderOption.auto,
        model=None,
    )
    _run_workflow(placeholder, checkpoint=checkpoint, resume=resume_run_id, clear_checkpoints=False, runs_dir=runs_dir)


def _run_workflow(request, checkpoint: bool, resume: str | None, clear_checkpoints: bool, runs_dir: Path | None) -> None:
    settings = load_settings()
    store = RunStore(runs_dir or settings.runs_dir)
    if clear_checkpoints:
        cleared = store.clear_checkpoints()
        console.print(f"[yellow]Cleared {cleared} checkpoint file(s).[/yellow]")
    if resume is None:
        _preflight_llm(request.llm_provider, request.model, bool(request.fixture), settings)
        _preflight_sentiment_runtime(request, settings)
        _print_request_review(request, checkpoint=checkpoint, runs_dir=store.base_dir, mode="fixture" if request.fixture else "live")

    dashboard = CLIRunDashboard(request)
    dashboard.add_message("System", f"Selected symbol: {request.symbol}")
    dashboard.add_message("System", f"Asset class: {request.asset_class}; horizon: {request.horizon}")
    dashboard.add_message("System", f"Research depth: {request.research_depth}; provider: {request.llm_provider}")
    live_holder: dict[str, Live] = {}

    def progress_callback(event: dict[str, Any]) -> None:
        dashboard.handle_event(event)
        live = live_holder.get("live")
        if live is not None:
            live.update(dashboard.render())

    workflow = ResearchWorkflow(settings=settings, runs_dir=runs_dir, progress_callback=progress_callback)
    console.print(_execution_contract_panel())
    try:
        with Live(dashboard.render(), console=console, refresh_per_second=4, transient=False) as live:
            live_holder["live"] = live
            state = workflow.run(request, checkpoint=checkpoint, resume_run_id=resume)
            live.update(dashboard.render())
    except Exception as exc:
        _exit_with_error(
            "Research workflow failed.",
            hints=[
                str(exc),
                "For Ollama, verify `ollama serve`, `ollama list`, and IRD_OLLAMA_BASE_URL=http://localhost:11434/v1.",
                "For live providers, run `ird config check` and confirm .env API keys.",
            ],
        )
    _print_run_summary(state)


@app.command()
def batch(
    symbols: str = typer.Option(..., "--symbols", help="Comma-separated instrument symbols."),
    asset_class: Optional[AssetClassOption] = typer.Option(None, "--asset-class", help="Optional provider route override."),
    horizon: HorizonOption = typer.Option(HorizonOption.short_term, "--horizon"),
    research_depth: ResearchDepthOption = typer.Option(ResearchDepthOption.standard, "--research-depth"),
    llm_provider: LLMProviderOption = typer.Option(LLMProviderOption.auto, "--llm-provider"),
    model: Optional[str] = typer.Option(None, "--model"),
    checkpoint: bool = typer.Option(False, "--checkpoint", help="Save checkpoints for each batch item."),
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir", help="Override runs output directory."),
) -> None:
    parsed_symbols = [item.strip().upper() for item in symbols.split(",") if item.strip()]
    if not parsed_symbols:
        _exit_with_error("--symbols must contain at least one symbol.")
    for symbol in parsed_symbols:
        console.rule(f"Running {symbol}")
        run_report(
            symbol=symbol,
            asset_class=asset_class,
            horizon=horizon,
            research_depth=research_depth,
            fixture=None,
            llm_provider=llm_provider,
            model=model,
            sentiment_provider=None,
            sentiment_base_model=None,
            sentiment_adapter_path=None,
            sentiment_score_batch_size=None,
            checkpoint=checkpoint,
            resume=None,
            clear_checkpoints=False,
            runs_dir=runs_dir,
        )


@app.command("runs")
def list_runs(
    runs_dir: Optional[Path] = typer.Option(None, "--runs-dir", help="Override runs output directory."),
    resumable_only: bool = typer.Option(False, "--resumable-only", help="Show only runs with checkpoint.json."),
) -> None:
    settings = load_settings()
    base_dir = runs_dir or settings.runs_dir
    rows = discover_runs(base_dir)
    if resumable_only:
        rows = [row for row in rows if row["status"] == "checkpoint"]
    table = Table(title=f"Runs: {base_dir}")
    table.add_column("Run ID")
    table.add_column("Status")
    table.add_column("Resume Command")
    for row in rows:
        resume_command = f"ird report --resume {row['run_id']} --runs-dir {base_dir}" if row["status"] == "checkpoint" else ""
        table.add_row(row["run_id"], row["status"], resume_command)
    if not rows:
        table.add_row("-", "none", "No run directories found.")
    console.print(table)


@app.command(name="eval")
def eval_command(
    suite: str = typer.Option(..., "--suite", help="schema, guardrail, single-vs-multi, consistency, compression, latency, lora, or sentiment-baseline."),
    llm_provider: LLMProviderOption = typer.Option(LLMProviderOption.ollama, "--llm-provider", help="Provider for model-based eval suites."),
    model: Optional[str] = typer.Option(None, "--model", help="Override model for model-based eval suites."),
    limit: Optional[int] = typer.Option(100, "--limit", help="Limit examples per sentiment dataset. Use 0 for the full held-out split."),
    dataset_dir: Optional[Path] = typer.Option(None, "--dataset-dir", help="Directory for cached held-out sentiment datasets."),
    train_manifest: Optional[Path] = typer.Option(None, "--train-manifest", help="Optional LoRA/SFT train manifest for leakage checks."),
    results_dir: Optional[Path] = typer.Option(None, "--results-dir", help="Override eval results directory."),
) -> None:
    result = run_eval_suite(
        suite,
        llm_provider=llm_provider.value,
        model=model,
        limit=limit,
        dataset_dir=dataset_dir,
        train_manifest=train_manifest,
        results_dir=results_dir,
    )  # type: ignore[arg-type]
    table = Table(title=f"Evaluation: {suite}")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in result.items():
        table.add_row(str(key), _format_eval_value(key, value))
    console.print(table)


def _format_eval_value(key: str, value: Any) -> str:
    if key == "datasets" and isinstance(value, dict):
        lines = []
        for name, dataset in value.items():
            if not isinstance(dataset, dict):
                continue
            lines.append(
                (
                    f"{name}: samples={dataset.get('samples')}, "
                    f"accuracy={_metric_float(dataset.get('accuracy'))}, "
                    f"macro_f1={_metric_float(dataset.get('macro_f1'))}"
                )
            )
        return "\n".join(lines)
    if key == "leakage_check" and isinstance(value, dict):
        return (
            f"status={value.get('status')}, "
            f"eval_samples={value.get('eval_samples')}, "
            f"split_overlaps={len(value.get('split_overlaps') or [])}"
        )
    if key == "output_contract" and isinstance(value, dict):
        violations = value.get("violations") if isinstance(value.get("violations"), dict) else {}
        violation_text = ", ".join(f"{name}={count}" for name, count in violations.items())
        return (
            f"reasoning_effort={value.get('reasoning_effort')}, "
            f"max_tokens={value.get('max_tokens')}, "
            f"samples_checked={value.get('samples_checked')}, "
            f"violations: {violation_text}"
        )
    if key == "artifacts" and isinstance(value, dict):
        return "\n".join(f"{name}: {path}" for name, path in value.items())
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=True, default=str)
    return str(value)


def _metric_float(value: Any) -> str:
    if isinstance(value, (float, int)):
        return f"{value:.4f}"
    return str(value)


@lora_app.command("prepare-data")
def lora_prepare_data(
    output_dir: Path = typer.Option(Path("lora_data/sentiment"), "--output-dir", help="Directory for LoRA JSONL and manifest files."),
    dataset_dir: Optional[Path] = typer.Option(None, "--dataset-dir", help="Directory for cached Hugging Face rows."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Optional per-split sample limit for local smoke tests."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Inspect split counts without writing files."),
) -> None:
    result = prepare_lora_data(output_dir=output_dir, dataset_dir=dataset_dir, limit=limit, dry_run=dry_run)
    _print_lora_result("LoRA Data Preparation", result)


@lora_app.command("train")
def lora_train(
    data_dir: Path = typer.Option(Path("lora_data/sentiment"), "--data-dir", help="Directory created by `ird lora prepare-data`."),
    output_root: Path = typer.Option(Path("models/investment-research-desk-lora-sentiment"), "--output-root", help="Root directory for timestamped adapter output."),
    epochs: float = typer.Option(2.0, "--epochs", min=0.1, help="Number of training epochs."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Validate config and show required packages without training."),
) -> None:
    config = LoraTrainingConfig(output_root=str(output_root), num_train_epochs=epochs)
    result = train_lora_sentiment(data_dir=data_dir, output_root=output_root, config=config, dry_run=dry_run)
    _print_lora_result("LoRA Training", result)


@lora_app.command("eval")
def lora_eval(
    adapter_path: Path = typer.Option(..., "--adapter-path", help="Path to a saved PEFT adapter directory."),
    output_dir: Optional[Path] = typer.Option(None, "--output-dir", help="Directory for heldout_eval_results.json."),
    dataset_dir: Optional[Path] = typer.Option(None, "--dataset-dir", help="Directory for cached Hugging Face rows."),
    limit: Optional[int] = typer.Option(None, "--limit", help="Optional per-dataset eval limit for smoke tests."),
    contract_limit: int = typer.Option(6, "--contract-limit", min=0, help="Per-dataset generative JSON contract-check sample size."),
    score_batch_size: int = typer.Option(4, "--score-batch-size", min=1, help="Batch size for forced-choice label scoring."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Inspect eval configuration without loading model weights."),
) -> None:
    target_output = output_dir or adapter_path.parent
    result = eval_lora_sentiment(
        adapter_path=adapter_path,
        output_dir=target_output,
        dataset_dir=dataset_dir,
        limit=limit,
        contract_limit=contract_limit,
        score_batch_size=score_batch_size,
        dry_run=dry_run,
    )
    _print_lora_result("LoRA Evaluation", result)


def _print_lora_result(title: str, result: dict[str, Any]) -> None:
    table = Table(title=title)
    table.add_column("Field")
    table.add_column("Value")
    for key, value in result.items():
        table.add_row(str(key), _format_lora_value(key, value))
    console.print(table)


def _format_lora_value(key: str, value: Any) -> str:
    if key == "datasets" and isinstance(value, dict):
        lines = []
        for name, dataset in value.items():
            if isinstance(dataset, dict):
                lines.append(
                    (
                        f"{name}: train={dataset.get('train_samples')}, "
                        f"dev={dataset.get('dev_samples')}, eval={dataset.get('eval_samples')}"
                    )
                )
        return "\n".join(lines)
    return _format_eval_value(key, value)


@config_app.command("check")
def config_check() -> None:
    settings = load_settings()
    table = Table(title="Investment Research Desk Config Check")
    table.add_column("Item")
    table.add_column("Status")
    table.add_column("Detail")

    ollama = OllamaLLMClient(settings.ollama_base_url, settings.ollama_model, timeout=settings.llm_timeout_sec)
    ok, detail = ollama.healthcheck()
    table.add_row("Ollama", "OK" if ok else "WARN", detail)
    table.add_row("Model", settings.ollama_model, "Qwen3-8B Instruct/Chat target; override with IRD_OLLAMA_MODEL")
    table.add_row("LLM Timeout", "OK", f"{settings.llm_timeout_sec}s; IRD_LLM_TIMEOUT_SEC")
    table.add_row("Agent Execution", "OK", f"{settings.agent_execution_mode}; IRD_AGENT_EXECUTION_MODE=sequential|parallel")
    table.add_row("Tool Loop Budget", "OK", f"timeout={settings.agent_tool_loop_timeout_sec}s; max_calls={settings.agent_max_tool_calls}")
    table.add_row("OKX", _http_status(f"{settings.okx_base_url}/api/v5/public/time"), settings.okx_base_url)
    table.add_row("Tavily", *_tavily_status(settings.tavily_base_url, settings.tavily_api_key))
    table.add_row("FMP", *_fmp_status(settings.fmp_base_url, settings.fmp_api_key))
    table.add_row("Finnhub", *_finnhub_status(settings.finnhub_base_url, settings.finnhub_api_key))
    table.add_row("Yahoo", *_simple_http_status("https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=5d&interval=1d", "chart endpoint reachable"))
    table.add_row("StockTwits", *_simple_http_status("https://api.stocktwits.com/api/2/streams/symbol/AAPL.json", "public stream reachable"))
    table.add_row("Reddit", *_simple_http_status("https://www.reddit.com/r/stocks/search.json?q=AAPL&restrict_sr=on&sort=new&t=week&limit=1", "public search reachable"))
    table.add_row("Jin10", "OK" if settings.jin10_api_url else "WARN", "JIN10_API_URL set" if settings.jin10_api_url else "JIN10_API_URL not set")
    adapter_path = settings.sentiment_adapter_path or discover_latest_adapter()
    missing = missing_runtime_packages()
    if settings.sentiment_provider == "hf-peft":
        if missing:
            sentiment_status = "WARN"
            sentiment_detail = f"hf-peft requested but missing packages: {', '.join(missing)}"
        elif adapter_path and adapter_path.exists():
            sentiment_status = "OK"
            sentiment_detail = f"hf-peft adapter={adapter_path}"
        else:
            sentiment_status = "WARN"
            sentiment_detail = "hf-peft requested but no adapter found; set IRD_SENTIMENT_ADAPTER_PATH"
    else:
        sentiment_status = "OK"
        sentiment_detail = f"{settings.sentiment_provider}; latest adapter: {adapter_path or 'not found'}"
    table.add_row("Sentiment Runtime", sentiment_status, sentiment_detail)
    table.add_row("Report Language", "OK", settings.report_language)
    console.print(table)


@okx_app.command("check")
def okx_check() -> None:
    settings = load_settings()
    provider = OkxMarketDataProvider(settings.okx_base_url)
    table = Table(title="OKX Public SWAP Market Check")
    table.add_column("Item")
    table.add_column("Status")
    table.add_column("Detail")
    table.add_row("Base URL", "OK", settings.okx_base_url)
    table.add_row("Scope", "OK", "public SWAP market data only; account/position APIs are disabled")
    try:
        request = build_run_request(
            symbol="ETH",
            asset_class="crypto",
            horizon="short_term",
            research_depth="standard",
            fixture=None,
            llm_provider="fake",
            model=None,
        )
        inst_id = provider.resolve_inst_id(request)
        bars = provider.fetch_ohlcv(request)
        context = provider.fetch_swap_market_context(request)
        table.add_row("Market data", "OK" if bars else "WARN", f"ETH resolved to {inst_id}; bars={len(bars)}")
        table.add_row("Funding", "OK" if context.get("funding_rate") else "WARN", str(context.get("funding_rate") or "missing"))
        table.add_row("Open interest", "OK" if context.get("open_interest") else "WARN", str(context.get("open_interest") or "missing"))
        table.add_row("Order book", "OK" if context.get("orderbook") else "WARN", f"imbalance={context.get('orderbook_imbalance')}")
    except Exception as exc:
        table.add_row("Market data", "WARN", str(exc))
    console.print(table)


def _welcome_panel() -> Panel:
    content = (
        f"[dim]{ASCII_LOGO}[/dim]\n"
        "[bold green]Investment Research Desk / 投研策略台[/bold green]\n\n"
        "[bold]Workflow Steps:[/bold]\n"
        "I. Analyst Team -> II. Bull/Bear Research Debate -> III. Research Reporter -> IV. final_market_context_cache\n\n"
        "[dim]Local CLI frontend for structured investment research context. No trading execution, orders, or position sizing.[/dim]"
    )
    return Panel(
        Align.center(content),
        title="Welcome to Investment Research Desk",
        subtitle="Multi-Agent Investment Research Framework",
        border_style="green",
    )


def _workflow_panel() -> Panel:
    table = Table.grid(padding=(0, 2))
    table.add_column(style="cyan")
    table.add_column()
    table.add_row("I.", "Analyst Team: Fundamental/Macro, News Impact, Sentiment, Technical")
    table.add_row("II.", "Bull/Bear Research Debate: constructive case and risk case")
    table.add_row("III.", "Research Reporter: structured context, brief, trace, metrics")
    table.add_row("IV.", "final_market_context_cache: downstream research context only")
    return Panel(table, title="Workflow", border_style="cyan")


def _step_panel(step: int, title: str, prompt: str, default: str | None = None) -> Panel:
    body = f"[bold]Step {step}: {title}[/bold]\n[dim]{prompt}[/dim]"
    if default:
        body += f"\n[dim]Default: {default}[/dim]"
    return Panel(body, border_style="bright_blue", padding=(1, 2))


def _runtime_header(request) -> Panel:
    subtitle = "Multi-agent investment research context system - CLI Frontend"
    body = (
        f"[bold green]{request.symbol}[/bold green] "
        f"[dim]| {request.asset_class} | {request.horizon} | {request.llm_provider}:{request.model or 'default'}[/dim]\n"
        f"[green]{subtitle}[/green]"
    )
    return Panel(Align.center(body), title="Investment Research Desk / Touyan Celue Tai", border_style="green")


def _runtime_progress_panel(statuses: dict[str, str]) -> Panel:
    table = Table(show_header=True, header_style="bold magenta", box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Team", style="cyan", justify="center", ratio=2)
    table.add_column("Agent", style="green", ratio=3)
    table.add_column("Status", justify="center", ratio=2)
    for team, agents in AGENT_TEAMS.items():
        for index, agent in enumerate(agents):
            status = statuses.get(agent, "pending")
            if status == "in_progress":
                status_cell = Spinner("dots", text="[blue]in_progress[/blue]", style="cyan")
            else:
                color = {"pending": "yellow", "completed": "green", "failed": "red"}.get(status, "white")
                status_cell = f"[{color}]{status}[/{color}]"
            table.add_row(team if index == 0 else "", AGENT_LABELS.get(agent, agent), status_cell)
        table.add_row("", "", "", style="dim")
    return Panel(table, title="Progress", border_style="cyan")


def _runtime_messages_panel(messages: deque[tuple[str, str, str]]) -> Panel:
    table = Table(show_header=True, header_style="bold magenta", box=box.MINIMAL, show_lines=True, expand=True)
    table.add_column("Time", style="cyan", width=8, justify="center")
    table.add_column("Type", style="green", width=10, justify="center")
    table.add_column("Content", ratio=1)
    for timestamp, message_type, content in list(messages)[-12:]:
        if len(content) > 180:
            content = content[:177] + "..."
        table.add_row(timestamp, message_type, Text(content, overflow="fold"))
    return Panel(table, title="Messages & Tools", border_style="blue")


def _markdown_for_agent_result(agent_name: str, output: dict[str, Any]) -> str:
    if agent_name == "fundamental_macro":
        return (
            "## Fundamental / Macro Analyst\n"
            f"- View: {output.get('fundamental_view', 'unknown')}\n"
            f"- Confidence: {output.get('confidence', 'unknown')}\n\n"
            f"### Key Drivers\n{_markdown_list(output.get('key_drivers'))}\n\n"
            f"### Concerns\n{_markdown_list(output.get('concerns'))}\n\n"
            f"### Evidence\n{_markdown_list(output.get('evidence'))}"
        )
    if agent_name == "news_impact":
        return (
            "## News / Macro Impact Analyst\n"
            f"- Impact logic: {output.get('impact_logic', 'unknown')}\n"
            f"- Confidence: {output.get('confidence', 'unknown')}\n\n"
            f"### Dominant Events\n{_markdown_list(output.get('dominant_events'))}\n\n"
            f"### Evidence\n{_markdown_list(output.get('evidence'))}"
        )
    if agent_name == "sentiment":
        return (
            "## Sentiment Analyst\n"
            f"- Mood: {output.get('crowd_mood', 'unknown')}\n"
            f"- Label: {output.get('sentiment_label', 'unknown')}\n"
            f"- Score: {output.get('sentiment_score', 'unknown')}\n"
            f"- Confidence: {output.get('confidence', 'unknown')}\n\n"
            f"### Evidence\n{_markdown_list(output.get('evidence'))}"
        )
    if agent_name == "technical":
        return (
            "## Technical Analyst\n"
            f"- View: {output.get('technical_view', 'unknown')}\n"
            f"- Trend: {output.get('trend', 'unknown')}\n"
            f"- Momentum: {output.get('momentum', 'unknown')}\n"
            f"- Volatility regime: {output.get('volatility_regime', 'unknown')}\n"
            f"- RSI 14: {output.get('rsi_14', 'unknown')}\n"
            f"- MACD: {output.get('macd_state', 'unknown')}\n"
            f"- OKX mark price: {output.get('mark_price', 'unknown')}\n"
            f"- OKX funding rate: {output.get('funding_rate', 'unknown')}\n"
            f"- OKX open interest: {output.get('open_interest', 'unknown')}\n"
            f"- OKX orderbook imbalance: {output.get('orderbook_imbalance', 'unknown')}\n"
            f"- Support zones: {', '.join(map(str, output.get('support_zones') or [])) or 'None'}\n"
            f"- Resistance zones: {', '.join(map(str, output.get('resistance_zones') or [])) or 'None'}"
        )
    return f"## {AGENT_LABELS.get(agent_name, agent_name)}\nCompleted."


def _markdown_for_state_progress(node_name: str, state: dict[str, Any]) -> str:
    if node_name == "bull_researcher" and state.get("constructive"):
        item = state["constructive"]
        return f"## Bull Researcher\n### Thesis\n{item.get('thesis')}\n\n### Evidence\n{_markdown_list(item.get('evidence'))}"
    if node_name == "bear_researcher" and state.get("risk"):
        item = state["risk"]
        return f"## Bear Researcher\n### Thesis\n{item.get('thesis')}\n\n### Evidence\n{_markdown_list(item.get('evidence'))}"
    if node_name == "research_reporter" and state.get("final_context"):
        final = state["final_context"]
        return (
            "## Research Reporter\n"
            f"- Balanced view: {final.get('balanced_view')}\n"
            f"- Risk level: {final.get('risk_level')}\n"
            f"- Confidence: {final.get('confidence')}\n\n"
            f"### Key Drivers\n{_markdown_list(final.get('key_drivers'))}\n\n"
            f"### Key Risks\n{_markdown_list(final.get('key_risks'))}"
        )
    return f"## {AGENT_LABELS.get(node_name, node_name)}\nCompleted."


def _markdown_list(items: Any) -> str:
    if not items:
        return "- None"
    return "\n".join(f"- {item}" for item in items)


def _count_status_entries(value: Any) -> int:
    if isinstance(value, list):
        return len(value)
    if isinstance(value, dict):
        return sum(_count_status_entries(item) for item in value.values()) or len(value)
    return 0


def _execution_contract_panel() -> Panel:
    table = Table(box=box.SIMPLE, show_header=True)
    table.add_column("Team")
    table.add_column("Nodes")
    for team, nodes in TEAM_FLOW:
        table.add_row(team, ", ".join(nodes))
    return Panel(table, title="Execution Contract", border_style="blue")


def _print_request_review(request, checkpoint: bool, runs_dir: Path, mode: str) -> None:
    table = Table(title="Run Contract Review", box=box.SIMPLE_HEAVY)
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("mode", mode)
    table.add_row("symbol", request.symbol)
    table.add_row("asset_class", request.asset_class)
    table.add_row("horizon", request.horizon)
    table.add_row("research_depth", request.research_depth)
    table.add_row("fixture", request.fixture or "")
    table.add_row("llm_provider", request.llm_provider)
    table.add_row("model", request.model or "")
    table.add_row("sentiment_provider", request.sentiment_provider or "settings/default")
    table.add_row("sentiment_base_model", request.sentiment_base_model or "settings/default")
    table.add_row("sentiment_adapter_path", request.sentiment_adapter_path or "settings/default")
    table.add_row("language", request.language)
    table.add_row("checkpoint", str(checkpoint))
    table.add_row("runs_dir", str(runs_dir))
    console.print(table)


def _running_table(started: float, completed: bool = False) -> Panel:
    table = Table(show_header=True, box=box.SIMPLE_HEAD)
    table.add_column("Team")
    table.add_column("Status")
    status = "completed" if completed else "running"
    elapsed = f"{time.perf_counter() - started:.1f}s"
    for team, _ in TEAM_FLOW:
        table.add_row(team, status, style="green" if completed else "yellow")
    return Panel(table, title=f"Multi-Agent Run | elapsed={elapsed}", border_style="cyan")


def _print_run_summary(state: dict) -> None:
    final = FinalResearchContext.model_validate(state["final_context"])
    _print_console_report(state)
    table = Table(title="Agent Trace")
    table.add_column("Agent/Node")
    table.add_column("Status")
    table.add_column("Latency")
    for agent in state["trace"]["agents"]:
        table.add_row(agent["name"], agent["status"], f"{agent['latency_sec']}s")
    console.print(table)
    if state.get("warnings"):
        console.print(Panel(_console_safe("\n".join(state["warnings"])), title="Warnings", style="yellow"))
    paths = Table(title="Output Paths")
    paths.add_column("Artifact")
    paths.add_column("Path")
    for name, path in state.get("output_paths", {}).items():
        paths.add_row(name, path)
    console.print(paths)
    _print_artifact_contract(state)


def _print_console_report(state: dict) -> None:
    final = FinalResearchContext.model_validate(state["final_context"])
    fundamental = state.get("fundamental", {})
    news = state.get("news", {})
    sentiment = state.get("sentiment", {})
    technical = state.get("technical", {})
    constructive = state.get("constructive", {})
    risk = state.get("risk", {})
    data = state.get("data", {})
    metrics = state.get("metrics") or {}
    debate = state.get("research_debate") or {}
    language = final.source_metadata.get("language", "en")
    t = _report_labels(language)

    executive = Table.grid(expand=True)
    executive.add_column(ratio=1)
    executive.add_column(ratio=3)
    executive.add_row(t["symbol"], final.symbol)
    executive.add_row(t["directional_view"], f"[{_direction_style(final.directional_view)}]{final.directional_view.upper()}[/]")
    executive.add_row(t["directional_rationale"], _console_safe(final.directional_rationale))
    executive.add_row(t["balanced_view"], final.balanced_view)
    executive.add_row(t["risk_level"], final.risk_level)
    executive.add_row(t["confidence"], str(final.confidence))
    executive.add_row(t["horizon"], final.horizon)
    executive.add_row(t["market_regime"], final.market_regime)
    console.print(
        Panel(
            executive,
            title=f"{t['final_report']} | {final.symbol}",
            subtitle=t["boundary"],
            border_style=_direction_style(final.directional_view),
        )
    )

    console.print(
        Group(
            _agent_panel(
                "Fundamental / Macro Analyst",
                [
                    ("View", fundamental.get("fundamental_view")),
                    ("Confidence", fundamental.get("confidence")),
                    ("Key Drivers", _plain_list(fundamental.get("key_drivers"))),
                    ("Concerns", _plain_list(fundamental.get("concerns"))),
                    ("Evidence", _plain_list(fundamental.get("evidence"))),
                ],
            ),
            _agent_panel(
                "News / Macro Impact Analyst",
                [
                    ("Impact Logic", news.get("impact_logic")),
                    ("Confidence", news.get("confidence")),
                    ("Asset Impact", (news.get("asset_impact") or {}).get(final.symbol, "mixed")),
                    ("Dominant Events", _plain_list(news.get("dominant_events"))),
                    ("Evidence", _plain_list(news.get("evidence"))),
                ],
            ),
            _agent_panel(
                "Sentiment Analyst",
                [
                    ("Crowd Mood", sentiment.get("crowd_mood")),
                    ("Label", sentiment.get("sentiment_label")),
                    ("Score", sentiment.get("sentiment_score")),
                    ("Confidence", sentiment.get("confidence")),
                    ("Evidence", _plain_list(sentiment.get("evidence"))),
                ],
            ),
            _agent_panel(
                "Technical Analyst",
                [
                    ("View", technical.get("technical_view")),
                    ("Trend", technical.get("trend")),
                    ("Momentum", technical.get("momentum")),
                    ("Volatility", technical.get("volatility_regime")),
                    ("RSI 14", technical.get("rsi_14")),
                    ("MACD", technical.get("macd_state")),
                    ("ATR 14", technical.get("atr_14")),
                    ("Realized Volatility", technical.get("realized_volatility")),
                    ("Max Drawdown", technical.get("max_drawdown")),
                    ("OKX Mark / Index", f"{technical.get('mark_price')} / {technical.get('index_price')}"),
                    ("OKX Funding", technical.get("funding_rate")),
                    ("OKX Open Interest", technical.get("open_interest")),
                    ("Orderbook Imbalance", technical.get("orderbook_imbalance")),
                    ("Support Zones", _plain_list(technical.get("support_zones"))),
                    ("Resistance Zones", _plain_list(technical.get("resistance_zones"))),
                ],
            ),
            _agent_panel(
                "Bull / Constructive Researcher",
                [
                    ("Thesis", constructive.get("thesis")),
                    ("Evidence", _plain_list(constructive.get("evidence"))),
                    ("Conditions", _plain_list(constructive.get("conditions"))),
                    ("Confidence", constructive.get("confidence")),
                ],
            ),
            _agent_panel(
                "Bear / Risk Researcher",
                [
                    ("Thesis", risk.get("thesis")),
                    ("Evidence", _plain_list(risk.get("evidence"))),
                    ("Conditions", _plain_list(risk.get("conditions"))),
                    ("Confidence", risk.get("confidence")),
                ],
            ),
            _agent_panel(
                t["research_reporter"],
                [
                    (t["fundamental_summary"], final.fundamental_summary),
                    (t["news_summary"], final.news_impact_summary),
                    (t["sentiment_summary"], final.sentiment_summary),
                    (t["technical_summary"], final.technical_summary),
                    (t["key_drivers"], _plain_list(final.key_drivers)),
                    (t["key_risks"], _plain_list(final.key_risks)),
                    (t["uncertainty"], _plain_list(final.uncertainty_factors)),
                ],
            ),
            _agent_panel(
                t["debate"],
                [
                    (t["points_agreement"], _plain_list(debate.get("points_of_agreement"))),
                    (t["key_tensions"], _plain_list(debate.get("key_tensions"))),
                    (t["evidence_quality"], _plain_list(debate.get("evidence_quality_notes"))),
                    (t["reporter_handoff"], debate.get("reporter_handoff")),
                ],
            ),
            _agent_panel(
                t["data_metadata"],
                [
                    ("OHLCV Bars", len(data.get("ohlcv") or [])),
                    ("Market Context Sections", ", ".join((data.get("market_context") or {}).keys()) or "None"),
                    ("News Events", len(data.get("news_events") or [])),
                    ("Sentiment Inputs", len(data.get("sentiment_inputs") or [])),
                    ("Provider Mode", (data.get("source_metadata") or {}).get("provider_mode", "unknown")),
                    ("Tool Policy", (data.get("source_metadata") or {}).get("tool_call_policy", "unknown")),
                    ("Agent Execution", (data.get("source_metadata") or {}).get("agent_execution_mode", "unknown")),
                    ("Provider Warnings", _plain_list((data.get("source_metadata") or {}).get("agent_tool_warnings"))),
                    ("Sentiment Runtime", (data.get("source_metadata") or {}).get("sentiment_runtime", "main")),
                    ("Guardrail Violations", ", ".join(metrics.get("guardrail_violations") or []) or "None"),
                ],
            ),
            _agent_panel(
                t["usage_boundary"],
                [
                    (t["constraints"], _plain_list(final.usage_constraints)),
                    (t["downstream"], final.downstream_agent_context),
                ],
            ),
        )
    )


def _report_labels(language: str) -> dict[str, str]:
    if language == "zh":
        return {
            "final_report": "最终投研上下文报告",
            "boundary": "仅作投研上下文，不是投资建议或交易执行指令",
            "symbol": "标的",
            "directional_view": "方向判断",
            "directional_rationale": "判断依据",
            "balanced_view": "综合观点",
            "risk_level": "风险等级",
            "confidence": "置信度",
            "horizon": "研究周期",
            "market_regime": "市场状态",
            "research_reporter": "最终研究报告",
            "fundamental_summary": "基本面/宏观摘要",
            "news_summary": "新闻影响摘要",
            "sentiment_summary": "情绪摘要",
            "technical_summary": "技术摘要",
            "key_drivers": "关键驱动",
            "key_risks": "关键风险",
            "uncertainty": "不确定因素",
            "debate": "Bull/Bear 辩论结论",
            "points_agreement": "共识",
            "key_tensions": "主要分歧",
            "evidence_quality": "证据质量说明",
            "reporter_handoff": "报告交接说明",
            "data_metadata": "数据与运行元信息",
            "usage_boundary": "使用边界",
            "constraints": "约束",
            "downstream": "下游使用说明",
        }
    return {
        "final_report": "Final Research Context Report",
        "boundary": "Research context only; not financial advice or execution instruction",
        "symbol": "Symbol",
        "directional_view": "Directional View",
        "directional_rationale": "Directional Rationale",
        "balanced_view": "Balanced View",
        "risk_level": "Risk Level",
        "confidence": "Confidence",
        "horizon": "Horizon",
        "market_regime": "Market Regime",
        "research_reporter": "Research Reporter",
        "fundamental_summary": "Fundamental Summary",
        "news_summary": "News Impact Summary",
        "sentiment_summary": "Sentiment Summary",
        "technical_summary": "Technical Summary",
        "key_drivers": "Key Drivers",
        "key_risks": "Key Risks",
        "uncertainty": "Uncertainty Factors",
        "debate": "Bull/Bear Debate Conclusion",
        "points_agreement": "Points Of Agreement",
        "key_tensions": "Key Tensions",
        "evidence_quality": "Evidence Quality Notes",
        "reporter_handoff": "Reporter Handoff",
        "data_metadata": "Data And Run Metadata",
        "usage_boundary": "Usage Boundary",
        "constraints": "Constraints",
        "downstream": "Downstream Context",
    }


def _agent_panel(title: str, rows: list[tuple[str, Any]]) -> Panel:
    table = Table.grid(expand=True, padding=(0, 1))
    table.add_column(ratio=1, style="cyan")
    table.add_column(ratio=4)
    for label, value in rows:
        table.add_row(label, _console_safe(str(value if value is not None else "None")))
    return Panel(table, title=title, border_style="green")


def _plain_list(items: Any) -> str:
    if not items:
        return "None"
    if isinstance(items, list):
        return "\n".join(f"- {item}" for item in items) if items else "None"
    return str(items)


def _plain_report_text(markdown_text: str) -> Text:
    lines = []
    for line in _console_safe(markdown_text).splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip().upper()
        elif stripped.startswith("- "):
            stripped = f"- {stripped[2:]}"
        lines.append(stripped)
    return Text("\n".join(lines), overflow="fold")


def _direction_style(direction: str) -> str:
    return "green" if direction == "bullish" else "red"


def _print_artifact_contract(state: dict) -> None:
    output_paths = state.get("output_paths", {})
    table = Table(title="Artifact Contract", box=box.SIMPLE)
    table.add_column("Artifact")
    table.add_column("Status")
    for artifact in REQUIRED_ARTIFACTS:
        if artifact in {"input.json", "agent_contracts.json", "normalized_data.json", "analyst_outputs.json", "analyst_team_outputs.json", "bull_risk_outputs.json", "research_debate.json"}:
            status = "written during run"
        else:
            key = artifact.removesuffix(".json").removesuffix(".md")
            status = "OK" if key in output_paths or artifact == "final_market_context_cache.json" else "check run directory"
        table.add_row(artifact, status)
    console.print(table)


def _select(message: str, choices):
    value = questionary.select(message, choices=choices, style=CLI_STYLE).ask()
    if value is None:
        raise typer.Exit(code=1)
    return value


def _enum_select(message: str, enum_type, default):
    choices = [
        questionary.Choice(f"{item.value}{' (default)' if item == default else ''}", item)
        for item in enum_type
    ]
    return _select(message, choices)


def _default_report_language_option(raw: str | None) -> ReportLanguageOption:
    try:
        return ReportLanguageOption((raw or "en").strip().lower())
    except ValueError:
        return ReportLanguageOption.en


def _confirm(message: str, default: bool) -> bool:
    value = questionary.confirm(message, default=default, style=CLI_STYLE).ask()
    if value is None:
        raise typer.Exit(code=1)
    return bool(value)


def _select_resume_run(runs_dir: Path) -> str:
    rows = [row for row in discover_runs(runs_dir) if row["status"] == "checkpoint"]
    if not rows:
        _exit_with_error(
            f"No resumable checkpoints found in {runs_dir}.",
            hints=["Run a report with `--checkpoint` first.", "Use `ird runs` to inspect completed and partial runs."],
        )
    return _select(
        "Select checkpoint to resume",
        [questionary.Choice(row["run_id"], row["run_id"]) for row in rows],
    )


def _clear_checkpoints(runs_dir: Path | None) -> None:
    settings = load_settings()
    store = RunStore(runs_dir or settings.runs_dir)
    cleared = store.clear_checkpoints()
    console.print(f"[yellow]Cleared {cleared} checkpoint file(s) from {store.base_dir}.[/yellow]")


def _preflight_llm(provider: str, model: str | None, fixture_mode: bool, settings) -> None:
    if provider == "fake":
        return
    ollama = OllamaLLMClient(settings.ollama_base_url, model or settings.ollama_model, timeout=settings.llm_timeout_sec)
    ok, detail = ollama.healthcheck()
    if ok:
        return
    if provider == "auto" and fixture_mode:
        console.print(f"[yellow]Ollama preflight warning: {detail}. Fixture mode can fall back to fake LLM.[/yellow]")
        return
    if provider in ALLOWED_LLM_PROVIDERS:
        _exit_with_error(
            "Ollama preflight failed.",
            hints=[
                detail,
                "Start Ollama with `ollama serve`.",
                f"Pull/list the target model with `ollama pull {model or settings.ollama_model}` and `ollama list`.",
                "Set IRD_OLLAMA_BASE_URL=http://localhost:11434/v1 in .env if you changed the endpoint.",
            ],
        )


def _preflight_sentiment_runtime(request, settings) -> None:
    configured_provider = "main" if request.llm_provider == "fake" and request.sentiment_provider is None else settings.sentiment_provider
    provider = (request.sentiment_provider or configured_provider or "main").strip().lower()
    if provider in {"", "main", "none", "disabled", "fake"}:
        return
    if provider != "hf-peft":
        _exit_with_error(f"Unsupported sentiment provider: {provider}")
    adapter_path = Path(request.sentiment_adapter_path) if request.sentiment_adapter_path else settings.sentiment_adapter_path
    if adapter_path is None:
        adapter_path = discover_latest_adapter()
    if adapter_path is None:
        _exit_with_error(
            "Sentiment adapter is enabled but no adapter path was provided or discovered.",
            hints=[
                "Set IRD_SENTIMENT_ADAPTER_PATH in .env.",
                "Or pass --sentiment-adapter-path models/investment-research-desk-lora-sentiment/<timestamp>/adapter.",
                "Or train an adapter under models/investment-research-desk-lora-sentiment/<timestamp>/adapter.",
            ],
        )
    if not adapter_path.exists():
        _exit_with_error(
            "Sentiment adapter path does not exist.",
            hints=[str(adapter_path), "Run `ird lora train ...` first or point to an existing PEFT adapter directory."],
        )
    missing = missing_runtime_packages()
    if missing:
        _exit_with_error(
            "Sentiment adapter runtime dependencies are not installed in this Python environment.",
            hints=[
                f"Missing packages: {', '.join(missing)}",
                "Run hf-peft adapter reports from the WSL CUDA training environment, or install the HF runtime packages in this environment.",
                "Use --sentiment-provider main if you want to run without the adapter.",
            ],
        )


def _exit_with_error(message: str, hints: list[str] | None = None) -> None:
    lines = [message]
    if hints:
        lines.append("")
        lines.extend(f"- {hint}" for hint in hints)
    console.print(Panel("\n".join(lines), title="CLI Contract Error", border_style="red"))
    raise typer.Exit(code=2)


def _console_safe(text: str) -> str:
    encoding = getattr(console.file, "encoding", None) or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def _http_status(url: str) -> str:
    try:
        with httpx.Client(timeout=5.0) as client:
            response = client.get(url)
        return "OK" if response.status_code < 400 else f"WARN HTTP {response.status_code}"
    except Exception as exc:
        return f"WARN {exc}"


def _fmp_status(base_url: str, api_key: str | None) -> tuple[str, str]:
    if not api_key:
        return "WARN", "FMP_API_KEY not set"
    try:
        with httpx.Client(timeout=8.0) as client:
            response = client.get(f"{base_url}/quote", params={"symbol": "AAPL", "apikey": api_key})
        if response.status_code >= 400:
            return "WARN", f"HTTP {response.status_code}"
        data = response.json()
        return ("OK", "quote endpoint reachable") if isinstance(data, list) else ("WARN", "unexpected response")
    except Exception as exc:
        return "WARN", str(exc)


def _finnhub_status(base_url: str, api_key: str | None) -> tuple[str, str]:
    if not api_key:
        return "WARN", "FINNHUB_API_KEY not set"
    try:
        with httpx.Client(timeout=8.0) as client:
            response = client.get(f"{base_url}/quote", params={"symbol": "AAPL", "token": api_key})
        if response.status_code >= 400:
            return "WARN", f"HTTP {response.status_code}"
        data = response.json()
        return ("OK", "quote endpoint reachable") if isinstance(data, dict) and "c" in data else ("WARN", "unexpected response")
    except Exception as exc:
        return "WARN", str(exc)


def _tavily_status(base_url: str, api_key: str | None) -> tuple[str, str]:
    if not api_key:
        return "WARN", "TAVILY_API_KEY not set"
    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.post(
                f"{base_url}/search",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={"query": "AAPL market news", "max_results": 1},
            )
        if response.status_code >= 400:
            return "WARN", f"HTTP {response.status_code}"
        return "OK", "search endpoint reachable"
    except Exception as exc:
        return "WARN", str(exc)


def _simple_http_status(url: str, ok_detail: str) -> tuple[str, str]:
    try:
        with httpx.Client(timeout=8.0, headers={"User-Agent": "investment-research-desk/0.1"}) as client:
            response = client.get(url)
        if response.status_code >= 400:
            return "WARN", f"HTTP {response.status_code}"
        return "OK", ok_detail
    except Exception as exc:
        return "WARN", str(exc)


def main() -> None:
    app()


if __name__ == "__main__":
    main()

