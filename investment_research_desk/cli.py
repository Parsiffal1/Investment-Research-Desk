from __future__ import annotations

import time
from pathlib import Path
from typing import Optional

import httpx
import questionary
import typer
from rich import box
from rich.align import Align
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table

from investment_research_desk import __version__
from investment_research_desk.cli_contract import (
    ALLOWED_LLM_PROVIDERS,
    AssetClassOption,
    CLIInteractionContract,
    HorizonOption,
    LLMProviderOption,
    REQUIRED_ARTIFACTS,
    ResearchDepthOption,
    TEAM_FLOW,
    build_run_request,
    discover_fixtures,
    discover_runs,
)
from investment_research_desk.config import load_settings
from investment_research_desk.eval import run_eval_suite
from investment_research_desk.graph import ResearchWorkflow
from investment_research_desk.llm import OllamaLLMClient
from investment_research_desk.persistence import RunStore
from investment_research_desk.schemas import FinalResearchContext

console = Console()
app = typer.Typer(
    name="ird",
    help="Investment Research Desk / 投研策略台 CLI",
    invoke_without_command=True,
    no_args_is_help=False,
)
config_app = typer.Typer(help="Configuration and runtime checks")
app.add_typer(config_app, name="config")


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
    contract = _collect_interactive_contract()
    if contract.mode == "config_check":
        config_check()
        return
    if contract.mode == "list_runs":
        list_runs(runs_dir=contract.runs_dir, resumable_only=False)
        return
    if contract.mode == "clear_checkpoints":
        _clear_checkpoints(contract.runs_dir)
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
        )
        return
    if contract.request is None:
        _exit_with_error("No run request was created by the interactive contract.")
    _run_workflow(contract.request, checkpoint=contract.checkpoint, resume=None, clear_checkpoints=False, runs_dir=contract.runs_dir)


def _collect_interactive_contract() -> CLIInteractionContract:
    settings = load_settings()
    console.print(_welcome_panel())
    console.print(_workflow_panel())

    action = _select(
        "Select action",
        [
            questionary.Choice("New research report", "new_report"),
            questionary.Choice("Resume from checkpoint", "resume"),
            questionary.Choice("List runs", "list_runs"),
            questionary.Choice("Config check", "config_check"),
            questionary.Choice("Clear unfinished checkpoints", "clear_checkpoints"),
            questionary.Choice("Exit", "exit"),
        ],
    )
    runs_dir = Path(
        questionary.text("Runs directory", default=str(settings.runs_dir)).ask() or str(settings.runs_dir)
    )

    if action in {"config_check", "list_runs", "clear_checkpoints", "exit"}:
        return CLIInteractionContract(mode=action, request=None, checkpoint=False, resume_run_id=None, runs_dir=runs_dir)

    if action == "resume":
        run_id = _select_resume_run(runs_dir)
        checkpoint = _confirm("Continue saving checkpoints after resume?", default=True)
        return CLIInteractionContract(mode="resume", request=None, checkpoint=checkpoint, resume_run_id=run_id, runs_dir=runs_dir)

    data_mode = _select(
        "Select data mode",
        [
            questionary.Choice("Fixture demo (stable local data)", "fixture"),
            questionary.Choice("Live providers (OKX/FMP/Finnhub/Tavily/public adapters)", "live"),
        ],
    )
    fixture = None
    symbol: str | None = None
    asset_class: str | AssetClassOption = AssetClassOption.crypto
    horizon: str | HorizonOption = HorizonOption.short_term
    if data_mode == "fixture":
        fixtures = discover_fixtures()
        fixture = _select(
            "Select fixture",
            [questionary.Choice(name, name) for name in fixtures] or [questionary.Choice("gold_cpi", "gold_cpi")],
        )
    else:
        symbol = questionary.text(
            "Symbol",
            default="BTC-USDT-SWAP",
            validate=lambda value: bool(value.strip()) or "Symbol is required.",
        ).ask()
        asset_class = _enum_select("Asset class", AssetClassOption, AssetClassOption.crypto)
        horizon = _enum_select("Horizon", HorizonOption, HorizonOption.short_term)

    research_depth = _enum_select("Research depth", ResearchDepthOption, ResearchDepthOption.standard)
    llm_provider = _enum_select("LLM provider", LLMProviderOption, LLMProviderOption.auto)
    default_model = settings.ollama_model if llm_provider in {LLMProviderOption.auto, LLMProviderOption.ollama} else ""
    model = questionary.text("Model override", default=default_model).ask() or None
    checkpoint = _confirm("Save checkpoints after graph steps?", default=True)

    try:
        request = build_run_request(
            symbol=symbol,
            asset_class=asset_class,
            horizon=horizon,
            research_depth=research_depth,
            fixture=fixture,
            llm_provider=llm_provider,
            model=model,
        )
    except ValueError as exc:
        _exit_with_error(str(exc))

    _print_request_review(request, checkpoint=checkpoint, runs_dir=runs_dir, mode=data_mode)
    if not _confirm("Start this research run?", default=True):
        raise typer.Exit()
    return CLIInteractionContract(
        mode="new_report",
        request=request,
        checkpoint=checkpoint,
        resume_run_id=None,
        runs_dir=runs_dir,
    )


@app.command()
def report(
    symbol: Optional[str] = typer.Option(None, "--symbol", help="Instrument symbol, e.g. BTC-USDT-SWAP."),
    asset_class: AssetClassOption = typer.Option(AssetClassOption.crypto, "--asset-class", help="Asset class."),
    horizon: HorizonOption = typer.Option(HorizonOption.short_term, "--horizon", help="Research horizon."),
    research_depth: ResearchDepthOption = typer.Option(ResearchDepthOption.standard, "--research-depth", help="quick, standard, or deep."),
    fixture: Optional[str] = typer.Option(None, "--fixture", help="Run from frozen fixture data."),
    llm_provider: LLMProviderOption = typer.Option(LLMProviderOption.auto, "--llm-provider", help="auto, fake, or ollama."),
    model: Optional[str] = typer.Option(None, "--model", help="Override model name."),
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
        checkpoint=checkpoint,
        resume=resume,
        clear_checkpoints=clear_checkpoints,
        runs_dir=runs_dir,
    )


def run_report(
    symbol: str | None,
    asset_class: str | AssetClassOption,
    horizon: str | HorizonOption,
    research_depth: str | ResearchDepthOption,
    fixture: str | None,
    llm_provider: str | LLMProviderOption,
    model: str | None,
    checkpoint: bool,
    resume: str | None,
    clear_checkpoints: bool,
    runs_dir: Path | None,
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
        _print_request_review(request, checkpoint=checkpoint, runs_dir=store.base_dir, mode="fixture" if request.fixture else "live")

    workflow = ResearchWorkflow(settings=settings, runs_dir=runs_dir)
    console.print(_execution_contract_panel())
    started = time.perf_counter()
    try:
        with Live(_running_table(started), console=console, refresh_per_second=4, transient=True) as live:
            state = workflow.run(request, checkpoint=checkpoint, resume_run_id=resume)
            live.update(_running_table(started, completed=True))
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
    asset_class: AssetClassOption = typer.Option(AssetClassOption.crypto, "--asset-class"),
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
    suite: str = typer.Option(..., "--suite", help="schema, guardrail, single-vs-multi, consistency, compression, latency, or lora."),
) -> None:
    result = run_eval_suite(suite)  # type: ignore[arg-type]
    table = Table(title=f"Evaluation: {suite}")
    table.add_column("Metric")
    table.add_column("Value")
    for key, value in result.items():
        table.add_row(str(key), str(value))
    console.print(table)


@config_app.command("check")
def config_check() -> None:
    settings = load_settings()
    table = Table(title="Investment Research Desk Config Check")
    table.add_column("Item")
    table.add_column("Status")
    table.add_column("Detail")

    ollama = OllamaLLMClient(settings.ollama_base_url, settings.ollama_model)
    ok, detail = ollama.healthcheck()
    table.add_row("Ollama", "OK" if ok else "WARN", detail)
    table.add_row("Model", settings.ollama_model, "Qwen3-8B Instruct/Chat target; override with IRD_OLLAMA_MODEL")
    table.add_row("OKX", _http_status(f"{settings.okx_base_url}/api/v5/public/time"), settings.okx_base_url)
    table.add_row("Tavily", *_tavily_status(settings.tavily_base_url, settings.tavily_api_key))
    table.add_row("FMP", *_fmp_status(settings.fmp_base_url, settings.fmp_api_key))
    table.add_row("Finnhub", *_finnhub_status(settings.finnhub_base_url, settings.finnhub_api_key))
    table.add_row("Yahoo", *_simple_http_status("https://query1.finance.yahoo.com/v8/finance/chart/AAPL?range=5d&interval=1d", "chart endpoint reachable"))
    table.add_row("StockTwits", *_simple_http_status("https://api.stocktwits.com/api/2/streams/symbol/AAPL.json", "public stream reachable"))
    table.add_row("Reddit", *_simple_http_status("https://www.reddit.com/r/stocks/search.json?q=AAPL&restrict_sr=on&sort=new&t=week&limit=1", "public search reachable"))
    table.add_row("Jin10", "OK" if settings.jin10_api_url else "WARN", "JIN10_API_URL set" if settings.jin10_api_url else "JIN10_API_URL not set")
    table.add_row("Fixtures", "OK" if Path("data/fixtures/gold_cpi.json").exists() else "WARN", "gold_cpi fixture")
    console.print(table)


def _welcome_panel() -> Panel:
    return Panel(
        Align.center(
            "[bold]Investment Research Desk / 投研策略台[/bold]\n"
            "[dim]CLI multi-agent research context system, not a trading execution system.[/dim]"
        ),
        title="Welcome",
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
    console.print(Panel.fit(f"{final.symbol} | {final.balanced_view} | risk={final.risk_level}", title="Research Context"))
    table = Table(title="Agent Trace")
    table.add_column("Agent/Node")
    table.add_column("Status")
    table.add_column("Latency")
    for agent in state["trace"]["agents"]:
        table.add_row(agent["name"], agent["status"], f"{agent['latency_sec']}s")
    console.print(table)
    if state.get("warnings"):
        console.print(Panel("\n".join(state["warnings"]), title="Warnings", style="yellow"))
    paths = Table(title="Output Paths")
    paths.add_column("Artifact")
    paths.add_column("Path")
    for name, path in state.get("output_paths", {}).items():
        paths.add_row(name, path)
    console.print(paths)
    _print_artifact_contract(state)


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
    value = questionary.select(message, choices=choices).ask()
    if value is None:
        raise typer.Exit(code=1)
    return value


def _enum_select(message: str, enum_type, default):
    choices = [
        questionary.Choice(f"{item.value}{' (default)' if item == default else ''}", item)
        for item in enum_type
    ]
    return _select(message, choices)


def _confirm(message: str, default: bool) -> bool:
    value = questionary.confirm(message, default=default).ask()
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
    ollama = OllamaLLMClient(settings.ollama_base_url, model or settings.ollama_model)
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


def _exit_with_error(message: str, hints: list[str] | None = None) -> None:
    lines = [message]
    if hints:
        lines.append("")
        lines.extend(f"- {hint}" for hint in hints)
    console.print(Panel("\n".join(lines), title="CLI Contract Error", border_style="red"))
    raise typer.Exit(code=2)


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
