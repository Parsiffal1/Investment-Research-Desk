from __future__ import annotations

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
from rich.console import Console
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
    ResearchDepthOption,
    TEAM_FLOW,
    build_run_request,
    discover_fixtures,
    discover_runs,
)
from investment_research_desk.config import load_settings
from investment_research_desk.eval import run_eval_suite
from investment_research_desk.graph import ResearchWorkflow
from investment_research_desk.graph.workflow import render_markdown_report
from investment_research_desk.llm import OllamaLLMClient
from investment_research_desk.persistence import RunStore
from investment_research_desk.providers.okx import OkxMarketDataProvider
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
okx_app = typer.Typer(help="OKX read-only market/account checks")
app.add_typer(okx_app, name="okx")


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
    "Run Control": ["run_controller", "data_ingestion"],
    "Analyst Team": ["fundamental_macro", "news_impact", "sentiment", "technical"],
    "Bull/Bear Research Debate": ["bull_researcher", "bear_researcher", "bull_bear_research_debate"],
    "Research Reporter": ["research_reporter", "final_market_context_cache", "persist"],
}

AGENT_LABELS = {
    "run_controller": "Run Controller",
    "data_ingestion": "Data Ingestion",
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
        layout["report"].update(Panel(Text(self.current_report, overflow="fold"), title="Current Report", border_style="green"))
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

    console.print(_step_panel(1, "Workflow Action", "Choose whether to start research, resume, inspect runs, or check configuration."))
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
        questionary.text("Runs directory", default=str(settings.runs_dir), style=CLI_STYLE).ask() or str(settings.runs_dir)
    )

    if action in {"config_check", "list_runs", "clear_checkpoints", "exit"}:
        return CLIInteractionContract(mode=action, request=None, checkpoint=False, resume_run_id=None, runs_dir=runs_dir)

    if action == "resume":
        console.print(_step_panel(2, "Checkpoint", "Select a resumable run and continue from the last saved graph step."))
        run_id = _select_resume_run(runs_dir)
        checkpoint = _confirm("Continue saving checkpoints after resume?", default=True)
        return CLIInteractionContract(mode="resume", request=None, checkpoint=checkpoint, resume_run_id=run_id, runs_dir=runs_dir)

    console.print(_step_panel(2, "Data Source", "Use stable fixture data for demos/tests, or live providers for a current research run."))
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
        console.print(_step_panel(3, "Fixture Scenario", "Select a frozen scenario for repeatable local research.", "gold_cpi"))
        fixture = _select(
            "Select fixture",
            [questionary.Choice(name, name) for name in fixtures] or [questionary.Choice("gold_cpi", "gold_cpi")],
        )
    else:
        console.print(_step_panel(3, "Instrument", "Enter the exact research symbol. Examples: NVDA, AAPL, BTC-USDT-SWAP.", "BTC-USDT-SWAP"))
        symbol = questionary.text(
            "Symbol",
            default="BTC-USDT-SWAP",
            validate=lambda value: bool(value.strip()) or "Symbol is required.",
            style=CLI_STYLE,
        ).ask()
        console.print(_step_panel(4, "Asset Class", "Select the asset class so providers, prompts, and validation use the right contract."))
        asset_class = _enum_select("Asset class", AssetClassOption, AssetClassOption.crypto)
        console.print(_step_panel(5, "Research Horizon", "Select the time horizon for analysis framing and prompt context.", HorizonOption.short_term.value))
        horizon = _enum_select("Horizon", HorizonOption, HorizonOption.short_term)

    console.print(_step_panel(6, "Research Depth", "Select how much reasoning/debate depth to request from the workflow.", ResearchDepthOption.standard.value))
    research_depth = _enum_select("Research depth", ResearchDepthOption, ResearchDepthOption.standard)
    console.print(_step_panel(7, "LLM Provider", "Select the LLM runtime. Ollama with qwen3:8b remains the primary local path.", LLMProviderOption.auto.value))
    llm_provider = _enum_select("LLM provider", LLMProviderOption, LLMProviderOption.auto)
    default_model = settings.ollama_model if llm_provider in {LLMProviderOption.auto, LLMProviderOption.ollama} else ""
    model = questionary.text("Model override", default=default_model, style=CLI_STYLE).ask() or None
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


@okx_app.command("check")
def okx_check() -> None:
    settings = load_settings()
    provider = _make_okx_provider(settings)
    table = Table(title="OKX Read-Only Check")
    table.add_column("Item")
    table.add_column("Status")
    table.add_column("Detail")
    table.add_row("Base URL", "OK", settings.okx_base_url)
    table.add_row("Profile", settings.okx_profile, f"demo={settings.okx_demo}; site={settings.okx_site}")
    table.add_row("Read-only", "OK" if settings.okx_read_only else "WARN", str(settings.okx_read_only))
    table.add_row("TradeKit modules", settings.okx_tradekit_modules, "market recommended for this project")
    table.add_row("Private credentials", "OK" if provider.private_available() else "WARN", "configured" if provider.private_available() else "missing")
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
        table.add_row("Market data", "OK" if bars else "WARN", f"ETH resolved to {inst_id}; bars={len(bars)}")
    except Exception as exc:
        table.add_row("Market data", "WARN", _redact_okx_error(str(exc), settings))
    if provider.private_available():
        try:
            provider.account_config()
            table.add_row("Private read", "OK", "account config reachable")
        except Exception as exc:
            table.add_row("Private read", "WARN", _redact_okx_error(str(exc), settings))
    console.print(table)


@okx_app.command("account")
def okx_account(
    ccy: Optional[str] = typer.Option(None, "--ccy", help="Optional comma-separated balance currencies, e.g. BTC,ETH,USDT."),
    inst_type: Optional[str] = typer.Option(None, "--inst-type", help="Optional position instrument type: MARGIN, SWAP, FUTURES, OPTION."),
    inst_id: Optional[str] = typer.Option(None, "--inst-id", help="Optional OKX instrument ID, e.g. ETH-USDT-SWAP."),
    json_output: bool = typer.Option(False, "--json", help="Print raw JSON returned by OKX."),
) -> None:
    settings = load_settings()
    provider = _make_okx_provider(settings)
    if not provider.private_available():
        _exit_with_error(
            "OKX private API credentials are not configured.",
            hints=["Fill OKX_API_KEY, OKX_SECRET_KEY, and OKX_PASSPHRASE in .env.", "Keep OKX_READ_ONLY=true for this project."],
        )
    try:
        balance = provider.account_balance(ccy=ccy)
        positions = provider.positions(inst_type=inst_type, inst_id=inst_id)
        risk = provider.account_position_risk(inst_type=inst_type)
    except Exception as exc:
        _exit_with_error("OKX account read failed.", hints=[_redact_okx_error(str(exc), settings)])
    if json_output:
        console.print_json(data={"balance": balance, "positions": positions, "risk": risk})
        return
    console.print(_okx_balance_table(balance))
    console.print(_okx_positions_table(positions))
    console.print(_okx_risk_table(risk))


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


def _make_okx_provider(settings) -> OkxMarketDataProvider:
    return OkxMarketDataProvider(
        settings.okx_base_url,
        api_key=settings.okx_api_key,
        secret_key=settings.okx_secret_key,
        passphrase=settings.okx_passphrase,
        demo=settings.okx_demo,
        read_only=settings.okx_read_only,
    )


def _redact_okx_error(text: str, settings) -> str:
    redacted = text
    for secret in [settings.okx_api_key, settings.okx_secret_key, settings.okx_passphrase]:
        if secret:
            redacted = redacted.replace(secret, "***")
    return redacted


def _okx_balance_table(payload: dict[str, Any]) -> Table:
    table = Table(title="OKX Account Balance")
    table.add_column("Currency")
    table.add_column("Equity")
    table.add_column("Available")
    table.add_column("USD Equity")
    table.add_column("Unrealized PnL")
    rows = []
    for account in payload.get("data", []):
        for detail in account.get("details", []):
            rows.append(
                [
                    str(detail.get("ccy", "")),
                    str(detail.get("eq", "")),
                    str(detail.get("availEq") or detail.get("availBal") or ""),
                    str(detail.get("eqUsd", "")),
                    str(detail.get("upl", "")),
                ]
            )
    if not rows:
        table.add_row("-", "-", "-", "-", "No non-zero balances returned.")
    for row in rows:
        table.add_row(*row)
    return table


def _okx_positions_table(payload: dict[str, Any]) -> Table:
    table = Table(title="OKX Positions")
    table.add_column("Instrument")
    table.add_column("Type")
    table.add_column("Side")
    table.add_column("Position")
    table.add_column("Avg Px")
    table.add_column("Mark Px")
    table.add_column("Notional USD")
    table.add_column("UPL")
    rows = payload.get("data", [])
    if not rows:
        table.add_row("-", "-", "-", "-", "-", "-", "-", "No open positions returned.")
        return table
    for item in rows:
        table.add_row(
            str(item.get("instId", "")),
            str(item.get("instType", "")),
            str(item.get("posSide", "")),
            str(item.get("pos", "")),
            str(item.get("avgPx", "")),
            str(item.get("markPx", "")),
            str(item.get("notionalUsd", "")),
            str(item.get("upl", "")),
        )
    return table


def _okx_risk_table(payload: dict[str, Any]) -> Table:
    table = Table(title="OKX Account / Position Risk")
    table.add_column("Currency")
    table.add_column("Equity")
    table.add_column("Adjusted Equity")
    table.add_column("Margin Ratio")
    table.add_column("Details")
    rows = payload.get("data", [])
    if not rows:
        table.add_row("-", "-", "-", "-", "No account risk rows returned.")
        return table
    for item in rows:
        table.add_row(
            str(item.get("ccy", "")),
            str(item.get("eq", "")),
            str(item.get("adjEq", "")),
            str(item.get("mgnRatio", "")),
            f"positions={len(item.get('posData', []) or [])}",
        )
    return table


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
    console.print(
        Panel(
            Text(render_markdown_report(state), overflow="fold"),
            title=f"Research Context Report | {final.symbol} | {final.balanced_view} | risk={final.risk_level}",
            border_style="green",
        )
    )
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
