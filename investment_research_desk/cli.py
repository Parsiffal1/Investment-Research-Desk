from __future__ import annotations

from pathlib import Path
from typing import Optional

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.prompt import Prompt
from rich.table import Table

from investment_research_desk import __version__
from investment_research_desk.config import load_settings
from investment_research_desk.eval import run_eval_suite
from investment_research_desk.graph import ResearchWorkflow
from investment_research_desk.llm import OllamaLLMClient
from investment_research_desk.persistence import RunStore
from investment_research_desk.providers.fixtures import FixtureProvider
from investment_research_desk.schemas import FinalResearchContext, RunRequest

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
    console.print(Panel.fit("Investment Research Desk / 投研策略台", subtitle="CLI research workflow"))
    fixture = Prompt.ask("Fixture name, or blank for live providers", default="gold_cpi")
    symbol = Prompt.ask("Symbol", default="XAU-USDT-SWAP")
    asset_class = Prompt.ask("Asset class", default="precious_metal")
    horizon = Prompt.ask("Horizon", default="short_term")
    depth = Prompt.ask("Research depth", default="standard")
    provider = Prompt.ask("LLM provider", choices=["auto", "fake", "ollama"], default="auto")
    run_report(
        symbol=symbol,
        asset_class=asset_class,
        horizon=horizon,
        research_depth=depth,
        fixture=fixture or None,
        llm_provider=provider,
        model=None,
        checkpoint=True,
        resume=None,
        clear_checkpoints=False,
        runs_dir=None,
    )


@app.command()
def report(
    symbol: Optional[str] = typer.Option(None, "--symbol", help="Instrument symbol, e.g. BTC-USDT-SWAP."),
    asset_class: str = typer.Option("crypto", "--asset-class", help="Asset class."),
    horizon: str = typer.Option("short_term", "--horizon", help="intraday, short_term, swing, or medium_term."),
    research_depth: str = typer.Option("standard", "--research-depth", help="quick, standard, or deep."),
    fixture: Optional[str] = typer.Option(None, "--fixture", help="Run from frozen fixture data."),
    llm_provider: str = typer.Option("auto", "--llm-provider", help="auto, fake, or ollama."),
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
    asset_class: str,
    horizon: str,
    research_depth: str,
    fixture: str | None,
    llm_provider: str,
    model: str | None,
    checkpoint: bool,
    resume: str | None,
    clear_checkpoints: bool,
    runs_dir: Path | None,
) -> None:
    settings = load_settings()
    store = RunStore(runs_dir or settings.runs_dir)
    if clear_checkpoints:
        cleared = store.clear_checkpoints()
        console.print(f"Cleared {cleared} checkpoint file(s).")
    if resume:
        request = RunRequest(symbol=symbol or "RESUME", llm_provider=llm_provider, model=model)
    elif fixture:
        request = FixtureProvider().request(fixture)
        request.llm_provider = llm_provider
        request.model = model
        request.research_depth = research_depth  # type: ignore[assignment]
    else:
        if not symbol:
            raise typer.BadParameter("--symbol is required when --fixture is not used")
        request = RunRequest(
            symbol=symbol,
            asset_class=asset_class,  # type: ignore[arg-type]
            horizon=horizon,  # type: ignore[arg-type]
            research_depth=research_depth,  # type: ignore[arg-type]
            llm_provider=llm_provider,  # type: ignore[arg-type]
            model=model,
        )
    workflow = ResearchWorkflow(settings=settings, runs_dir=runs_dir)
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), console=console, transient=True) as progress:
        task = progress.add_task("Running multi-agent research workflow...", total=None)
        state = workflow.run(request, checkpoint=checkpoint, resume_run_id=resume)
        progress.update(task, description="Research workflow complete")
    _print_run_summary(state)


@app.command()
def batch(
    symbols: str = typer.Option(..., "--symbols", help="Comma-separated instrument symbols."),
    asset_class: str = typer.Option("crypto", "--asset-class"),
    horizon: str = typer.Option("short_term", "--horizon"),
    llm_provider: str = typer.Option("auto", "--llm-provider"),
    model: Optional[str] = typer.Option(None, "--model"),
) -> None:
    for symbol in [item.strip() for item in symbols.split(",") if item.strip()]:
        console.rule(f"Running {symbol}")
        run_report(
            symbol=symbol,
            asset_class=asset_class,
            horizon=horizon,
            research_depth="standard",
            fixture=None,
            llm_provider=llm_provider,
            model=model,
            checkpoint=False,
            resume=None,
            clear_checkpoints=False,
            runs_dir=None,
        )


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


def main() -> None:
    app()


if __name__ == "__main__":
    main()
