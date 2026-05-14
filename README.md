# Investment Research Desk / 投研策略台

Investment Research Desk is a local CLI-first multi-agent investment research system. It converts market data, macro/news events, sentiment signals, and technical indicators into structured research context for human review and downstream strategy workflows.

It is **not** an autonomous trading or execution system. It does not place orders, manage positions, produce position sizing, or claim profitable trading performance.

## Quick Start

```powershell
uv sync
uv run ird --help
uv run ird config check
uv run ird report --fixture gold_cpi --llm-provider fake
uv run ird eval --suite schema
uv run ird eval --suite guardrail
```

For Ollama-backed local inference, set:

```powershell
$env:IRD_OLLAMA_BASE_URL="http://localhost:11434/v1"
$env:IRD_OLLAMA_MODEL="qwen3:8b"
uv run ird report --fixture gold_cpi --llm-provider ollama
```

API keys can be placed in a local `.env` file in the project root. The loader also accepts `notepad.env` for this workspace. Start from `.env.example` and do not commit real keys:

```powershell
Copy-Item .env.example .env
notepad .env
```

If your key file is named `notepad.env`, that is also loaded automatically.

Supported key names:

```text
TAVILY_API_KEY=
FMP_API_KEY=
FINNHUB_API_KEY=
```

The baseline model target is Qwen3-8B Instruct/Chat. LoRA integration remains pending until training artifacts are produced; do not report improvement metrics until measured results exist.

## CLI

- `ird` starts an interactive research flow.
- `ird report` generates one research context.
- `ird batch` runs multiple symbols.
- `ird runs` lists completed, partial, and resumable run directories.
- `ird eval` runs evaluation suites.
- `ird config check` validates runtime configuration.

### CLI Interaction Contract

`ird` without subcommands opens a menu-driven flow modeled after TradingAgents' CLI experience:

```text
New research report
  -> choose fixture or live providers
  -> choose symbol, asset class, horizon, research depth, LLM provider, model
  -> review the run contract
  -> run Analyst Team -> Bull/Bear Research Debate -> Research Reporter

Resume from checkpoint
  -> choose an existing run_id with checkpoint.json
  -> continue from the latest completed graph step

List runs / Config check / Clear unfinished checkpoints
  -> operational CLI actions with explicit status output
```

Validated option domains:

```text
asset_class: crypto, precious_metal, equity_index, commodity, fx, equity, other
horizon: intraday, short_term, swing, medium_term
research_depth: quick, standard, deep
llm_provider: auto, fake, ollama
```

Errors are reported as `CLI Contract Error` panels with actionable hints. Live Ollama runs preflight `http://localhost:11434/v1/models` before the workflow starts; fixture + `auto` can fall back to the deterministic fake LLM for stable local demos.

Each report run writes:

```text
runs/{run_id}/
  input.json
  agent_contracts.json
  normalized_data.json
  analyst_outputs.json
  analyst_team_outputs.json
  bull_risk_outputs.json
  research_debate.json
  final_research_context.json
  final_market_context_cache.json
  research_brief.md
  trace.json
  metrics.json
```

The workflow follows a TradingAgents-style research structure while stopping before any trading or portfolio execution layer:

```text
Data Ingestion
  -> Analyst Team
     -> Fundamental/Macro Analyst
     -> News/Macro Impact Analyst
     -> Sentiment Analyst
     -> Technical Analyst
  -> Bull/Bear Research Debate
     -> Bull Researcher
     -> Bear Researcher
  -> Research Reporter
  -> final_market_context_cache
```

## Agent Contracts

Each workflow node has an explicit contract covering role, allowed inputs, allowed tools, forbidden actions, output schema, and system prompt. Contracts are persisted per run in `agent_contracts.json`.

The seven analysis/research/reporting agents call the configured LLM through their contracts. Deterministic Python code first prepares factual candidate outputs, then the LLM reads the evidence and returns schema-validated JSON. If JSON generation or schema validation fails, the workflow falls back to the deterministic candidate and records warnings where applicable.

The workflow enforces tool boundaries by having each analyst call only its allowed dataflow tools. `Data Ingestion` prepares fixture data or a live-run data shell; in live mode, the analyst workers fetch their own inputs and the workflow later merges the resulting normalized data into `normalized_data.json`:

- `Fundamental/Macro Analyst`: calls `get_fundamentals` and macro/news tools, then reads fundamental metadata, quote metadata, and macro/news context.
- `News/Macro Impact Analyst`: calls `get_news`, then reads news events only.
- `Sentiment Analyst`: calls `get_sentiment_inputs`, then reads sentiment inputs only.
- `Technical Analyst`: calls `get_market_data`, calculates deterministic indicators in Python, then the LLM reads OHLCV and indicator results.
- `Bull/Bear Researchers`: analyst outputs only, no direct external provider calls.
- `Research Reporter`: analyst/debate outputs and warnings only.

## Guardrails

Allowed output includes market regime, key drivers, key risks, constructive/risk cases, and usage constraints. Prohibited output includes direct buy/sell instructions, exact position sizing as financial advice, order placement instructions, and guaranteed profit claims.
