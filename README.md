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

The baseline model target is Qwen3-8B Instruct/Chat. LoRA integration is intentionally separate from the MVP path; do not report improvement metrics until evaluation suites produce measured results.

## CLI

- `ird` starts an interactive research flow.
- `ird report` generates one research context.
- `ird batch` runs multiple symbols.
- `ird eval` runs evaluation suites.
- `ird config check` validates runtime configuration.

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

The MVP enforces tool boundaries by scoping each analyst's input data before execution:

- `Fundamental/Macro Analyst`: fundamental metadata, quote metadata, and macro/news context.
- `News/Macro Impact Analyst`: news events only.
- `Sentiment Analyst`: sentiment inputs only.
- `Technical Analyst`: OHLCV and deterministic indicator results only; indicators are calculated by Python, then read by the LLM.
- `Bull/Bear Researchers`: analyst outputs only, no direct external provider calls.
- `Research Reporter`: analyst/debate outputs and warnings only.

## Guardrails

Allowed output includes market regime, key drivers, key risks, constructive/risk cases, and usage constraints. Prohibited output includes direct buy/sell instructions, exact position sizing as financial advice, order placement instructions, and guaranteed profit claims.
