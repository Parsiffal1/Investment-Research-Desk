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
  normalized_data.json
  analyst_outputs.json
  final_research_context.json
  research_brief.md
  trace.json
  metrics.json
```

## Guardrails

Allowed output includes market regime, key drivers, key risks, constructive/risk cases, and usage constraints. Prohibited output includes direct buy/sell instructions, exact position sizing as financial advice, order placement instructions, and guaranteed profit claims.
