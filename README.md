# Investment Research Desk / 投研策略台

Investment Research Desk is a local CLI-first multi-agent research system that turns market data, news, macro context, sentiment inputs, technical indicators, and bull/bear debate into structured investment research context. It is designed for research assistance and human review, not trade execution.

投研策略台是一个本地 CLI-first 多 Agent 投研上下文生成系统，用于把市场数据、新闻与宏观事件、情绪输入、技术指标和 Bull/Bear debate 整理成结构化投研材料，供人工复核和后续策略研究使用。它不是交易执行系统。

## What It Is Not

This project does not place orders, manage accounts, read balances, manage positions, provide position sizing, promise returns, or provide financial advice. The output is research context only.

本项目不下单、不管理账户或持仓、不读取账户余额、不输出仓位 sizing、不承诺收益，也不构成投资建议。所有输出仅作为投研上下文。

## Features

- Menu-style and scriptable CLI through `ird`.
- LangGraph-based multi-agent workflow.
- Analyst team for fundamental/macro, news impact, sentiment, and technical analysis.
- Bull/Bear research debate and final research reporter.
- ReAct-style agent tool loops with tool budgets and relevance filtering.
- OKX public SWAP market context, Yahoo Finance, FMP, Finnhub, Tavily, StockTwits, Reddit, Jin10, and fixture fallback.
- Structured Pydantic outputs, run artifacts, traces, metrics, and guardrails.
- Optional Qwen3 sentiment LoRA adapter through Hugging Face PEFT.
- English and Chinese report modes.

## Screenshots

### Interactive CLI Workflow

![Interactive CLI workflow](docs/assets/screenshots/cli-interactive-menu.png)

### Live Multi-Agent Progress

![Live multi-agent progress](docs/assets/screenshots/cli-live-progress.png)

### Final Research Context Report

![Final research context report](docs/assets/screenshots/cli-final-report.png)

## Quick Start

```powershell
uv sync
Copy-Item .env.example .env
notepad .env
uv run ird config check
uv run pytest
```

Interactive CLI:

```powershell
uv run ird
```

Run a report:

```powershell
uv run ird report --symbol ETH-USDT-SWAP --asset-class crypto --horizon short_term --llm-provider ollama --language zh
```

Run an offline fixture demo:

```powershell
uv run ird demo
```

## Configuration

Common `.env` settings:

```text
IRD_OLLAMA_BASE_URL=http://localhost:11434/v1
IRD_OLLAMA_MODEL=qwen3:8b
IRD_DEFAULT_LLM_PROVIDER=auto

OKX_BASE_URL=https://www.okx.com
TAVILY_API_KEY=
FMP_API_KEY=
FINNHUB_API_KEY=
JIN10_API_URL=
JIN10_API_KEY=

IRD_AGENT_EXECUTION_MODE=sequential
IRD_LLM_TIMEOUT_SEC=180
IRD_AGENT_TOOL_LOOP_TIMEOUT_SEC=240
IRD_AGENT_MAX_TOOL_CALLS=8
IRD_REPORT_LANGUAGE=en
```

Do not commit `.env`. Use `.env.example` for shareable configuration.

## CLI

```text
ird                         Interactive menu
ird report                  Single research report
ird batch                   Batch reports
ird runs                    List run artifacts and checkpoints
ird config check            Environment and provider preflight
ird okx check               OKX public SWAP market data check
ird eval                    Lightweight regression/evaluation suites
ird lora prepare-data       Prepare sentiment LoRA data
ird lora train              Train sentiment LoRA adapter
ird lora eval               Evaluate sentiment LoRA adapter
```

## Workflow

```text
Run Controller
  -> Analyst Team
     -> Fundamental / Macro Analyst
     -> News / Macro Impact Analyst
     -> Sentiment Analyst
     -> Technical Analyst
  -> Bull/Bear Research Debate
     -> Bull Researcher
     -> Bear Researcher
     -> Debate Moderator
  -> Research Reporter
  -> final_market_context_cache
  -> persist artifacts
```

Live Ollama runs default to `IRD_AGENT_EXECUTION_MODE=sequential` to keep local Qwen3-8B execution stable. Fixture and fake-LLM test paths can still run in parallel.

Each agent owns its tool boundary. The LLM decides whether to call tools, which query to use, how many times to call, and when to stop. The system enforces budgets, financial-scope query constraints, relevance filtering, required tool floors, and partial-evidence fallback.

## Data Sources

- OKX: public SWAP OHLCV, mark/index price, funding, open interest, recent trades, and order book context. Account, balance, position, and order endpoints are intentionally out of scope.
- Yahoo Finance: equity OHLCV and ticker news fallback.
- FMP: quote/profile/news and free-tier compatible fallback.
- Finnhub: quote and company/general news.
- Tavily: search enrichment.
- StockTwits and Reddit: sentiment inputs.
- Jin10: macro/news adapter when configured.
- Fixtures: stable offline tests and demos.

Provider errors such as free-tier `402/403` responses are recorded in provider status and traces, but they are not promoted into final business warnings.

## Run Artifacts

Each run writes:

```text
runs/{run_id}/
  input.json
  agent_contracts.json
  normalized_data.json
  analyst_outputs.json
  analyst_team_outputs.json
  bull_risk_outputs.json
  research_debate.json
  final_market_context_cache.json
  final_research_context.json
  research_brief.md
  trace.json
  metrics.json
```

Run artifacts are ignored by git because they may contain provider outputs, local paths, and research history.

## Sentiment LoRA Adapter

The first adapter is limited to Sentiment Analyst classification. It does not replace the main report LLM, the analyst team, or the Bull/Bear debate.

Training and adapter runtime are intended for a WSL2 + CUDA environment:

```bash
bash scripts/wsl/setup_lora_env.sh
bash scripts/wsl/run_lora_pipeline.sh smoke
```

Run with adapter:

```bash
export IRD_SENTIMENT_ADAPTER_PATH=models/investment-research-desk-lora-sentiment/<timestamp>/adapter
bash scripts/wsl/run_adapter_report.sh ETH-USDT-SWAP
```

If the adapter is published with the repository, `adapter_model.safetensors` is tracked through Git LFS.

## Documentation

- `docs/current_implementation.md`: current implemented behavior.
- `docs/windows_cli_guide.md`: Windows CLI usage.
- `docs/wsl_lora_adapter_guide.md`: WSL + LoRA adapter usage.
- `docs/lora_training_wsl.md`: LoRA training workflow.

## Tests

```powershell
uv run pytest
```

WSL runtime:

```bash
cd <PROJECT_DIR>
source <WSL_VENV>/bin/activate
python -m pytest
```

## License

MIT License. See `LICENSE`.

## Disclaimer

Investment Research Desk is research software. It generates structured context for human review. It is not a broker, exchange, financial advisor, trading system, or execution engine.
