# Investment Research Desk / 投研策略台

Investment Research Desk 是一个本地 CLI-first 多 Agent 投研上下文生成系统。它把市场数据、新闻/宏观事件、情绪输入、技术指标和 Bull/Bear debate 整理成结构化投研材料，供人工复核和下游策略研究使用。

本项目不是交易执行系统：不下单、不管理账户或持仓、不给仓位比例、不承诺收益。

## Quick Start

Windows 普通运行：

```powershell
uv sync
uv run ird --help
uv run ird config check
uv run ird report --symbol NVDA --horizon short_term --llm-provider ollama
```

交互式入口：

```powershell
uv run ird
```

fixture demo：

```powershell
uv run ird demo
```

API key 放在项目根目录 `.env`，可从 `.env.example` 复制：

```powershell
Copy-Item .env.example .env
notepad .env
```

核心配置：

```text
IRD_OLLAMA_BASE_URL=http://localhost:11434/v1
IRD_OLLAMA_MODEL=qwen3:8b
OKX_BASE_URL=https://www.okx.com
TAVILY_API_KEY=
FMP_API_KEY=
FINNHUB_API_KEY=
```

## CLI

- `ird`：菜单式交互研究流程。
- `ird report`：非交互式单标的报告。
- `ird batch`：批量报告。
- `ird runs`：查看 run 目录和 checkpoint。
- `ird config check`：检查 Ollama、provider、adapter runtime。
- `ird okx check`：检查 OKX public SWAP market data。
- `ird lora ...`：情绪分类 LoRA 数据准备、训练、评估。

例子：

```powershell
uv run ird report --symbol ETH-USDT-SWAP --asset-class crypto --horizon short_term --llm-provider ollama
uv run ird report --symbol SPY --horizon short_term --llm-provider ollama --language zh
```

每次运行写入：

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

## Workflow

当前工作流：

```text
Run Controller
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

live + Ollama 默认使用 `IRD_AGENT_EXECUTION_MODE=sequential`，优先保证本地 Qwen3-8B 稳定运行。fixture/fake 测试仍可并行。

每个 analyst 都通过自己的工具边界获取数据。LLM 决定是否调用工具、如何优化 query、调用几次；系统负责预算、金融范围约束、relevance filtering 和必要的 contract floor。

## Data Sources

已接入的数据源：

- OKX：public SWAP K 线、mark/index、funding、open interest、order book 等。只使用 public market endpoint，不接 account/balance/position/order。
- FMP：quote/profile、免费版可用范围内的新闻和 OHLCV fallback。
- Finnhub：quote、company/general news。
- Yahoo Finance：equity OHLCV 和 ticker news。
- Tavily：搜索增强，作为补充来源。
- StockTwits / Reddit：情绪输入。
- Jin10：宏观新闻接口配置入口。
- Fixtures：稳定测试和 demo。

Provider 402/403 会记录在 provider status 中，但不会作为业务 warning 污染最终报告。

## Sentiment LoRA

第一阶段 LoRA 只用于 Sentiment Analyst 的金融情绪分类。主报告、其他 analyst、Bull/Bear debate 仍由 `--llm-provider` 指定的主模型执行。

训练和 adapter 运行建议放在 WSL2 CUDA 环境：

```bash
bash scripts/wsl/setup_lora_env.sh
bash scripts/wsl/run_lora_pipeline.sh smoke
```

带 adapter 运行：

```powershell
.\scripts\wsl\start_ollama_bridge.ps1
wsl -d Ubuntu -- bash scripts/wsl/run_adapter_report.sh ETH-USDT-SWAP
```

也可以显式指定：

```bash
ird report \
  --symbol ETH-USDT-SWAP \
  --asset-class crypto \
  --horizon short_term \
  --llm-provider ollama \
  --model qwen3:8b \
  --sentiment-provider hf-peft \
  --sentiment-adapter-path models/investment-research-desk-lora-sentiment/<run>/adapter
```

如果 `.env` 设置 `IRD_SENTIMENT_PROVIDER=hf-peft` 但没有提供 adapter path，系统会尝试自动发现 `models/investment-research-desk-lora-sentiment/<timestamp>/adapter` 下最新 adapter。Windows 环境缺少 `torch/transformers/peft/bitsandbytes/accelerate` 时会 preflight fail，不会静默降级。

## Tests

```powershell
uv run pytest
```

WSL training/runtime 环境：

```bash
cd /mnt/c/Users/saton/Documents/Codex/2026-05-14/files-mentioned-by-the-user-finsight
source /home/parsiffal/.venvs/ird-lora/bin/activate
python -m pytest
```

## Documentation

- `docs/current_implementation.md`：当前真实实现说明。
- `docs/windows_cli_guide.md`：Windows 普通 CLI 使用指南。
- `docs/wsl_lora_adapter_guide.md`：WSL + LoRA adapter 使用指南。
- `docs/lora_training_wsl.md`：LoRA 训练环境和流程。
