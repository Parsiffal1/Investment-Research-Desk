# Investment Research Desk 当前实现说明

项目名：Investment Research Desk / 投研策略台  
Python package：`investment_research_desk`  
CLI 命令：`ird`

本文只描述当前代码已经实现的行为，不把计划中的能力写成已完成能力。

## 当前定位

Investment Research Desk 是本地 CLI-first 多 Agent 投研上下文生成系统。它面向“研究辅助”和“上下文整理”，不是交易执行系统。

明确不做：

- 不下单。
- 不管理账户、余额、仓位或订单。
- 不输出仓位 sizing。
- 不承诺收益。
- 不把报告写成投资建议。

## 技术栈

- CLI：Typer、Questionary、Rich。
- Workflow：LangGraph `StateGraph`。
- Schema：Pydantic v2。
- LLM：Ollama OpenAI-compatible `/v1/chat/completions`，默认 `qwen3:8b`。
- 测试：pytest。
- 数据源路由：统一 `route_to_vendor()`。
- LoRA：Transformers + PEFT/QLoRA，用于 Sentiment Analyst。

## CLI 行为

主入口：

```powershell
uv run ird
```

菜单包含：

- New research report
- Resume previous run
- View run history
- System check
- Exit

非交互式报告：

```powershell
uv run ird report --symbol ETH-USDT-SWAP --asset-class crypto --horizon short_term --llm-provider ollama
```

支持 `--language en|zh`，schema key 不变，仅影响报告中的人类可读字段和 Markdown section。

## Workflow

当前 graph：

```text
run_controller
  -> analyst_team
  -> bull_researcher
  -> bear_researcher
  -> bull_bear_research_debate
  -> research_reporter
  -> final_market_context_cache
  -> persist
```

`analyst_team` 包含四个 analyst：

- Fundamental / Macro Analyst
- News / Macro Impact Analyst
- Sentiment Analyst
- Technical Analyst

live + Ollama 默认 `IRD_AGENT_EXECUTION_MODE=sequential`，减少本地 Qwen3-8B 并发 timeout。fixture/fake 测试可并行。

## Tool Calling

每个 analyst 都通过自己的工具边界取数：

- Fundamental/Macro：`get_fundamentals`、`get_news`
- News/Macro Impact：`get_news`、`get_global_news`
- Sentiment：`get_sentiment_inputs`
- Technical：`get_market_data`、`get_swap_market_context`

LLM 在 tool loop 中决定是否调用工具、query 如何写、调用几次。系统负责：

- tool call budget。
- 金融 query 范围约束。
- required tool floor。
- relevance filtering。
- partial evidence 保留。

当前可配置项：

```text
IRD_AGENT_EXECUTION_MODE=sequential
IRD_LLM_TIMEOUT_SEC=180
IRD_AGENT_TOOL_LOOP_TIMEOUT_SEC=240
IRD_AGENT_MAX_TOOL_CALLS=8
IRD_REPORT_LANGUAGE=en
```

## 数据源

已实现 provider：

- OKX：public SWAP market data。
- FMP：quote/profile、新闻、OHLCV fallback。
- Finnhub：quote、news。
- Yahoo Finance：OHLCV、ticker/global news。
- Tavily：搜索增强。
- StockTwits / Reddit：情绪输入。
- Jin10：宏观新闻配置入口。
- Fixtures：测试与 demo。

FMP/Finnhub/Yahoo/OKX 这类 ticker-scoped API 优先。Tavily 用作补充搜索，不作为第一证据源。

402/403 这类免费版限制不会作为业务 warning 污染最终报告，但 provider status 中会保留失败原因。

## LoRA Adapter

当前 LoRA 只接入 Sentiment Analyst。它使用 PEFT adapter 做 forced-choice label scoring：

```text
bearish / bullish / neutral
```

启用方式：

```bash
ird report --symbol ETH-USDT-SWAP \
  --llm-provider ollama \
  --sentiment-provider hf-peft \
  --sentiment-adapter-path models/investment-research-desk-lora-sentiment/<run>/adapter
```

如果 `.env` 中设置 `IRD_SENTIMENT_PROVIDER=hf-peft` 但未设置 path，系统会自动发现 `models/investment-research-desk-lora-sentiment/<timestamp>/adapter` 下最新 adapter。

Windows 环境缺少 HF runtime 时会 preflight fail；WSL CUDA 环境可运行 adapter。

## 输出产物

每个 run 写入：

```text
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

`metrics.json` 记录 token 压缩、schema/guardrail 状态和 runtime 元信息。  
`trace.json` 记录 agent latency、warnings、completed steps。

## 当前仍需注意

- Checkpoint/resume 仍是基础实现，不是完整节点级跳过重放系统。
- 中文报告依赖 LLM 输出质量；fallback 仍可能是英文。
- adapter 运行需要 WSL CUDA 或 Windows 完整 HF runtime。
