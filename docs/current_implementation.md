# Investment Research Desk 当前实现说明

项目名：Investment Research Desk / 投研策略台  
Python package：`investment_research_desk`  
CLI 命令：`ird`

本文只描述当前代码已经实现的能力，不把计划中的能力写成已完成能力。

## 产品定位

Investment Research Desk 是一个本地 CLI-first 多 Agent 投研上下文生成系统。它面向研究辅助、信息整理和人工复核，不是交易执行系统。

明确不做：

- 不下单。
- 不管理账户、余额、持仓或订单。
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
- LoRA：Transformers + PEFT/QLoRA，仅用于 Sentiment Analyst。

## CLI 行为

交互式入口：

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

支持 `--language en|zh`。schema key 和 enum 保持英文，报告中的人类可读字段按语言输出。

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

live + Ollama 默认 `IRD_AGENT_EXECUTION_MODE=sequential`，用于减少本地 Qwen3-8B 并发 timeout。fixture/fake 测试路径可并行。

## Tool Calling

每个 analyst 通过自己的工具边界获取数据：

- Fundamental/Macro：`get_fundamentals`、`get_news`
- News/Macro Impact：`get_news`、`get_global_news`
- Sentiment：`get_sentiment_inputs`
- Technical：`get_market_data`、`get_swap_market_context`

LLM 在 tool loop 中决定是否调用工具、query 如何写、调用几次。系统负责：

- tool call budget
- 金融 query 范围约束
- required tool floor
- relevance filtering
- partial evidence 保留

## 数据源

已实现 provider：

- OKX public SWAP market context
- Yahoo Finance
- FMP
- Finnhub
- Tavily
- StockTwits
- Reddit
- Jin10
- Fixtures

OKX 只使用 public market endpoint，不接 account、balance、position、order。

## 报告与产物

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

最终报告包含 Directional View、Confidence、Risk Level、Market Regime、Analyst Team 摘要、Bull Case、Bear Case、Debate Conclusion、Key Evidence、Key Risks、Invalidating Conditions、Data Coverage 和 Usage Constraints。

## Sentiment LoRA

LoRA adapter 只用于 Sentiment Analyst 的情绪分类。主报告、其他 analyst 和 Bull/Bear debate 仍由主 LLM 执行。

默认 adapter 运行环境是 WSL2 + CUDA。Windows 普通 CLI 不要求安装 `torch/transformers/peft/bitsandbytes/accelerate`。

## 已知限制

- 最终 `confidence` 是启发式证据置信度，不是概率校准结果。
- `bullish/bearish` 是研究方向判断，不是交易信号。
- 免费 API endpoint 可能受 402/403 限制，系统会记录 provider 状态并降级。
- live provider 数据质量依赖外部 API 覆盖度。
