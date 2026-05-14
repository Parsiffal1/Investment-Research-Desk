# Investment Research Desk 当前实现说明

生成时间：2026-05-14  
代码基准：`006985e`  
项目名：Investment Research Desk / 投研策略台  
Python package：`investment_research_desk`  
CLI 命令：`ird`

本文档只描述当前代码已经实现的行为，不把规划中的功能写成已完成能力。

## 1. 项目定位

Investment Research Desk 是一个本地 CLI-first 的多 Agent 投研上下文生成系统。当前实现目标是：用户通过 `ird` 输入标的、周期、研究深度和模型配置后，系统调用本地 Ollama 或 fake LLM，结合多源市场数据、新闻、情绪输入、技术指标和 Bull/Bear debate，生成结构化投研上下文、控制台报告、Markdown brief、trace、metrics 和可恢复 checkpoint。

系统明确不做以下事情：

- 不输出直接买入、卖出、下单、持仓调整指令。
- 不提供仓位比例或账户资金配置建议。
- 不承诺收益或胜率。
- 不做真实交易执行。
- 不管理 OKX 账户、余额、订单或持仓。

## 2. 技术栈

当前 `pyproject.toml` 项目使用 Python，核心运行时模块如下：

- CLI：Typer、Questionary、Rich。
- Workflow：LangGraph `StateGraph`。
- Schema：Pydantic v2。
- LLM：Ollama OpenAI-compatible `/v1/chat/completions`，以及 deterministic fake LLM。
- HTTP：httpx。
- 测试：pytest。
- 数据源封装：项目内 provider 类和统一 `route_to_vendor()`。

## 3. 配置系统

配置入口位于 `investment_research_desk/config.py`。

`load_settings()` 会加载：

1. `.env`
2. `notepad.env`，但不覆盖已存在变量

当前 `Settings` 字段：

| 字段 | 默认值 | 用途 |
|---|---|---|
| `IRD_OLLAMA_BASE_URL` | `http://localhost:11434/v1` | Ollama OpenAI-compatible endpoint |
| `IRD_OLLAMA_MODEL` | `qwen3:8b` | 默认 Ollama 模型 |
| `IRD_DEFAULT_LLM_PROVIDER` | `auto` | 默认 LLM provider |
| `OKX_BASE_URL` | `https://www.okx.com` | OKX public API |
| `TAVILY_API_KEY` | 空 | Tavily 搜索 API key |
| `TAVILY_BASE_URL` | `https://api.tavily.com` | Tavily endpoint |
| `FMP_API_KEY` | 空 | Financial Modeling Prep API key |
| `FMP_BASE_URL` | `https://financialmodelingprep.com/stable` | FMP endpoint |
| `FINNHUB_API_KEY` | 空 | Finnhub API key |
| `FINNHUB_BASE_URL` | `https://finnhub.io/api/v1` | Finnhub endpoint |
| `JIN10_API_URL` | 空 | Jin10 endpoint |
| `JIN10_API_KEY` | 空 | Jin10 key |
| `IRD_RUNS_DIR` | `runs` | 运行产物目录 |
| `IRD_MARKET_DATA_VENDORS` | `okx,yahoo_finance,fmp` | 市场数据 provider 优先级 |
| `IRD_NEWS_DATA_VENDORS` | `jin10,finnhub,yahoo_finance` | 新闻 provider 优先级 |
| `IRD_SENTIMENT_DATA_VENDORS` | `tavily,stocktwits,reddit` | 情绪 provider 优先级 |
| `IRD_FUNDAMENTAL_DATA_VENDORS` | `fmp,finnhub` | 基本面 provider 优先级 |

## 4. CLI 入口和交互规则

CLI 主入口位于 `investment_research_desk/cli.py`。

### 4.1 主命令

运行：

```powershell
uv run ird
```

如果没有子命令，会进入交互式 CLI。主菜单当前包括：

- New research report
- Resume previous run
- View run history
- System check
- Exit

New research report 当前交互步骤：

1. 输入 `symbol`，例如 `NVDA`、`AAPL`、`BTC-USDT-SWAP`。
2. 选择 `horizon`：`intraday`、`short_term`、`swing`、`medium_term`。
3. 选择 `research_depth`：`quick`、`standard`、`deep`。
4. 默认使用 Ollama provider，模型来自 `IRD_OLLAMA_MODEL`。
5. 打印 Run Contract Review。
6. 用户确认后开始运行。

交互式入口不再要求用户选择 asset class。`build_run_request()` 会根据 symbol 推断：

- 以 `-SWAP` 结尾：`crypto`
- 常见 crypto ticker：`crypto`
- 其他：`equity`

该行为与 TradingAgents 风格一致：默认用户输入的是标准 symbol，不做拼写纠错或 ticker resolver。

### 4.2 非交互命令

当前命令包括：

```powershell
uv run ird report --symbol BTC-USDT-SWAP --llm-provider ollama
uv run ird report --symbol NVDA --horizon short_term --research-depth standard
uv run ird report --fixture gold_cpi --llm-provider fake
uv run ird demo --fixture gold_cpi
uv run ird batch --symbols BTC-USDT-SWAP,ETH-USDT-SWAP
uv run ird runs
uv run ird config check
uv run ird okx ...
uv run ird eval --suite schema
```

`ird eval` 命令仍存在，但按当前开发约束，evaluation/ablation 不作为近期推进重点。

### 4.3 运行中 UI

运行时使用 Rich Live dashboard：

- 顶部显示 Investment Research Desk 标识、symbol、asset class、horizon、provider。
- 左侧显示各 team/agent 状态。
- 右侧显示系统消息和工具/推理进度。
- 下方显示当前 agent report 片段。
- footer 显示 Tool Calls、Generated Reports、Elapsed。

运行完成后，控制台直接展示完整结构化报告，不只打印输出路径。

## 5. 输入和核心 Schema

核心 schema 位于 `investment_research_desk/schemas.py`。

### 5.1 RunRequest

```python
class RunRequest(BaseModel):
    symbol: str
    asset_class: AssetClass = "crypto"
    horizon: Horizon = "short_term"
    research_depth: ResearchDepth = "standard"
    run_mode: Literal["snapshot", "batch"] = "snapshot"
    fixture: str | None = None
    llm_provider: Literal["auto", "fake", "ollama"] = "auto"
    model: str | None = None
```

`research_depth` 当前直接影响 Bull/Bear debate 轮数：

- `quick`：1 轮
- `standard`：2 轮
- `deep`：3 轮

### 5.2 NormalizedData

`NormalizedData` 是所有 agent 共享的标准数据容器：

- `symbol`
- `asset_class`
- `horizon`
- `ohlcv`
- `news_events`
- `sentiment_inputs`
- `market_context`
- `source_metadata`

当前实现会把各 analyst 自己调用工具得到的数据 slice 合并到一个最终 `NormalizedData` 中，并写入 `normalized_data.json`。

### 5.3 Agent 输出 Schema

当前核心 agent 输出：

- `FundamentalMacroResult`
- `NewsImpactResult`
- `SentimentResult`
- `TechnicalState`
- `ResearchCase`
- `ResearchDebateResult`
- `FinalResearchContext`

`FinalResearchContext` 当前包含明确的 `directional_view`，取值只允许：

- `bullish`
- `bearish`

这不是下单建议，而是最终投研上下文中的方向性研究判断。

## 6. LLM 集成

LLM 代码位于 `investment_research_desk/llm/clients.py`。

### 6.1 LLMClient Protocol

当前协议要求三个方法：

- `chat_json(system, user) -> dict`
- `chat_tools_json(system, user, tools, execute_tool, max_rounds) -> dict`
- `healthcheck() -> tuple[bool, str]`

### 6.2 OllamaLLMClient

`OllamaLLMClient` 使用 OpenAI-compatible endpoint：

- chat endpoint：`{base_url}/chat/completions`
- model list endpoint：`{base_url}/models`
- temperature：`0.1`

普通结构化调用：

1. 发送 system/user。
2. 要求 `response_format={"type": "json_object"}`。
3. 尝试解析 JSON。
4. 如果解析失败，执行一次 repair retry。

工具调用：

1. 构造 OpenAI-compatible `tools`。
2. `tool_choice="auto"`。
3. LLM 返回 `tool_calls` 时，调用传入的 `execute_tool()`。
4. 把工具结果作为 `role=tool` 消息追加回上下文。
5. 循环到 `max_rounds` 或 LLM 停止工具调用。
6. 如果达到轮数上限，发送最后一条 user message，要求停止调用工具并输出 final JSON。

### 6.3 FakeLLMClient

Fake LLM 用于测试和 fixture demo：

- `chat_json()` 从 prompt 中解析 `Candidate output JSON`，能稳定返回候选结果。
- `chat_tools_json()` 会按 tools 列表前两个工具进行 deterministic 调用。
- 不依赖 Ollama 或外部 LLM。

## 7. 数据源和工具路由

统一路由位于 `investment_research_desk/dataflows/interface.py`。

### 7.1 工具类别

当前工具类别：

| 类别 | 工具 |
|---|---|
| `market_data` | `get_market_data`, `get_swap_market_context` |
| `news_data` | `get_news`, `get_global_news` |
| `sentiment_data` | `get_sentiment_inputs` |
| `fundamental_data` | `get_fundamentals` |

### 7.2 Provider 列表

当前 provider：

- OKX
- FMP
- Finnhub
- Tavily
- Jin10
- Yahoo Finance
- StockTwits
- Reddit
- Fixtures

### 7.3 route_to_vendor()

`route_to_vendor(method, settings, request)` 的行为：

1. 根据 method 找到 category。
2. 从 settings 读取该 category 的 vendor 优先级。
3. 构造所有 provider 实例。
4. 按配置优先级和 fallback chain 依次调用 provider。
5. 成功数据合并到 combined。
6. 错误被捕获、脱敏，写入 warnings。
7. 返回 `VendorRouteResult(data, status, warnings)`。

`get_market_data` 特殊处理：只要已取得有效数据就停止 fallback。

其他 list 类结果会累加合并；dict 类结果会 merge。

### 7.4 OKX 当前能力边界

OKX provider 位于 `investment_research_desk/providers/okx.py`。

当前只使用 public SWAP/market context，不做 account 管理。

已接入能力：

- `fetch_ohlcv()`：K 线。
- `fetch_swap_market_context()`：public SWAP 市场上下文。

`fetch_swap_market_context()` 当前面向 SWAP/crypto/precious_metal/commodity 场景，主要用于：

- mark price
- index ticker
- funding rate
- open interest
- orderbook imbalance
- 其他 public SWAP context

### 7.5 FMP 和 Finnhub

FMP provider 当前能力：

- `quote(symbol)`
- `profile(symbol)`
- `fetch_ohlcv(request)`

FMP fundamentals 只在 asset class 属于 `equity`、`equity_index`、`other` 时返回。

Finnhub provider 当前能力：

- `quote(symbol)`
- `fetch_news(request)`
- `fetch_global_news(request)`

Finnhub fundamentals 只在 asset class 属于 `equity`、`other` 时返回。

### 7.6 Tavily、Yahoo、StockTwits、Reddit、Jin10

当前用途：

- Tavily：news events、sentiment inputs。
- Yahoo Finance：OHLCV、ticker news、global news。
- StockTwits：sentiment inputs。
- Reddit：sentiment inputs。
- Jin10：news、global news；需要配置 URL/key 才能正常使用。

## 8. Agent Contract

Agent contract 位于 `investment_research_desk/agents/contracts.py`。

每个 agent 定义：

- `name`
- `team`
- `role`
- `allowed_inputs`
- `allowed_tools`
- `forbidden_actions`
- `output_schema`
- `system_prompt`

所有 agent 共用禁止行为：

- 不发布直接 buy/sell/short/hold 指令。
- 不提供下单语言。
- 不提供仓位 sizing。
- 不保证盈利或收益。
- 不把研究上下文当作 financial advice。

当前 agent：

| Agent | Team | 输出 |
|---|---|---|
| `run_controller` | controller | `WorkflowState` |
| `fundamental_macro` | analyst | `FundamentalMacroResult` |
| `news_impact` | analyst | `NewsImpactResult` |
| `sentiment` | analyst | `SentimentResult` |
| `technical` | analyst | `TechnicalState` |
| `analyst_team` | analyst | `analyst_team_outputs.json` |
| `bull_researcher` | research | `ResearchCase` |
| `bear_researcher` | research | `ResearchCase` |
| `bull_bear_research_debate` | research | `research_debate.json` |
| `research_reporter` | reporting | `FinalResearchContext` |
| `final_market_context_cache` | cache | `final_market_context_cache.json` |
| `persist` | cache | run artifacts |

Prompts 位于 `investment_research_desk/agents/prompts.py`，contract 引用这些 system prompt。

## 9. Workflow 实现

Workflow 位于 `investment_research_desk/graph/workflow.py`。

### 9.1 LangGraph 拓扑

当前 LangGraph 节点顺序：

```text
START
  -> run_controller
  -> analyst_team
  -> bull_researcher
  -> bear_researcher
  -> bull_bear_research_debate
  -> research_reporter
  -> final_market_context_cache
  -> persist
  -> END
```

代码使用 `StateGraph(WorkflowState)`，每个节点通过 `_run_step()` 包裹，以统一处理：

- contract 检查
- progress event
- latency 记录
- warning 捕获
- checkpoint 保存
- trace append

### 9.2 Run Controller

`run_controller` 的职责：

- 如果 request 有 fixture，加载 fixture data，并将 `provider_mode` 设为 `fixture`。
- 如果不是 fixture，创建空 `NormalizedData` seed，标记 `provider_mode=live`。
- 写入初始 `normalized_data.json`。

注意：当前已删除单独的 Data Ingestion 节点，数据收集由 analyst 自主 tool loop 执行。

### 9.3 Analyst Team 并行执行

`_analyst_team()` 调用 `_run_analysts_parallel()`。

并行技术：

- Python `ThreadPoolExecutor`
- 每个 analyst 是一个 future
- 收集每个 agent 的 output、data slice、trace
- 最后 `_merge_agent_data()` 合并数据

并行执行的四个 analyst：

- `fundamental_macro`
- `news_impact`
- `sentiment`
- `technical`

`analyst_team` 汇总后写入：

- `analyst_outputs.json`
- `analyst_team_outputs.json`

### 9.4 TradingAgents-style LLM Tool Loop

当前 live 路径已经改成 TradingAgents 风格的 agent-level tool loop。入口为 `_run_agent_tool_loop()`。

流程：

1. 根据 agent 名称取得 contract。
2. 构造该 agent 可用工具列表。
3. 生成 tool-loop prompt，要求 LLM 在调用工具前细化 query/symbol。
4. 调用 `llm.chat_tools_json()`。
5. 每次 LLM 返回 tool call 时，由 workflow 执行 `_execute_agent_tool()`。
6. 工具执行仍统一通过 `route_to_vendor()`。
7. 如果 LLM 没有调用 required tool，则执行 contract floor call。
8. 汇总为该 agent 的 `NormalizedData` data slice。

当前每个 agent 的 live 工具配置：

| Agent | 工具 | Required |
|---|---|---|
| `fundamental_macro` | `get_fundamentals`, `get_news` | `get_fundamentals` |
| `news_impact` | `get_news`, `get_global_news` | `get_news` |
| `sentiment` | `get_sentiment_inputs` | `get_sentiment_inputs` |
| `technical` | `get_market_data`, `get_swap_market_context` | `get_market_data` |

工具调用边界：

- agent 总工具预算：`max_rounds * len(tool_names)`。
- 单个工具最多 4 次。
- 超出预算返回 error payload，不继续实际请求。

每个 data slice 的 metadata 会记录：

- `tool_call_policy=tradingagents_style_llm_tool_loop`
- `agent_tool_status`
- `warnings`
- `llm_tool_calls`
- `contract_floor_calls`
- `tool_call_budget`

Fixture 路径不会调用外部工具，而是从 fixture data 按 agent contract scope 切片。

### 9.5 Analyst 输出逻辑

#### Fundamental / Macro Analyst

输入：

- fundamental metadata
- quote/profile
- candidate news events
- ranked news relevance

fallback 逻辑：

- 从 FMP/Finnhub quote/profile 提取公司、行业、涨跌幅。
- 从新闻标题里识别 safe-haven、CPI/inflation、dollar 等宏观线索。
- 输出 drivers、concerns、evidence、confidence。

随后通过 `_llm_structured()` 调用 LLM，在 schema 约束下 refine。

#### News / Macro Impact Analyst

输入：

- LLM tool loop 收集到的 news/global news
- relevance ranking

fallback 逻辑：

- 根据标题/摘要中的 bullish/bearish 词计数。
- 构造 `dominant_events`、`asset_impact`、`impact_logic`。

LLM 输出必须符合 `NewsImpactResult`。

#### Sentiment Analyst

输入：

- Tavily/StockTwits/Reddit 等 sentiment inputs。

fallback 逻辑：

- 统计 bullish/bearish terms。
- 生成 `sentiment_score`、`sentiment_label`、`crowd_mood`。

LLM 进一步读取结构化 source blocks，区分新闻、社交、社区讨论和噪音。

#### Technical Analyst

输入：

- OHLCV
- OKX public SWAP market context

确定性指标：

- RSI
- MACD
- ATR
- Bollinger state
- realized volatility
- max drawdown
- support/resistance
- trend label

技术指标由 Python 计算，LLM 不重新计算，只解释结果。`_preserve_deterministic_fields()` 会保护数值字段，防止 LLM 改写 RSI、ATR、funding、open interest 等确定性数据。

### 9.6 Bull/Bear Research Debate

当前 research 阶段包括：

1. `bull_researcher`
2. `bear_researcher`
3. `bull_bear_research_debate`

`bull_researcher` 基于四个 analyst 输出构造 constructive case。

`bear_researcher` 基于同样 analyst 输出和 bull case 构造 risk case。

`bull_bear_research_debate` 会按 `research_depth` 做多轮交替：

- 第 1 轮来自前面两个节点已经生成的 bull/bear case。
- 第 2 轮起，在 debate node 内继续调用 `ConstructiveCaseAnalyst.run()` 和 `RiskCaseAnalyst.run()`。
- 新一轮 bull 会看到 `debate_history` 和上一轮 bear case。
- 新一轮 bear 会看到 `debate_history` 和最新 bull case。
- 每轮都写入 trace，名称如 `bull_researcher_round_2`。

最终 `DebateModerator` 生成：

- `points_of_agreement`
- `key_tensions`
- `evidence_quality_notes`
- `reporter_handoff`
- `confidence`

`research_debate.json` 包含：

- `round_count`
- `rounds`
- latest bull/bear researcher case
- moderator summary

### 9.7 Research Reporter

`ResearchReporter` 输入：

- `NormalizedData`
- 四个 analyst 输出
- bull case
- bear case
- research debate
- warnings

fallback 逻辑：

- 用 `_balance_view()` 综合 news、sentiment、technical。
- 用 `_directional_view()` 输出 `bullish` 或 `bearish`。
- 用 `_risk_level()` 输出 `low/medium/high`。
- 合成 final summary、drivers、risks、uncertainty factors。

随后调用 LLM 生成 `FinalResearchContext`。

当前为避免 provider 原文污染 final guardrail，`FinalResearchContext.source_metadata` 不再保存 raw tool result，而是保存摘要化 metadata：

- provider mode
- tool policy
- agent tool status
- warnings
- tool call budget
- summarized llm tool calls
- summarized contract floor calls

原始/半结构化数据仍保存在 `normalized_data.json`。

### 9.8 Final Market Context Cache

`final_market_context_cache` 节点写入：

- `final_market_context_cache.json`

内容包括：

- cache key：`symbol:asset_class:horizon`
- final context
- analyst team summary
- research debate summary
- usage boundary

### 9.9 Persist

`persist` 节点负责最终落盘：

- `final_research_context.json`
- `research_brief.md`
- `trace.json`
- `metrics.json`

metrics 当前包含：

- total latency
- raw input tokens
- final context tokens
- compression ratio
- JSON/schema valid flag
- guardrail violations

Guardrail 检查对象是：

- final context JSON
- 完整 Markdown report

## 10. 文件产物

每次运行目录格式：

```text
runs/{run_id}/
```

`run_id` 由 UTC 时间、symbol 和 8 位 uuid 组成，例如：

```text
20260514T162448Z_btc-usdt-swap_91c47b95
```

当前 required artifacts：

| 文件 | 说明 |
|---|---|
| `input.json` | 原始 RunRequest |
| `agent_contracts.json` | 本次运行使用的 agent contract manifest |
| `normalized_data.json` | 合并后的标准化数据，包含 raw/semi-structured tool results |
| `analyst_outputs.json` | 四个 analyst 输出 |
| `analyst_team_outputs.json` | Analyst Team handoff |
| `bull_risk_outputs.json` | 当前最新 bull/bear case |
| `research_debate.json` | 多轮 debate 记录和 moderator 总结 |
| `final_market_context_cache.json` | 下游缓存格式 |
| `final_research_context.json` | 最终结构化投研上下文 |
| `research_brief.md` | Markdown 报告 |
| `trace.json` | agent/node trace |
| `metrics.json` | token、压缩、guardrail 等指标 |
| `checkpoint.json` | 仅在启用 `--checkpoint` 时存在 |

## 11. Checkpoint 和 Resume

`RunStore` 位于 `investment_research_desk/persistence.py`。

Checkpoint 行为：

- `_run_step()` 在节点成功后，如果 `checkpoint_enabled=True`，写入 `checkpoint.json`。
- `persist` 节点不再写 checkpoint。
- resume 时通过 `RunStore.load_checkpoint(run_id)` 读取状态。
- 已完成节点会被 `_run_step()` 跳过。

CLI resume：

```powershell
uv run ird report --resume RUN_ID
```

清理 checkpoint：

```powershell
uv run ird report --clear-checkpoints ...
```

或通过交互式菜单选择 Resume previous run。

## 12. Guardrail

Guardrail 逻辑位于 `investment_research_desk/tools/guardrails.py`。

当前禁止模式包括：

- `direct_buy`
- `direct_sell`
- `position_sizing`
- `guaranteed_profit`
- `order_instruction`

还要求最终文本包含：

```text
Use as research context only
```

如果违反，`metrics.guardrail_violations` 会记录违规项，CLI summary 也会显示 warnings。

当前限制：guardrail 是 regex 规则，不是完整安全分类器。它能挡住明确禁用短语，但不能理解所有上下文。后续仍需要 evidence cleaning 和外部文本归因策略。

## 13. 测试覆盖

测试位于 `tests/`。

当前测试覆盖：

- agent contract 注册和 schema。
- CLI help/config/report smoke。
- dataflow routing。
- FMP provider free endpoint 行为。
- guardrail prohibited wording。
- technical indicators。
- LLM fake/JSON behavior。
- fixture workflow end-to-end。
- checkpoint/resume。
- live analyst tool calling mock。
- TradingAgents-style tool loop metadata。
- multi-round debate artifact。

最近一次验证：

```powershell
uv run pytest
```

结果：

```text
28 passed
```

## 14. 当前实现中的重要设计取舍

### 14.1 Analyst 并行，Research Debate 顺序

四个 analyst 当前用 `ThreadPoolExecutor` 并行执行，以缩短数据收集和初步分析时间。

Bull/Bear debate 是顺序交替执行，因为后续发言依赖前一轮 debate history。

### 14.2 Tool calling 是 LLM-driven，但工具执行由 workflow 控制

LLM 决定：

- 是否调用工具
- 调哪个工具
- query/symbol 怎么写
- 什么时候停止

Workflow 控制：

- 工具白名单
- 最大调用次数
- provider 路由
- 错误处理
- warnings 脱敏
- 结果合并

这符合 TradingAgents 风格，同时保留本项目的 provider abstraction。

### 14.3 技术指标由 Python 计算，LLM 解释

Technical Analyst 不让 LLM 自行计算指标，因为指标计算应可复现。LLM 只读取指标结果和 OKX public SWAP context，并生成解释。

### 14.4 Raw data 和 final context 分离

`normalized_data.json` 保存原始/半结构化工具结果，便于审计。

`final_research_context.json` 只保存最终投研上下文和来源摘要，避免 raw provider 广告文本、交易计划文本污染最终 guardrail。

## 15. 当前限制和待改进点

以下是当前代码真实存在的限制：

1. `asset_class` 推断很简单，只按 `-SWAP`、常见 crypto ticker 和默认 equity 判断。
2. 没有 ticker spell correction，也没有 symbol resolver。
3. News 和 sentiment 仍可能引入噪音，当前靠 relevance ranking、LLM 分析和 final guardrail 缓解，但 evidence cleaning 还不够强。
4. Sentiment evidence 可能包含较长外部原文，后续应做摘要化和广告/交易计划过滤。
5. Guardrail 是 regex，不是语义级审核。
6. Evaluation suite 命令存在，但当前开发策略中暂不推进 ablation/对照评测。
7. CLI 中文在某些 Windows code page 下可能出现乱码，因为 console encoding 依赖终端环境。
8. OKX 当前只使用 public market/SWAP context，不做 account、position、order。
9. 当前 LangGraph 拓扑是主流程顺序图；analyst 内部并行由 Python thread pool 实现，不是 LangGraph 多分支 conditional graph。
10. LLM tool loop 使用 OpenAI-compatible tool calls；如果本地 Ollama 模型对 tool calling 支持不稳定，可能触发 contract floor calls 或 warnings。

## 16. 关键文件索引

| 文件 | 作用 |
|---|---|
| `investment_research_desk/cli.py` | Typer CLI、交互式菜单、Rich dashboard、控制台报告 |
| `investment_research_desk/cli_contract.py` | CLI 输入 contract、enum、run request 构造、run discovery |
| `investment_research_desk/config.py` | `.env` 配置加载 |
| `investment_research_desk/schemas.py` | Pydantic 数据结构 |
| `investment_research_desk/llm/clients.py` | Ollama/Fake LLM 和 tool calling |
| `investment_research_desk/dataflows/interface.py` | 统一 provider routing |
| `investment_research_desk/providers/*.py` | 各数据源 provider |
| `investment_research_desk/agents/contracts.py` | Agent contract |
| `investment_research_desk/agents/prompts.py` | Agent system prompts |
| `investment_research_desk/agents/core.py` | Agent fallback、LLM structured output、reporter |
| `investment_research_desk/graph/workflow.py` | LangGraph workflow、tool loop、debate、persist |
| `investment_research_desk/persistence.py` | run artifact 和 checkpoint 写入 |
| `investment_research_desk/tools/indicators.py` | 技术指标 |
| `investment_research_desk/tools/guardrails.py` | guardrail regex |
| `tests/` | 当前测试套件 |

