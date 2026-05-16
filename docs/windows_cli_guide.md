# Windows CLI 使用指南

本文说明在 Windows 上运行 Investment Research Desk 的普通 CLI 流程。该流程不需要安装 LoRA 训练依赖。

## 1. 安装依赖

```powershell
cd <PROJECT_DIR>
uv sync
```

## 2. 配置环境变量

```powershell
Copy-Item .env.example .env
notepad .env
```

至少建议配置：

```text
IRD_OLLAMA_BASE_URL=http://localhost:11434/v1
IRD_OLLAMA_MODEL=qwen3:8b
OKX_BASE_URL=https://www.okx.com
TAVILY_API_KEY=
FMP_API_KEY=
FINNHUB_API_KEY=
```

`.env` 是本地私密文件，不要提交到 GitHub。

## 3. 启动 Ollama

```powershell
ollama serve
ollama pull qwen3:8b
ollama list
```

## 4. 检查系统

```powershell
uv run ird config check
```

该命令会检查 Ollama endpoint、模型、provider 配置和 adapter runtime 状态，但不会显示真实 API key。

## 5. 运行报告

交互式入口：

```powershell
uv run ird
```

非交互式报告：

```powershell
uv run ird report --symbol ETH-USDT-SWAP --asset-class crypto --horizon short_term --llm-provider ollama --language zh
```

离线 fixture demo：

```powershell
uv run ird demo
```

## 6. 查看产物

每次运行会写入：

```text
runs/{run_id}/
```

核心文件：

- `final_research_context.json`
- `research_brief.md`
- `trace.json`
- `metrics.json`

这些运行产物可能包含本地路径、外部 provider 返回内容和研究历史，默认不提交到 GitHub。
