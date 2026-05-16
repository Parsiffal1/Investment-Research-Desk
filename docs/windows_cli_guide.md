# Windows CLI 使用指南

本文面向普通投研报告运行，不启用 HF PEFT adapter。

## 环境检查

```powershell
cd C:\Users\saton\Documents\Codex\2026-05-14\files-mentioned-by-the-user-finsight
uv sync
uv run ird config check
```

如果 Ollama 未启动：

```powershell
ollama serve
ollama pull qwen3:8b
```

`.env` 至少建议包含：

```text
IRD_OLLAMA_BASE_URL=http://localhost:11434/v1
IRD_OLLAMA_MODEL=qwen3:8b
IRD_AGENT_EXECUTION_MODE=sequential
IRD_LLM_TIMEOUT_SEC=180
```

## 运行报告

交互式：

```powershell
uv run ird
```

非交互式：

```powershell
uv run ird report --symbol NVDA --horizon short_term --llm-provider ollama
uv run ird report --symbol ETH-USDT-SWAP --asset-class crypto --horizon short_term --llm-provider ollama
uv run ird report --symbol SPY --horizon short_term --llm-provider ollama --language zh
```

## 输出目录

运行结束后查看：

```powershell
uv run ird runs
```

每个 run 会生成 `final_research_context.json`、`research_brief.md`、`trace.json`、`metrics.json` 等产物。

## 常见问题

如果 `uv run pytest` 报 `.venv\lib64` 权限错误，说明当前目录曾被 WSL 创建过 Linux venv。删除或移走项目根目录 `.venv` 后重新运行 `uv run pytest`，让 Windows 重新创建自己的 venv。

如果启用 `--sentiment-provider hf-peft` 后报缺少 `torch/transformers/peft/bitsandbytes/accelerate`，这是预期行为：Windows 普通环境默认不运行 adapter。请改用 WSL adapter 指南，或显式使用 `--sentiment-provider main`。
