# WSL + LoRA Adapter 使用指南

本文面向已安装 WSL2 Ubuntu、CUDA 可见、并已存在 `/home/parsiffal/.venvs/ird-lora` 的本机环境。

## 1. 启动 Windows Ollama Bridge

WSL 里的 `localhost` 不是 Windows 的 `localhost`。先在 Windows PowerShell 中启动一个 WSL 可访问的 Ollama 端口：

```powershell
cd C:\Users\saton\Documents\Codex\2026-05-14\files-mentioned-by-the-user-finsight
.\scripts\wsl\start_ollama_bridge.ps1
```

默认监听：

```text
0.0.0.0:11435
```

WSL 内访问：

```text
http://172.24.48.1:11435/v1
```

## 2. 验证 WSL Runtime

```bash
cd /mnt/c/Users/saton/Documents/Codex/2026-05-14/files-mentioned-by-the-user-finsight
source /home/parsiffal/.venvs/ird-lora/bin/activate
python -m pytest
```

检查 GPU：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 3. 运行带 Adapter 的报告

快捷脚本：

```powershell
wsl -d Ubuntu -- bash scripts/wsl/run_adapter_report.sh ETH-USDT-SWAP
```

脚本默认设置：

```text
IRD_OLLAMA_BASE_URL=http://172.24.48.1:11435/v1
HF_HOME=/mnt/d/saton/ird_hf_cache
IRD_AGENT_EXECUTION_MODE=sequential
adapter=models/investment-research-desk-lora-sentiment/20260515T123418Z/adapter
```

如需覆盖 adapter：

```powershell
$env:IRD_SENTIMENT_ADAPTER_PATH="models/investment-research-desk-lora-sentiment/<timestamp>/adapter"
wsl -d Ubuntu -- bash scripts/wsl/run_adapter_report.sh ETH-USDT-SWAP
```

## 4. 判断 Adapter 是否生效

报告完成后检查 `analyst_outputs.json` 或控制台 Sentiment Analyst 区域。生效时 evidence 会出现：

```text
adapter_classification=neutral
adapter_classification=bullish
adapter_classification=bearish
```

`metrics.json` 的 `runtime.sentiment_runtime` 会记录：

```text
provider=hf-peft
base_model=Qwen/Qwen3-8B
adapter_path=...
method=forced_choice_label_scoring
```

## 5. 注意事项

- adapter 只影响 Sentiment Analyst，不替代主报告 LLM。
- 首次加载 Qwen3-8B + adapter 会较慢，热启动会快很多。
- 不要让 Windows 和 WSL 共用同一个 `.venv`。
- 如果 WSL 访问不到 Ollama，先确认 Windows bridge 端口 `11435` 已启动。
