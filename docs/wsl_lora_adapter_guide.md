# WSL + LoRA Adapter 使用指南

本文面向已经安装 WSL2 Ubuntu、CUDA 可见，并已经创建 LoRA runtime venv 的本地环境。

## 1. 准备变量

推荐通过环境变量传入本机路径，避免把个人路径写入脚本：

```bash
export IRD_PROJECT_DIR=<PROJECT_DIR>
export IRD_LORA_VENV=<WSL_VENV>
export HF_HOME=<HF_CACHE_DIR>
export IRD_SENTIMENT_ADAPTER_PATH=<ADAPTER_PATH>
```

示例占位符含义：

- `<PROJECT_DIR>`：项目根目录。
- `<WSL_VENV>`：WSL 内 Python venv，例如 `$HOME/.venvs/ird-lora`。
- `<HF_CACHE_DIR>`：Hugging Face cache 目录。
- `<ADAPTER_PATH>`：PEFT adapter 目录。

## 2. 启动 Windows Ollama Bridge

WSL 里的 `localhost` 不一定指向 Windows 的 `localhost`。如果 Ollama 跑在 Windows 上，先在 Windows PowerShell 启动 bridge：

```powershell
cd <PROJECT_DIR>
.\scripts\wsl\start_ollama_bridge.ps1
```

默认监听：

```text
0.0.0.0:11435
```

WSL 内使用：

```bash
export IRD_OLLAMA_BASE_URL=http://<WINDOWS_HOST_IP>:11435/v1
```

## 3. 验证 WSL Runtime

```bash
cd <PROJECT_DIR>
source <WSL_VENV>/bin/activate
python -m pytest
```

检查 GPU：

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. 运行带 Adapter 的报告

```bash
cd <PROJECT_DIR>
bash scripts/wsl/run_adapter_report.sh ETH-USDT-SWAP
```

也可以显式指定：

```bash
export IRD_SENTIMENT_ADAPTER_PATH=models/investment-research-desk-lora-sentiment/<timestamp>/adapter
export IRD_SENTIMENT_SCORE_BATCH_SIZE=4
bash scripts/wsl/run_adapter_report.sh ETH-USDT-SWAP
```

## 5. 判断 Adapter 是否生效

报告完成后检查 `analyst_outputs.json`、`metrics.json` 或控制台 Sentiment Analyst 区域。生效时 evidence 会出现：

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

## 6. 注意事项

- adapter 只影响 Sentiment Analyst，不替代主报告 LLM。
- 首次加载 Qwen3-8B + adapter 会比较慢，热启动会更快。
- 不要让 Windows 和 WSL 共用同一个 `.venv`。
- 如果 WSL 访问不到 Ollama，先确认 Windows bridge 端口已启动。
