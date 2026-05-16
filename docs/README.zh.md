[English](README.md) | [中文](README.zh.md)

# 文档导航

这个目录保存的是 **Investment Research Desk / 投研策略台** 的长文档说明。

如果你是从仓库首页点进来的，这一页就是最快的文档总入口。

## 按你的目标来读

### 我想先知道这个项目现在到底实现了什么
- [`current_implementation.md`](current_implementation.md)
  - 这是最准确的“当前实现状态”说明
  - 会讲清 workflow、tool 边界、约束和当前限制

### 我想在普通机器上跑常规 CLI
- [`windows_cli_guide.md`](windows_cli_guide.md)
  - 常规本地 CLI 路径
  - 包括环境配置、Ollama 启动、config check、报告运行和 demo 模式

### 我想在 WSL 中启用可选 adapter 路径
- [`wsl_lora_adapter_guide.md`](wsl_lora_adapter_guide.md)
  - 说明如何在 WSL 中用训练好的 adapter 跑报告
  - 包括 Ollama bridge 和运行时验证步骤

### 我想训练 QLoRA adapter 路径
- [`lora_training_wsl.md`](lora_training_wsl.md)
  - WSL2 + CUDA 训练流程
  - 包括数据源、split hygiene、产物结构和发布边界

## 这些文档之间是什么关系

```text
README.md / README.zh.md
  -> docs/README.md / docs/README.zh.md
     -> current_implementation.md
     -> windows_cli_guide.md
     -> wsl_lora_adapter_guide.md
     -> lora_training_wsl.md
```

推荐阅读顺序：
1. 先看根 README，理解项目定位和整体价值。
2. 再看 `current_implementation.md`，确认当前范围。
3. 然后根据你的目标，选择常规 CLI 路径或 WSL adapter / training 路径。

## 对应的代码入口

如果你看完文档后想直接看实现，优先从这里开始：
- [`../investment_research_desk/cli.py`](../investment_research_desk/cli.py) — 主 CLI 入口
- [`../investment_research_desk/graph/workflow.py`](../investment_research_desk/graph/workflow.py) — 工作流编排
- [`../investment_research_desk/agents/core.py`](../investment_research_desk/agents/core.py) — Agent 核心逻辑
- [`../investment_research_desk/sentiment_runtime.py`](../investment_research_desk/sentiment_runtime.py) — 可选 sentiment adapter 运行时

## 补充说明

- 这个项目是**投研上下文软件**，不是交易执行软件。
- LoRA 路径是可选能力，并且与常规本地 CLI 路径分离。
- 真实运行产物会写入 `runs/`，并且默认不纳入 git。
