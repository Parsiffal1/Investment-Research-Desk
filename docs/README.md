[English](README.md) | [中文](README.zh.md)

# Documentation Index

This directory holds the long-form project documentation for **Investment Research Desk**.

If you arrived here from the repository homepage, this page is the fastest way to find the right doc for your goal.

## Read this first depending on your goal

### I want to understand what the project already does
- [`current_implementation.md`](current_implementation.md)
  - The best source for the current implemented behavior
  - Describes the workflow, tool boundaries, constraints, and current limitations

### I want to run the normal CLI on a regular machine
- [`windows_cli_guide.md`](windows_cli_guide.md)
  - Standard local CLI path
  - Environment setup, Ollama startup, config check, report runs, and demo mode

### I want to run the optional adapter path in WSL
- [`wsl_lora_adapter_guide.md`](wsl_lora_adapter_guide.md)
  - How to run reports in WSL with a trained adapter
  - Includes Ollama bridge notes and runtime verification steps

### I want to train the QLoRA adapter path
- [`lora_training_wsl.md`](lora_training_wsl.md)
  - WSL2 + CUDA training workflow
  - Data sources, split hygiene, artifact structure, and release boundaries
- [`lora_experiment_details.md`](lora_experiment_details.md)
  - Full benchmark-backed experiment record
  - Hardware, runtime, held-out metrics, and artifact map

## How these docs fit together

```text
README.md / README.zh.md
  -> docs/README.md / docs/README.zh.md
     -> current_implementation.md
     -> windows_cli_guide.md
     -> wsl_lora_adapter_guide.md
     -> lora_training_wsl.md
```

Recommended reading order:
1. Start at the root README for product-level context.
2. Read `current_implementation.md` for exact scope.
3. Choose either the normal CLI path or the WSL adapter/training path.

## Related code entrypoints

If you want the implementation side after reading docs:
- [`../investment_research_desk/cli.py`](../investment_research_desk/cli.py) — main CLI
- [`../investment_research_desk/graph/workflow.py`](../investment_research_desk/graph/workflow.py) — workflow orchestration
- [`../investment_research_desk/agents/core.py`](../investment_research_desk/agents/core.py) — agent logic
- [`../investment_research_desk/sentiment_runtime.py`](../investment_research_desk/sentiment_runtime.py) — optional sentiment adapter runtime

## Notes

- The project is **research-context software**, not trade execution software.
- The LoRA path is optional and separated from the standard local CLI path.
- Real run artifacts are written to `runs/` and are intentionally gitignored.
