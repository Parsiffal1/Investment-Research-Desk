# LoRA Experiment Details

This document records the full sentiment QLoRA experiment for Investment Research Desk.

## Summary

- Experiment: sentiment classification adapter for Investment Research Desk
- Adapter output: `models/investment-research-desk-lora-sentiment/20260515T123418Z/adapter`
- Base model: `Qwen/Qwen3-8B`
- Method: QLoRA, 4-bit NF4 quantized base model plus PEFT LoRA adapter
- Task scope: financial sentiment classification only
- Evaluation method: forced-choice label scoring on held-out splits
- Contract check: small generative JSON sample with `/no_think`

## Hardware And Runtime

The run was executed in a WSL2 CUDA training environment, separate from the Windows CLI runtime.

| Item | Value |
|---|---|
| OS/runtime | WSL2 Ubuntu training environment |
| Python | 3.12.13 |
| PyTorch | 2.11.0+cu128 |
| CUDA availability | `true` |
| GPU count | 1 |
| GPU | NVIDIA GeForce RTX 5070 Ti Laptop GPU |
| bf16 support | `true` |
| Quantization backend | bitsandbytes |
| Quantization mode | 4-bit NF4 |
| Compute dtype | bfloat16 |

Source: `eval/results/lora_full/run.log`.

## Datasets

Only two datasets were used in this experiment.

| Dataset | Label space | Training split source | Held-out eval split |
|---|---|---|---|
| Financial PhraseBank | `positive`, `negative`, `neutral` | `train` + `validation` | `test` |
| Twitter Financial News Sentiment | `bullish`, `bearish`, `neutral` | `train` | `validation` |

## Dataset Scale And Splits

The data preparation step wrote `train_manifest.jsonl`, `dev_manifest.jsonl`, and `eval_manifest.jsonl`.
Each manifest entry includes dataset name, split, row index, label, text hash, and normalized text hash.

| Dataset | Raw train samples | Removed train/eval hash overlaps | Final train | Dev | Held-out eval |
|---|---:|---:|---:|---:|---:|
| Financial PhraseBank | 4,356 | 1 | 4,291 | 64 | 484 |
| Twitter Financial News Sentiment | 9,543 | 93 | 9,386 | 64 | 2,388 |
| Total | 13,899 | 94 | 13,677 | 128 | 2,872 |

Leakage checks:

- Held-out eval splits were excluded from training.
- Train/eval row overlap count: `0`
- Train/eval normalized text hash overlap count: `0`
- Dev/eval row overlap count: `0`
- Dev/eval normalized text hash overlap count: `0`
- Train/dev row overlap count: `0`
- Train/dev normalized text hash overlap count: `0`

Source: `models/investment-research-desk-lora-sentiment/20260515T123418Z/data_summary.json`.

## Training Configuration

| Parameter | Value |
|---|---:|
| Base model | `Qwen/Qwen3-8B` |
| Method | `qlora_4bit_nf4` |
| LoRA target modules | `all-linear` |
| LoRA rank `r` | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Epochs | 2 |
| Learning rate | 2e-4 |
| Per-device train batch size | 1 |
| Gradient accumulation steps | 16 |
| Effective batch size | 16 |
| Max sequence length | 256 |
| LR scheduler | cosine |
| Warmup ratio | 0.03 |
| Gradient checkpointing | enabled |
| Seed | 42 |

Source: `models/investment-research-desk-lora-sentiment/20260515T123418Z/training_config.json`.

## Training Runtime

| Metric | Value |
|---|---:|
| Training steps | 1,710 |
| Training runtime | 19,817.6667 seconds |
| Training runtime | about 5h 30m 18s |
| Train samples/second | 1.38 |
| Train steps/second | 0.086 |
| Final train loss | 0.8208 |
| Total FLOPs | 1.2155e17 |

Dev evaluation after training:

| Metric | Value |
|---|---:|
| Eval loss | 0.7706 |
| Eval runtime | 12.0598 seconds |
| Eval samples/second | 10.614 |
| Eval steps/second | 1.327 |

Source: `models/investment-research-desk-lora-sentiment/20260515T123418Z/dev_metrics.json`.

## Held-out Evaluation

Overall held-out result:

| Variant | Accuracy | Macro-F1 |
|---|---:|---:|
| Baseline Qwen3-8B | 0.7900 | 0.7771 |
| QLoRA adapter | 0.8926 | 0.8760 |
| Delta | +0.1026 | +0.0989 |

Output contract:

| Check | Result |
|---|---|
| `/no_think` contract | pass |
| Samples checked | 24 |
| `<think>` output violations | 0 |
| Non-JSON wrapper violations | 0 |
| Extra JSON field violations | 0 |
| Invalid label violations | 0 |
| LLM call failures | 0 |

### Financial PhraseBank

| Metric | Value |
|---|---:|
| Held-out split | `test` |
| Samples | 484 |
| Accuracy | 0.8740 |
| Macro-F1 | 0.8631 |

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| negative | 0.9123 | 0.8525 | 0.8814 |
| neutral | 0.8626 | 0.9408 | 0.9000 |
| positive | 0.8860 | 0.7426 | 0.8080 |

### Twitter Financial News Sentiment

| Metric | Value |
|---|---:|
| Held-out split | `validation` |
| Samples | 2,388 |
| Accuracy | 0.9112 |
| Macro-F1 | 0.8890 |

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| bearish | 0.8629 | 0.8703 | 0.8666 |
| bullish | 0.8838 | 0.8484 | 0.8657 |
| neutral | 0.9298 | 0.9393 | 0.9346 |

Source: `eval/results/lora_full/heldout_eval_results.json` and `eval/results/lora_full/heldout_eval_results.md`.

## Artifact Map

| Artifact | Path |
|---|---|
| Adapter | `models/investment-research-desk-lora-sentiment/20260515T123418Z/adapter` |
| Training config | `models/investment-research-desk-lora-sentiment/20260515T123418Z/training_config.json` |
| Data summary | `models/investment-research-desk-lora-sentiment/20260515T123418Z/data_summary.json` |
| Train manifest | `models/investment-research-desk-lora-sentiment/20260515T123418Z/train_manifest.jsonl` |
| Dev manifest | `models/investment-research-desk-lora-sentiment/20260515T123418Z/dev_manifest.jsonl` |
| Eval manifest | `models/investment-research-desk-lora-sentiment/20260515T123418Z/eval_manifest.jsonl` |
| Dev metrics | `models/investment-research-desk-lora-sentiment/20260515T123418Z/dev_metrics.json` |
| Full held-out JSON | `eval/results/lora_full/heldout_eval_results.json` |
| Full held-out Markdown | `eval/results/lora_full/heldout_eval_results.md` |

## Notes

- This experiment fine-tuned only the sentiment classification adapter.
- It does not fine-tune the main multi-agent research workflow.
- The held-out evaluation is split-protected by row identifiers and normalized text hashes.
- Public-facing claims should use only the held-out results above, not dev metrics.
