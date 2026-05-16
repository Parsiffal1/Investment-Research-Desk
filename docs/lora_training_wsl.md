# QLoRA Training on WSL2 CUDA

This project includes an optional QLoRA fine-tuning path for financial classification work. The standard Windows/local CLI remains usable without training dependencies; this workflow is for WSL2 + CUDA.

## Environment

Use a dedicated WSL virtual environment. Do not share the Windows `.venv`.

```bash
cd <PROJECT_DIR>
bash scripts/wsl/setup_lora_env.sh
```

Optional model-load verification:

```bash
source <WSL_VENV>/bin/activate
python scripts/wsl/verify_lora_env.py --load-model
```

## Data

The training pipeline uses:

- Financial PhraseBank
- Twitter Financial News Sentiment

Held-out splits are kept separate from training. Manifests include dataset name, split, row index, label, text hash, and normalized text hash to detect overlap.

Prepare data:

```bash
ird lora prepare-data --output-dir lora_data/sentiment
```

## Training

Smoke run:

```bash
bash scripts/wsl/run_lora_pipeline.sh smoke
```

Pilot run:

```bash
bash scripts/wsl/run_lora_pipeline.sh pilot
```

Full run:

```bash
bash scripts/wsl/run_lora_pipeline.sh full
```

The full run writes local artifacts such as:

- `lora_data/sentiment/train.jsonl`
- `lora_data/sentiment/dev.jsonl`
- `lora_data/sentiment/eval.jsonl`
- `models/investment-research-desk-lora-sentiment/<timestamp>/adapter`
- `models/investment-research-desk-lora-sentiment/<timestamp>/training_config.json`
- `models/investment-research-desk-lora-sentiment/<timestamp>/dev_metrics.json`
- `eval/results/lora_full/heldout_eval_results.json`
- `eval/results/lora_full/heldout_eval_results.md`

## Held-out benchmark reporting

The checked-in baseline lives in `investment_research_desk/lora/sentiment.py`:

| Variant | ACC | Macro-F1 |
| --- | ---: | ---: |
| Baseline Qwen3-8B forced-choice classifier | 0.7900 | 0.7771 |
| Fine-tuned adapter (`20260515T123418Z`) | 0.8926 | 0.8760 |

Held-out deltas versus baseline:

- Accuracy: **+0.1026**
- Macro-F1: **+0.0989**

The canonical result artifacts are:

- `eval/results/lora_full/heldout_eval_results.json`
- `eval/results/lora_full/heldout_eval_results.md`
- [`lora_experiment_details.md`](lora_experiment_details.md) — the full experiment record for public benchmark backing

## Publishing

Do not commit raw training data, intermediate trainer checkpoints, optimizer states, or local cache directories.

If publishing the final adapter in this repository, track `adapter_model.safetensors` with Git LFS and include only the final adapter directory plus metrics/config files.
