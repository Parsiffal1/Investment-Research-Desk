# Sentiment LoRA Training on WSL2 CUDA

This project trains the first adapter only for sentiment classification. The Windows CLI remains usable without these training dependencies; this workflow is for WSL2 + CUDA only.

## Current Windows Gate

The current machine has an NVIDIA RTX 5070 Ti Laptop GPU visible from Windows, but WSL is not installed/enabled in the current non-admin Codex session. Install WSL from an elevated PowerShell:

```powershell
Set-Location C:\Users\saton\Documents\Codex\2026-05-14\files-mentioned-by-the-user-finsight
.\scripts\wsl\install_wsl_ubuntu_admin.ps1
```

Restart Windows if prompted, then launch Ubuntu once and create the Linux user.

## WSL Setup

Copy or clone the project into WSL ext4 storage, for example:

```bash
mkdir -p ~/projects
cp -a /mnt/c/Users/saton/Documents/Codex/2026-05-14/files-mentioned-by-the-user-finsight ~/projects/investment-research-desk
cd ~/projects/investment-research-desk
```

Create the training environment:

```bash
bash scripts/wsl/setup_lora_env.sh
```

Run the heavier optional model-load check before training:

```bash
source .venv-lora/bin/activate
python scripts/wsl/verify_lora_env.py --load-model
```

## Training and Evaluation

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

The full run writes:

- `lora_data/sentiment/train.jsonl`
- `lora_data/sentiment/dev.jsonl`
- `lora_data/sentiment/eval.jsonl`
- `models/investment-research-desk-lora-sentiment/<timestamp>/adapter`
- `models/investment-research-desk-lora-sentiment/<timestamp>/training_config.json`
- `models/investment-research-desk-lora-sentiment/<timestamp>/dev_metrics.json`
- `eval/results/lora_full/heldout_eval_results.json`
- `eval/results/lora_full/heldout_eval_results.md`

Do not commit `models/` or `lora_data/`.

## Acceptance Gate

- `python scripts/wsl/verify_lora_env.py --load-model` succeeds.
- `ird lora prepare-data` reports `leakage_check=pass`.
- `train_eval_overlap`, `dev_eval_overlap`, and `train_dev_overlap` are all zero.
- `heldout_eval_results.json` includes accuracy, Macro-F1, per-class metrics, output contract status, and baseline deltas.
- README or resume metrics are updated only if measured LoRA held-out metrics beat the stored baseline.

The default evaluation path uses forced-choice label scoring for classification metrics. Generative JSON evaluation is limited to a small contract-check sample so full held-out evaluation does not spend hours generating short labels.
