#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-smoke}"
VENV_DIR="${VENV_DIR:-.venv-lora}"
MODEL_ROOT="${MODEL_ROOT:-models/investment-research-desk-lora-sentiment}"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
  echo "Missing $VENV_DIR. Run scripts/wsl/setup_lora_env.sh first." >&2
  exit 1
fi

source "$VENV_DIR/bin/activate"
python scripts/wsl/verify_lora_env.py

if [[ "$MODE" == "smoke" ]]; then
  DATA_DIR="${DATA_DIR:-lora_data/sentiment_smoke}"
  EVAL_DIR="${EVAL_DIR:-eval/results/lora_smoke}"
  ird lora prepare-data --output-dir "$DATA_DIR" --limit "${LORA_SMOKE_LIMIT:-24}"
elif [[ "$MODE" == "full" ]]; then
  DATA_DIR="${DATA_DIR:-lora_data/sentiment}"
  EVAL_DIR="${EVAL_DIR:-eval/results/lora_full}"
  ird lora prepare-data --output-dir "$DATA_DIR"
else
  echo "Usage: $0 [smoke|full]" >&2
  exit 1
fi

ird lora train --data-dir "$DATA_DIR" --output-root "$MODEL_ROOT"
ADAPTER_PATH="$(find "$MODEL_ROOT" -mindepth 2 -maxdepth 2 -type d -name adapter -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"

if [[ -z "$ADAPTER_PATH" ]]; then
  echo "No adapter directory was produced under $MODEL_ROOT." >&2
  exit 1
fi

mkdir -p "$EVAL_DIR"
ird lora eval --adapter-path "$ADAPTER_PATH" --output-dir "$EVAL_DIR"
echo "Adapter: $ADAPTER_PATH"
echo "Evaluation: $EVAL_DIR/heldout_eval_results.json"
