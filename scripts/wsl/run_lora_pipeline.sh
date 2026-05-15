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
  EVAL_LIMIT_ARGS=(--limit "${LORA_SMOKE_EVAL_LIMIT:-6}")
  ird lora prepare-data --output-dir "$DATA_DIR" --limit "${LORA_SMOKE_LIMIT:-24}"
  TRAIN_ARGS=(--epochs "${LORA_SMOKE_EPOCHS:-1}")
elif [[ "$MODE" == "pilot" ]]; then
  DATA_DIR="${DATA_DIR:-lora_data/sentiment_pilot}"
  EVAL_DIR="${EVAL_DIR:-eval/results/lora_pilot}"
  EVAL_LIMIT_ARGS=(--limit "${LORA_PILOT_EVAL_LIMIT:-300}" --contract-limit "${LORA_PILOT_CONTRACT_LIMIT:-6}")
  ird lora prepare-data --output-dir "$DATA_DIR" --limit "${LORA_PILOT_LIMIT:-900}"
  TRAIN_ARGS=(--epochs "${LORA_PILOT_EPOCHS:-1}")
elif [[ "$MODE" == "full" ]]; then
  DATA_DIR="${DATA_DIR:-lora_data/sentiment}"
  EVAL_DIR="${EVAL_DIR:-eval/results/lora_full}"
  EVAL_LIMIT_ARGS=(--contract-limit "${LORA_FULL_CONTRACT_LIMIT:-6}")
  ird lora prepare-data --output-dir "$DATA_DIR"
  TRAIN_ARGS=(--epochs "${LORA_FULL_EPOCHS:-2}")
else
  echo "Usage: $0 [smoke|pilot|full]" >&2
  exit 1
fi

ird lora train --data-dir "$DATA_DIR" --output-root "$MODEL_ROOT" "${TRAIN_ARGS[@]}"
ADAPTER_PATH="$(find "$MODEL_ROOT" -mindepth 2 -maxdepth 2 -type d -name adapter -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"

if [[ -z "$ADAPTER_PATH" ]]; then
  echo "No adapter directory was produced under $MODEL_ROOT." >&2
  exit 1
fi

mkdir -p "$EVAL_DIR"
ird lora eval --adapter-path "$ADAPTER_PATH" --output-dir "$EVAL_DIR" "${EVAL_LIMIT_ARGS[@]}"
echo "Adapter: $ADAPTER_PATH"
echo "Evaluation: $EVAL_DIR/heldout_eval_results.json"
