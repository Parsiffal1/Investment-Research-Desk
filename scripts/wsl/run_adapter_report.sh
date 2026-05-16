#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${IRD_PROJECT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
VENV="${IRD_LORA_VENV:-$HOME/.venvs/ird-lora}"
SYMBOL="${1:-ETH-USDT-SWAP}"
ADAPTER_PATH="${IRD_SENTIMENT_ADAPTER_PATH:-}"
OLLAMA_BASE_URL="${IRD_OLLAMA_BASE_URL:-http://172.24.48.1:11435/v1}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

cd "$PROJECT_DIR"
source "$VENV/bin/activate"

if [[ -z "$ADAPTER_PATH" ]]; then
  ADAPTER_PATH="$(ls -dt models/investment-research-desk-lora-sentiment/*/adapter 2>/dev/null | head -n 1 || true)"
fi

if [[ -z "$ADAPTER_PATH" ]]; then
  echo "No sentiment adapter found. Set IRD_SENTIMENT_ADAPTER_PATH or place an adapter under models/investment-research-desk-lora-sentiment/<timestamp>/adapter." >&2
  exit 1
fi

export IRD_OLLAMA_BASE_URL="$OLLAMA_BASE_URL"
export HF_HOME="$HF_HOME"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME}"
export IRD_AGENT_EXECUTION_MODE="${IRD_AGENT_EXECUTION_MODE:-sequential}"
export IRD_LLM_TIMEOUT_SEC="${IRD_LLM_TIMEOUT_SEC:-180}"

ird report \
  --symbol "$SYMBOL" \
  --asset-class crypto \
  --horizon short_term \
  --llm-provider ollama \
  --model qwen3:8b \
  --sentiment-provider hf-peft \
  --sentiment-adapter-path "$ADAPTER_PATH" \
  --sentiment-score-batch-size "${IRD_SENTIMENT_SCORE_BATCH_SIZE:-4}" \
  --runs-dir "${IRD_RUNS_DIR:-runs}"
