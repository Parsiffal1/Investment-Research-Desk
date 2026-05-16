#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${IRD_PROJECT_DIR:-/mnt/c/Users/saton/Documents/Codex/2026-05-14/files-mentioned-by-the-user-finsight}"
VENV="${IRD_LORA_VENV:-/home/parsiffal/.venvs/ird-lora}"
SYMBOL="${1:-ETH-USDT-SWAP}"
ADAPTER_PATH="${IRD_SENTIMENT_ADAPTER_PATH:-models/investment-research-desk-lora-sentiment/20260515T123418Z/adapter}"
OLLAMA_BASE_URL="${IRD_OLLAMA_BASE_URL:-http://172.24.48.1:11435/v1}"
HF_HOME="${HF_HOME:-/mnt/d/saton/ird_hf_cache}"

cd "$PROJECT_DIR"
source "$VENV/bin/activate"

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
