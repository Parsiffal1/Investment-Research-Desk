#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(pwd)}"
VENV_DIR="${VENV_DIR:-.venv-lora}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

cd "$PROJECT_DIR"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Missing $PYTHON_BIN. Install Python 3.11 or 3.12 in Ubuntu first." >&2
  exit 1
fi

"$PYTHON_BIN" -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip wheel setuptools
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install transformers datasets trl peft bitsandbytes accelerate evaluate scikit-learn
pip install -e .
pip install pytest

python scripts/wsl/verify_lora_env.py
