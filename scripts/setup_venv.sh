#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv-ai-music"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"   # override if needed: PYTHON_BIN=/opt/homebrew/bin/python3.10

PIPE_DIR="${ROOT_DIR}/vendor/all-in-one-ai-midi-pipeline"

echo "== ai-music-full-pipeline venv setup =="
echo "ROOT_DIR: ${ROOT_DIR}"
echo "VENV_DIR: ${VENV_DIR}"
echo "PYTHON_BIN: ${PYTHON_BIN}"
echo "PIPE_DIR: ${PIPE_DIR}"
echo

if [[ ! -d "${PIPE_DIR}" ]]; then
  echo "ERROR: missing submodule folder: ${PIPE_DIR}"
  echo "Did you run: git submodule update --init --recursive ?"
  exit 1
fi

# Create venv
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "== Creating venv =="
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
else
  echo "== Venv already exists =="
fi

# Activate
# shellcheck disable=SC1090
source "${VENV_DIR}/bin/activate"

echo
echo "== Upgrading pip tooling =="
python -m pip install -U pip setuptools wheel

echo
echo "== Installing top-level requirements (if present) =="
if [[ -f "${ROOT_DIR}/requirements.txt" ]]; then
  python -m pip install -r "${ROOT_DIR}/requirements.txt"
else
  echo "No ${ROOT_DIR}/requirements.txt found (skipping)."
fi

echo
echo "== Installing vendored pipeline requirements (if present) =="
if [[ -f "${PIPE_DIR}/requirements.txt" ]]; then
  python -m pip install -r "${PIPE_DIR}/requirements.txt"
else
  echo "No ${PIPE_DIR}/requirements.txt found (skipping)."
fi

echo
echo "== Sanity check =="
python - <<'PY'
import sys
print("python:", sys.executable)
print("version:", sys.version.split()[0])

# Torch + device
try:
    import torch
    print("torch:", torch.__version__)
    cuda = torch.cuda.is_available()
    mps = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    print("cuda available:", cuda)
    print("mps available:", mps)
except Exception as e:
    print("torch import FAILED:", e)

# torchcrepe
try:
    import torchcrepe
    print("torchcrepe:", torchcrepe.__version__)
except Exception as e:
    print("torchcrepe import FAILED (optional):", e)

# music21
try:
    import music21
    print("music21:", music21.__version__)
except Exception as e:
    print("music21 import FAILED:", e)

print("OK")
PY

echo
echo "Done."
echo "Activate later with:"
echo "  source ${VENV_DIR}/bin/activate"

