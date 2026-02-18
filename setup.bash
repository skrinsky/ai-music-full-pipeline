#!/usr/bin/env bash
set -euo pipefail

# ── ai-music-full-pipeline: uv environment setup ──
#
# Creates a uv-managed virtualenv with Python 3.10 and installs all
# dependencies for both the vendored audio→MIDI pipeline and the
# training/generation scripts.
#
# Usage:
#   bash setup.bash                  # default: .venv in repo root
#   VENV_DIR=/tmp/myvenv bash setup.bash   # custom location
#   PYTHON_VERSION=3.11 bash setup.bash    # different Python

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PIPE_DIR="${ROOT_DIR}/vendor/all-in-one-ai-midi-pipeline"

echo "== ai-music-full-pipeline: uv setup =="
echo "ROOT_DIR:        ${ROOT_DIR}"
echo "VENV_DIR:        ${VENV_DIR}"
echo "PYTHON_VERSION:  ${PYTHON_VERSION}"
echo "PIPE_DIR:        ${PIPE_DIR}"
echo

# ── Preflight checks ──

if ! command -v uv &>/dev/null; then
  echo "ERROR: uv not found. Install it first:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if [[ ! -d "${PIPE_DIR}" ]]; then
  echo "ERROR: missing submodule folder: ${PIPE_DIR}"
  echo "Run:  git submodule update --init --recursive"
  exit 1
fi

# ── Create venv ──

if [[ ! -d "${VENV_DIR}" ]]; then
  echo "== Creating venv (Python ${PYTHON_VERSION}) =="
  uv venv --python "${PYTHON_VERSION}" "${VENV_DIR}"
else
  echo "== Venv already exists at ${VENV_DIR} =="
fi

# ── Install setuptools (needed by torchcrepe and other pkg_resources users) ──

echo
echo "== Installing setuptools =="
uv pip install --python "${VENV_DIR}/bin/python" "setuptools<81"

# ── Install vendored pipeline requirements ──

echo
echo "== Installing vendored pipeline requirements =="
uv pip install --python "${VENV_DIR}/bin/python" \
  -r "${PIPE_DIR}/requirements.txt"

# ── Install top-level requirements (if any) ──

if [[ -f "${ROOT_DIR}/requirements.txt" ]]; then
  echo
  echo "== Installing top-level requirements =="
  uv pip install --python "${VENV_DIR}/bin/python" \
    -r "${ROOT_DIR}/requirements.txt"
fi

# ── Install any extra training deps not in the pipeline requirements ──
# (Currently the training scripts only need torch + numpy + pretty_midi,
#  all of which are already pulled in above.  Add extras here if needed.)

# ── Sanity check ──

echo
echo "== Sanity check =="
"${VENV_DIR}/bin/python" - <<'PY'
import sys
print(f"python:  {sys.executable}")
print(f"version: {sys.version.split()[0]}")

import torch
print(f"torch:   {torch.__version__}")
cuda = torch.cuda.is_available()
mps  = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
print(f"cuda:    {cuda}")
print(f"mps:     {mps}")

import numpy; print(f"numpy:   {numpy.__version__}")
import pretty_midi; print(f"pretty_midi: {pretty_midi.__version__}")
import librosa; print(f"librosa: {librosa.__version__}")

try:
    import torchcrepe; print(f"torchcrepe: {torchcrepe.__version__}")
except Exception as e:
    print(f"torchcrepe: not available ({e})")

print("OK")
PY

echo
echo "Done.  Activate with:"
echo "  source ${VENV_DIR}/bin/activate"
