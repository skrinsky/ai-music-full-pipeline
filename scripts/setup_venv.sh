#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
VENV_PY="${VENV_DIR}/bin/python"
PYTHON_BIN="${PYTHON_BIN:-python3.10}"   # override if needed: PYTHON_BIN=/opt/homebrew/bin/python3.10

PIPE_DIR="${ROOT_DIR}/vendor/all-in-one-ai-midi-pipeline"

echo "== ai-music-full-pipeline venv setup =="
echo "ROOT_DIR: ${ROOT_DIR}"
echo "VENV_DIR: ${VENV_DIR}"
echo "PYTHON_BIN: ${PYTHON_BIN}"
echo "PIPE_DIR: ${PIPE_DIR}"
echo

if ! command -v uv &>/dev/null; then
  echo "ERROR: uv not found. Install it first:"
  echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
  exit 1
fi

if [[ ! -d "${PIPE_DIR}" ]]; then
  echo "ERROR: missing submodule folder: ${PIPE_DIR}"
  echo "Did you run: git submodule update --init --recursive ?"
  exit 1
fi

# Create venv
if [[ ! -d "${VENV_DIR}" ]]; then
  echo "== Creating venv =="
  uv venv --python "${PYTHON_BIN}" "${VENV_DIR}"
else
  echo "== Venv already exists =="
fi

# setuptools is needed by torchcrepe and other pkg_resources users.
echo
echo "== Installing setuptools =="
uv pip install --python "${VENV_PY}" "setuptools<81"

# ── Smart PyTorch installation ─────────────────────────────────────────────────
# Detect platform and install the best available PyTorch build before anything
# else so that the vendor requirements.txt can't downgrade it.
echo
echo "== Detecting platform for PyTorch install =="

OS="$(uname -s)"
ARCH="$(uname -m)"
TORCH_CMD=""
TORCH_LABEL=""
UV_PIP="uv pip install --python ${VENV_PY}"

if [[ "$OS" == "Darwin" ]]; then
    MACOS_MAJOR="$(sw_vers -productVersion | cut -d. -f1)"
    if [[ "$ARCH" == "arm64" ]]; then
        if [[ "$MACOS_MAJOR" -ge 14 ]]; then
            TORCH_LABEL="Apple Silicon + macOS 14+ → latest PyTorch (MPS)"
            TORCH_CMD="${UV_PIP} --upgrade torch torchaudio"
        else
            # PyTorch ≥ 2.3 dropped MPS support for macOS 13; 2.2.x is the last
            # version that exposes MPS on Ventura.
            TORCH_LABEL="Apple Silicon + macOS 13 → PyTorch 2.2.x (MPS on Ventura)"
            TORCH_CMD="${UV_PIP} 'torch==2.2.2' 'torchaudio==2.2.2'"
        fi
    else
        TORCH_LABEL="Intel Mac → latest PyTorch (CPU)"
        TORCH_CMD="${UV_PIP} --upgrade torch torchaudio"
    fi
elif [[ "$OS" == "Linux" ]]; then
    if command -v nvidia-smi &>/dev/null 2>&1; then
        # Pick CUDA index matching the installed driver's CUDA version
        CUDA_TAG="$(nvidia-smi | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' || echo '12.1')"
        CUDA_MAJOR="$(echo "$CUDA_TAG" | cut -d. -f1)"
        CUDA_MINOR="$(echo "$CUDA_TAG" | cut -d. -f2)"
        if [[ "$CUDA_MAJOR" -ge 12 && "$CUDA_MINOR" -ge 4 ]]; then
            INDEX="https://download.pytorch.org/whl/cu124"
        elif [[ "$CUDA_MAJOR" -ge 12 ]]; then
            INDEX="https://download.pytorch.org/whl/cu121"
        elif [[ "$CUDA_MAJOR" -ge 11 && "$CUDA_MINOR" -ge 8 ]]; then
            INDEX="https://download.pytorch.org/whl/cu118"
        else
            INDEX="https://download.pytorch.org/whl/cu118"
        fi
        TORCH_LABEL="Linux + NVIDIA CUDA ${CUDA_TAG} → PyTorch with ${INDEX##*/}"
        TORCH_CMD="${UV_PIP} --upgrade torch torchaudio --index-url ${INDEX}"
    else
        TORCH_LABEL="Linux CPU → PyTorch CPU build"
        TORCH_CMD="${UV_PIP} --upgrade torch torchaudio --index-url https://download.pytorch.org/whl/cpu"
    fi
else
    # Windows / unknown — let pip pick the default wheel
    TORCH_LABEL="Unknown platform → PyTorch default wheel"
    TORCH_CMD="${UV_PIP} --upgrade torch torchaudio"
fi

echo "  ${TORCH_LABEL}"
eval "${TORCH_CMD}"

# ── Other requirements ─────────────────────────────────────────────────────────
echo
echo "== Installing top-level requirements (if present) =="
if [[ -f "${ROOT_DIR}/requirements.txt" ]]; then
  uv pip install --python "${VENV_PY}" -r "${ROOT_DIR}/requirements.txt"
else
  echo "No ${ROOT_DIR}/requirements.txt found (skipping)."
fi

echo
echo "== Installing vendored pipeline requirements (torch pins excluded) =="
if [[ -f "${PIPE_DIR}/requirements.txt" ]]; then
  # Strip torch / torchaudio / torchvision pins so the platform-chosen build
  # above isn't downgraded. A bare torchaudio==X pin drags torch down with it,
  # so all three must go. torchcrepe and other torch* packages are kept.
  TMPFILE="$(mktemp)"
  grep -vE '^(torch|torchaudio|torchvision)([=<>!~ ]|$)' "${PIPE_DIR}/requirements.txt" > "${TMPFILE}" || true
  uv pip install --python "${VENV_PY}" -r "${TMPFILE}"
  rm -f "${TMPFILE}"
else
  echo "No ${PIPE_DIR}/requirements.txt found (skipping)."
fi

echo
echo "== Sanity check =="
"${VENV_PY}" - <<'PY'
import sys
print("python:", sys.executable)
print("version:", sys.version.split()[0])

try:
    import torch
    print("torch:", torch.__version__)
    cuda = torch.cuda.is_available()
    mps  = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    if cuda:
        accel = f"CUDA ({torch.cuda.get_device_name(0)})"
    elif mps:
        accel = "MPS (Apple Silicon)"
    else:
        accel = "CPU only"
    print("accelerator:", accel)
except Exception as e:
    print("torch import FAILED:", e)

try:
    import torchcrepe
    print("torchcrepe:", torchcrepe.__version__)
except Exception as e:
    print("torchcrepe import FAILED (optional):", e)

try:
    import music21
    print("music21:", music21.__version__)
except Exception as e:
    print("music21 import FAILED:", e)

print("OK")
PY

echo
echo "Done.  Activate later with:"
echo "  source ${VENV_DIR}/bin/activate"
