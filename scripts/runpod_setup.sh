#!/usr/bin/env bash
# RunPod setup for blues training
# ────────────────────────────────────────────────────────
# 1. Spin up a RunPod GPU pod (RTX 3090 recommended) with
#    the "RunPod Pytorch 2.1" or similar template.
#
# 2. Upload preprocessed data (from your Mac):
#      rsync -avz --progress runs/blues_events/ runpod:~/ai-music-full-pipeline/runs/blues_events/
#    Or use RunPod's web terminal upload, or scp.
#
# 3. SSH into the pod and run:
#      git clone https://github.com/josmithiii/ai-music-full-pipeline.git
#      cd ai-music-full-pipeline
#      git checkout jos
#      bash scripts/runpod_setup.sh
#
# 4. Start training:
#      make blues-train
# ────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "== RunPod training setup =="
echo "ROOT_DIR: $ROOT_DIR"

# ── Python venv ──────────────────────────────────────────
VENV_DIR="$ROOT_DIR/.venv-ai-music"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ ! -d "$VENV_DIR" ]]; then
    echo "== Creating venv =="
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
pip install -U pip setuptools wheel

# ── Training-only deps (no audio pipeline needed) ────────
pip install torch numpy

# ── Verify CUDA ──────────────────────────────────────────
python - <<'PY'
import torch
print(f"PyTorch:  {torch.__version__}")
print(f"CUDA:     {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU:      {torch.cuda.get_device_name(0)}")
    print(f"VRAM:     {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
else:
    print("WARNING: no CUDA GPU detected!")
PY

# ── Check that preprocessed data is in place ─────────────
DATA_DIR="$ROOT_DIR/runs/blues_events"
if [[ ! -f "$DATA_DIR/events_train.pkl" ]]; then
    echo ""
    echo "ERROR: preprocessed data not found at $DATA_DIR/"
    echo ""
    echo "Copy it from your Mac:"
    echo "  rsync -avz --progress runs/blues_events/ <runpod-host>:$ROOT_DIR/runs/blues_events/"
    exit 1
fi

echo ""
echo "== Ready to train =="
echo "  source $VENV_DIR/bin/activate"
echo "  make blues-train"
echo ""
echo "Or with custom args:"
echo "  make blues-train ARGS='--batch_size 128'"
