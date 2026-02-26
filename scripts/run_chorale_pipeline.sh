#!/usr/bin/env bash
# End-to-end pipeline for Bach chorale training.
# Usage: bash scripts/run_chorale_pipeline.sh [--device auto|cuda|mps|cpu]
set -euo pipefail

DEVICE="${1:-auto}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(dirname "$SCRIPT_DIR")"
cd "$ROOT"

echo "=== Step 1: Convert NPZ → MIDI ==="
python scripts/convert_chorales_npz_to_midi.py \
  --npz data/Jsb16thSeparated.npz \
  --out_dir data/chorales_midi --bpm 100

echo ""
echo "=== Step 2: Preprocess MIDI → events ==="
python training/pre.py \
  --midi_folder data/chorales_midi \
  --data_folder runs/chorale_events \
  --instrument_set chorale4

echo ""
echo "=== Step 3: Train ==="
python training/train.py \
  --data_dir runs/chorale_events \
  --train_pkl runs/chorale_events/events_train.pkl \
  --val_pkl runs/chorale_events/events_val.pkl \
  --vocab_json runs/chorale_events/event_vocab.json \
  --save_path runs/checkpoints/chorale_model.pt \
  --device "$DEVICE"

echo ""
echo "=== Step 4: Generate ==="
python training/generate.py \
  --ckpt runs/checkpoints/chorale_model.pt \
  --vocab_json runs/chorale_events/event_vocab.json \
  --out_midi runs/generated/chorale_out.mid \
  --device "$DEVICE" --drum_bonus 0.0

echo ""
echo "=== Done ==="
echo "Generated MIDI: runs/generated/chorale_out.mid"
