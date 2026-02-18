#!/usr/bin/env bash
set -euo pipefail

# Default settings (override via env vars or flags)
AUDIO_GLOB=${AUDIO_GLOB:-"data/raw/*.wav"}
TRACKS=${TRACKS:-""}            # e.g. "drums,bass,guitar"
PIPE_NORMALIZE_KEY=${PIPE_NORMALIZE_KEY:-1}
PIPE_NO_CLEAN=${PIPE_NO_CLEAN:-0}

# Training settings
MIDI_DIR=${MIDI_DIR:-"out_midis"}
EVENTS_DIR=${EVENTS_DIR:-"runs/events"}
CKPT_PATH=${CKPT_PATH:-"runs/checkpoints/es_model.pt"}
GEN_OUT=${GEN_OUT:-"runs/generated/out.mid"}
DEVICE=${DEVICE:-"auto"}        # auto|cuda|mps|cpu

usage() {
  cat <<USAGE
Usage:
  scripts/run_end_to_end.sh [options]

Options:
  --audio-glob <glob>     Audio input glob (default: data/raw/*.wav)
  --tracks <csv>          Track subset, e.g. drums,bass,guitar (optional)
  --normalize-key         Enable key normalization in the audio->MIDI pipeline
  --no-clean              Disable MIDI cleaning in the audio->MIDI pipeline
  --device <auto|cuda|mps|cpu>  Training/generation device (default: auto)

  --midi-dir <dir>        Where exported MIDIs go (default: out_midis)
  --events-dir <dir>      Where preprocessed event data goes (default: runs/events)
  --ckpt <path>           Where to save model checkpoint (default: runs/checkpoints/es_model.pt)
  --gen-out <path>        Output MIDI for generation (default: runs/generated/out.mid)

Examples:
  scripts/run_end_to_end.sh
  scripts/run_end_to_end.sh --tracks drums,bass,guitar --device cuda
  scripts/run_end_to_end.sh --audio-glob "data/raw/*.flac" --normalize-key
USAGE
}

# Parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --audio-glob) AUDIO_GLOB="$2"; shift 2 ;;
    --tracks) TRACKS="$2"; shift 2 ;;
    --normalize-key) PIPE_NORMALIZE_KEY=1; shift ;;
    --no-clean) PIPE_NO_CLEAN=1; shift ;;
    --device) DEVICE="$2"; shift 2 ;;
    --midi-dir) MIDI_DIR="$2"; shift 2 ;;
    --events-dir) EVENTS_DIR="$2"; shift 2 ;;
    --ckpt) CKPT_PATH="$2"; shift 2 ;;
    --gen-out) GEN_OUT="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1"; usage; exit 2 ;;
  esac
done

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIPE_DIR="$ROOT_DIR/vendor/all-in-one-ai-midi-pipeline"

# Make paths absolute so we can cd into the submodule safely
if [[ "$AUDIO_GLOB" != /* ]]; then
  AUDIO_GLOB="$ROOT_DIR/$AUDIO_GLOB"
fi
if [[ "$MIDI_DIR" != /* ]]; then
  MIDI_DIR="$ROOT_DIR/$MIDI_DIR"
fi
if [[ "$EVENTS_DIR" != /* ]]; then
  EVENTS_DIR="$ROOT_DIR/$EVENTS_DIR"
fi
if [[ "$CKPT_PATH" != /* ]]; then
  CKPT_PATH="$ROOT_DIR/$CKPT_PATH"
fi
if [[ "$GEN_OUT" != /* ]]; then
  GEN_OUT="$ROOT_DIR/$GEN_OUT"
fi


mkdir -p "$ROOT_DIR/runs" "$(dirname "$CKPT_PATH")" "$(dirname "$GEN_OUT")" "$MIDI_DIR" "$EVENTS_DIR"

echo "=== [1/4] Audio -> MIDI (vendored pipeline) ==="
PIPE_ARGS=()
PIPE_ARGS+=(run-batch "$AUDIO_GLOB")

if [[ "$PIPE_NORMALIZE_KEY" == "1" ]]; then
  PIPE_ARGS+=(--normalize-key)
fi
if [[ "$PIPE_NO_CLEAN" == "1" ]]; then
  PIPE_ARGS+=(--no-clean)
fi
if [[ -n "$TRACKS" ]]; then
  PIPE_ARGS+=(--tracks "$TRACKS")
fi

pushd "$PIPE_DIR" >/dev/null

python pipeline.py "${PIPE_ARGS[@]}"

python pipeline.py export-midi --out "$MIDI_DIR"

popd >/dev/null

echo "=== [2/4] Export MIDIs to: $MIDI_DIR ==="
pushd "$PIPE_DIR" >/dev/null
python pipeline.py export-midi --out "$MIDI_DIR"
popd >/dev/null

echo "=== [3/4] Preprocess MIDIs -> events: $EVENTS_DIR ==="
PRE_ARGS=(--midi_folder "$MIDI_DIR" --data_folder "$EVENTS_DIR")
if [[ -n "$TRACKS" ]]; then
  PRE_ARGS+=(--tracks "$TRACKS")
fi
python "$ROOT_DIR/training/pre.py" "${PRE_ARGS[@]}"

echo "=== [4/4] Train + Generate ==="
TRAIN_ARGS=(
  --data_dir "$EVENTS_DIR"
  --train_pkl "$EVENTS_DIR/events_train.pkl"
  --val_pkl "$EVENTS_DIR/events_val.pkl"
  --vocab_json "$EVENTS_DIR/event_vocab.json"
  --save_path "$CKPT_PATH"
  --device "$DEVICE"
)
python "$ROOT_DIR/training/train.py" "${TRAIN_ARGS[@]}"

GEN_ARGS=(
  --ckpt "$CKPT_PATH"
  --vocab_json "$EVENTS_DIR/event_vocab.json"
  --out_midi "$GEN_OUT"
  --device "$DEVICE"
)
if [[ -n "$TRACKS" ]]; then
  GEN_ARGS+=(--tracks "$TRACKS")
fi
python "$ROOT_DIR/training/generate.py" "${GEN_ARGS[@]}"

echo "DONE."
echo "Generated MIDI: $GEN_OUT"
