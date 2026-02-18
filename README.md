# AI Music Full Pipeline

This repo orchestrates an end-to-end workflow:

**Stereo audio → stems/MIDI (submodule) → event preprocessing → training → generation**

The audio→MIDI stage is included as a git submodule under `vendor/all-in-one-ai-midi-pipeline/`.

## Repo layout


ai-music-full-pipeline/
vendor/
all-in-one-ai-midi-pipeline/ # submodule (audio -> MIDI)
training/
pre.py # MIDI -> events (+ optional track filtering)
train.py # train model
generate.py # generate MIDI from trained model
configs/
scripts/
run_end_to_end.sh # orchestration script
README.md
.gitignore


## Clone (with submodule)

If you haven't cloned yet:

```bash
git clone --recurse-submodules <THIS_REPO_URL>


If you already cloned without submodules:

git submodule update --init --recursive

Setup

Create a Python environment and install dependencies (you may want separate envs for pipeline vs training later; start with one env for now):

python3.10 -m venv .venv
source .venv/bin/activate
pip install -r vendor/all-in-one-ai-midi-pipeline/requirements.txt
# plus whatever your training scripts require (torch, etc.)

End-to-end run

Put audio in data/raw/ (ignored by git):

mkdir -p data/raw
# copy audio into data/raw/


Run the full pipeline:

scripts/run_end_to_end.sh

Select a subset of tracks

Example: only drums + bass:

scripts/run_end_to_end.sh --tracks drums,bass


Example: drums + bass + guitar:

scripts/run_end_to_end.sh --tracks drums,bass,guitar

Device selection

Training/generation device defaults to auto:

CUDA if available

else Apple MPS if available

else CPU

Override:

scripts/run_end_to_end.sh --device cuda
scripts/run_end_to_end.sh --device mps
scripts/run_end_to_end.sh --device cpu

Outputs

Generated / intermediate outputs live in (all ignored by git):

out_midis/ exported MIDIs

runs/events/ preprocessed event datasets

runs/checkpoints/ trained checkpoints

runs/generated/ generated MIDI outputs
