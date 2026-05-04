#!/usr/bin/env python3
"""Generate a handful of bass augmentation WAVs for listening.

Picks N random Slakh tracks that have a bass stem, runs the full
training pipeline (demucs separation → NAM amp sim → augmentations),
and saves one WAV per augmentation to --out_dir.

Usage (on server):
  python scripts/preview_bass_augmentations.py \
      --slakh_dir data/slakh/train \
      --nam_dir   data/nam_models \
      --n_tracks  3 \
      --out_dir   /tmp/bass_previews

Then locally:
  scp -r server:/tmp/bass_previews ./bass_previews
"""

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import scipy.signal
import soundfile as sf

REPO = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(REPO))

from scripts.build_discriminator_data import (
    AUGMENTATIONS_BASS,
    apply_aug,
    apply_nam_amp,
    _prog_to_stem,
    _nam_models,
    get_program,
    _load_flac_mono,
)
import scripts.build_discriminator_data as _bdd


def load_nam_models(nam_dir: str):
    manifest_path = Path(nam_dir) / "manifest.json"
    if not manifest_path.exists():
        print(f"No manifest.json in {nam_dir} — skipping amp sim")
        return
    manifest = json.loads(manifest_path.read_text())
    try:
        from nam.models import init_from_nam
        for cat, paths in manifest.items():
            loaded = []
            for p in paths:
                try:
                    with open(p) as f:
                        cfg = json.load(f)
                    m = init_from_nam(cfg)
                    m.eval()
                    loaded.append(m)
                    print(f"  NAM loaded: {cat} → {Path(p).name}")
                except Exception as e:
                    print(f"  NAM skip {Path(p).name}: {e}")
            if loaded:
                _nam_models[cat] = loaded
    except Exception as e:
        print(f"  NAM load failed: {e}")


def separate_bass(mix_mono: np.ndarray, sr: int) -> np.ndarray | None:
    import torch
    model = _bdd._demucs_model
    if sr != model.samplerate:
        n_out = int(len(mix_mono) * model.samplerate / sr)
        mix_mono = np.interp(np.linspace(0, len(mix_mono) - 1, n_out),
                             np.arange(len(mix_mono)), mix_mono).astype(np.float32)
    stereo = torch.tensor(np.stack([mix_mono, mix_mono])).unsqueeze(0)
    with torch.no_grad():
        from demucs.apply import apply_model
        out = apply_model(model, stereo, progress=False)
    out_np = out.squeeze(0).mean(dim=1).cpu().numpy()
    sources = {name: out_np[i] for i, name in enumerate(model.sources)}
    return sources.get("bass")


def save_wav(path: Path, audio: np.ndarray, sr: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    scipy.io.wavfile.write(str(path), sr, pcm)


def process_track(track_dir: Path, out_dir: Path, sr: int = 44100) -> bool:
    midi_dir = track_dir / "MIDI"
    if not midi_dir.exists():
        return False

    all_audio = {}
    has_bass = False
    for midi_path in sorted(midi_dir.glob("*.mid")):
        flac = track_dir / "stems" / f"{midi_path.stem}.flac"
        if not flac.exists():
            continue
        try:
            prog, is_drum = get_program(midi_path)
            audio = _load_flac_mono(flac, sr)
            if audio is None or len(audio) == 0:
                continue
            all_audio[midi_path.stem] = audio
            if _prog_to_stem(prog, is_drum) == "bass":
                has_bass = True
        except Exception:
            continue

    if not has_bass or not all_audio:
        return False

    print(f"  Running demucs on {track_dir.name} ...")
    mix = np.zeros(max(len(a) for a in all_audio.values()), dtype=np.float32)
    for a in all_audio.values():
        mix[:len(a)] += a
    peak = np.abs(mix).max()
    if peak > 0:
        mix /= peak

    try:
        bass_audio = separate_bass(mix, sr)
    except Exception as e:
        print(f"  demucs failed: {e}")
        return False

    if bass_audio is None or len(bass_audio) == 0:
        return False

    # Resample back to original sr if needed
    if _bdd._demucs_model.samplerate != sr:
        n_out = int(len(bass_audio) * sr / _bdd._demucs_model.samplerate)
        bass_audio = np.interp(np.linspace(0, len(bass_audio) - 1, n_out),
                               np.arange(len(bass_audio)), bass_audio).astype(np.float32)

    # Raw demucs output — reference before any processing
    save_wav(out_dir / f"{track_dir.name}_00_demucs_raw.wav", bass_audio, sr)

    # NAM amp sim then each augmentation
    amped = apply_nam_amp(bass_audio, sr, "bass")
    save_wav(out_dir / f"{track_dir.name}_01_amped_clean.wav", amped, sr)

    for i, aug_name in enumerate(AUGMENTATIONS_BASS):
        augmented = apply_aug(amped, sr, aug_name)
        fname = f"{track_dir.name}_{i+2:02d}_{aug_name}.wav"
        save_wav(out_dir / fname, augmented, sr)
        print(f"    saved {fname}")

    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slakh_dir", default="data/slakh/train")
    ap.add_argument("--nam_dir",   default="data/nam_models")
    ap.add_argument("--out_dir",   default="/tmp/bass_previews")
    ap.add_argument("--n_tracks",  type=int, default=3)
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)

    print("Loading NAM models ...")
    load_nam_models(args.nam_dir)

    print("Loading demucs ...")
    from demucs.pretrained import get_model
    _bdd._demucs_model = get_model('htdemucs_6s')
    _bdd._demucs_model.eval()

    slakh_dir = Path(args.slakh_dir)
    track_dirs = sorted(slakh_dir.glob("Track*"))
    random.shuffle(track_dirs)

    out_dir = Path(args.out_dir)
    done = 0
    for td in track_dirs:
        if done >= args.n_tracks:
            break
        print(f"\nProcessing {td.name} ...")
        if process_track(td, out_dir):
            done += 1

    print(f"\nDone. {done} tracks → {out_dir}")
    print(f"Per track: raw demucs + amped + {len(AUGMENTATIONS_BASS)} augmentations")
    print(f"\nTo download:\n  scp -r <server>:{out_dir} ./bass_previews")


if __name__ == "__main__":
    main()
