#!/usr/bin/env python3
"""Build HDF5 training data for the note discriminator.

Pipeline: blues MIDIs → per-stem MIDI → FluidSynth render → augmentation
→ basic-pitch detection → GT alignment → feature extraction → HDF5.
"""

import argparse
import multiprocessing
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pretty_midi
import scipy.io.wavfile
import scipy.signal

# --------------- stem program maps ---------------
STEM_PROGRAMS = {
    "guitar": [24, 25, 26, 27, 28, 29, 30, 31],
    "bass":   [32, 33, 34, 35, 36, 37, 38, 39],
    "other":  [80, 81, 82, 83, 88, 89, 90, 91],
}
STEM_TO_SLOT = {"guitar": 2, "bass": 4, "other": 3}
STEM_LOCAL_ID = {"guitar": 0, "bass": 1, "other": 2}

FEATURE_NAMES = [
    "amplitude",
    "duration_s",
    "pitch",
    "stem_id",
    "polyphony",
    "density_100ms",
    "octave_rank",
    "duration_zscore",
    "pitch_rel",
    "hi_conf_flag",
    "short_flag",
    "hi_poly_flag",
]
N_FEATURES = len(FEATURE_NAMES)

# --------------- GM program → stem (inline replicate of _slot_from_gm_program logic) ---------------
def _prog_to_stem(prog: int):
    """Return stem name or None."""
    if 24 <= prog <= 31:
        return "guitar"
    if 32 <= prog <= 39:
        return "bass"
    if 16 <= prog <= 23:   # organ family → other
        return "other"
    if 80 <= prog <= 103:  # synth leads + pads + effects → other
        return "other"
    return None


# --------------- augmentation ---------------
def apply_distortion(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Soft-clip via tanh; renormalize to original peak."""
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    gain_lin = 10 ** (gain_db / 20.0)
    clipped = np.tanh(audio * gain_lin)
    new_peak = np.max(np.abs(clipped))
    if new_peak > 0:
        clipped = clipped * (peak / new_peak)
    return clipped


def apply_reverb(audio: np.ndarray, sr: int, rt60: float, wet: float = 0.3) -> np.ndarray:
    """Convolve with exponential-decay noise IR; wet=30%."""
    n_ir = int(rt60 * sr)
    if n_ir < 1:
        return audio
    t = np.arange(n_ir) / sr
    decay = np.exp(-6.908 * t / rt60)   # -60 dB at rt60
    ir = np.random.default_rng(0).standard_normal(n_ir) * decay
    ir = ir / (np.linalg.norm(ir) + 1e-8)
    wet_sig = scipy.signal.fftconvolve(audio, ir)[: len(audio)]
    return (1 - wet) * audio + wet * wet_sig.astype(audio.dtype)


AUGMENTATIONS = [
    "clean",
    "dist_light",
    "dist_crunch",
    "dist_heavy",
    "reverb_room",
    "reverb_hall",
    "dist_light+reverb_room",
    "dist_heavy+reverb_hall",
]


def apply_aug(audio: np.ndarray, sr: int, aug_name: str) -> np.ndarray:
    a = audio.astype(np.float32)
    if "dist_light" in aug_name:
        a = apply_distortion(a, 6.0)
    if "dist_crunch" in aug_name:
        a = apply_distortion(a, 18.0)
    if "dist_heavy" in aug_name:
        a = apply_distortion(a, 35.0)
    if "reverb_room" in aug_name:
        a = apply_reverb(a, sr, 0.3)
    if "reverb_hall" in aug_name:
        a = apply_reverb(a, sr, 1.2)
    return a


# --------------- SF2 / FluidSynth ---------------
def _find_sf2() -> str:
    candidates = [
        Path.home() / "Library/Audio/Sounds/Banks/FluidR3_GM.sf2",
        Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
        Path("/usr/share/soundfonts/default.sf2"),
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return ""


def _check_fluidsynth():
    if shutil.which("fluidsynth") is None:
        print(
            "ERROR: fluidsynth not found on PATH.\n"
            "Install with: brew install fluidsynth   (macOS)\n"
            "              apt-get install fluidsynth (Ubuntu)\n"
        )
        sys.exit(1)


def render_midi_to_wav(midi_path: str, wav_path: str, sf2: str, sr: int = 44100):
    """Render MIDI to WAV via FluidSynth CLI."""
    cmd = ["fluidsynth", "-ni", "-F", wav_path, "-r", str(sr), sf2, midi_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FluidSynth failed: {result.stderr[:200]}")
    if not Path(wav_path).exists():
        raise RuntimeError(f"FluidSynth produced no output file: {wav_path}")


def load_wav_mono(wav_path: str):
    """Load WAV, mix to mono float32 in [-1, 1]."""
    sr, data = scipy.io.wavfile.read(wav_path)
    data = data.astype(np.float32)
    if data.ndim == 2:
        data = data.mean(axis=1)
    peak = np.max(np.abs(data))
    if peak > 0:
        data /= peak
    return data, sr


# --------------- per-stem MIDI ---------------
def build_stem_midi(source_pm: pretty_midi.PrettyMIDI, stem: str, render_program: int) -> pretty_midi.PrettyMIDI:
    """Return a PrettyMIDI with only instruments matching stem, re-programmed."""
    out = pretty_midi.PrettyMIDI(initial_tempo=120.0)
    # copy tempo changes
    tc = source_pm.get_tempo_changes()
    if tc[1].size > 0:
        out = pretty_midi.PrettyMIDI(initial_tempo=float(tc[1][0]))

    for inst in source_pm.instruments:
        if inst.is_drum:
            continue
        if _prog_to_stem(inst.program) != stem:
            continue
        new_inst = pretty_midi.Instrument(program=render_program, is_drum=False, name=stem)
        new_inst.notes = [pretty_midi.Note(
            velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end
        ) for n in inst.notes]
        if new_inst.notes:
            out.instruments.append(new_inst)
    return out


# --------------- GT alignment ---------------
def align_notes(detected, gt_notes, pitch_tol=1, onset_tol=0.05):
    """Greedy nearest-neighbor TP/FP labelling.

    detected: list of (start_s, end_s, pitch, amplitude, pitch_bends)
    gt_notes: list of pretty_midi.Note
    Returns list of int labels (1=TP, 0=FP), same length as detected.
    """
    used = set()
    labels = []
    for det in detected:
        det_start, det_end, det_pitch, *_ = det
        best_idx = None
        best_dt = float("inf")
        for j, gn in enumerate(gt_notes):
            if j in used:
                continue
            if abs(int(gn.pitch) - int(det_pitch)) > pitch_tol:
                continue
            dt = abs(gn.start - det_start)
            if dt < onset_tol and dt < best_dt:
                best_dt = dt
                best_idx = j
        if best_idx is not None:
            used.add(best_idx)
            labels.append(1)
        else:
            labels.append(0)
    return labels


# --------------- feature extraction ---------------
def extract_features(note_events, stem_local_id: int) -> np.ndarray:
    """Return (N, 12) float32 feature matrix from basic-pitch note_events."""
    if not note_events:
        return np.zeros((0, N_FEATURES), dtype=np.float32)

    starts = np.array([e[0] for e in note_events], dtype=np.float32)
    ends   = np.array([e[1] for e in note_events], dtype=np.float32)
    pitches = np.array([int(e[2]) for e in note_events], dtype=np.float32)
    amps    = np.array([float(e[3]) for e in note_events], dtype=np.float32)
    durs    = ends - starts

    n = len(note_events)

    # polyphony: number of notes whose interval overlaps onset of this note
    polyphony = np.zeros(n, dtype=np.float32)
    for i in range(n):
        t = starts[i]
        polyphony[i] = float(np.sum((starts <= t) & (ends > t)))

    # density_100ms: notes starting within ±50ms
    density = np.zeros(n, dtype=np.float32)
    for i in range(n):
        density[i] = float(np.sum(np.abs(starts - starts[i]) <= 0.05))

    # octave_rank: rank among simultaneously sounding notes (0 = lowest)
    octave_rank = np.zeros(n, dtype=np.float32)
    for i in range(n):
        t = starts[i]
        sim_pitches = pitches[(starts <= t) & (ends > t)]
        octave_rank[i] = float(np.sum(sim_pitches < pitches[i]))

    # zscore stats across ALL notes in this render
    dur_mean = durs.mean()
    dur_std  = durs.std() + 1e-8
    pitch_mean = pitches.mean()
    pitch_std  = pitches.std() + 1e-8

    dur_z   = (durs   - dur_mean)   / dur_std
    pitch_r = (pitches - pitch_mean) / pitch_std

    feats = np.stack([
        amps,
        durs,
        pitches,
        np.full(n, stem_local_id, dtype=np.float32),
        polyphony,
        density,
        octave_rank,
        dur_z,
        pitch_r,
        (amps > 0.7).astype(np.float32),
        (durs < 0.05).astype(np.float32),
        (polyphony > 4).astype(np.float32),
    ], axis=1).astype(np.float32)
    return feats


# --------------- basic-pitch wrapper ---------------
def run_basic_pitch(wav_path: str):
    """Return note_events list; each item: (start_s, end_s, pitch, amplitude, pitch_bends)."""
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    _, _midi_data, note_events = predict(wav_path, model_or_model_path=ICASSP_2022_MODEL_PATH)
    return note_events


# --------------- worker ---------------
def process_task(task):
    """Worker: (midi_path, stem, program, aug_name, sf2) → (features, labels, stem_ids, midi_name)."""
    midi_path, stem, program, aug_name, sf2 = task
    midi_name = Path(midi_path).name
    print(f"  [{midi_name}] stem={stem} prog={program} aug={aug_name}", flush=True)

    try:
        source_pm = pretty_midi.PrettyMIDI(midi_path)
        stem_pm = build_stem_midi(source_pm, stem, program)
        if not stem_pm.instruments:
            return None

        gt_notes = []
        for inst in stem_pm.instruments:
            gt_notes.extend(inst.notes)
        if not gt_notes:
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            stem_midi_path = os.path.join(tmpdir, "stem.mid")
            stem_pm.write(stem_midi_path)

            raw_wav = os.path.join(tmpdir, "raw.wav")
            render_midi_to_wav(stem_midi_path, raw_wav, sf2)

            audio, sr = load_wav_mono(raw_wav)
            audio = apply_aug(audio, sr, aug_name)

            aug_wav = os.path.join(tmpdir, "aug.wav")
            out_int = (audio * 32767).clip(-32768, 32767).astype(np.int16)
            scipy.io.wavfile.write(aug_wav, sr, out_int)

            note_events = run_basic_pitch(aug_wav)

        if not note_events:
            return None

        labels = align_notes(note_events, gt_notes)
        stem_local = STEM_LOCAL_ID[stem]
        feats = extract_features(note_events, stem_local)
        labels_arr = np.array(labels, dtype=np.int8)
        stem_ids = np.full(len(note_events), stem_local, dtype=np.int8)
        names = [midi_name] * len(note_events)
        return feats, labels_arr, stem_ids, names

    except Exception as exc:
        print(f"  SKIP [{midi_name}] stem={stem} prog={program} aug={aug_name}: {exc}", flush=True)
        return None


# --------------- HDF5 helpers ---------------
def _init_h5(path: Path, n_features: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("features",   shape=(0, n_features), maxshape=(None, n_features), dtype="float32", chunks=(4096, n_features))
        f.create_dataset("labels",     shape=(0,),            maxshape=(None,),            dtype="int8",    chunks=(4096,))
        f.create_dataset("stem_ids",   shape=(0,),            maxshape=(None,),            dtype="int8",    chunks=(4096,))
        f.create_dataset("source_midi",shape=(0,),            maxshape=(None,),            dtype=dt,        chunks=(4096,))


def _append_h5(path: Path, feats, labels, stem_ids, names):
    n = len(labels)
    with h5py.File(path, "a") as f:
        for ds_name, data in [
            ("features",    feats),
            ("labels",      labels),
            ("stem_ids",    stem_ids),
            ("source_midi", np.array(names, dtype=object)),
        ]:
            ds = f[ds_name]
            old = ds.shape[0]
            ds.resize(old + n, axis=0)
            ds[old:] = data


# --------------- main ---------------
def main():
    ap = argparse.ArgumentParser("build_discriminator_data: build HDF5 note discriminator training set.")
    ap.add_argument("--midi_dir", default="data/blues_midi", help="Source MIDI directory.")
    ap.add_argument("--out",      default="runs/discriminator_data/notes.h5", help="Output HDF5 path.")
    ap.add_argument("--n_midis",  type=int, default=50,   help="Number of MIDIs to sample.")
    ap.add_argument("--workers",  type=int, default=4,    help="Multiprocessing pool size.")
    ap.add_argument("--sf2",      default="",             help="Override SF2 path (auto-detect if empty).")
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    _check_fluidsynth()

    sf2 = args.sf2 if args.sf2 else _find_sf2()
    if not sf2:
        print(
            "ERROR: no SF2 soundfont found. Download FluidR3_GM.sf2 and pass --sf2 <path>."
        )
        sys.exit(1)
    print(f"Using SF2: {sf2}")

    midi_dir = Path(args.midi_dir)
    midi_files = sorted(midi_dir.glob("*.mid")) + sorted(midi_dir.glob("*.midi"))
    if not midi_files:
        print(f"ERROR: no MIDI files found in {midi_dir}")
        sys.exit(1)

    random.seed(args.seed)
    sample = random.sample(midi_files, min(args.n_midis, len(midi_files)))
    print(f"Processing {len(sample)} MIDIs from {midi_dir}")

    # Build task list: (midi_path, stem, program, aug, sf2)
    tasks = []
    for midi_path in sample:
        for stem, programs in STEM_PROGRAMS.items():
            for program in programs:
                for aug in AUGMENTATIONS:
                    tasks.append((str(midi_path), stem, program, aug, sf2))

    print(f"Total tasks: {len(tasks)}")

    out_path = Path(args.out)
    _init_h5(out_path, N_FEATURES)

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for result in pool.imap_unordered(process_task, tasks, chunksize=1):
            if result is None:
                continue
            feats, labels, stem_ids, names = result
            _append_h5(out_path, feats, labels, stem_ids, names)

    with h5py.File(out_path, "r") as f:
        n_total = f["labels"].shape[0]
        n_tp = int(np.sum(f["labels"][:]))
    print(f"\nDone. {n_total} notes written to {out_path}  (TP={n_tp}, FP={n_total-n_tp})")


if __name__ == "__main__":
    main()
