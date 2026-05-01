#!/usr/bin/env python3
"""Build HDF5 training data for the note discriminator.

Pipeline: Slakh2100 per-stem MIDI → FluidSynth render → simulated bleed mix
→ augmentation → basic-pitch detection → GT alignment → feature extraction → HDF5.

We have Slakh's MIDI/ stems (not the pre-rendered FLACs — the streaming download
captured MIDI before the tar archive reached the stems/ directories).  FluidSynth
renders audio with the per-stem GM program, giving timbral diversity without
needing the 104 GB FLAC archive.
"""

import argparse
import multiprocessing
import os
import random
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pretty_midi
import scipy.io.wavfile
import scipy.signal
import soundfile as sf

# --------------- constants -----------------------------------------------

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

SF2_CANDIDATES = [
    "/usr/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/soundfonts/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/TimGM6mb.sf2",
    "/usr/local/share/sounds/sf2/FluidR3_GM.sf2",
    "/usr/share/sounds/sf2/FluidR3_GS.sf2",
]


def find_sf2() -> str:
    for p in SF2_CANDIDATES:
        if Path(p).exists():
            return p
    raise FileNotFoundError("No SF2 soundfont found. Pass --sf2 explicitly. Searched: " + ", ".join(SF2_CANDIDATES))


# --------------- stem program map ----------------------------------------

def _prog_to_stem(prog: int, is_drum: bool):
    if is_drum:
        return None
    if 0 <= prog <= 7:
        return "other"      # piano
    if 16 <= prog <= 23:
        return "other"      # organ
    if 24 <= prog <= 31:
        return "guitar"
    if 32 <= prog <= 39:
        return "bass"
    if 80 <= prog <= 103:
        return "other"      # synth leads / pads
    return None


# --------------- augmentation --------------------------------------------

def apply_distortion(audio: np.ndarray, gain_db: float) -> np.ndarray:
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
    n_ir = int(rt60 * sr)
    if n_ir < 1:
        return audio
    t = np.arange(n_ir) / sr
    decay = np.exp(-6.908 * t / rt60)
    ir = np.random.default_rng(0).standard_normal(n_ir) * decay
    ir = ir / (np.linalg.norm(ir) + 1e-8)
    wet_sig = scipy.signal.fftconvolve(audio, ir)[: len(audio)]
    return (1 - wet) * audio + wet * wet_sig.astype(audio.dtype)


def apply_aug(audio: np.ndarray, sr: int, aug_name: str) -> np.ndarray:
    a = audio.astype(np.float32)
    if "dist_light"  in aug_name: a = apply_distortion(a,  6.0)
    if "dist_crunch" in aug_name: a = apply_distortion(a, 18.0)
    if "dist_heavy"  in aug_name: a = apply_distortion(a, 35.0)
    if "reverb_room" in aug_name: a = apply_reverb(a, sr, 0.3)
    if "reverb_hall" in aug_name: a = apply_reverb(a, sr, 1.2)
    return a


# --------------- FluidSynth rendering ------------------------------------

def render_fluidsynth(midi_path: Path, sf2: str, sr: int = 44100) -> np.ndarray | None:
    """Render MIDI with FluidSynth; return mono float32 or None on failure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_out = os.path.join(tmpdir, "render.wav")
        result = subprocess.run(
            ["fluidsynth", "-ni", "-F", wav_out, "-r", str(sr), sf2, str(midi_path)],
            capture_output=True, timeout=300,
        )
        if result.returncode != 0 or not Path(wav_out).exists():
            return None
        try:
            data, _ = sf.read(wav_out, dtype="float32", always_2d=True)
        except Exception:
            return None
    if data.size == 0:
        return None
    return data.mean(axis=1)


def get_program(midi_path: Path) -> tuple:
    """Return (program_num, is_drum) from first instrument in MIDI file."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    if not pm.instruments:
        return 0, False
    inst = pm.instruments[0]
    return inst.program, inst.is_drum


# --------------- mix / align / features ----------------------------------

def mix_with_bleed(primary: np.ndarray, bleeds: list, bleed_db: float = -20.0) -> np.ndarray:
    bleed_gain = 10 ** (bleed_db / 20.0)
    mixed = primary.copy()
    for b in bleeds:
        n = min(len(mixed), len(b))
        mixed[:n] += b[:n] * bleed_gain
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed /= peak
    return mixed


def get_gt_notes(midi_path: Path) -> list:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for inst in pm.instruments:
        notes.extend(inst.notes)
    notes.sort(key=lambda n: n.start)
    return notes


def align_notes(detected, gt_notes, pitch_tol=1, onset_tol=0.05):
    used = set()
    labels = []
    for det in detected:
        det_start, _det_end, det_pitch, *_ = det
        best_idx, best_dt = None, float("inf")
        for j, gn in enumerate(gt_notes):
            if j in used:
                continue
            if abs(int(gn.pitch) - int(det_pitch)) > pitch_tol:
                continue
            dt = abs(gn.start - det_start)
            if dt < onset_tol and dt < best_dt:
                best_dt, best_idx = dt, j
        if best_idx is not None:
            used.add(best_idx)
            labels.append(1)
        else:
            labels.append(0)
    return labels


def extract_features(note_events, stem_local_id: int) -> np.ndarray:
    if not note_events:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    starts  = np.array([e[0] for e in note_events], dtype=np.float32)
    ends    = np.array([e[1] for e in note_events], dtype=np.float32)
    pitches = np.array([int(e[2])   for e in note_events], dtype=np.float32)
    amps    = np.array([float(e[3]) for e in note_events], dtype=np.float32)
    durs    = ends - starts
    n       = len(note_events)

    polyphony = np.array(
        [float(np.sum((starts <= starts[i]) & (ends > starts[i]))) for i in range(n)],
        dtype=np.float32,
    )
    density = np.array(
        [float(np.sum(np.abs(starts - starts[i]) <= 0.05)) for i in range(n)],
        dtype=np.float32,
    )
    oct_rank = np.array(
        [float(np.sum(pitches[(starts <= starts[i]) & (ends > starts[i])] < pitches[i])) for i in range(n)],
        dtype=np.float32,
    )
    dur_z   = (durs    - durs.mean())    / (durs.std()    + 1e-8)
    pitch_r = (pitches - pitches.mean()) / (pitches.std() + 1e-8)

    return np.stack([
        amps, durs, pitches, np.full(n, stem_local_id, dtype=np.float32),
        polyphony, density, oct_rank, dur_z, pitch_r,
        (amps    > 0.7).astype(np.float32),
        (durs    < 0.05).astype(np.float32),
        (polyphony > 4).astype(np.float32),
    ], axis=1).astype(np.float32)


def run_basic_pitch(wav_path: str):
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    _, _midi, note_events = predict(wav_path, model_or_model_path=ICASSP_2022_MODEL_PATH)
    return note_events


# --------------- per-track worker ----------------------------------------

def process_track(task):
    """Render all qualifying stems for a track, then run all augmentations.

    Returns list of (feats, labels, stem_ids, names).
    """
    track_dir_str, sf2 = task
    track_dir = Path(track_dir_str)
    midi_dir  = track_dir / "MIDI"
    if not midi_dir.exists():
        return []

    # Step 1: render all stems once (FluidSynth uses the GM program in each MIDI).
    primary_stems = {}   # stem_id -> (audio, category, midi_path)
    all_audio     = {}   # stem_id -> audio   (for bleed, including non-target stems)

    for midi_path in sorted(midi_dir.glob("*.mid")):
        stem_id = midi_path.stem
        try:
            prog, is_drum = get_program(midi_path)
            audio = render_fluidsynth(midi_path, sf2)
            if audio is None or len(audio) == 0:
                continue
            all_audio[stem_id] = audio
            category = _prog_to_stem(prog, is_drum)
            if category is not None:
                primary_stems[stem_id] = (audio, category, midi_path)
        except Exception as exc:
            print(f"  SKIP {track_dir.name}/{stem_id} render: {exc}", flush=True)

    if not primary_stems:
        return []

    results = []
    for stem_id, (primary_audio, category, midi_path) in primary_stems.items():
        try:
            gt_notes = get_gt_notes(midi_path)
            if not gt_notes:
                continue

            # Simulate Demucs bleed: mix other stems at -20 dB.
            bleed_audios = [a for sid, a in all_audio.items() if sid != stem_id]
            mixed = mix_with_bleed(primary_audio, bleed_audios)

            stem_local = STEM_LOCAL_ID[category]

            for aug_name in AUGMENTATIONS:
                augmented = apply_aug(mixed, 44100, aug_name)
                with tempfile.TemporaryDirectory() as tmpdir:
                    wav_path = os.path.join(tmpdir, "aug.wav")
                    out_int = (augmented * 32767).clip(-32768, 32767).astype(np.int16)
                    scipy.io.wavfile.write(wav_path, 44100, out_int)
                    try:
                        note_events = run_basic_pitch(wav_path)
                    except Exception as exc:
                        print(f"  SKIP {track_dir.name}/{stem_id} bp({aug_name}): {exc}", flush=True)
                        continue

                if not note_events:
                    continue

                labels     = align_notes(note_events, gt_notes)
                feats      = extract_features(note_events, stem_local)
                labels_arr = np.array(labels, dtype=np.int8)
                n_tp       = int(labels_arr.sum())
                print(
                    f"{track_dir.name} | {category} ({stem_id}) | {aug_name} "
                    f"| {len(labels_arr)} notes, {n_tp} TP",
                    flush=True,
                )
                stem_ids_arr = np.full(len(labels_arr), stem_local, dtype=np.int8)
                names        = [f"{track_dir.name}/{stem_id}"] * len(labels_arr)
                results.append((feats, labels_arr, stem_ids_arr, names))

        except Exception as exc:
            print(f"  SKIP {track_dir.name}/{stem_id}: {exc}", flush=True)

    return results


# --------------- HDF5 helpers --------------------------------------------

def _init_h5(path: Path, n_features: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("features",    shape=(0, n_features), maxshape=(None, n_features), dtype="float32", chunks=(4096, n_features))
        f.create_dataset("labels",      shape=(0,),            maxshape=(None,),            dtype="int8",    chunks=(4096,))
        f.create_dataset("stem_ids",    shape=(0,),            maxshape=(None,),            dtype="int8",    chunks=(4096,))
        f.create_dataset("source_midi", shape=(0,),            maxshape=(None,),            dtype=dt,        chunks=(4096,))


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


# --------------- main ----------------------------------------------------

def main():
    ap = argparse.ArgumentParser("build_discriminator_data")
    ap.add_argument("--slakh_dir", default="data/slakh/train", help="Path to Slakh train split.")
    ap.add_argument("--out",       default="runs/discriminator_data/notes.h5")
    ap.add_argument("--n_tracks",  type=int, default=100)
    ap.add_argument("--workers",   type=int, default=1)
    ap.add_argument("--sf2",       default="",  help="Path to SF2 soundfont (auto-detected if blank).")
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    sf2 = args.sf2 or find_sf2()
    print(f"Soundfont: {sf2}")

    slakh_dir  = Path(args.slakh_dir)
    track_dirs = sorted(slakh_dir.glob("Track*"))
    if not track_dirs:
        print(f"ERROR: no Track* directories found in {slakh_dir}")
        raise SystemExit(1)

    random.seed(args.seed)
    sample = random.sample(track_dirs, min(args.n_tracks, len(track_dirs)))
    print(f"Processing {len(sample)} tracks from {slakh_dir}")
    print(f"Augmentations per stem: {len(AUGMENTATIONS)}")

    tasks    = [(str(td), sf2) for td in sample]
    out_path = Path(args.out)
    _init_h5(out_path, N_FEATURES)

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for results in pool.imap_unordered(process_track, tasks, chunksize=1):
            for feats, labels, stem_ids, names in results:
                _append_h5(out_path, feats, labels, stem_ids, names)

    with h5py.File(out_path, "r") as f:
        n_total = int(f["labels"].shape[0])
        n_tp    = int(np.sum(f["labels"][:]))
    print(f"\nDone. {n_total} notes written to {out_path}  (TP={n_tp}, FP={n_total - n_tp})")


if __name__ == "__main__":
    main()
