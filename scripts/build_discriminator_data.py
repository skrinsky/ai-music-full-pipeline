#!/usr/bin/env python3
"""Build HDF5 training data for the note discriminator.

Pipeline: Slakh2100 per-stem FLAC → simulated bleed mix → augmentation
→ basic-pitch detection → GT alignment → feature extraction → HDF5.
"""

import argparse
import multiprocessing
import os
import random
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pretty_midi
import scipy.io.wavfile
import scipy.signal
import soundfile as sf
import yaml

# --------------- stem local IDs ---------------
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


# --------------- stem program map ---------------
def _prog_to_stem(prog: int, is_drum: bool):
    if is_drum:
        return None
    if 0 <= prog <= 7:
        return "other"    # piano
    if 16 <= prog <= 23:
        return "other"    # organ
    if 24 <= prog <= 31:
        return "guitar"
    if 32 <= prog <= 39:
        return "bass"
    if 80 <= prog <= 103:
        return "other"    # synth leads / pads
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


# --------------- Slakh helpers ---------------
def load_slakh_track(track_dir: Path) -> dict:
    """Return {stem_id: {flac_path, midi_path, stem_category, program_num}}."""
    meta_path = track_dir / "metadata.yaml"
    if not meta_path.exists():
        return {}
    with meta_path.open() as fh:
        meta = yaml.safe_load(fh)

    stems_meta = meta.get("stems", {})
    result = {}
    for stem_id, info in stems_meta.items():
        if not info.get("audio_rendered", False):
            continue
        is_drum = info.get("is_drum", False)
        prog = info.get("program_num", -1)
        category = _prog_to_stem(prog, is_drum)
        if category is None:
            continue
        flac_path = track_dir / "stems" / f"{stem_id}.flac"
        midi_path = track_dir / "MIDI" / f"{stem_id}.mid"
        if not flac_path.exists() or not midi_path.exists():
            continue
        result[stem_id] = {
            "flac_path": flac_path,
            "midi_path": midi_path,
            "stem_category": category,
            "program_num": prog,
        }
    return result


def get_gt_notes(midi_path: Path) -> list:
    """Load pretty_midi, return sorted list of Note objects."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for inst in pm.instruments:
        notes.extend(inst.notes)
    notes.sort(key=lambda n: n.start)
    return notes


def _load_flac_mono(flac_path: Path, sr_target: int = 44100) -> np.ndarray:
    """Load FLAC, mix to mono float32 in [-1, 1]."""
    data, sr = sf.read(str(flac_path), dtype="float32", always_2d=True)
    audio = data.mean(axis=1)
    if sr != sr_target:
        # simple linear resample via scipy
        n_out = int(len(audio) * sr_target / sr)
        audio = scipy.signal.resample(audio, n_out).astype(np.float32)
    return audio


def mix_with_bleed(primary_audio: np.ndarray, bleed_audios: list, bleed_db: float = -20.0) -> np.ndarray:
    """Mix primary + bleed stems at bleed_db; normalize to [-1, 1]."""
    if not bleed_audios:
        mixed = primary_audio.copy()
    else:
        bleed_gain = 10 ** (bleed_db / 20.0)
        mixed = primary_audio.copy()
        for b in bleed_audios:
            # match lengths
            n = min(len(mixed), len(b))
            mixed[:n] += b[:n] * bleed_gain
    peak = np.max(np.abs(mixed))
    if peak > 0:
        mixed /= peak
    return mixed


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


# --------------- per-stem processor ---------------
def process_stem(stem_info: dict, all_stem_infos: dict, track_dir: Path, aug_name: str, sr: int = 44100):
    """Load primary stem + cross-stem bleed, augment, run basic-pitch, align GT.

    Returns (features_array, labels_array) or None.
    """
    primary_audio = _load_flac_mono(stem_info["flac_path"], sr)
    if primary_audio is None or len(primary_audio) == 0:
        return None

    # Bleed: all OTHER stems except drums (is_drum filtered out by _prog_to_stem already)
    bleed_audios = []
    primary_id = stem_info.get("_stem_id")
    for sid, sinfo in all_stem_infos.items():
        if sid == primary_id:
            continue
        try:
            b = _load_flac_mono(sinfo["flac_path"], sr)
            bleed_audios.append(b)
        except Exception:
            pass

    mixed = mix_with_bleed(primary_audio, bleed_audios)
    augmented = apply_aug(mixed, sr, aug_name)

    gt_notes = get_gt_notes(stem_info["midi_path"])
    if not gt_notes:
        return None

    with tempfile.TemporaryDirectory() as tmpdir:
        wav_path = os.path.join(tmpdir, "stem_aug.wav")
        out_int = (augmented * 32767).clip(-32768, 32767).astype(np.int16)
        scipy.io.wavfile.write(wav_path, sr, out_int)
        note_events = run_basic_pitch(wav_path)

    if not note_events:
        return None

    labels = align_notes(note_events, gt_notes)
    stem_local = STEM_LOCAL_ID[stem_info["stem_category"]]
    feats = extract_features(note_events, stem_local)
    labels_arr = np.array(labels, dtype=np.int8)
    return feats, labels_arr


# --------------- track worker ---------------
def process_track(task):
    """Worker: (track_dir_str, aug_name) → list of (feats, labels, stem_local_id, track_name)."""
    track_dir_str, aug_name = task
    track_dir = Path(track_dir_str)
    track_name = track_dir.name

    try:
        stem_infos = load_slakh_track(track_dir)
        if not stem_infos:
            return []

        # Annotate each stem_info with its own stem_id for bleed exclusion.
        for sid, sinfo in stem_infos.items():
            sinfo["_stem_id"] = sid

        results = []
        for stem_id, sinfo in stem_infos.items():
            try:
                ret = process_stem(sinfo, stem_infos, track_dir, aug_name)
                if ret is None:
                    continue
                feats, labels_arr = ret
                stem_local = STEM_LOCAL_ID[sinfo["stem_category"]]
                n_det = len(labels_arr)
                n_tp = int(labels_arr.sum())
                print(
                    f"{track_name} | {sinfo['stem_category']} ({stem_id}) | {aug_name} | "
                    f"{n_det} notes detected, {n_tp} TP",
                    flush=True,
                )
                stem_ids = np.full(n_det, stem_local, dtype=np.int8)
                names = [f"{track_name}/{stem_id}"] * n_det
                results.append((feats, labels_arr, stem_ids, names))
            except Exception as exc:
                print(
                    f"  SKIP {track_name}/{stem_id} aug={aug_name}: {exc}",
                    flush=True,
                )
        return results

    except Exception as exc:
        print(f"  SKIP {track_name} aug={aug_name}: {exc}", flush=True)
        return []


# --------------- HDF5 helpers ---------------
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


# --------------- main ---------------
def main():
    ap = argparse.ArgumentParser("build_discriminator_data: build HDF5 note discriminator training set.")
    ap.add_argument("--slakh_dir", default="data/slakh/train", help="Slakh2100 train split directory.")
    ap.add_argument("--out",       default="runs/discriminator_data/notes.h5", help="Output HDF5 path.")
    ap.add_argument("--n_tracks",  type=int, default=100,  help="Number of tracks to sample.")
    ap.add_argument("--workers",   type=int, default=1,    help="Multiprocessing pool size.")
    ap.add_argument("--seed",      type=int, default=42)
    args = ap.parse_args()

    slakh_dir = Path(args.slakh_dir)
    track_dirs = sorted(slakh_dir.glob("Track*"))
    if not track_dirs:
        print(f"ERROR: no Track* directories found in {slakh_dir}")
        raise SystemExit(1)

    random.seed(args.seed)
    sample = random.sample(track_dirs, min(args.n_tracks, len(track_dirs)))
    print(f"Processing {len(sample)} tracks from {slakh_dir}")

    # Build task list: (track_dir_str, aug_name)
    tasks = []
    for track_dir in sample:
        for aug in AUGMENTATIONS:
            tasks.append((str(track_dir), aug))

    print(f"Total tasks: {len(tasks)} ({len(sample)} tracks × {len(AUGMENTATIONS)} augmentations)")

    out_path = Path(args.out)
    _init_h5(out_path, N_FEATURES)

    ctx = multiprocessing.get_context("spawn")
    with ctx.Pool(processes=args.workers) as pool:
        for results in pool.imap_unordered(process_track, tasks, chunksize=1):
            for feats, labels, stem_ids, names in results:
                _append_h5(out_path, feats, labels, stem_ids, names)

    with h5py.File(out_path, "r") as f:
        n_total = f["labels"].shape[0]
        n_tp = int(np.sum(f["labels"][:]))
    print(f"\nDone. {n_total} notes written to {out_path}  (TP={n_tp}, FP={n_total - n_tp})")


if __name__ == "__main__":
    main()
