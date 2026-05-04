#!/usr/bin/env python3
"""Build HDF5 training data for the combined note discriminator.

Default pipeline (--use-demucs, recommended):
  Slakh2100 per-stem audio → full mix → htdemucs_6s separation →
  augmentation → basic-pitch → GT alignment → scalar features + mel patches → HDF5.

Legacy pipeline (without --use-demucs):
  Slakh2100 per-stem audio → simulated bleed mix → augmentation →
  basic-pitch → GT alignment → scalar features + mel patches → HDF5.

HDF5 datasets written per note:
  features    (N, 12)           float32  — timbre-invariant scalar features
  spec_patches(N, n_mel, n_frames) float16 — log-mel patch centred on onset
  labels      (N,)              int8     — 1=TP, 0=FP
  stem_ids    (N,)              int8     — 0=guitar, 1=bass, 2=other
  source_midi (N,)              str      — "TrackXXXXX/SYY"
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

try:
    import librosa
    def compute_log_mel(audio: np.ndarray, sr: int, n_mels: int, hop_length: int) -> np.ndarray:
        mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
        return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
except ImportError:
    def _mel_filterbank(sr, n_fft, n_mels):
        hz2mel = lambda hz: 2595 * np.log10(1 + hz / 700.0)
        mel2hz = lambda m: 700 * (10 ** (m / 2595.0) - 1)
        mel_pts = np.linspace(hz2mel(0), hz2mel(sr / 2), n_mels + 2)
        hz_pts  = mel2hz(mel_pts)
        bins    = np.floor((n_fft + 1) * hz_pts / sr).astype(int).clip(0, n_fft // 2)
        fb      = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
        for i in range(n_mels):
            s, c, e = bins[i], bins[i + 1], bins[i + 2]
            if c > s: fb[i, s:c] = np.linspace(0, 1, c - s)
            if e > c: fb[i, c:e] = np.linspace(1, 0, e - c)
        return fb

    def compute_log_mel(audio: np.ndarray, sr: int, n_mels: int, hop_length: int) -> np.ndarray:
        n_fft = hop_length * 4
        _, _, Zxx = scipy.signal.stft(audio, fs=sr, nperseg=n_fft,
                                      noverlap=n_fft - hop_length, window="hann")
        power   = np.abs(Zxx) ** 2
        mel_fb  = _mel_filterbank(sr, n_fft, n_mels)
        mel     = mel_fb @ power
        log_mel = 10.0 * np.log10(mel + 1e-8)
        log_mel -= log_mel.max()
        return log_mel.astype(np.float32)


# --------------- constants -----------------------------------------------

STEM_LOCAL_ID = {"guitar": 0, "bass": 1, "other": 2}

FEATURE_NAMES = [
    "amplitude", "duration_s", "pitch", "stem_id", "polyphony",
    "density_100ms", "octave_rank", "duration_zscore", "pitch_rel",
    "hi_conf_flag", "short_flag", "hi_poly_flag",
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
    raise FileNotFoundError(
        "No SF2 soundfont found. Pass --sf2 explicitly. Searched: " + ", ".join(SF2_CANDIDATES)
    )


# --------------- demucs separation ---------------------------------------

# htdemucs_6s source order: drums, bass, other, vocals, guitar, piano
_DEMUCS_SOURCES = ['drums', 'bass', 'other', 'vocals', 'guitar', 'piano']
_DEMUCS_TARGET_STEMS = {'guitar', 'bass', 'other'}  # stems we care about

# Worker-local models (loaded once per process via initializer)
_demucs_model = None
_nam_models: dict = {}   # category → list of nam models


def _init_worker(use_demucs: bool, nam_dir: str = ""):
    global _demucs_model
    if use_demucs:
        from demucs.pretrained import get_model
        _demucs_model = get_model('htdemucs_6s')
        _demucs_model.eval()

    if nam_dir:
        import json as _json
        from pathlib import Path as _Path
        manifest_path = _Path(nam_dir) / "manifest.json"
        if manifest_path.exists():
            manifest = _json.loads(manifest_path.read_text())
            try:
                from nam.models import init_from_nam as _init_nam
                for cat, paths in manifest.items():
                    loaded = []
                    for p in paths:
                        try:
                            with open(p) as f:
                                cfg = _json.load(f)
                            m = _init_nam(cfg)
                            m.eval()
                            loaded.append(m)
                        except Exception as e:
                            print(f"  NAM skip {_Path(p).name}: {e}", flush=True)
                    if loaded:
                        _nam_models[cat] = loaded
                        print(f"  NAM loaded: {cat} → {len(loaded)} model(s)", flush=True)
            except Exception as e:
                print(f"  NAM load failed (skipping amp sim): {e}", flush=True)


def apply_nam_amp(audio: np.ndarray, sr: int, category: str) -> np.ndarray:
    """Run audio through a randomly-selected NAM amp model for this instrument category.
    Falls back to identity if no models loaded for this category."""
    models = _nam_models.get(category) or _nam_models.get("other")
    if not models:
        return audio
    model = random.choice(models)
    import torch
    nam_sr = 48000
    if sr != nam_sr:
        n_out = int(len(audio) * nam_sr / sr)
        a48 = np.interp(np.linspace(0, len(audio) - 1, n_out),
                        np.arange(len(audio)), audio).astype(np.float32)
    else:
        a48 = audio.astype(np.float32)
    with torch.no_grad():
        out48 = model(torch.from_numpy(a48).unsqueeze(0)).squeeze(0).numpy()
    if sr != nam_sr:
        out = np.interp(np.linspace(0, len(out48) - 1, len(audio)),
                        np.arange(len(out48)), out48).astype(np.float32)
    else:
        out = out48
    peak = np.abs(out).max()
    return out / peak if peak > 0 else out


def separate_with_demucs(mix_mono: np.ndarray, sr: int = 44100) -> dict:
    """Run htdemucs_6s on a mono mix; return {stem_name: mono_float32 array}."""
    import torch
    model = _demucs_model
    # demucs expects stereo (2, T) at model.samplerate
    if sr != model.samplerate:
        n_out = int(len(mix_mono) * model.samplerate / sr)
        mix_mono = np.interp(
            np.linspace(0, len(mix_mono) - 1, n_out),
            np.arange(len(mix_mono)),
            mix_mono,
        ).astype(np.float32)
    stereo = torch.tensor(np.stack([mix_mono, mix_mono])).unsqueeze(0)  # (1, 2, T)
    with torch.no_grad():
        from demucs.apply import apply_model
        out = apply_model(model, stereo, progress=False)  # (1, sources, 2, T)
    out_np = out.squeeze(0).mean(dim=1).cpu().numpy()     # (sources, T) mono
    return {name: out_np[i] for i, name in enumerate(model.sources)}


# --------------- stem program map ----------------------------------------

def _prog_to_stem(prog: int, is_drum: bool):
    if is_drum:
        return None
    if  0 <= prog <=  7: return "other"   # piano
    if 16 <= prog <= 23: return "other"   # organ
    if 24 <= prog <= 31: return "guitar"
    if 32 <= prog <= 39: return "bass"
    if 80 <= prog <= 103: return "other"  # synth leads / pads
    return None


# --------------- augmentation --------------------------------------------

def apply_distortion(audio: np.ndarray, gain_db: float) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    clipped = np.tanh(audio * 10 ** (gain_db / 20.0))
    new_peak = np.max(np.abs(clipped))
    return clipped * (peak / new_peak) if new_peak > 0 else clipped


def apply_reverb(audio: np.ndarray, sr: int, rt60: float, wet: float = 0.3) -> np.ndarray:
    n_ir = int(rt60 * sr)
    if n_ir < 1:
        return audio
    t      = np.arange(n_ir) / sr
    ir     = np.random.default_rng(0).standard_normal(n_ir) * np.exp(-6.908 * t / rt60)
    ir    /= np.linalg.norm(ir) + 1e-8
    wet_s  = scipy.signal.fftconvolve(audio, ir)[: len(audio)]
    return ((1 - wet) * audio + wet * wet_s).astype(audio.dtype)


def apply_aug(audio: np.ndarray, sr: int, aug_name: str) -> np.ndarray:
    a = audio.astype(np.float32)
    if "dist_light"  in aug_name: a = apply_distortion(a,  6.0)
    if "dist_crunch" in aug_name: a = apply_distortion(a, 18.0)
    if "dist_heavy"  in aug_name: a = apply_distortion(a, 35.0)
    if "reverb_room" in aug_name: a = apply_reverb(a, sr, 0.3)
    if "reverb_hall" in aug_name: a = apply_reverb(a, sr, 1.2)
    return a


# --------------- audio loading / rendering -------------------------------

def _load_flac_mono(flac_path: Path, sr_target: int = 44100) -> np.ndarray:
    data, sr = sf.read(str(flac_path), dtype="float32", always_2d=True)
    audio = data.mean(axis=1)
    if sr != sr_target:
        n_out = int(len(audio) * sr_target / sr)
        audio = scipy.signal.resample(audio, n_out).astype(np.float32)
    return audio


def render_fluidsynth(midi_path: Path, sf2: str, sr: int = 44100) -> np.ndarray | None:
    with tempfile.TemporaryDirectory() as tmpdir:
        wav_out = os.path.join(tmpdir, "render.wav")
        result  = subprocess.run(
            ["fluidsynth", "-ni", "-F", wav_out, "-r", str(sr), sf2, str(midi_path)],
            capture_output=True, timeout=300,
        )
        if result.returncode != 0 or not Path(wav_out).exists():
            return None
        try:
            data, _ = sf.read(wav_out, dtype="float32", always_2d=True)
        except Exception:
            return None
    return data.mean(axis=1) if data.size > 0 else None


def get_program(midi_path: Path) -> tuple:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    if not pm.instruments:
        return 0, False
    inst = pm.instruments[0]
    return inst.program, inst.is_drum


def load_audio(stem_id: str, track_dir: Path, midi_path: Path, sf2: str, sr: int = 44100):
    """Use pre-rendered FLAC if available, otherwise render with FluidSynth."""
    flac = track_dir / "stems" / f"{stem_id}.flac"
    if flac.exists():
        return _load_flac_mono(flac, sr)
    return render_fluidsynth(midi_path, sf2, sr)


# --------------- mix / GT -----------------------------------------------

def mix_with_bleed(primary: np.ndarray, bleeds: list, bleed_db: float = -20.0) -> np.ndarray:
    mixed = primary.copy()
    gain  = 10 ** (bleed_db / 20.0)
    for b in bleeds:
        n = min(len(mixed), len(b))
        mixed[:n] += b[:n] * gain
    peak = np.max(np.abs(mixed))
    return mixed / peak if peak > 0 else mixed


def get_gt_notes(midi_path: Path) -> list:
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    notes = []
    for inst in pm.instruments:
        notes.extend(inst.notes)
    notes.sort(key=lambda n: n.start)
    return notes


def align_notes(detected, gt_notes, pitch_tol=1, onset_tol=0.05):
    used, labels = set(), []
    for det in detected:
        det_start, _end, det_pitch, *_ = det
        best_idx, best_dt = None, float("inf")
        for j, gn in enumerate(gt_notes):
            if j in used or abs(int(gn.pitch) - int(det_pitch)) > pitch_tol:
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


# --------------- feature extraction -------------------------------------

def extract_features(note_events, stem_local_id: int) -> np.ndarray:
    if not note_events:
        return np.zeros((0, N_FEATURES), dtype=np.float32)
    starts  = np.array([e[0] for e in note_events], dtype=np.float32)
    ends    = np.array([e[1] for e in note_events], dtype=np.float32)
    pitches = np.array([int(e[2])   for e in note_events], dtype=np.float32)
    amps    = np.array([float(e[3]) for e in note_events], dtype=np.float32)
    durs    = ends - starts
    n       = len(note_events)

    polyphony = np.array([float(np.sum((starts <= starts[i]) & (ends > starts[i]))) for i in range(n)], dtype=np.float32)
    density   = np.array([float(np.sum(np.abs(starts - starts[i]) <= 0.05)) for i in range(n)], dtype=np.float32)
    oct_rank  = np.array([float(np.sum(pitches[(starts <= starts[i]) & (ends > starts[i])] < pitches[i])) for i in range(n)], dtype=np.float32)
    dur_z     = (durs    - durs.mean())    / (durs.std()    + 1e-8)
    pitch_r   = (pitches - pitches.mean()) / (pitches.std() + 1e-8)

    return np.stack([
        amps, durs, pitches, np.full(n, stem_local_id, dtype=np.float32),
        polyphony, density, oct_rank, dur_z, pitch_r,
        (amps > 0.7).astype(np.float32),
        (durs < 0.05).astype(np.float32),
        (polyphony > 4).astype(np.float32),
    ], axis=1).astype(np.float32)


# --------------- mel spectrogram patches ---------------------------------

def extract_spec_patches(
    log_mel: np.ndarray,
    note_events,
    sr: int,
    hop_length: int,
    n_mels: int,
    n_frames: int,
    pre_frac: float = 0.25,
) -> np.ndarray:
    """Slice (n_mels, n_frames) patches from log_mel, centred near each onset.

    pre_frac: fraction of n_frames to use before the onset (default 25%).
    Returns (N, n_mels, n_frames) float16.
    """
    pre     = int(n_frames * pre_frac)
    total_f = log_mel.shape[1]
    floor   = float(log_mel.min())
    patches = []

    for det in note_events:
        onset_s = float(det[0])
        centre  = int(onset_s * sr / hop_length)
        start   = max(0, centre - pre)
        end     = start + n_frames

        if end <= total_f:
            p = log_mel[:, start:end].copy()
        else:
            # Pad right with min value
            avail = log_mel[:, start:total_f]
            pad   = np.full((n_mels, end - total_f), floor, dtype=np.float32)
            p     = np.concatenate([avail, pad], axis=1)

        # Per-patch z-score normalisation
        p = (p - p.mean()) / (p.std() + 1e-8)
        patches.append(p.astype(np.float16))

    if not patches:
        return np.zeros((0, n_mels, n_frames), dtype=np.float16)
    return np.stack(patches)


# --------------- basic-pitch wrapper ------------------------------------

def run_basic_pitch(wav_path: str):
    from basic_pitch.inference import predict
    from basic_pitch import ICASSP_2022_MODEL_PATH
    _, _midi, note_events = predict(wav_path, model_or_model_path=ICASSP_2022_MODEL_PATH)
    return note_events


# --------------- per-track worker ---------------------------------------

def _process_stem_audio(stem_audio, category, midi_path, track_dir, sr,
                         n_mels, n_frames, hop_length):
    """Run amp sim → augmentations → basic-pitch → align → extract features+patches."""
    gt_notes   = get_gt_notes(midi_path)
    if not gt_notes:
        return []
    stem_local = STEM_LOCAL_ID[category]
    # Apply amp sim once (before augmentations — same model, varied conditions)
    amped = apply_nam_amp(stem_audio, sr, category)
    results    = []
    for aug_name in AUGMENTATIONS:
        augmented = apply_aug(amped, sr, aug_name)
        with tempfile.TemporaryDirectory() as tmpdir:
            wav_path = os.path.join(tmpdir, "aug.wav")
            scipy.io.wavfile.write(wav_path, sr,
                                   (augmented * 32767).clip(-32768, 32767).astype(np.int16))
            try:
                note_events = run_basic_pitch(wav_path)
            except Exception as exc:
                print(f"  SKIP {track_dir.name} bp({aug_name}): {exc}", flush=True)
                continue
        if not note_events:
            continue
        labels     = align_notes(note_events, gt_notes)
        feats      = extract_features(note_events, stem_local)
        labels_arr = np.array(labels, dtype=np.int8)
        log_mel    = compute_log_mel(augmented, sr, n_mels, hop_length)
        patches    = extract_spec_patches(log_mel, note_events, sr, hop_length, n_mels, n_frames)
        n_tp = int(labels_arr.sum())
        print(f"{track_dir.name} | {category} | {aug_name} | {len(labels_arr)} notes, {n_tp} TP",
              flush=True)
        stem_ids_arr = np.full(len(labels_arr), stem_local, dtype=np.int8)
        names        = [f"{track_dir.name}/{category}"] * len(labels_arr)
        results.append((feats, labels_arr, stem_ids_arr, names, patches))
    return results


def process_track(task):
    """Process one Slakh track.  Uses demucs separation if _demucs_model is loaded.

    Returns list of (feats, labels, stem_ids, names, spec_patches).
    """
    track_dir_str, sf2, n_mels, n_frames, hop_length = task
    track_dir = Path(track_dir_str)
    midi_dir  = track_dir / "MIDI"
    if not midi_dir.exists():
        return []

    sr = 44100

    # Load/render every stem.
    primary_stems = {}   # category → list of (audio, midi_path)
    all_audio     = {}   # stem_id → audio (for building full mix)

    for midi_path in sorted(midi_dir.glob("*.mid")):
        stem_id = midi_path.stem
        try:
            prog, is_drum = get_program(midi_path)
            audio = load_audio(stem_id, track_dir, midi_path, sf2, sr)
            if audio is None or len(audio) == 0:
                continue
            all_audio[stem_id] = audio
            category = _prog_to_stem(prog, is_drum)
            if category is not None:
                primary_stems.setdefault(category, []).append((audio, midi_path))
        except Exception as exc:
            print(f"  SKIP {track_dir.name}/{stem_id} render: {exc}", flush=True)

    if not primary_stems or not all_audio:
        return []

    results = []

    if _demucs_model is not None:
        # ── Demucs path ────────────────────────────────────────────────────
        # Build full mix from all stems, run demucs once, use separated outputs.
        try:
            lengths = [len(a) for a in all_audio.values()]
            max_len = max(lengths)
            mix = np.zeros(max_len, dtype=np.float32)
            for a in all_audio.values():
                mix[:len(a)] += a
            peak = np.max(np.abs(mix))
            if peak > 0:
                mix /= peak

            separated = separate_with_demucs(mix, sr)
        except Exception as exc:
            print(f"  SKIP {track_dir.name} demucs: {exc}", flush=True)
            return []

        # For each target category, use the demucs-separated stem audio.
        # Merge all MIDI notes for that category into a single GT set.
        for category in _DEMUCS_TARGET_STEMS:
            if category not in separated:
                continue
            stem_audio = separated[category]
            # Collect GT notes from all Slakh stems that map to this category
            gt_midi_paths = [mp for (_, mp) in primary_stems.get(category, [])]
            if not gt_midi_paths:
                continue
            # Write a merged GT by concatenating all MIDI notes across sub-stems
            for midi_path in gt_midi_paths:
                try:
                    for r in _process_stem_audio(stem_audio, category, midi_path,
                                                  track_dir, sr, n_mels, n_frames, hop_length):
                        results.append(r)
                except Exception as exc:
                    print(f"  SKIP {track_dir.name}/{category}: {exc}", flush=True)
    else:
        # ── Legacy path (clean stem + simulated bleed) ─────────────────────
        for stem_id_key, stems_list in primary_stems.items():
            for primary_audio, midi_path in stems_list:
                try:
                    bleeds = [a for sid, a in all_audio.items()]
                    mixed  = mix_with_bleed(primary_audio, bleeds)
                    for r in _process_stem_audio(mixed, stem_id_key, midi_path,
                                                  track_dir, sr, n_mels, n_frames, hop_length):
                        results.append(r)
                except Exception as exc:
                    print(f"  SKIP {track_dir.name}/{stem_id_key}: {exc}", flush=True)

    return results


# --------------- HDF5 helpers -------------------------------------------

def _init_h5(path: Path, n_features: int, n_mels: int, n_frames: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as f:
        dt = h5py.special_dtype(vlen=str)
        f.create_dataset("features",     shape=(0, n_features),        maxshape=(None, n_features),        dtype="float32", chunks=(4096, n_features))
        f.create_dataset("labels",       shape=(0,),                   maxshape=(None,),                   dtype="int8",    chunks=(4096,))
        f.create_dataset("stem_ids",     shape=(0,),                   maxshape=(None,),                   dtype="int8",    chunks=(4096,))
        f.create_dataset("source_midi",  shape=(0,),                   maxshape=(None,),                   dtype=dt,        chunks=(4096,))
        f.create_dataset("spec_patches", shape=(0, n_mels, n_frames),  maxshape=(None, n_mels, n_frames),  dtype="float16",
                         chunks=(512, n_mels, n_frames), compression="gzip", compression_opts=4)
        # Store patch params as attributes for training code to read
        f.attrs["n_mels"]   = n_mels
        f.attrs["n_frames"] = n_frames


def _append_h5(path: Path, feats, labels, stem_ids, names, spec_patches):
    n = len(labels)
    with h5py.File(path, "a") as f:
        for ds_name, data in [
            ("features",     feats),
            ("labels",       labels),
            ("stem_ids",     stem_ids),
            ("source_midi",  np.array(names, dtype=object)),
            ("spec_patches", spec_patches),
        ]:
            ds = f[ds_name]
            old = ds.shape[0]
            ds.resize(old + n, axis=0)
            ds[old:] = data


# --------------- main ---------------------------------------------------

def main():
    ap = argparse.ArgumentParser("build_discriminator_data")
    ap.add_argument("--slakh_dir",   default="data/slakh/train")
    ap.add_argument("--out",         default="runs/discriminator_data/notes.h5")
    ap.add_argument("--n_tracks",    type=int, default=100)
    ap.add_argument("--workers",     type=int, default=1)
    ap.add_argument("--sf2",         default="",    help="SF2 path (auto-detected if blank).")
    ap.add_argument("--n_mels",      type=int, default=64)
    ap.add_argument("--n_frames",    type=int, default=32)
    ap.add_argument("--hop_length",  type=int, default=512)
    ap.add_argument("--seed",        type=int, default=42)
    ap.add_argument("--use-demucs",  action="store_true", default=True,
                    help="Mix all stems → htdemucs_6s separation → use demucs output "
                         "for both basic-pitch and mel patches (recommended). "
                         "Pass --no-use-demucs to use the legacy clean-stem pipeline.")
    ap.add_argument("--no-use-demucs", dest="use_demucs", action="store_false")
    ap.add_argument("--nam_dir", default="data/nam_models",
                    help="Directory containing .nam captures + manifest.json "
                         "(from make nam-fetch). Leave blank to skip amp sim.")
    args = ap.parse_args()

    sf2 = args.sf2 or (find_sf2() if not args.use_demucs else "")
    if not args.use_demucs:
        print(f"Soundfont: {sf2}")
    nam_dir = args.nam_dir if Path(args.nam_dir).exists() else ""
    print(f"Pipeline: {'demucs (htdemucs_6s)' if args.use_demucs else 'legacy clean-stem+bleed'}")
    print(f"Amp sim:  {'NAM (' + args.nam_dir + ')' if nam_dir else 'disabled (run make nam-fetch to enable)'}")
    print(f"Mel patches: {args.n_mels} bands × {args.n_frames} frames (hop={args.hop_length})")

    slakh_dir  = Path(args.slakh_dir)
    track_dirs = sorted(slakh_dir.glob("Track*"))
    if not track_dirs:
        print(f"ERROR: no Track* directories found in {slakh_dir}")
        raise SystemExit(1)

    random.seed(args.seed)
    sample = random.sample(track_dirs, min(args.n_tracks, len(track_dirs)))
    print(f"Processing {len(sample)} tracks, {len(AUGMENTATIONS)} augmentations each")

    tasks    = [(str(td), sf2, args.n_mels, args.n_frames, args.hop_length) for td in sample]
    out_path = Path(args.out)
    _init_h5(out_path, N_FEATURES, args.n_mels, args.n_frames)

    ctx = multiprocessing.get_context("fork")
    initargs = (args.use_demucs, nam_dir)
    with ctx.Pool(processes=args.workers,
                  initializer=_init_worker, initargs=initargs) as pool:
        for results in pool.imap_unordered(process_track, tasks, chunksize=1):
            for feats, labels, stem_ids, names, patches in results:
                _append_h5(out_path, feats, labels, stem_ids, names, patches)

    with h5py.File(out_path, "r") as f:
        n_total = int(f["labels"].shape[0])
        n_tp    = int(np.sum(f["labels"][:]))
    print(f"\nDone. {n_total} notes written to {out_path}  (TP={n_tp}, FP={n_total - n_tp})")


if __name__ == "__main__":
    main()
