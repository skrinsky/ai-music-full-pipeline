#!/usr/bin/env python3
"""Note discriminator models, datasets, and event-filtering utilities.

Two model variants:
  NoteDiscriminator        — scalar-only MLP (12 features → logit)
  CombinedNoteDiscriminator — CNN on mel-patch + MLP on scalars, dual heads:
                               * combined_head: trained with both branches
                               * scalar_head:   trained with scalar branch only
                                               (used at inference time in pre.py)
"""

import random
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

# --------------- constants -----------------------------------------------

FEATURE_NAMES = [
    "amplitude", "duration_s", "pitch", "stem_id", "polyphony",
    "density_100ms", "octave_rank", "duration_zscore", "pitch_rel",
    "hi_conf_flag", "short_flag", "hi_poly_flag",
]
N_FEATURES = len(FEATURE_NAMES)

N_MEL    = 64
N_FRAMES = 32

# inst_idx → local stem id (guitar=0, bass=1, other=2); -1 = passthrough
_INST_TO_LOCAL = {2: 0, 4: 1, 3: 2}

# inst_idx → stem WAV filename inside htdemucs_6s/<track>/
_INST_TO_STEM_WAV = {2: "guitar.wav", 4: "bass.wav", 3: "other.wav"}


# --------------- datasets ------------------------------------------------

class NoteDataset(Dataset):
    """Scalar features only — for MLP-only model."""

    def __init__(self, h5_path: str, split: str = "train", val_frac: float = 0.15, seed: int = 42):
        with h5py.File(h5_path, "r") as f:
            features    = f["features"][:]
            labels      = f["labels"][:]
            source_midi = f["source_midi"][:].astype(str)

        unique = sorted(set(source_midi))
        rng    = random.Random(seed)
        rng.shuffle(unique)
        val_set = set(unique[: max(1, int(len(unique) * val_frac))])
        mask    = np.array([m in val_set for m in source_midi])
        idx     = np.where(mask)[0] if split == "val" else np.where(~mask)[0]

        self.features = torch.tensor(features[idx], dtype=torch.float32)
        self.labels   = torch.tensor(labels[idx],   dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]


class CombinedNoteDataset(Dataset):
    """Scalar features + mel spectrogram patches — for combined model."""

    def __init__(self, h5_path: str, split: str = "train", val_frac: float = 0.15, seed: int = 42):
        with h5py.File(h5_path, "r") as f:
            if "spec_patches" not in f:
                raise KeyError(
                    "HDF5 has no 'spec_patches' dataset. "
                    "Re-run build_discriminator_data.py to generate patches."
                )
            features     = f["features"][:]
            spec_patches = f["spec_patches"][:]      # (N, n_mel, n_frames) float16
            labels       = f["labels"][:]
            source_midi  = f["source_midi"][:].astype(str)

        unique  = sorted(set(source_midi))
        rng     = random.Random(seed)
        rng.shuffle(unique)
        val_set = set(unique[: max(1, int(len(unique) * val_frac))])
        mask    = np.array([m in val_set for m in source_midi])
        idx     = np.where(mask)[0] if split == "val" else np.where(~mask)[0]

        self.features     = torch.tensor(features[idx],                    dtype=torch.float32)
        self.spec_patches = torch.tensor(spec_patches[idx].astype("f"),    dtype=torch.float32)
        self.labels       = torch.tensor(labels[idx],                      dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.spec_patches[i], self.features[i], self.labels[i]


# --------------- scalar-only MLP -----------------------------------------

class NoteDiscriminator(nn.Module):
    """12-feature MLP → binary TP/FP logit."""

    def __init__(self, n_features: int = N_FEATURES, hidden=(64, 32)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, h1), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "NoteDiscriminator":
        ckpt  = torch.load(path, map_location=device)
        model = cls(
            n_features=ckpt.get("n_features", N_FEATURES),
            hidden=tuple(ckpt.get("hidden", (64, 32))),
        )
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device).eval()


# --------------- combined CNN + MLP model --------------------------------

class CombinedNoteDiscriminator(nn.Module):
    """CNN branch (mel patch) + MLP branch (12 scalars) with two output heads.

    combined_head  — used during training with both input branches
    scalar_head    — used at inference time in score_events() (no audio needed)

    Both heads receive gradient during training via an auxiliary scalar-only loss,
    so the scalar_head also benefits from the richer representation learned
    alongside the spectrogram features.
    """

    def __init__(self, n_scalar: int = N_FEATURES, n_mel: int = N_MEL, n_frames: int = N_FRAMES):
        super().__init__()
        self.n_scalar = n_scalar
        self.n_mel    = n_mel
        self.n_frames = n_frames

        # CNN branch: (B, 1, n_mel, n_frames) → (B, 64)
        self.mel_cnn = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),       # → (B, 64)
        )

        # Scalar MLP branch: (B, n_scalar) → (B, 32)
        self.scalar_mlp = nn.Sequential(
            nn.LayerNorm(n_scalar),
            nn.Linear(n_scalar, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
        )

        # Combined head (training)
        self.combined_head = nn.Linear(64 + 32, 1)

        # Scalar-only head (inference in pre.py)
        self.scalar_head = nn.Linear(32, 1)

    def forward(self, spec: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        """spec: (B, n_mel, n_frames); scalars: (B, n_scalar) → logit (B,)."""
        mel_emb    = self.mel_cnn(spec.unsqueeze(1))
        scalar_emb = self.scalar_mlp(scalars)
        return self.combined_head(torch.cat([mel_emb, scalar_emb], dim=1)).squeeze(-1)

    def forward_scalar_only(self, scalars: torch.Tensor) -> torch.Tensor:
        """Scalar-only inference path — no audio required."""
        return self.scalar_head(self.scalar_mlp(scalars)).squeeze(-1)

    def predict_proba(self, spec: torch.Tensor, scalars: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward(spec, scalars))

    def predict_proba_scalar(self, scalars: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return torch.sigmoid(self.forward_scalar_only(scalars))

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "CombinedNoteDiscriminator":
        ckpt  = torch.load(path, map_location=device)
        model = cls(
            n_scalar=ckpt.get("n_scalar", N_FEATURES),
            n_mel   =ckpt.get("n_mel",    N_MEL),
            n_frames=ckpt.get("n_frames", N_FRAMES),
        )
        model.load_state_dict(ckpt["state_dict"])
        return model.to(device).eval()


# --------------- factory -------------------------------------------------

def load_discriminator(path: str, device: str = "cpu"):
    """Load either NoteDiscriminator or CombinedNoteDiscriminator from checkpoint."""
    ckpt = torch.load(path, map_location=device)
    if ckpt.get("model_type") == "combined":
        return CombinedNoteDiscriminator.load(path, device)
    return NoteDiscriminator.load(path, device)


# --------------- event feature builder ----------------------------------

def _build_event_features(events: list, tempo_bpm: float) -> np.ndarray:
    """Derive (N, 12) feature matrix from pre.py event tuples (no audio needed).

    events: list of (start_sec, inst_idx, pitch, velocity, dur_qn)
    """
    n = len(events)
    if n == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32)

    starts   = np.array([e[0] for e in events], dtype=np.float32)
    inst_ids = np.array([e[1] for e in events], dtype=np.int32)
    pitches  = np.array([e[2] for e in events], dtype=np.float32)
    vels     = np.array([e[3] for e in events], dtype=np.float32)
    durs_qn  = np.array([e[4] for e in events], dtype=np.float32)

    durs_s   = durs_qn * 60.0 / max(tempo_bpm, 1.0)
    ends     = starts + durs_s
    amps     = vels / 127.0
    stem_ids = np.array([_INST_TO_LOCAL.get(int(i), -1) for i in inst_ids], dtype=np.float32)

    polyphony = np.zeros(n, dtype=np.float32)
    density   = np.zeros(n, dtype=np.float32)
    oct_rank  = np.zeros(n, dtype=np.float32)
    for i in range(n):
        t = starts[i]
        polyphony[i] = float(np.sum((starts <= t) & (ends > t)))
        density[i]   = float(np.sum(np.abs(starts - t) <= 0.05))
        sim          = pitches[(starts <= t) & (ends > t)]
        oct_rank[i]  = float(np.sum(sim < pitches[i]))

    dur_z   = np.zeros(n, dtype=np.float32)
    pitch_r = np.zeros(n, dtype=np.float32)
    for local_id in [0, 1, 2]:
        mask = stem_ids == local_id
        if mask.sum() < 2:
            continue
        dm, ds  = durs_s[mask].mean(), durs_s[mask].std() + 1e-8
        pm, ps  = pitches[mask].mean(), pitches[mask].std() + 1e-8
        dur_z[mask]   = (durs_s[mask]  - dm) / ds
        pitch_r[mask] = (pitches[mask] - pm) / ps

    return np.stack([
        amps, durs_s, pitches, stem_ids,
        polyphony, density, oct_rank, dur_z, pitch_r,
        (amps > 0.7).astype(np.float32),
        (durs_s < 0.05).astype(np.float32),
        (polyphony > 4).astype(np.float32),
    ], axis=1).astype(np.float32)


# --------------- inference entry point ----------------------------------

def score_events(
    events: list,
    model,
    tempo_bpm: float,
    threshold: float = 0.35,
) -> list:
    """Filter pre.py events through the discriminator; return filtered list.

    Works with both NoteDiscriminator and CombinedNoteDiscriminator (uses
    scalar-only head for the combined model — no audio needed at this stage).
    Non-stem events (not guitar/bass/other) pass through unfiltered.
    """
    if not events:
        return events

    feats  = _build_event_features(events, tempo_bpm)
    tensor = torch.tensor(feats, dtype=torch.float32)

    if isinstance(model, CombinedNoteDiscriminator):
        probs = model.predict_proba_scalar(tensor).numpy()
    else:
        probs = model.predict_proba(tensor).numpy()

    filtered = []
    for i, ev in enumerate(events):
        local_id = _INST_TO_LOCAL.get(int(ev[1]), -1)
        if local_id == -1 or probs[i] >= threshold:
            filtered.append(ev)
    return filtered


def _load_log_mel(wav_path: "Path", n_mel: int = N_MEL,
                  hop_length: int = 512, sr_target: int = 44100) -> "np.ndarray | None":
    """Load a WAV and return a (n_mel, T) log-mel spectrogram, or None on failure."""
    try:
        import soundfile as sf
        audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)  # mono
        if sr != sr_target:
            # Simple linear resampling — good enough for patch extraction
            ratio  = sr_target / sr
            n_new  = int(len(audio) * ratio)
            audio  = np.interp(
                np.linspace(0, len(audio) - 1, n_new),
                np.arange(len(audio)),
                audio,
            ).astype(np.float32)
        try:
            import librosa
            mel = librosa.feature.melspectrogram(
                y=audio, sr=sr_target, n_mels=n_mel, hop_length=hop_length)
            return librosa.power_to_db(mel, ref=np.max).astype(np.float32)
        except ImportError:
            import scipy.signal
            n_fft   = hop_length * 4
            _, _, Zxx = scipy.signal.stft(audio, fs=sr_target, nperseg=n_fft,
                                           noverlap=n_fft - hop_length, window="hann")
            power   = np.abs(Zxx) ** 2
            hz2mel  = lambda hz: 2595 * np.log10(1 + hz / 700.0)
            mel2hz  = lambda m:  700 * (10 ** (m / 2595.0) - 1)
            mel_pts = np.linspace(hz2mel(0), hz2mel(sr_target / 2), n_mel + 2)
            hz_pts  = mel2hz(mel_pts)
            bins    = np.floor((n_fft + 1) * hz_pts / sr_target).astype(int).clip(0, n_fft // 2)
            fb      = np.zeros((n_mel, n_fft // 2 + 1), dtype=np.float32)
            for i in range(n_mel):
                s, c, e = bins[i], bins[i + 1], bins[i + 2]
                if c > s: fb[i, s:c] = np.linspace(0, 1, c - s)
                if e > c: fb[i, c:e] = np.linspace(1, 0, e - c)
            mel  = fb @ power
            lm   = 10.0 * np.log10(mel + 1e-8)
            lm  -= lm.max()
            return lm.astype(np.float32)
    except Exception:
        return None


def _extract_mel_patches(log_mel: "np.ndarray", onsets_s: "np.ndarray",
                          sr: int = 44100, hop_length: int = 512,
                          n_frames: int = N_FRAMES,
                          pre_frac: float = 0.25) -> "np.ndarray":
    """Return (N, n_mel, n_frames) float32 patches centred just after each onset."""
    n_mel   = log_mel.shape[0]
    total_f = log_mel.shape[1]
    floor   = float(log_mel.min())
    pre     = int(n_frames * pre_frac)
    patches = []
    for onset_s in onsets_s:
        centre = int(float(onset_s) * sr / hop_length)
        start  = max(0, centre - pre)
        end    = start + n_frames
        if end <= total_f:
            p = log_mel[:, start:end].copy()
        else:
            avail = log_mel[:, start:total_f]
            pad   = np.full((n_mel, end - total_f), floor, dtype=np.float32)
            p     = np.concatenate([avail, pad], axis=1)
        p = (p - p.mean()) / (p.std() + 1e-8)
        patches.append(p.astype(np.float32))
    return np.stack(patches) if patches else np.zeros((0, n_mel, n_frames), dtype=np.float32)


def score_events_with_audio(
    events: list,
    model,
    tempo_bpm: float,
    stems_dir: "Path",
    track_name: str,
    threshold: float = 0.35,
    hop_length: int = 512,
) -> list:
    """Filter events using the combined CNN+scalar head when stem WAVs are available.

    For each instrument that has a matching stem WAV under
    stems_dir / track_name / <stem>.wav, extracts a mel patch per note and
    runs the full combined_head.  Falls back to scalar-only for any instrument
    whose WAV is missing (or if model is not CombinedNoteDiscriminator).
    Non-stem instruments (vox, drums) always pass through unfiltered.
    """
    if not events:
        return events

    if not isinstance(model, CombinedNoteDiscriminator):
        return score_events(events, model, tempo_bpm, threshold)

    feats  = _build_event_features(events, tempo_bpm)
    scalar_t = torch.tensor(feats, dtype=torch.float32)

    # Per-instrument log-mel (loaded lazily, None = unavailable)
    log_mels: dict = {}
    for inst_idx, stem_wav in _INST_TO_STEM_WAV.items():
        wav_path = Path(stems_dir) / track_name / stem_wav
        log_mels[inst_idx] = _load_log_mel(wav_path) if wav_path.exists() else None

    # Build spec tensor: use audio patches where available, zeros otherwise
    n_mel, n_frames = model.n_mel, model.n_frames
    spec_np = np.zeros((len(events), n_mel, n_frames), dtype=np.float32)
    has_audio = np.zeros(len(events), dtype=bool)

    # Group indices by instrument for batch patch extraction
    for inst_idx, lm in log_mels.items():
        if lm is None:
            continue
        idxs    = [i for i, e in enumerate(events) if int(e[1]) == inst_idx]
        onsets  = np.array([events[i][0] for i in idxs], dtype=np.float32)
        patches = _extract_mel_patches(lm, onsets, hop_length=hop_length,
                                        n_frames=n_frames)
        for j, i in enumerate(idxs):
            spec_np[i]    = patches[j]
            has_audio[i]  = True

    spec_t = torch.tensor(spec_np, dtype=torch.float32)

    # Run combined head for notes that have audio, scalar-only for the rest
    with torch.no_grad():
        probs_combined = model.predict_proba(spec_t, scalar_t).numpy()
        probs_scalar   = model.predict_proba_scalar(scalar_t).numpy()

    probs = np.where(has_audio, probs_combined, probs_scalar)

    filtered = []
    for i, ev in enumerate(events):
        local_id = _INST_TO_LOCAL.get(int(ev[1]), -1)
        if local_id == -1 or probs[i] >= threshold:
            filtered.append(ev)
    return filtered
