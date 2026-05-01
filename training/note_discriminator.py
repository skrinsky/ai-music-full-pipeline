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
