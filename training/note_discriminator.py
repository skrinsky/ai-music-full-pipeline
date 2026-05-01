#!/usr/bin/env python3
"""Note discriminator model, dataset, and event-filtering utilities."""

import random
from pathlib import Path
from typing import List

import h5py
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

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

# inst_idx → local stem id (guitar=0, bass=1, other=2); None = passthrough
_INST_TO_LOCAL = {2: 0, 4: 1, 3: 2}


class NoteDataset(Dataset):
    """Load features/labels from HDF5; split by unique source_midi to avoid leakage."""

    def __init__(self, h5_path: str, split: str = "train", val_frac: float = 0.15, seed: int = 42):
        with h5py.File(h5_path, "r") as f:
            features    = f["features"][:]
            labels      = f["labels"][:]
            source_midi = f["source_midi"][:].astype(str)

        unique_midis = sorted(set(source_midi))
        rng = random.Random(seed)
        rng.shuffle(unique_midis)
        n_val = max(1, int(len(unique_midis) * val_frac))
        val_set = set(unique_midis[:n_val])

        mask = np.array([m in val_set for m in source_midi])
        if split == "val":
            idx = np.where(mask)[0]
        else:
            idx = np.where(~mask)[0]

        self.features = torch.tensor(features[idx], dtype=torch.float32)
        self.labels   = torch.tensor(labels[idx],   dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        return self.features[i], self.labels[i]


class NoteDiscriminator(nn.Module):
    """Small MLP: 12 features → binary TP/FP logit."""

    def __init__(self, n_features: int = N_FEATURES, hidden=(64, 32)):
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.LayerNorm(n_features),
            nn.Linear(n_features, h1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return P(TP) for each note."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x))

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "NoteDiscriminator":
        """Load checkpoint; return model in eval mode."""
        ckpt = torch.load(path, map_location=device)
        model = cls(
            n_features=ckpt.get("n_features", N_FEATURES),
            hidden=tuple(ckpt.get("hidden", (64, 32))),
        )
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        model.eval()
        return model


def _build_event_features(events: list, tempo_bpm: float) -> np.ndarray:
    """Derive (N, 12) feature matrix from pre.py event tuples without audio."""
    # events: (start_sec, inst_idx, pitch, velocity, dur_qn)
    n = len(events)
    if n == 0:
        return np.zeros((0, N_FEATURES), dtype=np.float32)

    starts   = np.array([e[0] for e in events], dtype=np.float32)
    inst_ids = np.array([e[1] for e in events], dtype=np.int32)
    pitches  = np.array([e[2] for e in events], dtype=np.float32)
    vels     = np.array([e[3] for e in events], dtype=np.float32)
    durs_qn  = np.array([e[4] for e in events], dtype=np.float32)

    durs_s = durs_qn * 60.0 / max(tempo_bpm, 1.0)
    ends   = starts + durs_s
    amps   = vels / 127.0

    stem_ids = np.array([_INST_TO_LOCAL.get(int(i), -1) for i in inst_ids], dtype=np.float32)

    polyphony = np.zeros(n, dtype=np.float32)
    density   = np.zeros(n, dtype=np.float32)
    oct_rank  = np.zeros(n, dtype=np.float32)
    for i in range(n):
        t = starts[i]
        polyphony[i] = float(np.sum((starts <= t) & (ends > t)))
        density[i]   = float(np.sum(np.abs(starts - t) <= 0.05))
        sim = pitches[(starts <= t) & (ends > t)]
        oct_rank[i]  = float(np.sum(sim < pitches[i]))

    # zscore stats computed per stem
    dur_z    = np.zeros(n, dtype=np.float32)
    pitch_r  = np.zeros(n, dtype=np.float32)
    for local_id in [0, 1, 2]:
        mask = stem_ids == local_id
        if mask.sum() < 2:
            continue
        dm, ds = durs_s[mask].mean(), durs_s[mask].std() + 1e-8
        pm_, ps = pitches[mask].mean(), pitches[mask].std() + 1e-8
        dur_z[mask]   = (durs_s[mask]  - dm) / ds
        pitch_r[mask] = (pitches[mask] - pm_) / ps

    feats = np.stack([
        amps,
        durs_s,
        pitches,
        stem_ids,
        polyphony,
        density,
        oct_rank,
        dur_z,
        pitch_r,
        (amps > 0.7).astype(np.float32),
        (durs_s < 0.05).astype(np.float32),
        (polyphony > 4).astype(np.float32),
    ], axis=1).astype(np.float32)
    return feats


def score_events(
    events: list,
    model: NoteDiscriminator,
    tempo_bpm: float,
    threshold: float = 0.35,
) -> list:
    """Filter pre.py events through the discriminator; return filtered event list.

    Non-stem events (not guitar/bass/other) pass through unfiltered.
    """
    if not events:
        return events

    stem_indices = {_INST_TO_LOCAL[k] for k in _INST_TO_LOCAL}
    passthrough_mask = [_INST_TO_LOCAL.get(int(e[1]), -1) == -1 for e in events]

    feats = _build_event_features(events, tempo_bpm)
    tensor = torch.tensor(feats, dtype=torch.float32)
    probs = model.predict_proba(tensor).numpy()

    filtered = []
    for i, ev in enumerate(events):
        if passthrough_mask[i] or probs[i] >= threshold:
            filtered.append(ev)
    return filtered
