#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dense chorale preprocessing: NPZ → compact token sequences.

Reads Jsb16thSeparated.npz directly (TonicNet format), preserving the
original train/valid/test splits.  Each chorale is tokenized as:

    BOS chord₀ S₀ B₀ A₀ T₀  chord₁ S₁ B₁ A₁ T₁  ...  EOS

Parallel continuation counters (per-voice held-note counts, 0-31) are
stored alongside token sequences for the ContinuationEmbedding in the
model.

Transposition augmentation (±6 semitones, train only) expands 229 train
chorales to ~2900+ sequences.

Vocab: 100 tokens (PAD/BOS/EOS + 46 pitches + REST + 50 chords).
"""

import argparse
import json
import os
import pickle
import sys
from typing import Optional

import numpy as np
import pretty_midi

# ────────────────────── VOCAB LAYOUT ──────────────────────
# Tokens 0-2: special
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2

# Tokens 3-48: pitches (MIDI 36-81, covering all 4 voices)
PITCH_OFFSET = 3      # token 3 = MIDI 36
MIDI_LO = 36
MIDI_HI = 81
NUM_PITCHES = MIDI_HI - MIDI_LO + 1  # 46

# Token 49: REST
REST_ID = PITCH_OFFSET + NUM_PITCHES  # 49

# Tokens 50-99: chords (50 slots)
CHORD_OFFSET = 50
NUM_CHORDS = 50

VOCAB_SIZE = CHORD_OFFSET + NUM_CHORDS  # 100

# ────────────────────── VOICE RANGES ──────────────────────
# MIDI ranges for transposition validation
VOICE_RANGES = {
    "soprano": (60, 81),
    "bass":    (36, 64),
    "alto":    (53, 77),
    "tenor":   (45, 72),
}
# NPZ column order: soprano=0, alto=1, tenor=2, bass=3
NPZ_COL_NAMES = ["soprano", "alto", "tenor", "bass"]

# Output voice order per timestep: chord, soprano, bass, alto, tenor
# (matches TonicNet)
VOICE_ORDER = [0, 3, 1, 2]  # indices into NPZ columns
VOICE_ORDER_NAMES = ["soprano", "bass", "alto", "tenor"]

REST_THRESHOLD = 36  # pitches below this are rests in the NPZ

# ────────────────────── CHORD ANALYSIS ──────────────────────

# We map chords to a fixed vocabulary of 50 tokens:
# 12 major + 12 minor + 12 diminished + 12 augmented + 1 other + 1 rest-chord
CHORD_QUALITY_OFFSETS = {
    "major":      0,
    "minor":      12,
    "diminished": 24,
    "augmented":  36,
}
CHORD_OTHER = 48   # unrecognized quality
CHORD_REST = 49    # all voices resting

# Pitch-class names for chord root
_PC_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _pitch_classes_from_midi(pitches: list[int]) -> list[int]:
    """Return sorted unique pitch classes from a list of MIDI pitches."""
    return sorted(set(p % 12 for p in pitches))


def _classify_chord_pcs(pcs: list[int]) -> tuple[int, str]:
    """Classify pitch classes into (root_pc, quality) using interval matching.

    Returns (root_pc, quality_string).  Falls back to ("other") if no match.
    Handles incomplete triads (2 PCs with doubled notes, common in chorales).
    """
    if len(pcs) == 0:
        return (0, "rest")

    pcs_set = set(pcs)

    # Triad definitions: (intervals_from_root, quality_name)
    TRIADS = [
        ({0, 4, 7}, "major"),
        ({0, 3, 7}, "minor"),
        ({0, 3, 6}, "diminished"),
        ({0, 4, 8}, "augmented"),
    ]

    # First pass: exact match (3+ PCs that contain a full triad)
    for root in pcs:
        intervals = set((pc - root) % 12 for pc in pcs_set)
        for triad_ivs, quality in TRIADS:
            if triad_ivs <= intervals:
                return (root, quality)

    # Second pass: incomplete triads (2 PCs that are a subset of a triad)
    # Prioritize by triad order (major > minor > dim > aug)
    if len(pcs_set) == 2:
        for root in pcs:
            intervals = set((pc - root) % 12 for pc in pcs_set)
            for triad_ivs, quality in TRIADS:
                if intervals <= triad_ivs and 0 in intervals:
                    return (root, quality)

    # No clean triad found — use bass note as root, mark as "other"
    return (min(pcs), "other")


def analyze_chord_at_timestep(pitches: np.ndarray) -> int:
    """Analyze a 4-voice chord and return a chord token ID (0-49).

    pitches: shape (4,) — soprano, alto, tenor, bass MIDI values.
    Values < REST_THRESHOLD or NaN are treated as rests.
    """
    valid = []
    for p in pitches:
        if np.isnan(p) or p < REST_THRESHOLD:
            continue
        valid.append(int(round(p)))

    if len(valid) == 0:
        return CHORD_REST

    pcs = _pitch_classes_from_midi(valid)
    root_pc, quality = _classify_chord_pcs(pcs)

    if quality == "rest":
        return CHORD_REST
    if quality in CHORD_QUALITY_OFFSETS:
        return CHORD_QUALITY_OFFSETS[quality] + (root_pc % 12)
    return CHORD_OTHER


def chord_token_to_label(chord_local: int) -> str:
    """Convert a local chord index (0-49) to a human-readable label."""
    if chord_local == CHORD_REST:
        return "REST_CHORD"
    if chord_local == CHORD_OTHER:
        return "OTHER_CHORD"
    for quality, offset in CHORD_QUALITY_OFFSETS.items():
        if offset <= chord_local < offset + 12:
            root_pc = chord_local - offset
            return f"{_PC_NAMES[root_pc]}_{quality}"
    return f"CHORD_{chord_local}"


# ────────────────────── PITCH ↔ TOKEN ──────────────────────

def midi_to_token(midi_pitch: int) -> int:
    """MIDI pitch → vocab token.  Clamps to valid range."""
    return PITCH_OFFSET + max(0, min(NUM_PITCHES - 1, midi_pitch - MIDI_LO))


def token_to_midi(tok: int) -> int:
    """Vocab token → MIDI pitch."""
    return (tok - PITCH_OFFSET) + MIDI_LO


def is_pitch_token(tok: int) -> bool:
    return PITCH_OFFSET <= tok < PITCH_OFFSET + NUM_PITCHES


def is_rest_token(tok: int) -> bool:
    return tok == REST_ID


def is_chord_token(tok: int) -> bool:
    return CHORD_OFFSET <= tok < CHORD_OFFSET + NUM_CHORDS


# ────────────────────── CONTINUATION COUNTERS ──────────────────────

def compute_continuation_counters(arr: np.ndarray) -> list[list[int]]:
    """Compute per-voice held-note counters for each timestep.

    arr: shape (T, 4) — soprano, alto, tenor, bass MIDI pitches.

    Returns: list of T entries, each a list of 4 ints (one per voice in
    NPZ column order).  Counter is 0 at onset, increments each timestep
    the same pitch is held, capped at 31.
    """
    T = arr.shape[0]
    counters: list[list[int]] = []
    prev_pitch = [None, None, None, None]
    prev_count = [0, 0, 0, 0]

    for t in range(T):
        row = []
        for v in range(4):
            p = arr[t, v]
            is_rest = np.isnan(p) or p < REST_THRESHOLD
            if is_rest:
                pitch_val = None
            else:
                pitch_val = int(round(p))

            if pitch_val is not None and pitch_val == prev_pitch[v]:
                cnt = min(31, prev_count[v] + 1)
            else:
                cnt = 0
            row.append(cnt)
            prev_pitch[v] = pitch_val
            prev_count[v] = cnt
        counters.append(row)
    return counters


# ────────────────────── TOKENIZE ──────────────────────

def tokenize_chorale(arr: np.ndarray) -> tuple[list[int], list[int]]:
    """Tokenize a (T, 4) chorale array to dense token + continuation sequences.

    Voice order per timestep: chord, soprano, bass, alto, tenor.
    (Matches TonicNet convention.)

    Returns:
        tokens: [BOS, chord₀, S₀, B₀, A₀, T₀, ..., EOS]
        conts:  parallel continuation counters (0 for BOS/EOS/chord positions)
    """
    T = arr.shape[0]
    raw_counters = compute_continuation_counters(arr)

    tokens: list[int] = [BOS_ID]
    conts: list[int] = [0]

    for t in range(T):
        # Chord token
        chord_local = analyze_chord_at_timestep(arr[t])
        tokens.append(CHORD_OFFSET + chord_local)
        conts.append(0)

        # Voice tokens in order: soprano(0), bass(3), alto(1), tenor(2)
        for npz_col in VOICE_ORDER:
            p = arr[t, npz_col]
            if np.isnan(p) or p < REST_THRESHOLD:
                tokens.append(REST_ID)
            else:
                tokens.append(midi_to_token(int(round(p))))
            conts.append(raw_counters[t][npz_col])

    tokens.append(EOS_ID)
    conts.append(0)

    return tokens, conts


# ────────────────────── TRANSPOSITION ──────────────────────

def transpose_chorale(arr: np.ndarray, semitones: int) -> Optional[np.ndarray]:
    """Transpose all voices by semitones.  Returns None if any voice goes
    out of range.
    """
    if semitones == 0:
        return arr.copy()

    out = arr.copy().astype(np.float64)
    for col, name in enumerate(NPZ_COL_NAMES):
        lo, hi = VOICE_RANGES[name]
        mask = ~np.isnan(out[:, col]) & (out[:, col] >= REST_THRESHOLD)
        transposed = out[:, col].copy()
        transposed[mask] += semitones
        # Check range
        valid = transposed[mask]
        if len(valid) > 0 and (valid.min() < lo or valid.max() > hi):
            return None
        out[:, col] = transposed

    return out


# ────────────────────── DECODE TO MIDI ──────────────────────

def decode_tokens_to_midi(tokens: list[int], path: str,
                          bpm: float = 100.0) -> None:
    """Convert a dense token sequence back to a 4-track MIDI file."""
    sec_per_step = 60.0 / bpm / 4.0  # 16th note duration

    pm = pretty_midi.PrettyMIDI(initial_tempo=bpm, resolution=960)

    # GM programs and instrument names for each voice
    gm_voices: dict[str, tuple[int, str]] = {
        "soprano": (73, "Flute"),
        "bass":    (70, "Bassoon"),
        "alto":    (69, "English Horn"),
        "tenor":   (71, "Clarinet"),
    }
    voice_names = VOICE_ORDER_NAMES  # soprano, bass, alto, tenor
    instruments = {}
    for vname in voice_names:
        prog, gm_name = gm_voices[vname]
        inst = pretty_midi.Instrument(program=prog,
                                       is_drum=False, name=gm_name)
        instruments[vname] = inst

    # Parse tokens: skip BOS, stop at EOS
    # Each timestep is 5 tokens: chord, soprano, bass, alto, tenor
    body = []
    for tok in tokens:
        if tok == BOS_ID:
            continue
        if tok == EOS_ID:
            break
        body.append(tok)

    # Group into timesteps of 5
    n_steps = len(body) // 5
    # Track active notes per voice
    active: dict[str, tuple[int, int] | None] = {v: None for v in voice_names}  # (midi_pitch, start_step)

    for step in range(n_steps):
        base = step * 5
        # chord token at base+0 (ignored for MIDI reconstruction)
        for vi, vname in enumerate(voice_names):
            tok = body[base + 1 + vi]
            if is_pitch_token(tok):
                midi_p = token_to_midi(tok)
            elif is_rest_token(tok):
                midi_p = None
            else:
                midi_p = None  # unexpected token, treat as rest

            prev = active[vname]
            if prev is not None:
                prev_pitch, prev_start = prev
                # End note if pitch changed or rest
                if midi_p is None or midi_p != prev_pitch:
                    start_sec = prev_start * sec_per_step
                    end_sec = step * sec_per_step
                    if end_sec > start_sec:
                        instruments[vname].notes.append(pretty_midi.Note(
                            velocity=80, pitch=prev_pitch,
                            start=start_sec, end=end_sec,
                        ))
                    active[vname] = None

            if midi_p is not None and active[vname] is None:
                active[vname] = (midi_p, step)

    # Close any still-open notes
    for vname in voice_names:
        prev = active[vname]
        if prev is not None:
            prev_pitch, prev_start = prev
            start_sec = prev_start * sec_per_step
            end_sec = n_steps * sec_per_step
            if end_sec > start_sec:
                instruments[vname].notes.append(pretty_midi.Note(
                    velocity=80, pitch=prev_pitch,
                    start=start_sec, end=end_sec,
                ))

    for vname in voice_names:
        pm.instruments.append(instruments[vname])

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    pm.write(path)


# ────────────────────── BUILD VOCAB JSON ──────────────────────

def build_vocab_dict() -> dict:
    """Build the dense chorale vocab dictionary."""
    # Pitch labels
    note_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    pitch_labels = {}
    for i in range(NUM_PITCHES):
        midi_p = MIDI_LO + i
        octave = midi_p // 12 - 1
        name = note_names[midi_p % 12]
        pitch_labels[str(PITCH_OFFSET + i)] = f"{name}{octave}"

    # Chord labels
    chord_labels = {}
    for i in range(NUM_CHORDS):
        chord_labels[str(CHORD_OFFSET + i)] = chord_token_to_label(i)

    return {
        "vocab_size": VOCAB_SIZE,
        "PAD_ID": PAD_ID,
        "BOS_ID": BOS_ID,
        "EOS_ID": EOS_ID,
        "REST_ID": REST_ID,
        "PITCH_OFFSET": PITCH_OFFSET,
        "MIDI_LO": MIDI_LO,
        "MIDI_HI": MIDI_HI,
        "NUM_PITCHES": NUM_PITCHES,
        "CHORD_OFFSET": CHORD_OFFSET,
        "NUM_CHORDS": NUM_CHORDS,
        "VOICE_ORDER": VOICE_ORDER_NAMES,
        "VOICE_RANGES": VOICE_RANGES,
        "pitch_labels": pitch_labels,
        "chord_labels": chord_labels,
    }


# ────────────────────── MAIN ──────────────────────

def preprocess_split(chorales: list[np.ndarray], split_name: str,
                     augment: bool = False,
                     semitone_range: int = 6) -> list[dict]:
    """Tokenize a list of chorales.  If augment=True, apply ±semitone_range
    transpositions (train only).

    Returns list of dicts with keys: tokens, conts, n_timesteps, transposition.
    """
    results: list[dict] = []
    rejected = 0

    for idx, arr in enumerate(chorales):
        arr = arr.astype(np.float64)
        transpositions = [0]
        if augment:
            transpositions = list(range(-semitone_range, semitone_range + 1))

        for semi in transpositions:
            transposed = transpose_chorale(arr, semi)
            if transposed is None:
                rejected += 1
                continue
            tokens, conts = tokenize_chorale(transposed)
            results.append({
                "tokens": tokens,
                "conts": conts,
                "n_timesteps": transposed.shape[0],
                "transposition": semi,
                "original_idx": idx,
            })

    if rejected > 0:
        print(f"  {split_name}: rejected {rejected} transpositions (out of range)")

    return results


def main():
    ap = argparse.ArgumentParser(description="Dense chorale preprocessing: NPZ → token sequences.")
    ap.add_argument("--npz", default="data/Jsb16thSeparated.npz",
                    help="Path to Jsb16thSeparated.npz")
    ap.add_argument("--data_folder", default="runs/chorale_dense_events",
                    help="Output directory")
    ap.add_argument("--semitone_range", type=int, default=6,
                    help="Transposition range (±N semitones, default ±6)")
    ap.add_argument("--no_augment", action="store_true",
                    help="Disable transposition augmentation")
    ap.add_argument("--sample_midis", type=int, default=5,
                    help="Number of sample MIDIs to write (0 to skip)")
    args = ap.parse_args()

    if not os.path.isfile(args.npz):
        print(f"ERROR: NPZ file not found: {args.npz}", file=sys.stderr)
        sys.exit(1)

    data = np.load(args.npz, allow_pickle=True, encoding="latin1")
    os.makedirs(args.data_folder, exist_ok=True)

    # ── Process each split ──
    splits = {}
    for split_name, npz_key in [("train", "train"), ("val", "valid"), ("test", "test")]:
        chorales = list(data[npz_key])
        augment = (split_name == "train") and (not args.no_augment)
        print(f"Processing {split_name}: {len(chorales)} chorales"
              f"{' (augmenting ±' + str(args.semitone_range) + ' semitones)' if augment else ''}")
        results = preprocess_split(
            chorales, split_name,
            augment=augment,
            semitone_range=args.semitone_range,
        )
        splits[split_name] = results
        print(f"  → {len(results)} sequences"
              f" (avg {np.mean([len(r['tokens']) for r in results]):.0f} tokens)")

    # ── Save pickles ──
    for split_name, results in splits.items():
        pkl_path = os.path.join(args.data_folder, f"dense_{split_name}.pkl")
        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)
        print(f"Saved {pkl_path} ({len(results)} sequences)")

    # ── Save vocab JSON ──
    vocab = build_vocab_dict()
    vocab_path = os.path.join(args.data_folder, "dense_vocab.json")
    with open(vocab_path, "w") as f:
        json.dump(vocab, f, indent=2)
    print(f"Saved {vocab_path} (vocab_size={vocab['vocab_size']})")

    # ── Stats ──
    print("\n── Stats ──")
    for split_name, results in splits.items():
        token_lens = [len(r["tokens"]) for r in results]
        print(f"  {split_name}: {len(results)} seqs, "
              f"tokens min={min(token_lens)} avg={np.mean(token_lens):.0f} max={max(token_lens)}")

    # ── Sample MIDIs ──
    if args.sample_midis > 0:
        sample_dir = os.path.join(args.data_folder, "_samples")
        os.makedirs(sample_dir, exist_ok=True)
        # Write a few from each split
        for split_name, results in splits.items():
            for i in range(min(args.sample_midis, len(results))):
                # Pick original (transposition=0) if possible
                orig = [r for r in results if r["transposition"] == 0]
                r = orig[i] if i < len(orig) else results[i]
                out_path = os.path.join(sample_dir, f"{split_name}_{i:03d}.mid")
                decode_tokens_to_midi(r["tokens"], out_path)
        print(f"Wrote sample MIDIs to {sample_dir}")

    print("\nDone.")


if __name__ == "__main__":
    main()
