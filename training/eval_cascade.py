#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Evaluation metrics for cascade-generated music.

Metrics:
  - chord_tone_coverage: fraction of notes that belong to the local chord
  - range_violations: count of notes outside standard instrument ranges
  - parallel_fifths_octaves: count of parallel 5th/octave voice-leading violations
  - note_density_per_instrument: notes per quarter note per instrument
  - pitch_class_entropy: Shannon entropy of pitch class distribution
  - compare_ablations: summary comparison of two generated MIDIs

Usage:
  python training/eval_cascade.py \\
      --midi runs/generated/cascade_out.mid \\
      --vocab_json runs/cascade_events/cascade_vocab.json
"""

import os
import json
import math
import argparse
from collections import Counter
from typing import Dict, List, Tuple, Optional

import numpy as np
import pretty_midi

from training.pre import (
    InstrumentConfig,
    make_instrument_config,
    INSTRUMENT_PRESETS,
    extract_multitrack_events,
    qn_between,
    is_drum_slot,
)
from training.pre_cascade import (
    extract_chord_labels,
    CHORD_QUALITIES,
)


# ── Standard instrument ranges (MIDI note numbers) ───────────

INSTRUMENT_RANGES: Dict[str, Tuple[int, int]] = {
    "voxlead": (48, 84),   # C3–C6
    "voxharm": (48, 84),   # C3–C6
    "guitar":  (40, 88),   # E2–E6
    "other":   (21, 108),  # A0–C8 (generous for synths/piano)
    "bass":    (24, 60),   # C1–C4
    "drums":   (27, 87),   # GM percussion range
    # Chorale voices
    "soprano": (57, 84),
    "alto":    (50, 77),
    "tenor":   (43, 72),
    "bassvox": (33, 69),
}


# ── Chord-tone coverage ──────────────────────────────────────

def chord_tone_coverage(
    ev: List[Tuple[float, int, int, int, float]],
    chord_labels: List[Tuple[float, int, int]],
    tempo_bpm: float,
    config: InstrumentConfig,
) -> float:
    """Fraction of melodic notes whose pitch class belongs to the local chord.

    Returns float in [0, 1]. Higher is better for tonal coherence.
    """
    if not ev or not chord_labels:
        return 0.0

    # Build chord intervals: chord at each beat covers [beat_qn, beat_qn+1)
    # (time_qn, root_pc, quality_idx)
    chord_at_beat: Dict[int, Tuple[int, int]] = {}
    for (t_qn, root, qual) in chord_labels:
        chord_at_beat[int(round(t_qn))] = (root, qual)

    # Chord tones by quality
    QUALITY_PCS = {
        0: {0, 4, 7},          # maj: root, M3, P5
        1: {0, 3, 7},          # min: root, m3, P5
        2: {0, 4, 7, 10},      # dom7: root, M3, P5, m7
        3: {0, 3, 6},          # dim: root, m3, d5
        4: {0, 4, 8},          # aug: root, M3, A5
    }

    on_chord = 0
    total = 0

    for (start_s, inst, midi, vel, dur_qn) in ev:
        if is_drum_slot(inst, config):
            continue
        total += 1
        onset_qn = qn_between(0.0, start_s, tempo_bpm)
        beat = int(onset_qn)

        # Find nearest chord
        chord_info = chord_at_beat.get(beat)
        if chord_info is None:
            # Try adjacent beats
            for offset in [-1, 1, -2, 2]:
                chord_info = chord_at_beat.get(beat + offset)
                if chord_info is not None:
                    break
        if chord_info is None:
            continue

        root, qual = chord_info
        chord_pcs = {(root + interval) % 12 for interval in QUALITY_PCS.get(qual, {0, 4, 7})}
        if (midi % 12) in chord_pcs:
            on_chord += 1

    return float(on_chord) / max(1, total)


# ── Range violations ──────────────────────────────────────────

def range_violations(
    ev: List[Tuple[float, int, int, int, float]],
    config: InstrumentConfig,
) -> Dict[str, int]:
    """Count notes outside standard instrument ranges.

    Returns {instrument_name: violation_count}.
    """
    violations: Dict[str, int] = {}
    for (_, inst, midi, _, _) in ev:
        if inst >= len(config.names):
            continue
        name = config.names[inst]
        lo, hi = INSTRUMENT_RANGES.get(name, (0, 127))
        if midi < lo or midi > hi:
            violations[name] = violations.get(name, 0) + 1
    return violations


# ── Parallel fifths and octaves ───────────────────────────────

def parallel_fifths_octaves(
    ev: List[Tuple[float, int, int, int, float]],
    tempo_bpm: float,
    config: InstrumentConfig,
    time_tolerance_qn: float = 0.125,
) -> int:
    """Count parallel 5th and octave voice-leading violations.

    Checks consecutive note pairs between all melodic instrument pairs.
    Returns total violation count.
    """
    # Group notes by instrument
    by_inst: Dict[int, List[Tuple[float, int]]] = {}
    for (start_s, inst, midi, _, _) in ev:
        if is_drum_slot(inst, config):
            continue
        onset_qn = qn_between(0.0, start_s, tempo_bpm)
        by_inst.setdefault(inst, []).append((onset_qn, midi))

    # Sort each by onset
    for inst in by_inst:
        by_inst[inst].sort()

    violations = 0
    inst_list = sorted(by_inst.keys())

    for i in range(len(inst_list)):
        for j in range(i + 1, len(inst_list)):
            notes_i = by_inst[inst_list[i]]
            notes_j = by_inst[inst_list[j]]

            # Find simultaneous note pairs
            pairs = []
            ji = 0
            for (t_i, p_i) in notes_i:
                while ji < len(notes_j) and notes_j[ji][0] < t_i - time_tolerance_qn:
                    ji += 1
                jj = ji
                while jj < len(notes_j) and notes_j[jj][0] <= t_i + time_tolerance_qn:
                    pairs.append((t_i, p_i, notes_j[jj][1]))
                    jj += 1

            # Check consecutive pairs for parallel 5ths/octaves
            for k in range(1, len(pairs)):
                _, p1_i, p1_j = pairs[k - 1]
                _, p2_i, p2_j = pairs[k]

                interval1 = abs(p1_i - p1_j) % 12
                interval2 = abs(p2_i - p2_j) % 12

                # Both voices move (not oblique/contrary)
                if p2_i == p1_i or p2_j == p1_j:
                    continue
                # Same direction (parallel)
                dir_i = (p2_i - p1_i) > 0
                dir_j = (p2_j - p1_j) > 0
                if dir_i != dir_j:
                    continue

                # Parallel 5th (interval 7) or octave (interval 0)
                if interval1 in (0, 7) and interval1 == interval2:
                    violations += 1

    return violations


# ── Note density ──────────────────────────────────────────────

def note_density_per_instrument(
    ev: List[Tuple[float, int, int, int, float]],
    tempo_bpm: float,
    config: InstrumentConfig,
) -> Dict[str, float]:
    """Notes per quarter note for each instrument.

    Returns {instrument_name: density}.
    """
    if not ev:
        return {}

    max_time = max(qn_between(0.0, e[0], tempo_bpm) for e in ev)
    duration_qn = max(1.0, max_time)

    counts: Dict[str, int] = {}
    for (_, inst, _, _, _) in ev:
        if inst < len(config.names):
            name = config.names[inst]
            counts[name] = counts.get(name, 0) + 1

    return {name: count / duration_qn for name, count in counts.items()}


# ── Pitch class entropy ──────────────────────────────────────

def pitch_class_entropy(
    ev: List[Tuple[float, int, int, int, float]],
    config: InstrumentConfig,
) -> float:
    """Shannon entropy of pitch class distribution (melodic instruments only).

    Returns float in [0, log2(12)] ≈ [0, 3.585]. Higher = more diverse.
    """
    pc_counts = Counter()
    for (_, inst, midi, _, _) in ev:
        if not is_drum_slot(inst, config):
            pc_counts[midi % 12] += 1

    total = sum(pc_counts.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in pc_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)

    return entropy


# ── Ablation comparison ───────────────────────────────────────

def compare_ablations(
    midi_a: str,
    midi_b: str,
    vocab_path: str,
) -> Dict[str, Dict[str, float]]:
    """Run all metrics on two MIDIs and return comparison summary."""
    with open(vocab_path, "r") as f:
        vocab = json.load(f)

    config = make_instrument_config(
        vocab.get("instrument_names", INSTRUMENT_PRESETS["blues6"])
    )

    results = {}
    for label, midi_path in [("A", midi_a), ("B", midi_b)]:
        if not os.path.isfile(midi_path):
            print(f"WARNING: {midi_path} not found, skipping")
            continue

        ev, tempo, _, _ = extract_multitrack_events(midi_path, config)
        chords = extract_chord_labels(ev, tempo, config)

        results[label] = {
            "chord_tone_coverage": chord_tone_coverage(ev, chords, tempo, config),
            "range_violations": sum(range_violations(ev, config).values()),
            "parallel_fifths_octaves": parallel_fifths_octaves(ev, tempo, config),
            "pitch_class_entropy": pitch_class_entropy(ev, config),
            "total_notes": len(ev),
            **{f"density_{k}": v for k, v in note_density_per_instrument(ev, tempo, config).items()},
        }

    return results


# ── CLI ───────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser("eval_cascade: evaluate generated MIDI.")
    ap.add_argument("--midi", required=True, help="Path to generated MIDI")
    ap.add_argument("--vocab_json", required=True, help="Path to cascade_vocab.json")
    ap.add_argument("--midi_b", default="", help="Optional second MIDI for A/B comparison")
    ap.add_argument("--instrument_set", default="blues6")
    args = ap.parse_args()

    with open(args.vocab_json, "r") as f:
        vocab = json.load(f)

    config = make_instrument_config(
        vocab.get("instrument_names", INSTRUMENT_PRESETS[args.instrument_set])
    )

    if args.midi_b:
        results = compare_ablations(args.midi, args.midi_b, args.vocab_json)
        print("\n── Ablation Comparison ─────────────────────")
        for label, metrics in results.items():
            print(f"\n  {label}:")
            for k, v in metrics.items():
                print(f"    {k:>30}: {v:.4f}" if isinstance(v, float) else f"    {k:>30}: {v}")
        print("────────────────────────────────────────────")
        return

    # Single MIDI evaluation
    ev, tempo, _, _ = extract_multitrack_events(args.midi, config)
    chords = extract_chord_labels(ev, tempo, config)

    print(f"\n── Evaluation: {os.path.basename(args.midi)} ──")
    print(f"  Total notes:             {len(ev)}")

    ctc = chord_tone_coverage(ev, chords, tempo, config)
    print(f"  Chord-tone coverage:     {ctc:.3f}")

    rv = range_violations(ev, config)
    total_rv = sum(rv.values())
    print(f"  Range violations:        {total_rv}")
    for name, count in sorted(rv.items()):
        print(f"    {name:>10}: {count}")

    pfo = parallel_fifths_octaves(ev, tempo, config)
    print(f"  Parallel 5ths/8ves:      {pfo}")

    density = note_density_per_instrument(ev, tempo, config)
    print(f"  Note density (per QN):")
    for name, d in sorted(density.items()):
        print(f"    {name:>10}: {d:.2f}")

    pce = pitch_class_entropy(ev, config)
    print(f"  Pitch class entropy:     {pce:.3f}  (max={math.log2(12):.3f})")
    print(f"──────────────────────────────────────────────")


if __name__ == "__main__":
    main()
