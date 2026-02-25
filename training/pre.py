#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Event-stream preprocessing with configurable instrument sets and train-only augmentation.

Instrument presets:
  blues6:   voxlead, voxharm, guitar, other, bass, drums  (original 6)
  chorale4: soprano, alto, tenor, bassvox                  (Bach chorales)

Augmentations (TRAIN ONLY):
  • Pitch: ±{1,3,5} semitones on melodic instruments (drums untouched)
  • Velocity: ±10 additive (clipped to [1,127])

Outputs
-------
DATA_FOLDER/
  events_train.pkl         # {"sequences": [...], "aux": [...]}
  events_val.pkl           # {"sequences": [...], "aux": [...]}
  event_vocab.json
  _samples/tokenized_000.mid ...
  vocab_summary.txt

Event order per note: TIME_SHIFT → BAR (optional) → INST → VEL → PITCH → DUR.
BAR tokens encode (steps_per_bar, bar_position) pairs so mixed meters work.

Aux targets ("polyphony instructor") are computed per 512-token window from note intervals.
Aux_dim depends on instrument config (blues6=36, chorale4=24).
"""

import os
import glob
import json
import random
import pickle
import argparse
import itertools
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Set, Optional

import numpy as np
import pretty_midi
from collections import Counter

# -------------- PATHS --------------
MIDI_FOLDER   = "midi_songs5"
DATA_FOLDER   = "data_events6"
os.makedirs(DATA_FOLDER, exist_ok=True)
SAMPLES_DIR   = os.path.join(DATA_FOLDER, "_samples")
os.makedirs(SAMPLES_DIR, exist_ok=True)

# -------------- INSTRUMENT CONFIG --------------
@dataclass
class InstrumentConfig:
    names: List[str]
    drum_idx: Optional[int] = None
    guitar_idx: Optional[int] = None
    other_idx: Optional[int] = None
    bass_idx: Optional[int] = None
    voxlead_idx: Optional[int] = None
    voxharm_idx: Optional[int] = None
    # Augmentation settings (per-preset)
    aug_transposes: List[int] = field(default_factory=lambda: [-5, -3, -1, 1, 3, 5])
    aug_vel_deltas: List[int] = field(default_factory=lambda: [-10, 10])
    # Optional voice range limits for transposition validation: {voice_name: (min_midi, max_midi)}
    voice_ranges: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    @property
    def num_instruments(self) -> int:
        return len(self.names)

    def has_drums(self) -> bool:
        return self.drum_idx is not None


def make_instrument_config(names: List[str]) -> InstrumentConfig:
    """Auto-detect role indices from instrument names."""
    def _find(name: str) -> Optional[int]:
        try:
            return names.index(name)
        except ValueError:
            return None

    cfg = InstrumentConfig(
        names=list(names),
        drum_idx=_find("drums"),
        guitar_idx=_find("guitar"),
        other_idx=_find("other"),
        bass_idx=_find("bass"),
        voxlead_idx=_find("voxlead"),
        voxharm_idx=_find("voxharm"),
    )

    # Chorale-specific settings
    if set(names) == {"soprano", "alto", "tenor", "bassvox"}:
        cfg.aug_transposes = [-3, -2, -1, 1, 2, 3]
        cfg.aug_vel_deltas = []  # uniform velocity in chorales
        cfg.voice_ranges = {
            "soprano": (57, 84),
            "alto": (50, 77),
            "tenor": (43, 72),
            "bassvox": (33, 69),
        }

    return cfg


INSTRUMENT_PRESETS: Dict[str, List[str]] = {
    "blues6":   ["voxlead", "voxharm", "guitar", "other", "bass", "drums"],
    "chorale4": ["soprano", "alto", "tenor", "bassvox"],
}

# Default config (blues6) — used by legacy callers and module-level constants
_DEFAULT_CONFIG = make_instrument_config(INSTRUMENT_PRESETS["blues6"])

# Legacy globals for backward compatibility (used by tests, generate.py seed import, etc.)
INSTRUMENT_NAMES = _DEFAULT_CONFIG.names
NUM_INSTRUMENTS = _DEFAULT_CONFIG.num_instruments
DRUM_IDX = _DEFAULT_CONFIG.drum_idx  # type: ignore[assignment]
OTHER_IDX = _DEFAULT_CONFIG.other_idx  # type: ignore[assignment]
GUITAR_IDX = _DEFAULT_CONFIG.guitar_idx  # type: ignore[assignment]
BASS_IDX = _DEFAULT_CONFIG.bass_idx  # type: ignore[assignment]
VOXLEAD_IDX = _DEFAULT_CONFIG.voxlead_idx  # type: ignore[assignment]
VOXHARM_IDX = _DEFAULT_CONFIG.voxharm_idx  # type: ignore[assignment]

# -------------- CORE SETTINGS --------------
# Windowing for LM training
SEQ_LEN    = 512
SEQ_STRIDE = 256

# ----- DATA AUGMENTATION (TRAIN ONLY) -----
AUG_TRANSPOSES = [-5, -3, -1, 1, 3, 5]  # semitone shifts for melodic instruments (all common blues keys)
AUG_VEL_DELTAS = [-10, 10]    # additive velocity shifts for ALL instruments (clip to [1,127])
AUG_ENABLE     = True


# -------------- TRACK SELECTION --------------
# Keep a fixed canonical instrument index space (size=6) for compatibility,
# but allow selecting a subset of tracks to KEEP when building the dataset.
# This affects which events are included (and thus what the model learns to generate).
#
# Canonical instruments:
#   voxlead, voxharm, guitar, other, bass, drums
#
# Aliases accepted in --tracks:
#   voxbg, bgvox, backingvox, auxvox  -> voxharm
TRACK_ALIASES = {
    "voxbg": "voxharm",
    "bgvox": "voxharm",
    "backingvox": "voxharm",
    "auxvox": "voxharm",
    "voxharm": "voxharm",
    "voxlead": "voxlead",
    "guitar": "guitar",
    "other": "other",
    "bass": "bass",
    "drums": "drums",
}

def parse_tracks_arg(s: str, config: Optional[InstrumentConfig] = None) -> List[str]:
    if config is None:
        config = _DEFAULT_CONFIG
    s = (s or "").strip().lower()
    if (not s) or s in ("all", "*"):
        return list(config.names)
    parts = [p.strip().lower() for p in s.split(",") if p.strip()]
    out = []
    for p in parts:
        # Try alias first (blues6 only), then direct name match
        canon = TRACK_ALIASES.get(p, None)
        if canon is None:
            # Check if it's a direct name in config.names
            if p in config.names:
                canon = p
            else:
                raise ValueError(
                    f"Unknown track '{p}'. Valid: {sorted(set(config.names) | set(TRACK_ALIASES.keys()))} or 'all'."
                )
        if canon not in config.names:
            raise ValueError(f"Track '{canon}' not in instrument config: {config.names}")
        if canon not in out:
            out.append(canon)
    return out

# ---------- DRUM DIAGNOSTICS (optional) ----------
GM_MIN, GM_MAX = 32, 81  # GM percussion nominal range

def _is_drum_track(inst, name_keywords=None) -> bool:
    if name_keywords is None:
        name_keywords = (
            "drum","kick","kik","bd","snare","sd","hat","hh",
            "ride","crash","shaker","cymbal","toms","tom","perc","clap"
        )
    lname = (inst.name or "").lower()
    return bool(getattr(inst, "is_drum", False) or any(k in lname for k in name_keywords))

def diagnose_drum_midi_anomalies(midi_folder: str, strict_range_only: bool = True):
    paths = sorted(
        glob.glob(os.path.join(midi_folder, "*.mid")) +
        glob.glob(os.path.join(midi_folder, "*.midi"))
    )
    results = {}
    global_out = Counter()
    total_files = len(paths)
    flagged_files = 0

    print(f"\n── Drum Diagnostics: scanning {total_files} files in '{midi_folder}' ──")
    for p in paths:
        try:
            pm = pretty_midi.PrettyMIDI(p)
        except Exception as e:
            print(f"  [skip] {os.path.basename(p)}: {e}")
            continue

        drum_pitches = []
        for inst in pm.instruments:
            if _is_drum_track(inst):
                drum_pitches.extend(int(n.pitch) for n in inst.notes)

        if not drum_pitches:
            continue

        uniq = sorted({m for m in drum_pitches})
        oob = sorted(m for m in uniq if (m < GM_MIN or m > GM_MAX))

        if (not strict_range_only) or oob:
            flagged_files += (1 if oob else 0)
            for m in oob:
                global_out[m] += 1
            name = os.path.basename(p)
            if oob:
                print(f"  [non-GM range] {name}: out-of-range={oob} (min={min(uniq)}, max={max(uniq)})")
            elif not strict_range_only:
                print(f"  [drums ok]     {name}: range OK (min={min(uniq)}, max={max(uniq)})")

        results[p] = {"out_of_range": oob, "all_drums": uniq, "counts": Counter(drum_pitches)}

    print("\n── Summary ─────────────────────────────────")
    print(f"Files scanned: {total_files}")
    print(f"Files with out-of-range drum notes (<{GM_MIN} or >{GM_MAX}): {flagged_files}")
    if global_out:
        top = ", ".join(f"{m}:{c}" for m, c in global_out.most_common(12))
        print(f"Most common offending pitches (MIDI): {top}")
    else:
        print("No out-of-range drum notes found.")
    print("──────────────────────────────────────────\n")
    return results
# ---------- end DRUM DIAGNOSTICS ----------

# -------------- GRID / BINS --------------
BASE_SUBDIV = 4  # steps per quarter note group used for BAR positions
TIME_SHIFT_QN_STEP = 1.0/24.0
TIME_SHIFT_QN_MAX  = 4.0   # in quarter notes (multi-token encoding handles longer gaps)

def make_duration_bins_qn(max_qn=8.0):
    base = [1/24, 1/12, 1/8, 1/6, 1/4, 1/3, 3/8, 1/2, 2/3, 3/4, 1.0,
            1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
    xs = sorted({x for x in base if 0 < x <= max_qn})
    return xs
DURATION_BINS_QN = make_duration_bins_qn()

VELOCITY_BINS = list(range(0, 128, 16))  # 0,16,...,112

BOS = "<BOS>"
EOS = "<EOS>"

# -------------- HELPERS: BAR GRID --------------
def build_bar_grid(pm: pretty_midi.PrettyMIDI, base_subdiv=BASE_SUBDIV):
    downbeats = pm.get_downbeats()
    if len(downbeats) == 0:
        beats = pm.get_beats()
        if len(beats) < 2:
            raise ValueError("Cannot infer bars (no downbeats or beats).")
        downbeats = beats[::4]
    db = np.asarray(downbeats, dtype=float)

    if len(db) >= 2:
        bar_durs = np.diff(db)
        median_dur = float(np.median(bar_durs))
    else:
        median_dur = 2.0

    ts_changes = sorted(pm.time_signature_changes, key=lambda ts: ts.time)
    if not ts_changes:
        ts_changes = [pretty_midi.containers.TimeSignature(4, 4, 0.0)]

    def ts_at(time_sec: float):
        idx = 0
        for i, ts in enumerate(ts_changes):
            if ts.time <= time_sec:
                idx = i
            else:
                break
        ts = ts_changes[idx]
        return int(ts.numerator), int(ts.denominator)

    bars = []
    for i in range(len(db)):
        s = db[i]
        e = db[i+1] if i+1 < len(db) else db[i] + median_dur
        numer, denom = ts_at(s)
        steps = int(numer * (4.0/denom) * base_subdiv)
        steps = max(1, steps)
        bars.append((s, e, steps))

    return db, bars

def time_to_barpos(t_sec: float, bar_starts: np.ndarray, bars_meta: List[Tuple[float,float,int]]) -> Tuple[int,int]:
    i = int(np.searchsorted(bar_starts, t_sec, side='right') - 1)
    i = max(0, min(i, len(bars_meta)-1))
    s, e, steps = bars_meta[i]
    dur = max(e - s, 1e-9)
    phase = (t_sec - s) / dur
    pos = int(np.floor(np.clip(phase, 0.0, 0.999999) * steps)) % steps
    return pos, steps

# -------------- QUANTIZATION --------------
def nearest_bin(x: float, bins: List[float]) -> float:
    if not bins:
        return 0.0
    arr = np.asarray(bins, dtype=float)
    idx = int(np.argmin(np.abs(arr - float(x))))
    return float(arr[idx])

def qn_between(a_sec: float, b_sec: float, tempo_bpm: float) -> float:
    return (b_sec - a_sec) * tempo_bpm / 60.0

def qn_to_sec(qn: float, tempo_bpm: float) -> float:
    return qn * 60.0 / tempo_bpm

# -------------- NOTE EXTRACTION / MAPPING --------------
DRUM_KEYWORDS = ("drum","kick","kik","bd","snare","sd","hat","hh","ride","crash","shaker","cymbal","toms","tom","perc","clap")

ALIAS_TO_CANON = {
    "auxvox": "voxharm",
    "voxbg":  "voxharm",
    "bgvox":  "voxharm",
    "backingvox": "voxharm",
    "synth":  "other",
}

VOXLEAD_IDX = INSTRUMENT_NAMES.index("voxlead")
VOXHARM_IDX = INSTRUMENT_NAMES.index("voxharm")
BASS_IDX    = INSTRUMENT_NAMES.index("bass")

def _slot_from_gm_program(prog: int) -> Optional[int]:
    """Map GM program number to canonical instrument slot.

    Returns None for programs we don't recognise — caller falls through to OTHER_IDX.
    """
    if 0 <= prog <= 7:     return OTHER_IDX       # piano
    if 16 <= prog <= 20:   return OTHER_IDX       # organ family (Hammond, etc.)
    if prog == 22:         return OTHER_IDX       # harmonica (very common in blues)
    if 24 <= prog <= 31:   return GUITAR_IDX      # guitar family
    if 32 <= prog <= 39:   return BASS_IDX        # bass family
    if prog == 52:         return VOXLEAD_IDX     # choir aahs → lead vox
    if 53 <= prog <= 54:   return VOXHARM_IDX     # voice oohs / synth voice
    # Fallback mappings
    if 48 <= prog <= 51:   return OTHER_IDX       # strings ensemble
    if prog == 55:         return OTHER_IDX       # orchestra hit
    if 56 <= prog <= 63:   return OTHER_IDX       # brass
    if 64 <= prog <= 71:   return OTHER_IDX       # reed
    if 80 <= prog <= 87:   return OTHER_IDX       # synth lead
    return None


def map_name_to_slot(inst: pretty_midi.Instrument,
                     config: Optional[InstrumentConfig] = None) -> int:
    if config is None:
        config = _DEFAULT_CONFIG

    lname = (inst.name or "").strip().lower()

    # For non-blues6 configs: try exact match on config.names first
    if config.names != INSTRUMENT_PRESETS["blues6"]:
        for idx, cname in enumerate(config.names):
            if lname == cname:
                return idx
        # No exact match — skip heuristics for non-blues configs
        raise ValueError(
            f"Track name '{inst.name}' not in config.names={config.names}. "
            f"Converter should write exact names."
        )

    # --- blues6 heuristics below ---
    # canonicalize common aliases first
    for alias, canon in ALIAS_TO_CANON.items():
        if alias in lname:
            lname = lname.replace(alias, canon)

    # Drums
    if inst.is_drum or any(k in lname for k in DRUM_KEYWORDS):
        assert config.drum_idx is not None
        return config.drum_idx

    # Vox lead
    if "voxlead" in lname or ("lead" in lname and "vox" in lname):
        assert config.voxlead_idx is not None
        return config.voxlead_idx

    # Vox harm (includes auxvox/backing)
    if ("voxharm" in lname) or ("harmony" in lname and "vox" in lname):
        assert config.voxharm_idx is not None
        return config.voxharm_idx

    # Guitar
    if "guitar" in lname or "gtr" in lname:
        assert config.guitar_idx is not None
        return config.guitar_idx

    # Bass
    if "bass" in lname:
        assert config.bass_idx is not None
        return config.bass_idx

    # GM program fallback (for files with empty/generic track names)
    gm_slot = _slot_from_gm_program(inst.program)
    if gm_slot is not None:
        return gm_slot

    # Everything else → other
    assert config.other_idx is not None
    return config.other_idx

def extract_multitrack_events(path: str,
                              config: Optional[InstrumentConfig] = None):
    """
    Returns:
      events: list of (start_sec, inst_idx, midi_pitch, velocity, dur_qn)
      tempo_bpm
      bar_starts, bars_meta
    """
    if config is None:
        config = _DEFAULT_CONFIG

    pm = pretty_midi.PrettyMIDI(path)
    tc = pm.get_tempo_changes()
    tempo = float(tc[1][0]) if tc[1].size > 0 else 120.0

    bar_starts, bars_meta = build_bar_grid(pm, BASE_SUBDIV)

    ev = []
    for inst in pm.instruments:
        slot = map_name_to_slot(inst, config)
        for n in inst.notes:
            start = float(n.start)
            dur_qn = qn_between(n.start, n.end, tempo)
            ev.append((start, slot, int(n.pitch), int(n.velocity), float(dur_qn)))

    ev.sort(key=lambda x: x[0])
    return ev, tempo, bar_starts, bars_meta

# -------------- AUGMENTATION HELPERS --------------
def is_drum_slot(inst_idx: int, config: Optional[InstrumentConfig] = None) -> bool:
    if config is None:
        config = _DEFAULT_CONFIG
    return config.drum_idx is not None and inst_idx == config.drum_idx

def clip_midi_pitch(p: int) -> int:
    return max(0, min(127, int(p)))

def clip_velocity(v: int) -> int:
    return max(1, min(127, int(v)))

def augment_events_additive(
    ev: List[Tuple[float,int,int,int,float]],
    semitone_shift: int,
    vel_delta: int,
    config: Optional[InstrumentConfig] = None,
) -> Optional[List[Tuple[float,int,int,int,float]]]:
    """
    Return a new list:
      • melodic pitches shifted by semitone_shift (drums untouched)
      • velocities shifted by vel_delta (all instruments), clipped to [1,127]

    If config has voice_ranges, rejects the entire transposition if any note
    falls outside the voice's range.  Returns None if rejected.
    """
    if config is None:
        config = _DEFAULT_CONFIG

    out = []
    for (start_s, inst, midi, vel, dur_qn) in ev:
        if is_drum_slot(inst, config):
            midi_out = midi
        else:
            midi_out = clip_midi_pitch(midi + semitone_shift)
            # Voice range check
            if config.voice_ranges:
                voice_name = config.names[inst] if inst < len(config.names) else None
                if voice_name and voice_name in config.voice_ranges:
                    lo, hi = config.voice_ranges[voice_name]
                    if midi_out < lo or midi_out > hi:
                        return None  # reject this transposition
        v = clip_velocity(vel + vel_delta) if vel_delta != 0 else vel
        out.append((start_s, inst, midi_out, v, dur_qn))
    return out

# -------------- VOCAB BUILD --------------
def build_pitch_maps(all_events, config: Optional[InstrumentConfig] = None):
    """
    Build per-family pitch maps:
      general (melodic instruments), drums (all percussive pitches).
    When config has no drums, the drums map is empty.
    """
    if config is None:
        config = _DEFAULT_CONFIG

    gen: Set[int] = set()
    drums: Set[int] = set()
    for _, inst, midi, _, _ in all_events:
        if is_drum_slot(inst, config):
            drums.add(midi)
        else:
            gen.add(midi)

    maps: Dict[str, Dict[int, int]] = {
        "general": {p: i+1 for i, p in enumerate(sorted(gen))},   # 0 reserved
    }
    if config.has_drums():
        maps["drums"] = {p: i+1 for i, p in enumerate(sorted(drums))}  # 0 reserved

    return maps

def gather_bar_pairs(all_bars_meta: List[List[Tuple[float,float,int]]]) -> List[Tuple[int,int]]:
    pairs: Set[Tuple[int,int]] = set()
    for bars in all_bars_meta:
        for (_, _, steps) in bars:
            for pos in range(steps):
                pairs.add((steps, pos))
    return sorted(pairs)

def quantize_velocity(v: int, bins=VELOCITY_BINS) -> int:
    idx = int(np.argmin([abs(v - b) for b in bins]))
    return idx

def quantize_duration_qn(d_qn: float) -> int:
    val = nearest_bin(d_qn, DURATION_BINS_QN)
    return int(DURATION_BINS_QN.index(val))

# -------------- EVENT VOCAB LAYOUT --------------
def compute_aux_layout(config: InstrumentConfig) -> Dict:
    """Compute the auxiliary feature layout for a given instrument config.

    Returns dict with 'aux_dim', 'fields', 'enabled'.
    """
    N = config.num_instruments
    fields: List[str] = []
    dim = 0

    # Always present: per-instrument polyphony features
    fields.append(f"max_polyphony[{N}]")
    dim += N
    fields.append(f"mean_polyphony[{N}]")
    dim += N
    fields.append(f"overlap_ratio[{N}]")
    dim += N

    # Chord stats only for configs with guitar + other
    has_chords = config.guitar_idx is not None and config.other_idx is not None
    if has_chords:
        fields.append("chord_mean_guitar, chord_max_guitar, chord_mean_other, chord_max_other")
        dim += 4

    # Pitch class histogram (always)
    fields.append("pitch_class_histogram[12] (non-drums, normalized)")
    dim += 12

    # Swing/blues only for configs with drums
    has_swing_blues = config.has_drums()
    if has_swing_blues:
        fields.append("swing_score[1] (0=straight, 1=triplet-shuffle)")
        dim += 1
        fields.append("blues_scale_score[1] (0=none, 1=all in blues scale)")
        dim += 1

    return {
        "aux_dim": dim,
        "fields": fields,
        "enabled": True,
        "has_chords": has_chords,
        "has_swing_blues": has_swing_blues,
    }


def build_event_vocab(pitch_maps, bar_pairs: List[Tuple[int,int]],
                      config: Optional[InstrumentConfig] = None):
    if config is None:
        config = _DEFAULT_CONFIG

    vocab = {}
    idx = 0
    vocab["PAD"] = {"start": idx, "size": 1}; idx += 1
    vocab["BOS"] = {"start": idx, "size": 1}; idx += 1
    vocab["EOS"] = {"start": idx, "size": 1}; idx += 1

    bar_pair_to_local = {pair: i for i, pair in enumerate(bar_pairs)}
    vocab["BAR"] = {"start": idx, "size": len(bar_pairs)}; idx += len(bar_pairs)

    n_time = int(round(TIME_SHIFT_QN_MAX / TIME_SHIFT_QN_STEP))
    n_time = max(1, n_time)
    vocab["TIME_SHIFT"] = {"start": idx, "size": n_time}; idx += n_time

    vocab["INST"] = {"start": idx, "size": config.num_instruments}; idx += config.num_instruments
    vocab["VEL"]  = {"start": idx, "size": len(VELOCITY_BINS)}; idx += len(VELOCITY_BINS)
    vocab["DUR"]  = {"start": idx, "size": len(DURATION_BINS_QN)}; idx += len(DURATION_BINS_QN)

    gen_sz = len(pitch_maps["general"]) + 1
    vocab["PITCH_GENERAL"] = {"start": idx, "size": gen_sz}; idx += gen_sz

    if config.has_drums() and "drums" in pitch_maps:
        drums_sz = len(pitch_maps["drums"]) + 1
        vocab["PITCH_DRUMS"] = {"start": idx, "size": drums_sz}; idx += drums_sz

    pitch_space_for_inst = {}
    for i in range(config.num_instruments):
        if is_drum_slot(i, config):
            pitch_space_for_inst[str(i)] = "PITCH_DRUMS"
        else:
            pitch_space_for_inst[str(i)] = "PITCH_GENERAL"

    aux_layout = compute_aux_layout(config)

    vocab_meta = {
        "type": f"event_vocab_v2_{config.num_instruments}inst",
        "base_subdiv": BASE_SUBDIV,
        "time_shift_qn_step": TIME_SHIFT_QN_STEP,
        "time_shift_qn_max": TIME_SHIFT_QN_MAX,
        "velocity_bins": VELOCITY_BINS,
        "duration_bins_qn": DURATION_BINS_QN,
        "pitch_maps": pitch_maps,
        "pitch_space_for_inst": pitch_space_for_inst,
        "bar_pairs": bar_pairs,
        "bar_pair_to_local": {f"{s}:{p}": i for (s,p), i in bar_pair_to_local.items()},
        "layout": vocab,
        "instrument_names": config.names,
        "drum_index": config.drum_idx,
        "aux": aux_layout,
    }
    return vocab_meta

# -------------- ENCODER / DECODER --------------
def tok_of(vocab, group, local_idx):
    return vocab["layout"][group]["start"] + int(local_idx)

def encode_bar_pair(vocab, steps: int, pos: int) -> Optional[int]:
    key = f"{steps}:{pos}"
    local = vocab["bar_pair_to_local"].get(key, None)
    if local is None:
        return None
    return tok_of(vocab, "BAR", local)

def encode_time_shift_qn(vocab, delta_qn: float) -> List[int]:
    out = []
    step = float(vocab["time_shift_qn_step"])
    max_local = vocab["layout"]["TIME_SHIFT"]["size"]
    steps = int(round(delta_qn / step))
    if steps <= 0:
        return out
    while steps > 0:
        take = min(steps, max_local)
        out.append(tok_of(vocab, "TIME_SHIFT", take - 1))  # local 0 → 1 step
        steps -= take
    return out

def encode_inst(vocab, inst_idx: int) -> int:
    return tok_of(vocab, "INST", inst_idx)

def encode_vel(vocab, v: int) -> int:
    return tok_of(vocab, "VEL", quantize_velocity(v))

def encode_dur(vocab, d_qn: float) -> int:
    return tok_of(vocab, "DUR", quantize_duration_qn(d_qn))

def encode_pitch(vocab, inst_idx: int, midi_pitch: int) -> Optional[int]:
    space_name = vocab["pitch_space_for_inst"][str(inst_idx)]
    pmaps = vocab["pitch_maps"]["general"] if space_name == "PITCH_GENERAL" else vocab["pitch_maps"]["drums"]
    # JSON round-trips keys as strings, so try both int and str lookups
    local = pmaps.get(int(midi_pitch)) or pmaps.get(str(midi_pitch))
    if local is None:
        return None
    return tok_of(vocab, space_name, local)

def decode_to_midi(seq: List[int], vocab: dict, out_path: str, tempo_bpm=120.0):
    layout = vocab["layout"]
    inv_token = {}
    for g, spec in layout.items():
        s, n = spec["start"], spec["size"]
        for j in range(n):
            inv_token[s + j] = (g, j)

    inv_pitch = {}
    for short_name, mp in vocab["pitch_maps"].items():
        inv = {int(v): int(k) for k, v in mp.items()}
        inv_pitch[short_name] = inv
        inv_pitch["PITCH_" + short_name.upper()] = inv

    pitch_space_for_inst = vocab["pitch_space_for_inst"]
    DRUM_INSTS = {int(i_str) for i_str, sp in pitch_space_for_inst.items() if "DRUM" in str(sp).upper()}

    def is_drum_inst(i: int) -> bool:
        return i in DRUM_INSTS

    inst_names = vocab.get("instrument_names") or [f"inst_{i}" for i in range(layout["INST"]["size"])]

    pm = pretty_midi.PrettyMIDI(resolution=960)
    tracks = []
    for i, nm in enumerate(inst_names):
        inst = pretty_midi.Instrument(program=0, name=nm)
        inst.is_drum = is_drum_inst(i)
        tracks.append(inst)

    cur_time_qn = 0.0
    cur_inst = 0
    cur_vel  = 64
    cur_dur  = 0.25

    def _qn_to_sec(qn: float, tempo: float) -> float:
        return qn * 60.0 / tempo

    for t in seq:
        pair = inv_token.get(t)
        if pair is None:
            continue
        group, local = pair

        if group in ("BOS", "EOS"):
            continue
        elif group == "TIME_SHIFT":
            steps = local + 1
            cur_time_qn += steps * float(vocab["time_shift_qn_step"])
        elif group == "BAR":
            pass
        elif group == "INST":
            cur_inst = int(local)
        elif group == "VEL":
            vbin = vocab["velocity_bins"][local] if "velocity_bins" in vocab else (local * 16)
            cur_vel = int(max(1, min(127, vbin)))
        elif group == "DUR":
            cur_dur = float(vocab["duration_bins_qn"][local])
        elif group.startswith("PITCH"):
            space = vocab["pitch_space_for_inst"].get(str(cur_inst), "PITCH_GENERAL")
            inv_map = inv_pitch.get(space) or (inv_pitch.get("drums") if "DRUM" in space else inv_pitch.get("general"))
            if not inv_map:
                continue
            midi = inv_map.get(int(local))
            if midi is None:
                continue

            start = _qn_to_sec(cur_time_qn, tempo_bpm)
            end   = _qn_to_sec(cur_time_qn + cur_dur, tempo_bpm)
            note  = pretty_midi.Note(velocity=int(cur_vel), pitch=int(midi), start=start, end=end)
            if 0 <= cur_inst < len(tracks):
                tracks[cur_inst].notes.append(note)

    pm.instruments.extend(tracks)
    pm.write(out_path)

# -------------- TOKENIZER --------------
def tokenize_song(ev, tempo_bpm, bar_starts, bars_meta, vocab):
    tokens = [vocab["layout"]["BOS"]["start"]]
    last_time_qn = 0.0
    last_bar_key = None

    for (start_s, inst, midi, vel, dur_qn) in ev:
        cur_time_qn = qn_between(0.0, start_s, tempo_bpm)
        delta_qn = max(0.0, cur_time_qn - last_time_qn)
        tokens += encode_time_shift_qn(vocab, delta_qn)
        last_time_qn = cur_time_qn

        pos, steps = time_to_barpos(start_s, bar_starts, bars_meta)
        btok = encode_bar_pair(vocab, steps, pos)
        key = (steps, pos)
        if btok is not None and key != last_bar_key:
            tokens.append(btok)
            last_bar_key = key

        tokens.append(encode_inst(vocab, inst))
        tokens.append(encode_vel(vocab, vel))
        ptok = encode_pitch(vocab, inst, midi)
        if ptok is None:
            continue
        tokens.append(ptok)
        tokens.append(encode_dur(vocab, dur_qn))

    tokens.append(vocab["layout"]["EOS"]["start"])
    return tokens

# -------------- AUX: WINDOW TIME BOUNDS FROM TOKENS --------------
def token_time_qn_prefix(tokens: List[int], vocab: dict) -> np.ndarray:
    """
    prefix[i] = time_qn before consuming tokens[i], for i in [0..L]
    prefix[L] = time_qn after consuming all tokens
    """
    layout = vocab["layout"]
    ts_start = layout["TIME_SHIFT"]["start"]
    ts_end   = ts_start + layout["TIME_SHIFT"]["size"]
    step_qn  = float(vocab["time_shift_qn_step"])

    L = len(tokens)
    prefix = np.zeros(L + 1, dtype=np.float32)
    t = 0.0
    for i, tok in enumerate(tokens):
        prefix[i] = t
        if ts_start <= tok < ts_end:
            local = tok - ts_start
            t += (local + 1) * step_qn
    prefix[L] = t
    return prefix

def window_slices_with_time(tokens: List[int], vocab: dict, max_len=SEQ_LEN, stride=SEQ_STRIDE):
    """
    Yields (window_tokens, t0_qn, t1_qn).
    """
    pref = token_time_qn_prefix(tokens, vocab)
    L = len(tokens)

    if L <= max_len:
        yield tokens, float(pref[0]), float(pref[L])
        return

    start = 0
    while start < L:
        end = min(L, start + max_len)
        window = tokens[start:end]
        if len(window) < 4:
            break
        t0 = float(pref[start])
        t1 = float(pref[end])
        yield window, t0, t1
        if end >= L:
            break
        start += stride

# -------------- AUX: NOTE INTERVALS + FEATURES --------------
def events_to_intervals_qn(ev, tempo_bpm: float):
    """
    From ev list (start_sec, inst, pitch, vel, dur_qn) produce intervals in QN:
      (start_qn, end_qn, inst, pitch)
    """
    out = []
    for (start_s, inst, midi, vel, dur_qn) in ev:
        start_qn = qn_between(0.0, start_s, tempo_bpm)
        end_qn   = start_qn + float(dur_qn)
        out.append((float(start_qn), float(end_qn), int(inst), int(midi)))
    out.sort(key=lambda x: x[0])
    return out

def compute_aux_for_window(intervals,
                           t0_qn: float,
                           t1_qn: float,
                           config: Optional[InstrumentConfig] = None,
                           # Legacy keyword args for backward compat
                           num_instruments: Optional[int] = None,
                           drum_idx: Optional[int] = None,
                           guitar_idx: Optional[int] = None,
                           other_idx: Optional[int] = None,
                           onset_bin_qn: float = 1.0/24.0) -> np.ndarray:
    """
    Returns aux vector float32.  Shape depends on config:
      blues6  → (36,)
      chorale4 → (24,)
    """
    if config is None:
        config = _DEFAULT_CONFIG
    # Legacy callers may pass num_instruments etc. — use config instead
    n_inst = config.num_instruments
    _drum_idx = config.drum_idx
    _guitar_idx = config.guitar_idx
    _other_idx = config.other_idx
    aux_info = compute_aux_layout(config)

    if t1_qn <= t0_qn:
        t1_qn = t0_qn + 1e-3
    win_dur = t1_qn - t0_qn

    segs: List[List[Tuple[float, float]]] = [[] for _ in range(n_inst)]
    pitches_in_win: List[int] = []
    onsets_qn: List[float] = []

    for (s, e, inst, pitch) in intervals:
        if e <= t0_qn:
            continue
        if s >= t1_qn:
            break
        ss = max(s, t0_qn)
        ee = min(e, t1_qn)
        if ee > ss:
            if inst < n_inst:
                segs[inst].append((ss, ee))
            if not is_drum_slot(inst, config):
                pitches_in_win.append(pitch)
                if s >= t0_qn:
                    onsets_qn.append(s)

    def union_len(segments: List[Tuple[float,float]]) -> float:
        if not segments:
            return 0.0
        segments = sorted(segments)
        total = 0.0
        cs, ce = segments[0]
        for s, e in segments[1:]:
            if s <= ce:
                ce = max(ce, e)
            else:
                total += (ce - cs)
                cs, ce = s, e
        total += (ce - cs)
        return total

    def poly_stats(segments: List[Tuple[float,float]]) -> Tuple[float, float]:
        if not segments:
            return 0.0, 0.0
        events = []
        for s, e in segments:
            events.append((s, +1))
            events.append((e, -1))
        events.sort()
        cur = 0
        last_t = events[0][0]
        area = 0.0
        mx = 0
        for t, d in events:
            dt = t - last_t
            if dt > 0:
                area += cur * dt
            cur += d
            mx = max(mx, cur)
            last_t = t
        mean = area / win_dur
        return float(mx), float(mean)

    max_poly = np.zeros(n_inst, dtype=np.float32)
    mean_poly = np.zeros(n_inst, dtype=np.float32)
    overlap = np.zeros(n_inst, dtype=np.float32)

    for i in range(n_inst):
        segments = segs[i]
        overlap[i] = float(union_len(segments) / win_dur) if win_dur > 0 else 0.0
        mx, mn = poly_stats(segments)
        max_poly[i] = mx
        mean_poly[i] = mn

    parts: List[np.ndarray] = [max_poly, mean_poly, overlap]

    # Chord stats (only for configs with guitar + other)
    if aux_info["has_chords"]:
        assert _guitar_idx is not None and _other_idx is not None

        def chord_stats_for_inst(inst_idx: int) -> Tuple[float, float]:
            ons = []
            for (s, e, inst, pitch) in intervals:
                if inst != inst_idx:
                    continue
                if s < t0_qn or s >= t1_qn:
                    continue
                b = int(round((s - t0_qn) / onset_bin_qn))
                ons.append(b)
            if not ons:
                return 0.0, 0.0
            c = Counter(ons)
            sizes = np.array(list(c.values()), dtype=np.float32)
            return float(sizes.mean()), float(sizes.max())

        mean_ch_g, max_ch_g = chord_stats_for_inst(_guitar_idx)
        mean_ch_o, max_ch_o = chord_stats_for_inst(_other_idx)
        parts.append(np.array([mean_ch_g, max_ch_g, mean_ch_o, max_ch_o], dtype=np.float32))

    # Pitch class histogram (always)
    pc = np.zeros(12, dtype=np.float32)
    for p in pitches_in_win:
        pc[int(p) % 12] += 1.0
    s_pc = float(pc.sum())
    pc_norm = (pc / s_pc) if s_pc > 0 else pc
    parts.append(pc_norm)

    # Swing & blues (only for configs with drums)
    if aux_info["has_swing_blues"]:
        swing_score = calculate_swing_score(onsets_qn)
        blues_score = calculate_blues_scale_score(pitches_in_win)
        parts.append(np.array([swing_score, blues_score], dtype=np.float32))

    aux = np.concatenate(parts, axis=0).astype(np.float32)
    assert aux.shape[0] == aux_info["aux_dim"], \
        f"aux shape mismatch: got {aux.shape[0]}, expected {aux_info['aux_dim']}"
    return aux

# -------------- WINDOWING (legacy, not used now) --------------
def window_sequences(tokens: List[int], max_len=SEQ_LEN, stride=SEQ_STRIDE) -> List[List[int]]:
    out = []
    if len(tokens) <= max_len:
        out.append(tokens)
        return out
    start = 0
    while start < len(tokens):
        window = tokens[start:start+max_len]
        if len(window) < 4:
            break
        out.append(window)
        if start + max_len >= len(tokens):
            break
        start += stride
    return out

# -------------- VOCAB COMPACTION --------------
def compact_vocab(
    train_seqs: List[List[int]],
    val_seqs: List[List[int]],
    vocab: dict,
) -> dict:
    """Remove unused token values from vocab and remap all sequences in-place.

    After the first tokenization pass, many (group, local_idx) slots are allocated
    but never appear in any sequence.  This function:
      1. Scans every sequence to find which global tokens are actually used.
      2. Builds a per-group old_local → new_local remapping (contiguous from 0).
      3. Rebuilds vocab metadata (layout, bar_pairs, bins, pitch_maps).
      4. Remaps every token in every sequence in-place.
    """
    old_layout = vocab["layout"]

    # -- Step 1: collect used global token IDs --
    used_globals: Set[int] = set()
    for seq in itertools.chain(train_seqs, val_seqs):
        used_globals.update(seq)

    # -- Step 2: per-group used locals & remapping --
    # Data-driven: only include groups that exist in old_layout
    GROUP_ORDER = [g for g in ["PAD", "BOS", "EOS", "BAR", "TIME_SHIFT", "INST",
                               "VEL", "DUR", "PITCH_GENERAL", "PITCH_DRUMS"]
                   if g in old_layout]
    KEEP_GROUPS = {"PAD", "BOS", "EOS", "INST"}
    COMPACT_GROUPS = [g for g in GROUP_ORDER if g not in KEEP_GROUPS]

    old_to_new_local: Dict[str, Dict[int, int]] = {}
    new_sizes: Dict[str, int] = {}

    for group in COMPACT_GROUPS:
        start, size = old_layout[group]["start"], old_layout[group]["size"]
        used_locals = sorted(
            local for local in range(size)
            if (start + local) in used_globals
        )
        old_to_new_local[group] = {old: new for new, old in enumerate(used_locals)}
        new_sizes[group] = len(used_locals)

    for group in KEEP_GROUPS:
        new_sizes[group] = old_layout[group]["size"]
        old_to_new_local[group] = {i: i for i in range(old_layout[group]["size"])}

    # -- Step 3: rebuild layout with sequential offsets --
    new_layout: Dict[str, dict] = {}
    idx = 0
    for group in GROUP_ORDER:
        new_layout[group] = {"start": idx, "size": new_sizes[group]}
        idx += new_sizes[group]

    # -- Step 4: build global remap (old global → new global) --
    global_remap: Dict[int, int] = {}
    for group in GROUP_ORDER:
        old_start = old_layout[group]["start"]
        new_start = new_layout[group]["start"]
        for old_local, new_local in old_to_new_local[group].items():
            global_remap[old_start + old_local] = new_start + new_local

    # -- Step 5: remap all sequences in-place --
    for seq in itertools.chain(train_seqs, val_seqs):
        for i, tok in enumerate(seq):
            seq[i] = global_remap[tok]

    # -- Step 6: rebuild vocab metadata --

    # BAR: keep only pairs whose local index was used
    old_bar_pairs = vocab["bar_pairs"]
    bar_remap = old_to_new_local["BAR"]
    new_bar_pairs: List[Tuple[int, int]] = []
    new_bar_pair_to_local: Dict[str, int] = {}
    for old_local in sorted(bar_remap.keys()):
        pair = old_bar_pairs[old_local]
        new_bar_pairs.append(pair)
        s, p = pair
        new_bar_pair_to_local[f"{s}:{p}"] = bar_remap[old_local]

    # VEL: keep only bins at used indices
    vel_remap = old_to_new_local["VEL"]
    new_velocity_bins = [vocab["velocity_bins"][old] for old in sorted(vel_remap.keys())]

    # DUR: keep only bins at used indices
    dur_remap = old_to_new_local["DUR"]
    new_duration_bins_qn = [vocab["duration_bins_qn"][old] for old in sorted(dur_remap.keys())]

    # PITCH: remap local indices in pitch maps
    gen_remap = old_to_new_local["PITCH_GENERAL"]
    new_pitch_maps: Dict[str, Dict] = {
        "general": {midi_pitch: gen_remap[local]
                     for midi_pitch, local in vocab["pitch_maps"]["general"].items()
                     if local in gen_remap},
    }
    if "PITCH_DRUMS" in old_to_new_local and "drums" in vocab["pitch_maps"]:
        drums_remap = old_to_new_local["PITCH_DRUMS"]
        new_pitch_maps["drums"] = {
            midi_pitch: drums_remap[local]
            for midi_pitch, local in vocab["pitch_maps"]["drums"].items()
            if local in drums_remap
        }

    # -- Step 7: report savings --
    print("\n── Vocab Compaction ─────────────────────────")
    total_removed = 0
    for group in GROUP_ORDER:
        old_sz = old_layout[group]["size"]
        new_sz = new_sizes[group]
        removed = old_sz - new_sz
        total_removed += removed
        if removed > 0:
            print(f"  {group:>15}: {old_sz:4d} → {new_sz:4d}  (removed {removed})")
    old_total = max(v["start"] + v["size"] for v in old_layout.values())
    new_total = max(v["start"] + v["size"] for v in new_layout.values())
    print(f"  {'TOTAL':>15}: {old_total:4d} → {new_total:4d}  (removed {total_removed})")
    print("─────────────────────────────────────────────")

    # -- Update vocab dict --
    vocab["layout"] = new_layout
    vocab["bar_pairs"] = new_bar_pairs
    vocab["bar_pair_to_local"] = new_bar_pair_to_local
    vocab["velocity_bins"] = new_velocity_bins
    vocab["duration_bins_qn"] = new_duration_bins_qn
    vocab["pitch_maps"] = new_pitch_maps

    return vocab


def calculate_swing_score(onsets_qn: List[float]) -> float:
    """Measure fraction of off-beat onsets that are closer to triplet (2/3) than straight (1/2)."""
    off_beats = [o % 1.0 for o in onsets_qn if 0.25 <= (o % 1.0) <= 0.75]
    if not off_beats:
        return 0.0
    # 0.5 is straight 8th, 0.666 is triplet 8th
    triplet_hits = sum(1 for p in off_beats if abs(p - 0.666) < abs(p - 0.5))
    return float(triplet_hits) / len(off_beats)

def calculate_blues_scale_score(pitches: List[int]) -> float:
    """Find most likely root from pitch-class histogram, then check adherence to blues scale."""
    if not pitches:
        return 0.0
    pc = Counter(p % 12 for p in pitches)
    # Most common pitch class as a candidate for the root (I)
    root = pc.most_common(1)[0][0]
    blues_intervals = {0, 3, 5, 6, 7, 10} # root, b3, 4, b5, 5, b7
    blues_pcs = {(root + i) % 12 for i in blues_intervals}
    on_scale = sum(pc[p] for p in blues_pcs)
    return float(on_scale) / len(pitches)

def is_track_bluesy(ev, tempo_bpm: float, min_scale_score: float = 0.60, min_swing_score: float = 0.50,
                    config: Optional[InstrumentConfig] = None) -> bool:
    """Quick check for bluesiness (swing or scale) to filter training data."""
    if config is None:
        config = _DEFAULT_CONFIG
    if not ev:
        return False

    pitches = [e[2] for e in ev if not is_drum_slot(e[1], config)]
    scale_score = calculate_blues_scale_score(pitches)

    onsets_qn = [qn_between(0.0, e[0], tempo_bpm) for e in ev]
    swing_score = calculate_swing_score(onsets_qn)

    return (scale_score >= min_scale_score) or (swing_score >= min_swing_score)

# -------------- MAIN PIPELINE --------------
def main():
    global MIDI_FOLDER, DATA_FOLDER, SAMPLES_DIR, AUG_ENABLE
    ap = argparse.ArgumentParser("preES4: preprocess multitrack MIDI into factored event-stream windows.")
    ap.add_argument("--midi_folder", default=MIDI_FOLDER, help="Folder containing per-song multi-track MIDI files.")
    ap.add_argument("--data_folder", default=DATA_FOLDER, help="Output folder for events_train/val + vocab.")
    ap.add_argument("--tracks", default="all", help="Comma-separated subset of tracks to keep (default: all). Examples: drums,bass,guitar  |  drums,bass  |  all. Aliases: voxbg/bgvox/backingvox/auxvox -> voxharm.")
    ap.add_argument("--no-aug", action="store_true", help="Disable train-time augmentation (transpose/velocity).")
    ap.add_argument("--diagnose-drums", action="store_true", help="Run a quick scan for out-of-range drum pitches and exit.")
    ap.add_argument("--blues_only", action="store_true", help="Filter out songs that don't meet minimum blues criteria (blues scale adherence or swing).")
    ap.add_argument("--instrument_set", default="blues6", choices=list(INSTRUMENT_PRESETS.keys()),
                    help="Preset instrument configuration (default: blues6).")
    ap.add_argument("--instruments", default="", help="Arbitrary comma-separated instrument names (overrides --instrument_set). E.g.: soprano,alto,tenor,bassvox")
    args = ap.parse_args()

    if args.diagnose_drums:
        diagnose_drum_midi_anomalies(args.midi_folder, strict_range_only=True)
        return

    # Resolve instrument config
    if args.instruments:
        inst_names = [n.strip() for n in args.instruments.split(",") if n.strip()]
        config = make_instrument_config(inst_names)
    else:
        config = make_instrument_config(INSTRUMENT_PRESETS[args.instrument_set])
    print(f"Instrument config: {config.names} (drums={config.drum_idx})")

    # Apply CLI overrides
    MIDI_FOLDER = args.midi_folder
    DATA_FOLDER = args.data_folder
    AUG_ENABLE = False if args.no_aug else True

    os.makedirs(DATA_FOLDER, exist_ok=True)
    SAMPLES_DIR = os.path.join(DATA_FOLDER, "_samples")
    os.makedirs(SAMPLES_DIR, exist_ok=True)

    selected_tracks = parse_tracks_arg(args.tracks, config)
    allowed_inst_idx = {config.names.index(t) for t in selected_tracks}
    print(f"Keeping tracks: {selected_tracks} (indices={sorted(allowed_inst_idx)})")

    # 1) Collect & split
    paths = sorted(glob.glob(os.path.join(MIDI_FOLDER, "*.mid"))) + \
            sorted(glob.glob(os.path.join(MIDI_FOLDER, "*.midi")))
    if not paths:
        raise RuntimeError(f"No MIDI files found in '{MIDI_FOLDER}'.")
    if len(paths) < 2:
        raise RuntimeError(f"Need at least 2 MIDI files for train/val split, got {len(paths)} in '{MIDI_FOLDER}'.")

    random.seed(42)
    random.shuffle(paths)

    n_train = int(0.8 * len(paths))
    train_paths, val_paths = paths[:n_train], paths[n_train:]

    # 2) Gather base events + bar metadata
    song_meta: Dict[str, Tuple[list, float, np.ndarray, list]] = {}
    all_bars_meta = []
    skipped_not_bluesy = 0
    for p in paths:
        try:
            ev, tempo, bar_starts, bars_meta = extract_multitrack_events(p, config)
            # Filter to selected tracks (keep canonical index space)
            ev = [e for e in ev if e[1] in allowed_inst_idx]
            if not ev:
                raise RuntimeError('No events remain after track filtering.')

            # Optional: Blues Filter
            if args.blues_only and not is_track_bluesy(ev, tempo, config=config):
                skipped_not_bluesy += 1
                continue

        except Exception as e:
            print(f"Skipping {os.path.basename(p)}: {e}")
            continue
        song_meta[p] = (ev, tempo, bar_starts, bars_meta)
        all_bars_meta.append(bars_meta)

    if skipped_not_bluesy > 0:
        print(f"Filtered out {skipped_not_bluesy} songs for not meeting blues criteria (--blues_only).")

    if not any(song_meta[p][0] for p in song_meta):
        raise RuntimeError("No events extracted. Check your MIDI folder & mapping heuristics.")

    # Determine augmentation ranges from config
    aug_transposes = config.aug_transposes
    aug_vel_deltas = config.aug_vel_deltas if config.aug_vel_deltas else [0]

    # 3) Build vocab from: VAL (base) + TRAIN (base + augmented)
    events_for_vocab = []

    # add val base
    for p in val_paths:
        if p in song_meta:
            events_for_vocab.extend(song_meta[p][0])

    # add train base + aug
    for p in train_paths:
        if p not in song_meta:
            continue
        ev, _, _, _ = song_meta[p]
        events_for_vocab.extend(ev)
        if AUG_ENABLE:
            for s in aug_transposes:
                for dv in aug_vel_deltas:
                    ev_aug = augment_events_additive(ev, semitone_shift=s, vel_delta=dv, config=config)
                    if ev_aug is not None:
                        events_for_vocab.extend(ev_aug)

    # 4) Build vocab
    bar_pairs  = gather_bar_pairs(all_bars_meta)
    pitch_maps = build_pitch_maps(events_for_vocab, config)
    vocab      = build_event_vocab(pitch_maps, bar_pairs, config)
    vocab["tracks"] = {
        "selected": selected_tracks,
        "allowed_inst_indices": sorted(list(allowed_inst_idx)),
        "canonical_instrument_names": config.names,
    }

    # 5) Tokenize & window (train = base + aug; val = base only), and compute aux per window.
    def tokenize_group_with_aux(group_paths: List[str], do_aug: bool, split_name: str = "train"):
        seqs: List[List[int]] = []
        auxs: List[np.ndarray] = []
        for i, p in enumerate(group_paths):
            if i % 10 == 0:
                print(f"Tokenizing ({split_name}): {i}/{len(group_paths)} {os.path.basename(p)}")
            if p not in song_meta:
                continue
            ev, tempo, bar_starts, bars_meta = song_meta[p]

            def add_one(ev_local):
                toks = tokenize_song(ev_local, tempo, bar_starts, bars_meta, vocab)
                intervals = events_to_intervals_qn(ev_local, tempo)
                for window, t0, t1 in window_slices_with_time(toks, vocab, SEQ_LEN, SEQ_STRIDE):
                    seqs.append(window)
                    auxs.append(compute_aux_for_window(
                        intervals, t0, t1, config=config,
                    ))

            # base
            add_one(ev)

            # augmented (train only)
            if do_aug and AUG_ENABLE:
                for s in aug_transposes:
                    for dv in aug_vel_deltas:
                        ev_aug = augment_events_additive(ev, semitone_shift=s, vel_delta=dv, config=config)
                        if ev_aug is not None:
                            add_one(ev_aug)

        return seqs, auxs

    train_seqs, train_aux = tokenize_group_with_aux(train_paths, do_aug=True, split_name="train")
    val_seqs,   val_aux   = tokenize_group_with_aux(val_paths, do_aug=False, split_name="val")

    # 5b) Compact vocab — remove dead tokens, remap sequences in-place
    vocab = compact_vocab(train_seqs, val_seqs, vocab)

    # 6) Save
    with open(os.path.join(DATA_FOLDER, "events_train.pkl"), "wb") as f:
        pickle.dump({"sequences": train_seqs, "aux": train_aux}, f)
    with open(os.path.join(DATA_FOLDER, "events_val.pkl"), "wb") as f:
        pickle.dump({"sequences": val_seqs, "aux": val_aux}, f)
    with open(os.path.join(DATA_FOLDER, "event_vocab.json"), "w") as f:
        json.dump(vocab, f, indent=2)

    # 7) Round-trip a few samples to MIDI (for sanity)
    for i in range(min(3, len(train_seqs))):
        outp = os.path.join(SAMPLES_DIR, f"tokenized_{i:03d}.mid")
        decode_to_midi(train_seqs[i], vocab, outp, tempo_bpm=120.0)

    # 8) Report
    layout = vocab["layout"]
    total_vocab = max(v["start"] + v["size"] for v in layout.values())
    meters = sorted({s for (s, _) in vocab["bar_pairs"]})
    sizes: Dict[str, object] = {
        "TOTAL_VOCAB": total_vocab,
        "BAR_pairs": layout["BAR"]["size"],
        "TIME_SHIFT": layout["TIME_SHIFT"]["size"],
        "INST": layout["INST"]["size"],
        "VEL": layout["VEL"]["size"],
        "DUR": layout["DUR"]["size"],
        "PITCH_GENERAL": layout["PITCH_GENERAL"]["size"],
    }
    if "PITCH_DRUMS" in layout:
        sizes["PITCH_DRUMS"] = layout["PITCH_DRUMS"]["size"]
    sizes.update({
        "meters_steps_per_bar": meters,
        "time_shift_qn_step": vocab["time_shift_qn_step"],
        "instruments": config.names,
        "selected_tracks": selected_tracks,
        "drum_index": config.drum_idx,
        "aux_dim": vocab["aux"]["aux_dim"] if "aux" in vocab else 0,
        "train_windows": len(train_seqs),
        "val_windows": len(val_seqs),
    })

    print("\n── Vocab Summary ───────────────────────────")
    for k, v in sizes.items():
        print(f"{k:>22}: {v}")
    print("────────────────────────────────────────────")
    print(f"Wrote {os.path.join(DATA_FOLDER, 'events_train.pkl')}  (seqs={len(train_seqs)}, aux={len(train_aux)})")
    print(f"Wrote {os.path.join(DATA_FOLDER, 'events_val.pkl')}    (seqs={len(val_seqs)}, aux={len(val_aux)})")
    print(f"Wrote {os.path.join(DATA_FOLDER, 'event_vocab.json')}")
    print(f"Round-trip MIDIs in {SAMPLES_DIR}")

    # quick aux sanity
    n_inst = config.num_instruments
    if train_aux:
        a0 = train_aux[0]
        print("\n── Aux Sanity ──────────────────────────────")
        print(f"aux[0] shape: {np.asarray(a0).shape}  dtype={np.asarray(a0).dtype}")
        max_poly = np.asarray(a0[:n_inst])
        overlap = np.asarray(a0[2*n_inst:3*n_inst])
        print(f"max_polyphony[0]: {max_poly}")
        print(f"overlap_ratio[0]: {overlap}")
        print("────────────────────────────────────────────")

    with open(os.path.join(DATA_FOLDER, "vocab_summary.txt"), "w") as f:
        f.write("Event Vocab Summary\n")
        for k, v in sizes.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    main()