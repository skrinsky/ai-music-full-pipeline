#!/usr/bin/env python3
"""Audition tool for GigaMIDI files — uses the *same* mapping as training/pre.py.

Subcommands:
  stats   Aggregate instrumentation table across all files
  list    One line per file: filename, duration, note count, instrument breakdown
  info    Per-track detail for a single file
  play    Print info then open file for macOS playback
"""

import argparse
import glob
import os
import subprocess
import sys
from collections import Counter
from typing import List

import pretty_midi

# Import the canonical mapping from training code
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from training.pre import (
    INSTRUMENT_PRESETS,
    InstrumentConfig,
    make_instrument_config,
    map_name_to_slot,
)

# GM program name table (General MIDI Level 1)
_GM_NAMES = [
    "Acoustic Grand Piano", "Bright Acoustic Piano", "Electric Grand Piano",
    "Honky-tonk Piano", "Electric Piano 1", "Electric Piano 2", "Harpsichord",
    "Clavinet", "Celesta", "Glockenspiel", "Music Box", "Vibraphone",
    "Marimba", "Xylophone", "Tubular Bells", "Dulcimer", "Drawbar Organ",
    "Percussive Organ", "Rock Organ", "Church Organ", "Reed Organ",
    "Accordion", "Harmonica", "Tango Accordion", "Acoustic Guitar (nylon)",
    "Acoustic Guitar (steel)", "Electric Guitar (jazz)", "Electric Guitar (clean)",
    "Electric Guitar (muted)", "Overdriven Guitar", "Distortion Guitar",
    "Guitar Harmonics", "Acoustic Bass", "Electric Bass (finger)",
    "Electric Bass (pick)", "Fretless Bass", "Slap Bass 1", "Slap Bass 2",
    "Synth Bass 1", "Synth Bass 2", "Violin", "Viola", "Cello", "Contrabass",
    "Tremolo Strings", "Pizzicato Strings", "Orchestral Harp", "Timpani",
    "String Ensemble 1", "String Ensemble 2", "Synth Strings 1",
    "Synth Strings 2", "Choir Aahs", "Voice Oohs", "Synth Voice",
    "Orchestra Hit", "Trumpet", "Trombone", "Tuba", "Muted Trumpet",
    "French Horn", "Brass Section", "Synth Brass 1", "Synth Brass 2",
    "Soprano Sax", "Alto Sax", "Tenor Sax", "Baritone Sax", "Oboe",
    "English Horn", "Bassoon", "Clarinet", "Piccolo", "Flute", "Recorder",
    "Pan Flute", "Blown Bottle", "Shakuhachi", "Whistle", "Ocarina",
    "Lead 1 (square)", "Lead 2 (sawtooth)", "Lead 3 (calliope)",
    "Lead 4 (chiff)", "Lead 5 (charang)", "Lead 6 (voice)",
    "Lead 7 (fifths)", "Lead 8 (bass + lead)", "Pad 1 (new age)",
    "Pad 2 (warm)", "Pad 3 (polysynth)", "Pad 4 (choir)", "Pad 5 (bowed)",
    "Pad 6 (metallic)", "Pad 7 (halo)", "Pad 8 (sweep)", "FX 1 (rain)",
    "FX 2 (soundtrack)", "FX 3 (crystal)", "FX 4 (atmosphere)",
    "FX 5 (brightness)", "FX 6 (goblins)", "FX 7 (echoes)",
    "FX 8 (sci-fi)", "Sitar", "Banjo", "Shamisen", "Koto", "Kalimba",
    "Bagpipe", "Fiddle", "Shanai", "Tinkle Bell", "Agogo", "Steel Drums",
    "Woodblock", "Taiko Drum", "Melodic Tom", "Synth Drum", "Reverse Cymbal",
    "Guitar Fret Noise", "Breath Noise", "Seashore", "Bird Tweet",
    "Telephone Ring", "Helicopter", "Applause", "Gunshot",
]


def gm_name(prog: int) -> str:
    if 0 <= prog < len(_GM_NAMES):
        return _GM_NAMES[prog]
    return f"prog_{prog}"


def _midi_paths(folder: str) -> List[str]:
    paths = sorted(
        glob.glob(os.path.join(folder, "*.mid"))
        + glob.glob(os.path.join(folder, "*.midi"))
    )
    if not paths:
        print(f"ERROR: no MIDI files found in '{folder}'", file=sys.stderr)
        sys.exit(1)
    return paths


def _get_config(args: argparse.Namespace) -> InstrumentConfig:
    """Build InstrumentConfig from --instrument_set flag."""
    preset = getattr(args, "instrument_set", "blues6")
    return make_instrument_config(INSTRUMENT_PRESETS[preset])


def cmd_stats(args: argparse.Namespace) -> None:
    """Aggregate instrumentation table across all files."""
    config = _get_config(args)
    paths = _midi_paths(args.folder)
    slot_notes: Counter = Counter()
    slot_files: Counter = Counter()
    total_files = 0
    skipped = 0

    for p in paths:
        try:
            pm = pretty_midi.PrettyMIDI(p)
        except Exception as e:
            print(f"  [skip] {os.path.basename(p)}: {e}", file=sys.stderr)
            skipped += 1
            continue
        total_files += 1
        file_slots: set = set()
        for inst in pm.instruments:
            slot = map_name_to_slot(inst, config)
            n = len(inst.notes)
            slot_notes[slot] += n
            if n > 0:
                file_slots.add(slot)
        for s in file_slots:
            slot_files[s] += 1

    total_notes = sum(slot_notes.values())
    print(f"\n{'Slot':<12} {'Name':<10} {'Notes':>10} {'%Notes':>8} {'Files':>8} {'%Files':>8}")
    print("-" * 60)
    for i, name in enumerate(config.names):
        n = slot_notes.get(i, 0)
        f = slot_files.get(i, 0)
        pn = 100.0 * n / total_notes if total_notes else 0
        pf = 100.0 * f / total_files if total_files else 0
        print(f"{i:<12} {name:<10} {n:>10,} {pn:>7.1f}% {f:>8,} {pf:>7.1f}%")
    print("-" * 60)
    print(f"Total: {total_files} files, {total_notes:,} notes (skipped {skipped})\n")


def cmd_list(args: argparse.Namespace) -> None:
    """One line per file."""
    config = _get_config(args)
    paths = _midi_paths(args.folder)
    for p in paths:
        try:
            pm = pretty_midi.PrettyMIDI(p)
        except Exception as e:
            print(f"[skip] {os.path.basename(p)}: {e}")
            continue
        dur = pm.get_end_time()
        notes_per_slot: Counter = Counter()
        for inst in pm.instruments:
            slot = map_name_to_slot(inst, config)
            notes_per_slot[slot] += len(inst.notes)
        total = sum(notes_per_slot.values())
        breakdown = "  ".join(
            f"{config.names[i]}={notes_per_slot.get(i, 0)}"
            for i in range(len(config.names))
            if notes_per_slot.get(i, 0) > 0
        )
        print(f"{os.path.basename(p):<40} {dur:6.1f}s  {total:>6} notes  {breakdown}")


def cmd_info(args: argparse.Namespace) -> None:
    """Per-track detail for a single file."""
    config = _get_config(args)
    path = args.file
    if not os.path.isfile(path):
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        sys.exit(1)
    pm = pretty_midi.PrettyMIDI(path)
    print(f"\nFile: {path}")
    print(f"Duration: {pm.get_end_time():.1f}s")
    tc = pm.get_tempo_changes()
    if tc[1].size > 0:
        print(f"Tempo: {tc[1][0]:.1f} BPM")
    print()
    fmt = "{:<4} {:<25} {:<6} {:<5} {:<30} {:<10} {:>6} {:>10}"
    print(fmt.format("#", "Raw Name", "Drum?", "Prog", "GM Name", "Slot", "Notes", "Pitch"))
    print("-" * 100)
    for i, inst in enumerate(pm.instruments):
        slot = map_name_to_slot(inst, config)
        n = len(inst.notes)
        if n > 0:
            pitches = [note.pitch for note in inst.notes]
            prange = f"{min(pitches)}-{max(pitches)}"
        else:
            prange = "-"
        print(fmt.format(
            i,
            (inst.name or "")[:25],
            "Y" if inst.is_drum else "N",
            inst.program,
            gm_name(inst.program)[:30],
            config.names[slot],
            n,
            prange,
        ))
    print()


def cmd_play(args: argparse.Namespace) -> None:
    """Print info then open for macOS playback."""
    cmd_info(args)
    print(f"Opening {args.file} ...")
    subprocess.run(["open", args.file], check=False)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Audition GigaMIDI files using the same mapping as training/pre.py"
    )
    sub = ap.add_subparsers(dest="cmd")
    sub.required = True

    _iset_choices = list(INSTRUMENT_PRESETS.keys())

    p_stats = sub.add_parser("stats", help="Aggregate instrumentation table")
    p_stats.add_argument("--folder", default="data/blues_midi", help="MIDI folder")
    p_stats.add_argument("--instrument_set", default="blues6", choices=_iset_choices, help="Instrument set")
    p_stats.set_defaults(func=cmd_stats)

    p_list = sub.add_parser("list", help="One line per file")
    p_list.add_argument("--folder", default="data/blues_midi", help="MIDI folder")
    p_list.add_argument("--instrument_set", default="blues6", choices=_iset_choices, help="Instrument set")
    p_list.set_defaults(func=cmd_list)

    p_info = sub.add_parser("info", help="Per-track detail for one file")
    p_info.add_argument("file", help="Path to MIDI file")
    p_info.add_argument("--instrument_set", default="blues6", choices=_iset_choices, help="Instrument set")
    p_info.set_defaults(func=cmd_info)

    p_play = sub.add_parser("play", help="Info + open for macOS playback")
    p_play.add_argument("file", help="Path to MIDI file")
    p_play.add_argument("--instrument_set", default="blues6", choices=_iset_choices, help="Instrument set")
    p_play.set_defaults(func=cmd_play)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
