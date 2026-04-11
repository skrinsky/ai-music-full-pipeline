#!/usr/bin/env python3
"""
Generate MIDI using a fine-tuned Notochord model.

Uses Notochord's built-in query() API for autoregressive generation.
Optionally primes the hidden state from one of your own tracks so the
model starts with context from your style.

Usage:
    python finetune/notochord_generate.py \\
        --checkpoint finetune/runs/noto_finetuned.pt \\
        --data_dir   finetune/runs/noto_data \\
        --out_midi   finetune/runs/generated/noto_out.mid

    # Prime from one of your tracks:
    python finetune/notochord_generate.py \\
        --checkpoint  finetune/runs/noto_finetuned.pt \\
        --data_dir    finetune/runs/noto_data \\
        --prompt_midi summer_midi/my_song.mid \\
        --out_midi    finetune/runs/generated/noto_continuation.mid
"""

import argparse
import json
from pathlib import Path

import pretty_midi
import torch
from notochord import Notochord


def events_to_midi(events: list[dict], out_path: Path,
                   max_note_dur: float = 2.0, max_polyphony: int = 0):
    """Convert a list of {instrument, pitch, time, velocity} dicts to a MIDI file."""
    pm = pretty_midi.PrettyMIDI()

    tracks: dict[int, pretty_midi.Instrument] = {}
    # (inst, pitch) → (note_on absolute time, velocity)
    pending: dict[tuple, tuple[float, int]] = {}
    # inst → ordered list of open pitches (oldest first)
    open_by_inst: dict[int, list[int]] = {}

    def close_note(inst_id, pitch, end_time):
        key = (inst_id, pitch)
        if key not in pending:
            return
        on_time, on_vel = pending.pop(key)
        dur = max(end_time - on_time, 0.01)
        tracks[inst_id].notes.append(pretty_midi.Note(
            velocity=on_vel, pitch=pitch,
            start=on_time, end=on_time + dur,
        ))
        if inst_id in open_by_inst and pitch in open_by_inst[inst_id]:
            open_by_inst[inst_id].remove(pitch)

    abs_time = 0.0
    for ev in events:
        abs_time += ev["time"]
        inst_id = int(ev["instrument"])
        pitch   = max(0, min(127, int(ev["pitch"])))
        vel     = max(0, min(127, int(ev["velocity"])))

        if inst_id not in tracks:
            is_drum = (inst_id == 128)
            program = max(0, min(127, 0 if is_drum else inst_id))
            tracks[inst_id] = pretty_midi.Instrument(
                program=program, is_drum=is_drum,
                name=f"inst_{inst_id}")

        key = (inst_id, pitch)
        if vel > 0:
            # Polyphony cap: close oldest note before opening a new one
            if max_polyphony > 0:
                open_now = open_by_inst.get(inst_id, [])
                while len(open_now) >= max_polyphony:
                    close_note(inst_id, open_now[0], abs_time)
                    open_now = open_by_inst.get(inst_id, [])
            if key in pending:
                close_note(inst_id, pitch, abs_time)
            pending[key] = (abs_time, max(1, vel))
            open_by_inst.setdefault(inst_id, [])
            if pitch not in open_by_inst[inst_id]:
                open_by_inst[inst_id].append(pitch)
        else:
            close_note(inst_id, pitch, abs_time)

    # Close open notes — cap duration so nothing hangs to end of piece
    for (inst_id, pitch), (on_time, on_vel) in list(pending.items()):
        end_time = min(on_time + max_note_dur, abs_time + 0.25)
        tracks[inst_id].notes.append(pretty_midi.Note(
            velocity=on_vel, pitch=pitch,
            start=on_time, end=end_time,
        ))

    for inst in tracks.values():
        pm.instruments.append(inst)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    pm.write(str(out_path))
    print(f"Saved MIDI → {out_path}  ({len(events)} events, {abs_time:.1f}s)")


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--checkpoint",      required=True)
    ap.add_argument("--base_checkpoint", default=None,
                    help="Original pre-trained checkpoint — needed when fine-tuned "
                         "ckpt was saved without architecture kwargs (kw={})")
    ap.add_argument("--data_dir",        default=None,
                    help="notochord_convert.py output dir — reads instrument list "
                         "from meta.json to keep generation in-distribution")
    ap.add_argument("--out_midi",        required=True)
    ap.add_argument("--prompt_midi",     default=None,
                    help="Prime the model's hidden state from this MIDI file")

    ap.add_argument("--n_events",        type=int,   default=1000,
                    help="Events to generate (~1000 ≈ 60-90s)")

    ap.add_argument("--pitch_temp",      type=float, default=0.9,
                    help="Pitch temperature")
    ap.add_argument("--rhythm_temp",     type=float, default=0.9,
                    help="Rhythm/timing temperature")
    ap.add_argument("--velocity_temp",   type=float, default=0.9,
                    help="Velocity temperature")
    ap.add_argument("--instrument_temp", type=float, default=0.9,
                    help="Instrument temperature")

    ap.add_argument("--max_time",        type=float, default=2.0,
                    help="Max seconds between events")
    ap.add_argument("--min_time",        type=float, default=0.03,
                    help="Min seconds between events")

    ap.add_argument("--max_note_dur",    type=float, default=2.0,
                    help="Cap note duration (seconds)")
    ap.add_argument("--max_polyphony",   type=int,   default=0,
                    help="Max simultaneous notes per instrument (0 = let model decide)")

    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = (
        "mps"  if torch.backends.mps.is_available()  else
        "cuda" if torch.cuda.is_available()           else
        "cpu"
    ) if args.device == "auto" else args.device
    print(f"Device: {device}")

    # Load model ---------------------------------------------------------------
    print(f"Loading: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if not ckpt.get("kw") and args.base_checkpoint:
        print(f"  kw empty — loading architecture from {args.base_checkpoint}")
        model = Notochord.from_checkpoint(args.base_checkpoint)
        model.load_state_dict(ckpt["model_state"])
    else:
        model = Notochord.from_checkpoint(args.checkpoint)
    model = model.to(device)
    model.eval()

    # Instrument filter from training data ------------------------------------
    include_inst = None
    if args.data_dir:
        meta_path = Path(args.data_dir) / "meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            include_inst = meta.get("instruments")
            if include_inst:
                print(f"Instruments ({len(include_inst)}): {include_inst}")
            else:
                print("meta.json missing 'instruments' — re-run noto-convert to add it")
        else:
            print(f"Warning: meta.json not found at {meta_path}")

    # Prime --------------------------------------------------------------------
    model.reset()
    if args.prompt_midi:
        print(f"Priming from {args.prompt_midi} …")
        try:
            model.prompt(args.prompt_midi)
        except Exception as exc:
            print(f"  prompt() failed: {exc} — starting from scratch")

    # Generate -----------------------------------------------------------------
    print(f"Generating {args.n_events} events …  "
          f"pitch_temp={args.pitch_temp}  rhythm_temp={args.rhythm_temp}")
    events = []
    with torch.no_grad():
        for i in range(args.n_events):
            try:
                ev = model.query(
                    min_time=args.min_time,
                    max_time=args.max_time,
                    include_inst=include_inst,
                    pitch_temp=args.pitch_temp,
                    rhythm_temp=args.rhythm_temp,
                    timing_temp=args.rhythm_temp,
                    velocity_temp=args.velocity_temp,
                    instrument_temp=args.instrument_temp,
                )

                inst  = int(ev.get("instrument", ev.get("inst", 0)))
                pitch = int(ev.get("pitch", 60))
                dt    = float(ev.get("time", 0.1))
                vel   = float(ev.get("vel", ev.get("velocity", 64)))

                events.append({"instrument": inst, "pitch": pitch,
                                "time": dt, "velocity": vel})
                model.feed(inst=inst, pitch=pitch, time=dt, vel=vel)

            except Exception as exc:
                print(f"  Event {i}: {exc}")
                break

    print(f"Generated {len(events)} events")
    events_to_midi(events, Path(args.out_midi),
                   max_note_dur=args.max_note_dur,
                   max_polyphony=args.max_polyphony)


if __name__ == "__main__":
    main()
