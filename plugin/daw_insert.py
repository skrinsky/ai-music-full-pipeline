"""
Auto-insert a generated MIDI file into the currently running DAW.

Supported:
  Reaper  — requires: pip install python-reapy
            one-time setup: python -c "import reapy; reapy.config.configure_reaper()"
            (run while REAPER is open)

  Ableton — requires: pip install python-osc mido
            one-time setup: install AbletonOSC as a Control Surface in Live
            (https://github.com/ideoforms/AbletonOSC — free, drop in Live's
             MIDI Remote Scripts folder, then Preferences → MIDI → Control Surface)

Returns 'reaper' | 'ableton' | 'reaper_error' | 'ableton_error' | 'unsupported'
"""

import subprocess
import threading
import time
from pathlib import Path


# ── process detection ─────────────────────────────────────────────────────────

def _proc_running(*names: str) -> bool:
    for name in names:
        try:
            if subprocess.run(["pgrep", "-xi", name],
                              capture_output=True).returncode == 0:
                return True
        except Exception:
            pass
    return False


# ── Reaper ────────────────────────────────────────────────────────────────────

def _insert_reaper(midi_path: str) -> bool:
    try:
        import reapy  # type: ignore
        # InsertMedia mode 2 = insert as new tracks at edit cursor
        reapy.reaper.InsertMedia(midi_path, 2)
        return True
    except ImportError:
        print("[daw_insert] python-reapy not installed — run: pip install python-reapy")
        return False
    except Exception as e:
        print(f"[daw_insert] Reaper error: {e}")
        print("[daw_insert] Make sure the reapy bridge is configured: "
              "python -c \"import reapy; reapy.config.configure_reaper()\"")
        return False


# ── Ableton Live (via AbletonOSC) ─────────────────────────────────────────────

def _parse_midi_notes(midi_path: str):
    """Return (notes, clip_length_beats) from a MIDI file."""
    import mido  # type: ignore
    mid = mido.MidiFile(midi_path)
    tpb = mid.ticks_per_beat
    notes = []
    clip_end = 0.0

    for track in mid.tracks:
        abs_ticks = 0
        active = {}
        for msg in track:
            abs_ticks += msg.time
            t = abs_ticks / tpb
            if msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = (t, msg.velocity)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                if key in active:
                    start, vel = active.pop(key)
                    dur = max(t - start, 0.0625)
                    notes.append((int(msg.note), float(start), float(dur), int(vel)))
                    clip_end = max(clip_end, t)

    clip_len = max(clip_end + 1.0, 4.0)
    return notes, clip_len


def _insert_ableton(midi_path: str) -> bool:
    try:
        from pythonosc import udp_client, dispatcher, osc_server  # type: ignore
        import mido  # type: ignore
    except ImportError as e:
        print(f"[daw_insert] Missing dep for Ableton: {e} — run: pip install python-osc mido")
        return False

    try:
        notes, clip_len = _parse_midi_notes(midi_path)
    except Exception as e:
        print(f"[daw_insert] MIDI parse error: {e}")
        return False

    HOST, SEND_PORT, RECV_PORT = "127.0.0.1", 11000, 11001
    client = udp_client.SimpleUDPClient(HOST, SEND_PORT)

    # Listen for the track index that AbletonOSC echoes back after creation
    track_idx_holder = [None]
    done = threading.Event()

    d = dispatcher.Dispatcher()

    def on_track_created(addr, *args):
        if args:
            track_idx_holder[0] = int(args[0])
        done.set()

    d.map("/live/song/create_midi_track", on_track_created)

    try:
        recv_server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", RECV_PORT), d)
    except OSError as e:
        print(f"[daw_insert] Cannot bind OSC recv port {RECV_PORT}: {e}")
        return False

    recv_thread = threading.Thread(target=recv_server.serve_forever, daemon=True)
    recv_thread.start()

    # Ask Live to create a new MIDI track at the end (-1 = append)
    client.send_message("/live/song/create_midi_track", [-1])
    got_response = done.wait(timeout=3.0)
    recv_server.shutdown()

    if not got_response or track_idx_holder[0] is None:
        print("[daw_insert] Ableton: no response — is AbletonOSC control surface active?")
        return False

    track_idx = track_idx_holder[0]
    clip_slot = 0

    # Create a blank clip of the right length
    client.send_message("/live/clip_slot/create_clip", [track_idx, clip_slot, float(clip_len)])
    time.sleep(0.3)

    # Add notes: flat list of [pitch, start, dur, vel, muted, ...]
    # AbletonOSC /live/clip/add_new_notes <track> <slot> pitch start dur vel mute ...
    if notes:
        flat = [track_idx, clip_slot]
        for pitch, start, dur, vel in notes:
            flat.extend([pitch, start, dur, vel, 0])
        client.send_message("/live/clip/add_new_notes", flat)
        time.sleep(0.1)

    return True


# ── public entry point ────────────────────────────────────────────────────────

def insert_midi(midi_path: str) -> str:
    """
    Detect the running DAW and insert midi_path as a new track.
    Returns one of: 'reaper', 'ableton', 'reaper_error', 'ableton_error', 'unsupported'
    """
    if _proc_running("REAPER"):
        return "reaper" if _insert_reaper(midi_path) else "reaper_error"

    # Ableton Live process name varies by OS version ("Live" on macOS)
    if _proc_running("Live", "Ableton Live"):
        return "ableton" if _insert_ableton(midi_path) else "ableton_error"

    print("[daw_insert] No supported DAW detected — MIDI saved to disk only")
    return "unsupported"
