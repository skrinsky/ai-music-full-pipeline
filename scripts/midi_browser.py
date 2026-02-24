#!/usr/bin/env python3
"""Interactive MIDI browser & player — tkinter GUI with fluidsynth playback.

Browse a folder of MIDI files, inspect per-track metadata (using the same
mapping as training/pre.py), and listen via fluidsynth.

Usage:
    python scripts/midi_browser.py --folder data/blues_midi
"""

import argparse
import os
import signal
import subprocess
import sys
import tkinter as tk
from tkinter import ttk
from typing import List, Optional

import pretty_midi

# ── Reuse project code ──────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from scripts.audition_gigamidi import gm_name, _midi_paths
from training.pre import INSTRUMENT_NAMES, map_name_to_slot

# ── Fluidsynth config ───────────────────────────────────────────────
FLUIDSYNTH_BIN = "/opt/homebrew/bin/fluidsynth"
SOUNDFONT = (
    "/opt/homebrew/Cellar/fluid-synth/2.5.2/"
    "share/fluid-synth/sf2/VintageDreamsWaves-v2.sf2"
)


class MidiBrowser(tk.Tk):
    def __init__(self, folder: str) -> None:
        super().__init__()
        self.title("MIDI Browser")
        self.geometry("960x620")
        self.minsize(800, 500)

        self.folder = folder
        self.paths: List[str] = _midi_paths(folder)
        self.current_idx: int = -1

        # Playback state
        self.proc: Optional[subprocess.Popen] = None
        self.paused: bool = False

        self._build_ui()
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # Select the first file
        if self.paths:
            self.listbox.selection_set(0)
            self.listbox.event_generate("<<ListboxSelect>>")

    # ── UI construction ─────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Title bar
        title = f"MIDI Browser — {len(self.paths)} files in {self.folder}"
        title_lbl = tk.Label(self, text=title, font=("Helvetica", 14, "bold"),
                             anchor="w", padx=8, pady=4)
        title_lbl.pack(fill="x")

        # Main paned window: left listbox | right detail
        paned = tk.PanedWindow(self, orient="horizontal", sashwidth=6)
        paned.pack(fill="both", expand=True, padx=4, pady=2)

        # Left: scrollable listbox
        left = tk.Frame(paned)
        paned.add(left, width=240)

        scrollbar = tk.Scrollbar(left, orient="vertical")
        self.listbox = tk.Listbox(left, yscrollcommand=scrollbar.set,
                                  font=("Menlo", 11), exportselection=False)
        scrollbar.config(command=self.listbox.yview)
        scrollbar.pack(side="right", fill="y")
        self.listbox.pack(side="left", fill="both", expand=True)

        for p in self.paths:
            self.listbox.insert("end", os.path.basename(p))

        self.listbox.bind("<<ListboxSelect>>", self._on_select)

        # Right: detail panel
        right = tk.Frame(paned)
        paned.add(right)

        # File info labels
        info_frame = tk.Frame(right, padx=8, pady=4)
        info_frame.pack(fill="x")

        self.lbl_file = tk.Label(info_frame, text="File: —",
                                 font=("Menlo", 12, "bold"), anchor="w")
        self.lbl_file.pack(fill="x")

        self.lbl_meta = tk.Label(info_frame, text="", font=("Menlo", 11),
                                 anchor="w", justify="left")
        self.lbl_meta.pack(fill="x")

        ttk.Separator(right, orient="horizontal").pack(fill="x", padx=8, pady=2)

        # Track treeview
        cols = ("raw_name", "drum", "prog", "gm_name", "slot", "notes", "pitch")
        self.tree = ttk.Treeview(right, columns=cols, show="headings", height=10)
        self.tree.heading("#0", text="#")
        for col, hdr, w in [
            ("raw_name", "Raw Name", 160),
            ("drum", "Drum?", 50),
            ("prog", "Prog", 45),
            ("gm_name", "GM Name", 180),
            ("slot", "Slot", 70),
            ("notes", "Notes", 60),
            ("pitch", "Pitch", 70),
        ]:
            self.tree.heading(col, text=hdr)
            self.tree.column(col, width=w, minwidth=40)

        tree_scroll = ttk.Scrollbar(right, orient="vertical",
                                    command=self.tree.yview)
        self.tree.configure(yscrollcommand=tree_scroll.set)
        tree_scroll.pack(side="right", fill="y", padx=(0, 8))
        self.tree.pack(fill="both", expand=True, padx=(8, 0), pady=4)

        # Bottom: transport controls
        transport = tk.Frame(self, pady=6)
        transport.pack(fill="x")

        self.btn_prev = tk.Button(transport, text="<< Prev", width=8,
                                  command=self._prev)
        self.btn_prev.pack(side="left", padx=(12, 4))

        self.btn_play = tk.Button(transport, text="Play \u25B6", width=12,
                                  command=self._play_pause)
        self.btn_play.pack(side="left", padx=4)

        self.btn_stop = tk.Button(transport, text="Stop \u25A0", width=8,
                                  command=self._stop)
        self.btn_stop.pack(side="left", padx=4)

        self.btn_next = tk.Button(transport, text="Next >>", width=8,
                                  command=self._next)
        self.btn_next.pack(side="left", padx=4)

        self.lbl_pos = tk.Label(transport, text="— / —", font=("Menlo", 11))
        self.lbl_pos.pack(side="right", padx=12)

    # ── Selection & detail ──────────────────────────────────────────

    def _on_select(self, event: tk.Event) -> None:
        sel = self.listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        if idx == self.current_idx:
            return
        self.current_idx = idx
        self._load_detail(idx)

    def _load_detail(self, idx: int) -> None:
        path = self.paths[idx]
        basename = os.path.basename(path)
        self.lbl_file.config(text=f"File: {basename}")
        self.lbl_pos.config(text=f"{idx + 1} / {len(self.paths)}")

        try:
            pm = pretty_midi.PrettyMIDI(path)
        except Exception as e:
            self.lbl_meta.config(text=f"Error loading: {e}")
            self._clear_tree()
            return

        dur = pm.get_end_time()
        tc = pm.get_tempo_changes()
        tempo_str = f"{tc[1][0]:.1f} BPM" if tc[1].size > 0 else "unknown"
        total_notes = sum(len(inst.notes) for inst in pm.instruments)

        self.lbl_meta.config(
            text=f"Duration: {dur:.1f}s   Tempo: {tempo_str}   Total notes: {total_notes}"
        )
        self._show_track_table(pm)

    def _clear_tree(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)

    def _show_track_table(self, pm: pretty_midi.PrettyMIDI) -> None:
        self._clear_tree()
        for i, inst in enumerate(pm.instruments):
            slot = map_name_to_slot(inst)
            n = len(inst.notes)
            if n > 0:
                pitches = [note.pitch for note in inst.notes]
                prange = f"{min(pitches)}-{max(pitches)}"
            else:
                prange = "—"

            self.tree.insert("", "end", text=str(i), values=(
                (inst.name or "(empty)")[:25],
                "Y" if inst.is_drum else "N",
                inst.program,
                gm_name(inst.program)[:30],
                INSTRUMENT_NAMES[slot],
                n,
                prange,
            ))

    # ── Playback ────────────────────────────────────────────────────

    def _play_pause(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            # Process running — toggle pause/resume
            self._pause_resume()
        else:
            # No process — start playback
            self._play()

    def _play(self) -> None:
        if self.current_idx < 0:
            return
        self._kill_proc()

        path = self.paths[self.current_idx]
        try:
            self.proc = subprocess.Popen(
                [FLUIDSYNTH_BIN, "-a", "coreaudio", "-n", "-i", SOUNDFONT, path],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            print(f"ERROR: fluidsynth not found at {FLUIDSYNTH_BIN}", file=sys.stderr)
            sys.exit(1)

        self.paused = False
        self.btn_play.config(text="Pause \u23F8")
        self._check_playback()

    def _pause_resume(self) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        if self.paused:
            os.kill(self.proc.pid, signal.SIGCONT)
            self.paused = False
            self.btn_play.config(text="Pause \u23F8")
        else:
            os.kill(self.proc.pid, signal.SIGSTOP)
            self.paused = True
            self.btn_play.config(text="Resume \u25B6")

    def _stop(self) -> None:
        self._kill_proc()
        self.btn_play.config(text="Play \u25B6")

    def _kill_proc(self) -> None:
        if self.proc is None:
            return
        if self.proc.poll() is not None:
            self.proc = None
            return
        # Resume first if paused (otherwise terminate hangs)
        if self.paused:
            try:
                os.kill(self.proc.pid, signal.SIGCONT)
            except OSError:
                pass
        self.proc.terminate()
        try:
            self.proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.proc.kill()
        self.proc = None
        self.paused = False

    def _check_playback(self) -> None:
        if self.proc is not None and self.proc.poll() is not None:
            # Playback ended naturally
            self.proc = None
            self.paused = False
            self.btn_play.config(text="Play \u25B6")
            return
        if self.proc is not None:
            self.after(500, self._check_playback)

    # ── Navigation ──────────────────────────────────────────────────

    def _next(self) -> None:
        if self.current_idx < len(self.paths) - 1:
            self._stop()
            new_idx = self.current_idx + 1
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(new_idx)
            self.listbox.see(new_idx)
            self.current_idx = new_idx
            self._load_detail(new_idx)
            self._play()

    def _prev(self) -> None:
        if self.current_idx > 0:
            self._stop()
            new_idx = self.current_idx - 1
            self.listbox.selection_clear(0, "end")
            self.listbox.selection_set(new_idx)
            self.listbox.see(new_idx)
            self.current_idx = new_idx
            self._load_detail(new_idx)
            self._play()

    # ── Cleanup ─────────────────────────────────────────────────────

    def _on_close(self) -> None:
        self._kill_proc()
        self.destroy()


def main() -> None:
    ap = argparse.ArgumentParser(description="Interactive MIDI browser & player")
    ap.add_argument("--folder", default="data/blues_midi",
                    help="Folder containing MIDI files")
    args = ap.parse_args()

    if not os.path.isdir(args.folder):
        print(f"ERROR: folder not found: {args.folder}", file=sys.stderr)
        sys.exit(1)

    app = MidiBrowser(args.folder)
    app.mainloop()


if __name__ == "__main__":
    main()
