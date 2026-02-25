"""Tests for NPZ → multi-track MIDI chorale converter."""

import os
import tempfile
import numpy as np
import pretty_midi
import pytest

from scripts.convert_chorales_npz_to_midi import chorale_to_midi, VOICE_NAMES, REST_THRESHOLD


def _simple_chorale(n_steps: int = 32) -> np.ndarray:
    """Create a synthetic (T, 4) chorale array."""
    arr = np.zeros((n_steps, 4), dtype=np.float16)
    # Soprano: alternating C5 (72) and D5 (74) every 4 steps
    for t in range(n_steps):
        arr[t, 0] = 72 if (t // 4) % 2 == 0 else 74
    # Alto: constant E4 (64)
    arr[:, 1] = 64
    # Tenor: constant C4 (60)
    arr[:, 2] = 60
    # Bass: constant C3 (48)
    arr[:, 3] = 48
    return arr


class TestChoraleToMidi:
    def test_four_tracks(self):
        arr = _simple_chorale()
        pm = chorale_to_midi(arr, bpm=100.0)
        assert len(pm.instruments) == 4

    def test_track_names(self):
        arr = _simple_chorale()
        pm = chorale_to_midi(arr, bpm=100.0)
        names = [inst.name for inst in pm.instruments]
        assert names == VOICE_NAMES

    def test_no_drums(self):
        arr = _simple_chorale()
        pm = chorale_to_midi(arr, bpm=100.0)
        for inst in pm.instruments:
            assert not inst.is_drum

    def test_velocity_uniform(self):
        arr = _simple_chorale()
        pm = chorale_to_midi(arr, bpm=100.0)
        for inst in pm.instruments:
            for note in inst.notes:
                assert note.velocity == 80

    def test_soprano_has_correct_pitches(self):
        arr = _simple_chorale(32)
        pm = chorale_to_midi(arr, bpm=100.0)
        soprano = pm.instruments[0]
        pitches = sorted(set(n.pitch for n in soprano.notes))
        assert pitches == [72, 74]

    def test_sustained_note_detection(self):
        """Repeated pitch = one long note, not many short ones."""
        arr = np.full((16, 4), 60.0, dtype=np.float16)  # all C4, 16 steps
        pm = chorale_to_midi(arr, bpm=100.0)
        for inst in pm.instruments:
            # Should be exactly 1 note per voice (sustained 16 steps)
            assert len(inst.notes) == 1

    def test_pitch_change_creates_new_onset(self):
        """Pitch change = end old note + start new note."""
        arr = np.full((8, 4), 60.0, dtype=np.float16)
        arr[4:, 0] = 62  # soprano changes at step 4
        pm = chorale_to_midi(arr, bpm=100.0)
        soprano = pm.instruments[0]
        assert len(soprano.notes) == 2
        assert soprano.notes[0].pitch == 60
        assert soprano.notes[1].pitch == 62

    def test_rest_handling(self):
        arr = np.full((16, 4), 60.0, dtype=np.float16)
        arr[4:8, 0] = 0  # rest in soprano from step 4-7
        pm = chorale_to_midi(arr, bpm=100.0)
        soprano = pm.instruments[0]
        # Should have 2 notes (before rest and after rest)
        assert len(soprano.notes) == 2

    def test_writes_midi_file(self):
        arr = _simple_chorale()
        pm = chorale_to_midi(arr, bpm=100.0)
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as f:
            path = f.name
        try:
            pm.write(path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0
            # Re-read to verify
            pm2 = pretty_midi.PrettyMIDI(path)
            assert len(pm2.instruments) == 4
        finally:
            os.unlink(path)

    def test_pitch_ranges_reasonable(self):
        """All pitches should be in standard vocal ranges."""
        arr = _simple_chorale()
        pm = chorale_to_midi(arr, bpm=100.0)
        for inst in pm.instruments:
            for note in inst.notes:
                assert 30 <= note.pitch <= 90, f"Pitch {note.pitch} on {inst.name} out of range"
