"""Test preprocessing a chorale MIDI through the event-stream pipeline."""

import os
import tempfile
import numpy as np
import pretty_midi
import pytest

from training.pre import (
    InstrumentConfig,
    make_instrument_config,
    INSTRUMENT_PRESETS,
    extract_multitrack_events,
    build_pitch_maps,
    build_event_vocab,
    gather_bar_pairs,
    tokenize_song,
    decode_to_midi,
    compute_aux_for_window,
    compute_aux_layout,
    events_to_intervals_qn,
    is_drum_slot,
)
from scripts.convert_chorales_npz_to_midi import chorale_to_midi


def _write_test_chorale_midi() -> str:
    """Create a small test chorale MIDI and return its path."""
    arr = np.zeros((32, 4), dtype=np.float16)
    arr[:, 0] = 72  # soprano C5
    arr[:, 1] = 67  # alto G4
    arr[:, 2] = 60  # tenor C4
    arr[:, 3] = 48  # bass C3
    # Add a pitch change in soprano at step 16
    arr[16:, 0] = 74  # D5
    pm = chorale_to_midi(arr, bpm=100.0)
    path = tempfile.mktemp(suffix=".mid")
    pm.write(path)
    return path


@pytest.fixture
def chorale_config():
    return make_instrument_config(INSTRUMENT_PRESETS["chorale4"])


@pytest.fixture
def chorale_midi_path():
    path = _write_test_chorale_midi()
    yield path
    if os.path.exists(path):
        os.unlink(path)


class TestChoraleExtraction:
    def test_extract_events(self, chorale_midi_path, chorale_config):
        ev, tempo, bar_starts, bars_meta = extract_multitrack_events(
            chorale_midi_path, chorale_config
        )
        assert len(ev) > 0
        # All inst indices should be in [0, 3]
        inst_indices = {e[1] for e in ev}
        assert all(0 <= i < 4 for i in inst_indices)

    def test_no_drum_events(self, chorale_midi_path, chorale_config):
        ev, tempo, bar_starts, bars_meta = extract_multitrack_events(
            chorale_midi_path, chorale_config
        )
        for e in ev:
            assert not is_drum_slot(e[1], chorale_config)


class TestChoraleVocab:
    def test_no_pitch_drums_in_vocab(self, chorale_midi_path, chorale_config):
        ev, tempo, bar_starts, bars_meta = extract_multitrack_events(
            chorale_midi_path, chorale_config
        )
        pitch_maps = build_pitch_maps(ev, chorale_config)
        assert "drums" not in pitch_maps
        bar_pairs = gather_bar_pairs([bars_meta])
        vocab = build_event_vocab(pitch_maps, bar_pairs, chorale_config)
        assert "PITCH_DRUMS" not in vocab["layout"]

    def test_four_inst_tokens(self, chorale_midi_path, chorale_config):
        ev, tempo, bar_starts, bars_meta = extract_multitrack_events(
            chorale_midi_path, chorale_config
        )
        pitch_maps = build_pitch_maps(ev, chorale_config)
        bar_pairs = gather_bar_pairs([bars_meta])
        vocab = build_event_vocab(pitch_maps, bar_pairs, chorale_config)
        assert vocab["layout"]["INST"]["size"] == 4

    def test_instrument_names_in_vocab(self, chorale_midi_path, chorale_config):
        ev, tempo, bar_starts, bars_meta = extract_multitrack_events(
            chorale_midi_path, chorale_config
        )
        pitch_maps = build_pitch_maps(ev, chorale_config)
        bar_pairs = gather_bar_pairs([bars_meta])
        vocab = build_event_vocab(pitch_maps, bar_pairs, chorale_config)
        assert vocab["instrument_names"] == ["soprano", "alto", "tenor", "bassvox"]

    def test_aux_dim_24(self, chorale_midi_path, chorale_config):
        ev, tempo, bar_starts, bars_meta = extract_multitrack_events(
            chorale_midi_path, chorale_config
        )
        pitch_maps = build_pitch_maps(ev, chorale_config)
        bar_pairs = gather_bar_pairs([bars_meta])
        vocab = build_event_vocab(pitch_maps, bar_pairs, chorale_config)
        assert vocab["aux"]["aux_dim"] == 24


class TestChoraleTokenize:
    def test_tokenize_roundtrip(self, chorale_midi_path, chorale_config):
        """Tokenize a chorale and decode back to MIDI."""
        ev, tempo, bar_starts, bars_meta = extract_multitrack_events(
            chorale_midi_path, chorale_config
        )
        pitch_maps = build_pitch_maps(ev, chorale_config)
        bar_pairs = gather_bar_pairs([bars_meta])
        vocab = build_event_vocab(pitch_maps, bar_pairs, chorale_config)

        tokens = tokenize_song(ev, tempo, bar_starts, bars_meta, vocab)
        assert len(tokens) > 5  # at least BOS + some events + EOS

        # Decode back
        out_path = tempfile.mktemp(suffix=".mid")
        try:
            decode_to_midi(tokens, vocab, out_path, tempo_bpm=tempo)
            assert os.path.isfile(out_path)
            pm = pretty_midi.PrettyMIDI(out_path)
            assert len(pm.instruments) == 4
            # Check we got notes back
            total_notes = sum(len(inst.notes) for inst in pm.instruments)
            assert total_notes > 0
        finally:
            if os.path.exists(out_path):
                os.unlink(out_path)


class TestChoraleAux:
    def test_aux_vector_shape(self, chorale_midi_path, chorale_config):
        ev, tempo, bar_starts, bars_meta = extract_multitrack_events(
            chorale_midi_path, chorale_config
        )
        intervals = events_to_intervals_qn(ev, tempo)
        aux = compute_aux_for_window(intervals, 0.0, 10.0, config=chorale_config)
        expected_dim = compute_aux_layout(chorale_config)["aux_dim"]
        assert aux.shape == (expected_dim,)
        assert aux.dtype == np.float32
