#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the dense chorale pipeline (pre, model, decode)."""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from training.pre_chorale_dense import (
    PAD_ID, BOS_ID, EOS_ID, REST_ID, VOCAB_SIZE,
    PITCH_OFFSET, MIDI_LO, MIDI_HI, NUM_PITCHES,
    CHORD_OFFSET, NUM_CHORDS, CHORD_REST, CHORD_OTHER,
    VOICE_RANGES, VOICE_ORDER, REST_THRESHOLD,
    midi_to_token, token_to_midi, is_pitch_token, is_rest_token, is_chord_token,
    analyze_chord_at_timestep, chord_token_to_label,
    compute_continuation_counters, transpose_chorale,
    tokenize_chorale, decode_tokens_to_midi, build_vocab_dict,
)
from training.model_chorale_dense import ChoraleDenseModel


# ────────────────── Fixtures ──────────────────

def _make_chorale(n_steps: int = 8) -> np.ndarray:
    """Create a simple 4-voice chorale array (T, 4) = soprano, alto, tenor, bass."""
    # C major chord held for all steps, transposing up once
    arr = np.zeros((n_steps, 4), dtype=np.float64)
    arr[:, 0] = 72  # soprano: C5
    arr[:, 1] = 64  # alto: E4
    arr[:, 2] = 60  # tenor: C4 (middle C)
    arr[:, 3] = 48  # bass: C3
    # Add a pitch change midway for soprano
    arr[n_steps // 2:, 0] = 74  # soprano: D5
    return arr


def _make_chorale_with_rest(n_steps: int = 8) -> np.ndarray:
    """Create a chorale with some rest timesteps."""
    arr = _make_chorale(n_steps)
    arr[2, 1] = 0  # alto rests at step 2
    arr[3, 3] = np.nan  # bass rests at step 3
    return arr


# ────────────────── Pitch token roundtrip ──────────────────

class TestPitchTokens:
    def test_roundtrip_all_pitches(self):
        for midi_p in range(MIDI_LO, MIDI_HI + 1):
            tok = midi_to_token(midi_p)
            assert is_pitch_token(tok), f"token {tok} for MIDI {midi_p} should be pitch"
            assert token_to_midi(tok) == midi_p

    def test_rest_is_not_pitch(self):
        assert not is_pitch_token(REST_ID)
        assert is_rest_token(REST_ID)

    def test_chord_tokens(self):
        for i in range(NUM_CHORDS):
            tok = CHORD_OFFSET + i
            assert is_chord_token(tok)
            assert not is_pitch_token(tok)

    def test_special_tokens(self):
        assert not is_pitch_token(PAD_ID)
        assert not is_pitch_token(BOS_ID)
        assert not is_pitch_token(EOS_ID)

    def test_vocab_size(self):
        assert VOCAB_SIZE == 100


# ────────────────── Chord analysis ──────────────────

class TestChordAnalysis:
    def test_c_major(self):
        pitches = np.array([72, 64, 60, 48], dtype=np.float64)  # C5, E4, C4, C3
        chord_local = analyze_chord_at_timestep(pitches)
        label = chord_token_to_label(chord_local)
        assert "C_major" in label

    def test_a_minor(self):
        pitches = np.array([69, 64, 60, 45], dtype=np.float64)  # A4, E4, C4, A2
        chord_local = analyze_chord_at_timestep(pitches)
        label = chord_token_to_label(chord_local)
        assert "A_minor" in label

    def test_all_rest(self):
        pitches = np.array([0, 0, 0, 0], dtype=np.float64)
        chord_local = analyze_chord_at_timestep(pitches)
        assert chord_local == CHORD_REST

    def test_nan_rest(self):
        pitches = np.array([np.nan, np.nan, np.nan, np.nan])
        chord_local = analyze_chord_at_timestep(pitches)
        assert chord_local == CHORD_REST

    def test_chord_label_roundtrip(self):
        for i in range(NUM_CHORDS):
            label = chord_token_to_label(i)
            assert isinstance(label, str)
            assert len(label) > 0


# ────────────────── Continuation counters ──────────────────

class TestContinuationCounters:
    def test_held_notes(self):
        arr = _make_chorale(6)
        counters = compute_continuation_counters(arr)
        assert len(counters) == 6
        # Step 0: all onsets
        assert counters[0] == [0, 0, 0, 0]
        # Step 1: all held
        assert counters[1] == [1, 1, 1, 1]
        # Step 2: all held
        assert counters[2] == [2, 2, 2, 2]
        # Step 3 (n_steps//2=3): soprano changes pitch
        assert counters[3][0] == 0  # soprano onset
        assert counters[3][1] == 3  # alto still held
        assert counters[3][2] == 3  # tenor still held
        assert counters[3][3] == 3  # bass still held

    def test_rest_resets_counter(self):
        arr = _make_chorale_with_rest(8)
        counters = compute_continuation_counters(arr)
        # Step 2: alto rests (pitch 0 < 36)
        assert counters[2][1] == 0  # reset on rest
        # Step 3: alto back to E4
        assert counters[3][1] == 0  # new onset

    def test_cap_at_31(self):
        arr = np.full((40, 4), 60.0, dtype=np.float64)
        counters = compute_continuation_counters(arr)
        assert counters[31] == [31, 31, 31, 31]
        assert counters[32] == [31, 31, 31, 31]  # capped


# ────────────────── Transposition ──────────────────

class TestTransposition:
    def test_identity(self):
        arr = _make_chorale(4)
        result = transpose_chorale(arr, 0)
        assert result is not None
        np.testing.assert_array_equal(result, arr)

    def test_up_one(self):
        arr = _make_chorale(4)
        result = transpose_chorale(arr, 1)
        assert result is not None
        np.testing.assert_array_equal(result[:, 0], arr[:, 0] + 1)

    def test_reject_out_of_range(self):
        arr = np.full((4, 4), 81.0, dtype=np.float64)  # soprano at max
        result = transpose_chorale(arr, 1)  # soprano would go to 82 > 81
        assert result is None

    def test_reject_bass_below_range(self):
        arr = np.full((4, 4), 60.0, dtype=np.float64)
        arr[:, 3] = 36  # bass at minimum
        result = transpose_chorale(arr, -1)  # bass would go to 35 < 36
        assert result is None

    def test_rest_not_transposed(self):
        arr = _make_chorale_with_rest(8)
        result = transpose_chorale(arr, 2)
        if result is not None:
            # Step 2 alto was 0 (rest) — should stay 0
            assert result[2, 1] == 0


# ────────────────── Sequence structure ──────────────────

class TestTokenizeChorale:
    def test_basic_structure(self):
        arr = _make_chorale(4)
        tokens, conts = tokenize_chorale(arr)
        assert tokens[0] == BOS_ID
        assert tokens[-1] == EOS_ID
        assert len(tokens) == len(conts)
        # 4 timesteps * 5 tokens/step + BOS + EOS = 22
        assert len(tokens) == 4 * 5 + 2

    def test_voice_order(self):
        """Check that token positions follow chord, soprano, bass, alto, tenor."""
        arr = _make_chorale(2)
        tokens, _ = tokenize_chorale(arr)
        # Step 0 starts at index 1 (after BOS)
        # Index 1: chord (should be chord token)
        assert is_chord_token(tokens[1])
        # Index 2: soprano
        assert is_pitch_token(tokens[2]) or is_rest_token(tokens[2])
        # Index 3: bass
        assert is_pitch_token(tokens[3]) or is_rest_token(tokens[3])
        # Index 4: alto
        assert is_pitch_token(tokens[4]) or is_rest_token(tokens[4])
        # Index 5: tenor
        assert is_pitch_token(tokens[5]) or is_rest_token(tokens[5])

    def test_rest_tokens(self):
        arr = _make_chorale_with_rest(4)
        tokens, _ = tokenize_chorale(arr)
        # Step 2 (index = 1 + 2*5 = 11): chord at 11, soprano 12, bass 13, alto 14, tenor 15
        # Alto rests at step 2 → token at index 14 should be REST
        assert tokens[14] == REST_ID

    def test_no_pad_in_output(self):
        arr = _make_chorale(4)
        tokens, _ = tokenize_chorale(arr)
        assert PAD_ID not in tokens

    def test_conts_zeros_for_special(self):
        arr = _make_chorale(4)
        _, conts = tokenize_chorale(arr)
        assert conts[0] == 0  # BOS
        assert conts[-1] == 0  # EOS
        # Chord positions (every 5th token starting from 1)
        for i in range(1, len(conts) - 1, 5):
            assert conts[i] == 0


# ────────────────── Model forward/backward ──────────────────

class TestModel:
    def test_forward_shape(self):
        model = ChoraleDenseModel(vocab_size=100, d_model=32, n_heads=2,
                                   n_layers=1, ff_mult=2, dropout=0.0)
        x = torch.randint(0, 100, (2, 20))
        cont = torch.randint(0, 32, (2, 20))
        logits = model(x, cont)
        assert logits.shape == (2, 20, 100)

    def test_forward_no_cont(self):
        model = ChoraleDenseModel(vocab_size=100, d_model=32, n_heads=2,
                                   n_layers=1, ff_mult=2, dropout=0.0)
        x = torch.randint(0, 100, (2, 20))
        logits = model(x)
        assert logits.shape == (2, 20, 100)

    def test_backward(self):
        model = ChoraleDenseModel(vocab_size=100, d_model=32, n_heads=2,
                                   n_layers=1, ff_mult=2, dropout=0.0)
        x = torch.randint(0, 100, (2, 20))
        cont = torch.randint(0, 32, (2, 20))
        logits = model(x, cont)
        loss = logits.sum()
        loss.backward()
        for p in model.parameters():
            if p.requires_grad:
                assert p.grad is not None

    def test_param_count(self):
        model = ChoraleDenseModel(vocab_size=100, d_model=128, n_heads=4,
                                   n_layers=4, ff_mult=3, dropout=0.15)
        params = model.count_parameters()
        # Should be in the ballpark of 685K
        assert 500_000 < params < 1_000_000, f"params={params}"


# ────────────────── Decode to MIDI roundtrip ──────────────────

class TestDecodeMIDI:
    def test_roundtrip(self):
        arr = _make_chorale(8)
        tokens, conts = tokenize_chorale(arr)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.mid")
            decode_tokens_to_midi(tokens, path)
            assert os.path.isfile(path)
            assert os.path.getsize(path) > 0

    def test_with_rests(self):
        arr = _make_chorale_with_rest(8)
        tokens, _ = tokenize_chorale(arr)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test_rest.mid")
            decode_tokens_to_midi(tokens, path)
            assert os.path.isfile(path)


# ────────────────── Vocab dict ──────────────────

class TestVocabDict:
    def test_structure(self):
        vocab = build_vocab_dict()
        assert vocab["vocab_size"] == 100
        assert vocab["PAD_ID"] == 0
        assert vocab["BOS_ID"] == 1
        assert vocab["EOS_ID"] == 2
        assert vocab["REST_ID"] == 49
        assert vocab["CHORD_OFFSET"] == 50
        assert len(vocab["pitch_labels"]) == NUM_PITCHES
        assert len(vocab["chord_labels"]) == NUM_CHORDS

    def test_json_serializable(self):
        vocab = build_vocab_dict()
        s = json.dumps(vocab)
        loaded = json.loads(s)
        assert loaded["vocab_size"] == 100
