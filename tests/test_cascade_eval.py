"""Tests for cascade evaluation metrics (eval_cascade.py)."""

import math
import pytest

from training.pre import (
    make_instrument_config,
    INSTRUMENT_PRESETS,
)
from training.pre_cascade import (
    extract_chord_labels,
    CHORD_QUALITIES,
)
from training.eval_cascade import (
    chord_tone_coverage,
    range_violations,
    parallel_fifths_octaves,
    note_density_per_instrument,
    pitch_class_entropy,
)


@pytest.fixture
def blues6_config():
    return make_instrument_config(INSTRUMENT_PRESETS["blues6"])


# ── Chord-tone coverage ──────────────────────────────────────

class TestChordToneCoverage:
    def test_perfect_coverage(self, blues6_config):
        """All notes are chord tones of C major (C, E, G)."""
        guitar_idx = blues6_config.guitar_idx  # 2
        # C major chord tones at beat 0: C=60, E=64, G=67
        ev = [
            (0.0, guitar_idx, 60, 80, 1.0),  # C4
            (0.0, guitar_idx, 64, 80, 1.0),  # E4
            (0.0, guitar_idx, 67, 80, 1.0),  # G4
        ]
        # Chord labels: C major at beat 0
        chords = [(0.0, 0, 0)]  # (time_qn=0, root=C, quality=maj)
        cov = chord_tone_coverage(ev, chords, 120.0, blues6_config)
        assert cov == 1.0

    def test_zero_coverage(self, blues6_config):
        """All notes are non-chord tones."""
        guitar_idx = blues6_config.guitar_idx
        # C major chord, but play F#, Bb
        ev = [
            (0.0, guitar_idx, 66, 80, 1.0),  # F#4
            (0.0, guitar_idx, 70, 80, 1.0),  # Bb4
        ]
        chords = [(0.0, 0, 0)]  # C major
        cov = chord_tone_coverage(ev, chords, 120.0, blues6_config)
        assert cov == 0.0

    def test_empty_events(self, blues6_config):
        cov = chord_tone_coverage([], [], 120.0, blues6_config)
        assert cov == 0.0

    def test_drums_excluded(self, blues6_config):
        """Drum notes should not affect coverage."""
        drums_idx = blues6_config.drum_idx
        ev = [
            (0.0, drums_idx, 36, 100, 0.25),
            (0.0, drums_idx, 42, 100, 0.25),
        ]
        chords = [(0.0, 0, 0)]
        cov = chord_tone_coverage(ev, chords, 120.0, blues6_config)
        assert cov == 0.0  # no melodic notes → 0/0 → 0

    def test_coverage_between_0_and_1(self, blues6_config):
        guitar_idx = blues6_config.guitar_idx
        ev = [
            (0.0, guitar_idx, 60, 80, 1.0),  # C (chord tone)
            (0.0, guitar_idx, 61, 80, 1.0),  # C# (non-chord tone)
        ]
        chords = [(0.0, 0, 0)]
        cov = chord_tone_coverage(ev, chords, 120.0, blues6_config)
        assert 0.0 < cov < 1.0


# ── Range violations ──────────────────────────────────────────

class TestRangeViolations:
    def test_no_violations(self, blues6_config):
        guitar_idx = blues6_config.guitar_idx
        ev = [
            (0.0, guitar_idx, 60, 80, 1.0),  # in range
            (0.5, guitar_idx, 72, 80, 1.0),  # in range
        ]
        rv = range_violations(ev, blues6_config)
        assert sum(rv.values()) == 0

    def test_bass_too_high(self, blues6_config):
        bass_idx = blues6_config.bass_idx
        ev = [
            (0.0, bass_idx, 72, 80, 1.0),  # C5 — above bass range (24-60)
        ]
        rv = range_violations(ev, blues6_config)
        assert rv.get("bass", 0) == 1

    def test_guitar_too_low(self, blues6_config):
        guitar_idx = blues6_config.guitar_idx
        ev = [
            (0.0, guitar_idx, 20, 80, 1.0),  # well below guitar range (40-88)
        ]
        rv = range_violations(ev, blues6_config)
        assert rv.get("guitar", 0) == 1

    def test_empty_events(self, blues6_config):
        rv = range_violations([], blues6_config)
        assert sum(rv.values()) == 0


# ── Parallel fifths/octaves ───────────────────────────────────

class TestParallelFifths:
    def test_parallel_fifths_detected(self, blues6_config):
        """Two voices moving in parallel 5ths."""
        guitar_idx = blues6_config.guitar_idx
        bass_idx = blues6_config.bass_idx
        # Beat 0: bass C2, guitar G3 (interval: P5)
        # Beat 1: bass D2, guitar A3 (interval: P5, both up by step)
        ev = [
            (0.0, bass_idx, 36, 80, 1.0),   # C2
            (0.0, guitar_idx, 55, 80, 1.0),  # G3
            (0.5, bass_idx, 38, 80, 1.0),   # D2
            (0.5, guitar_idx, 57, 80, 1.0),  # A3
        ]
        count = parallel_fifths_octaves(ev, 120.0, blues6_config)
        assert count >= 1

    def test_no_parallel_motion(self, blues6_config):
        """Contrary motion — should not detect parallels."""
        guitar_idx = blues6_config.guitar_idx
        bass_idx = blues6_config.bass_idx
        ev = [
            (0.0, bass_idx, 36, 80, 1.0),   # C2
            (0.0, guitar_idx, 60, 80, 1.0),  # C4
            (0.5, bass_idx, 38, 80, 1.0),   # D2 (up)
            (0.5, guitar_idx, 59, 80, 1.0),  # B3 (down) — contrary
        ]
        count = parallel_fifths_octaves(ev, 120.0, blues6_config)
        assert count == 0

    def test_empty_events(self, blues6_config):
        count = parallel_fifths_octaves([], 120.0, blues6_config)
        assert count == 0


# ── Note density ──────────────────────────────────────────────

class TestNoteDensity:
    def test_basic_density(self, blues6_config):
        guitar_idx = blues6_config.guitar_idx
        # 4 notes spanning 2 QN = density 2.0
        ev = [
            (0.0, guitar_idx, 60, 80, 0.25),
            (0.25, guitar_idx, 62, 80, 0.25),
            (0.5, guitar_idx, 64, 80, 0.25),
            (0.75, guitar_idx, 65, 80, 0.25),
        ]
        density = note_density_per_instrument(ev, 120.0, blues6_config)
        # At 120 BPM, 0.75s = 1.5 QN. 4 notes / 1.5 QN ≈ 2.67
        assert "guitar" in density
        assert density["guitar"] > 0

    def test_empty_events(self, blues6_config):
        density = note_density_per_instrument([], 120.0, blues6_config)
        assert density == {}


# ── Pitch class entropy ──────────────────────────────────────

class TestPitchClassEntropy:
    def test_single_pitch_class(self, blues6_config):
        """All same pitch class → entropy 0."""
        guitar_idx = blues6_config.guitar_idx
        ev = [
            (0.0, guitar_idx, 60, 80, 1.0),  # C
            (0.5, guitar_idx, 72, 80, 1.0),  # C (octave higher)
            (1.0, guitar_idx, 48, 80, 1.0),  # C (octave lower)
        ]
        ent = pitch_class_entropy(ev, blues6_config)
        assert ent == 0.0

    def test_uniform_distribution(self, blues6_config):
        """All 12 pitch classes equally → max entropy."""
        guitar_idx = blues6_config.guitar_idx
        ev = [(i * 0.1, guitar_idx, 60 + i, 80, 0.1) for i in range(12)]
        ent = pitch_class_entropy(ev, blues6_config)
        assert abs(ent - math.log2(12)) < 0.01

    def test_empty_events(self, blues6_config):
        ent = pitch_class_entropy([], blues6_config)
        assert ent == 0.0

    def test_drums_excluded(self, blues6_config):
        """Only drum events → entropy 0 (no melodic notes)."""
        drums_idx = blues6_config.drum_idx
        ev = [
            (0.0, drums_idx, 36, 100, 0.25),
            (0.5, drums_idx, 38, 90, 0.25),
        ]
        ent = pitch_class_entropy(ev, blues6_config)
        assert ent == 0.0

    def test_entropy_increases_with_variety(self, blues6_config):
        guitar_idx = blues6_config.guitar_idx
        ev_few = [
            (0.0, guitar_idx, 60, 80, 1.0),  # C
            (0.5, guitar_idx, 64, 80, 1.0),  # E
        ]
        ev_many = [
            (0.0, guitar_idx, 60, 80, 1.0),  # C
            (0.25, guitar_idx, 62, 80, 1.0),  # D
            (0.5, guitar_idx, 64, 80, 1.0),  # E
            (0.75, guitar_idx, 65, 80, 1.0),  # F
            (1.0, guitar_idx, 67, 80, 1.0),  # G
        ]
        ent_few = pitch_class_entropy(ev_few, blues6_config)
        ent_many = pitch_class_entropy(ev_many, blues6_config)
        assert ent_many > ent_few
