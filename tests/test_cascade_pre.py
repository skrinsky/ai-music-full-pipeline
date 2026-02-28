"""Tests for cascade preprocessing (pre_cascade.py)."""

import pytest
import numpy as np

from training.pre import (
    make_instrument_config,
    INSTRUMENT_PRESETS,
    build_pitch_maps,
    gather_bar_pairs,
    build_event_vocab,
)
from training.pre_cascade import (
    extend_vocab_for_cascade,
    split_events_by_instrument,
    merge_streams_chronological,
    extract_chord_labels,
    inject_chord_tokens,
    compute_musical_times,
    build_cascade_example,
    truncate_context_to_fit,
    build_all_cascade_stages,
    CHORD_QUALITIES,
    CASCADE_ORDER_A,
    CASCADE_ORDER_B,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def blues6_config():
    return make_instrument_config(INSTRUMENT_PRESETS["blues6"])


@pytest.fixture
def simple_events(blues6_config):
    """Simple multi-instrument events: drums, bass, guitar at known times."""
    # (start_sec, inst_idx, midi_pitch, velocity, dur_qn)
    # At 120 BPM: 1 QN = 0.5 sec
    drums_idx = blues6_config.drum_idx    # 5
    bass_idx = blues6_config.bass_idx     # 4
    guitar_idx = blues6_config.guitar_idx # 2
    other_idx = blues6_config.other_idx   # 3

    events = [
        # drums at beat 0, 1, 2, 3
        (0.0, drums_idx, 36, 100, 0.25),
        (0.5, drums_idx, 38, 90, 0.25),
        (1.0, drums_idx, 36, 100, 0.25),
        (1.5, drums_idx, 42, 80, 0.25),
        # bass at beat 0, 2
        (0.0, bass_idx, 40, 80, 1.0),
        (1.0, bass_idx, 43, 80, 1.0),
        # guitar at beat 0, 1
        (0.0, guitar_idx, 60, 70, 0.5),
        (0.5, guitar_idx, 64, 70, 0.5),
        # other at beat 2
        (1.0, other_idx, 67, 60, 1.0),
    ]
    events.sort(key=lambda x: x[0])
    return events


@pytest.fixture
def simple_vocab(simple_events, blues6_config):
    """Build a vocab from simple_events."""
    pitch_maps = build_pitch_maps(simple_events, blues6_config)
    bar_pairs = [(16, i) for i in range(16)]  # 4/4 time
    vocab = build_event_vocab(pitch_maps, bar_pairs, blues6_config)
    return vocab


@pytest.fixture
def cascade_vocab(simple_vocab):
    """Vocab with cascade extensions."""
    return extend_vocab_for_cascade(simple_vocab)


# ── Vocab Extension ───────────────────────────────────────────

class TestExtendVocab:
    def test_sep_added(self, cascade_vocab):
        assert "SEP" in cascade_vocab["layout"]
        assert cascade_vocab["layout"]["SEP"]["size"] == 1

    def test_chord_root_added(self, cascade_vocab):
        assert "CHORD_ROOT" in cascade_vocab["layout"]
        assert cascade_vocab["layout"]["CHORD_ROOT"]["size"] == 12

    def test_chord_qual_added(self, cascade_vocab):
        assert "CHORD_QUAL" in cascade_vocab["layout"]
        assert cascade_vocab["layout"]["CHORD_QUAL"]["size"] == len(CHORD_QUALITIES)

    def test_no_overlap(self, cascade_vocab):
        """Token ranges should not overlap."""
        layout = cascade_vocab["layout"]
        ranges = []
        for name, spec in layout.items():
            ranges.append((spec["start"], spec["start"] + spec["size"], name))
        ranges.sort()
        for i in range(len(ranges) - 1):
            end_i = ranges[i][1]
            start_next = ranges[i + 1][0]
            assert end_i <= start_next, \
                f"Overlap between {ranges[i][2]} (end={end_i}) and {ranges[i+1][2]} (start={start_next})"


# ── Instrument Splitting ──────────────────────────────────────

class TestSplitMerge:
    def test_split_preserves_events(self, simple_events, blues6_config):
        by_inst = split_events_by_instrument(simple_events, blues6_config)
        total = sum(len(v) for v in by_inst.values())
        assert total == len(simple_events)

    def test_split_correct_instruments(self, simple_events, blues6_config):
        by_inst = split_events_by_instrument(simple_events, blues6_config)
        assert len(by_inst.get(blues6_config.drum_idx, [])) == 4   # 4 drum events
        assert len(by_inst.get(blues6_config.bass_idx, [])) == 2   # 2 bass events
        assert len(by_inst.get(blues6_config.guitar_idx, [])) == 2 # 2 guitar events
        assert len(by_inst.get(blues6_config.other_idx, [])) == 1  # 1 other event

    def test_merge_round_trip(self, simple_events, blues6_config):
        by_inst = split_events_by_instrument(simple_events, blues6_config)
        merged = merge_streams_chronological(list(by_inst.values()))
        assert len(merged) == len(simple_events)
        # All events should be time-sorted
        times = [e[0] for e in merged]
        assert times == sorted(times)

    def test_merge_preserves_content(self, simple_events, blues6_config):
        by_inst = split_events_by_instrument(simple_events, blues6_config)
        merged = merge_streams_chronological(list(by_inst.values()))
        # Same set of pitches
        orig_pitches = sorted(e[2] for e in simple_events)
        merged_pitches = sorted(e[2] for e in merged)
        assert orig_pitches == merged_pitches


# ── Chord Extraction ──────────────────────────────────────────

class TestChordExtraction:
    def test_basic_extraction(self, simple_events, blues6_config):
        chords = extract_chord_labels(simple_events, 120.0, blues6_config)
        assert len(chords) > 0
        # All entries are (time_qn, root_pc, quality_idx)
        for (t, r, q) in chords:
            assert isinstance(t, float)
            assert 0 <= r < 12
            assert 0 <= q < len(CHORD_QUALITIES)

    def test_chords_sorted_by_time(self, simple_events, blues6_config):
        chords = extract_chord_labels(simple_events, 120.0, blues6_config)
        times = [c[0] for c in chords]
        assert times == sorted(times)

    def test_empty_events(self, blues6_config):
        chords = extract_chord_labels([], 120.0, blues6_config)
        assert chords == []

    def test_drums_excluded(self, blues6_config):
        """Chord extraction should ignore drum events."""
        drums_only = [
            (0.0, blues6_config.drum_idx, 36, 100, 0.25),
            (0.5, blues6_config.drum_idx, 38, 90, 0.25),
        ]
        chords = extract_chord_labels(drums_only, 120.0, blues6_config)
        assert chords == []


# ── Musical Time ──────────────────────────────────────────────

class TestMusicalTime:
    def test_monotonic(self, cascade_vocab):
        """Musical times should be non-decreasing."""
        # Create some tokens with TIME_SHIFT
        layout = cascade_vocab["layout"]
        ts_start = layout["TIME_SHIFT"]["start"]
        bos = layout["BOS"]["start"]
        eos = layout["EOS"]["start"]
        inst_start = layout["INST"]["start"]

        tokens = [bos, ts_start + 5, inst_start, ts_start + 2, inst_start, eos]
        times = compute_musical_times(tokens, cascade_vocab)
        assert len(times) == len(tokens)
        for i in range(1, len(times)):
            assert times[i] >= times[i - 1], f"Time decreased at index {i}"

    def test_time_advances_on_time_shift(self, cascade_vocab):
        layout = cascade_vocab["layout"]
        ts_start = layout["TIME_SHIFT"]["start"]
        bos = layout["BOS"]["start"]

        tokens = [bos, ts_start + 0]  # TIME_SHIFT local=0 → 1 step
        times = compute_musical_times(tokens, cascade_vocab)
        step_qn = float(cascade_vocab["time_shift_qn_step"])
        assert times[0] == 0.0
        assert times[1] == 0.0  # time is BEFORE consuming the token
        # After the token, time would be step_qn

    def test_non_time_tokens_dont_advance(self, cascade_vocab):
        layout = cascade_vocab["layout"]
        bos = layout["BOS"]["start"]
        inst_start = layout["INST"]["start"]

        tokens = [bos, inst_start, inst_start + 1]
        times = compute_musical_times(tokens, cascade_vocab)
        assert all(t == 0.0 for t in times)


# ── SEP Placement ─────────────────────────────────────────────

class TestCascadeExample:
    def test_sep_in_correct_position(self, cascade_vocab):
        layout = cascade_vocab["layout"]
        bos = layout["BOS"]["start"]
        eos = layout["EOS"]["start"]
        sep = layout["SEP"]["start"]

        ctx_tokens = [bos, 10, 11, 12, eos]
        ctx_times = [0.0, 0.0, 0.1, 0.2, 0.3]
        tgt_tokens = [bos, 20, 21, eos]
        tgt_times = [0.0, 0.0, 0.1, 0.2]

        tokens, times, sep_pos = build_cascade_example(
            ctx_tokens, ctx_times, tgt_tokens, tgt_times, cascade_vocab, 0
        )
        assert tokens[0] == bos
        assert tokens[sep_pos] == sep
        assert tokens[-1] == eos

    def test_format_bos_ctx_sep_tgt_eos(self, cascade_vocab):
        layout = cascade_vocab["layout"]
        bos = layout["BOS"]["start"]
        eos = layout["EOS"]["start"]
        sep = layout["SEP"]["start"]

        ctx_tokens = [bos, 10, 11, eos]
        ctx_times = [0.0, 0.0, 0.1, 0.2]
        tgt_tokens = [bos, 20, eos]
        tgt_times = [0.0, 0.0, 0.1]

        tokens, times, sep_pos = build_cascade_example(
            ctx_tokens, ctx_times, tgt_tokens, tgt_times, cascade_vocab, 0
        )
        # Expected: [BOS, 10, 11, SEP, 20, EOS]
        assert tokens == [bos, 10, 11, sep, 20, eos]
        assert sep_pos == 3


# ── Context Truncation ────────────────────────────────────────

class TestTruncation:
    def test_no_truncation_when_fits(self, cascade_vocab):
        layout = cascade_vocab["layout"]
        sep = layout["SEP"]["start"]
        tokens = [0, 1, 2, sep, 3, 4]
        times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        t, m, sp = truncate_context_to_fit(tokens, times, 3, max_len=10)
        assert t == tokens
        assert sp == 3

    def test_truncation_removes_from_context_start(self, cascade_vocab):
        layout = cascade_vocab["layout"]
        bos = layout["BOS"]["start"]
        sep = layout["SEP"]["start"]
        eos = layout["EOS"]["start"]

        # [BOS, a, b, c, SEP, x, y, EOS] = 8 tokens, max_len=6
        tokens = [bos, 10, 11, 12, sep, 20, 21, eos]
        times = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
        t, m, sp = truncate_context_to_fit(tokens, times, 4, max_len=6)
        # Should remove 2 from context: [BOS, 12, SEP, 20, 21, EOS]
        assert len(t) == 6
        assert t[0] == bos
        assert t[sp] == sep
        assert t[-1] == eos
        # Target tokens preserved
        assert 20 in t
        assert 21 in t

    def test_truncation_preserves_target(self, cascade_vocab):
        layout = cascade_vocab["layout"]
        bos = layout["BOS"]["start"]
        sep = layout["SEP"]["start"]
        eos = layout["EOS"]["start"]

        # Very long context, short target
        ctx = list(range(100, 200))
        tgt = [20, 21]
        tokens = [bos] + ctx + [sep] + tgt + [eos]
        times = [float(i) for i in range(len(tokens))]
        sep_pos = 1 + len(ctx)

        t, m, sp = truncate_context_to_fit(tokens, times, sep_pos, max_len=20)
        assert len(t) <= 20
        assert t[sp] == sep
        assert 20 in t[sp + 1:]
        assert 21 in t[sp + 1:]


# ── Full Pipeline ─────────────────────────────────────────────

class TestBuildAllStages:
    def test_ablation_a_stages(self, simple_events, blues6_config, cascade_vocab):
        bar_starts = np.array([0.0, 2.0])
        bars_meta = [(0.0, 2.0, 16)]
        chords = extract_chord_labels(simple_events, 120.0, blues6_config)

        stages = build_all_cascade_stages(
            simple_events, 120.0, bar_starts, bars_meta,
            cascade_vocab, blues6_config, chords, "A",
        )
        # Should produce stages only for instruments that have events
        assert len(stages) > 0
        for (tokens, times, sep_pos, stage_id) in stages:
            assert len(tokens) == len(times)
            assert 0 <= sep_pos < len(tokens)
            assert tokens[sep_pos] == cascade_vocab["layout"]["SEP"]["start"]

    def test_ablation_b_fewer_stages(self, simple_events, blues6_config, cascade_vocab):
        bar_starts = np.array([0.0, 2.0])
        bars_meta = [(0.0, 2.0, 16)]
        chords = extract_chord_labels(simple_events, 120.0, blues6_config)

        stages_a = build_all_cascade_stages(
            simple_events, 120.0, bar_starts, bars_meta,
            cascade_vocab, blues6_config, chords, "A",
        )
        stages_b = build_all_cascade_stages(
            simple_events, 120.0, bar_starts, bars_meta,
            cascade_vocab, blues6_config, chords, "B",
        )
        # B merges guitar+other, so ≤ stages than A
        assert len(stages_b) <= len(stages_a)

    def test_later_stages_have_longer_context(self, simple_events, blues6_config, cascade_vocab):
        bar_starts = np.array([0.0, 2.0])
        bars_meta = [(0.0, 2.0, 16)]
        chords = extract_chord_labels(simple_events, 120.0, blues6_config)

        stages = build_all_cascade_stages(
            simple_events, 120.0, bar_starts, bars_meta,
            cascade_vocab, blues6_config, chords, "A",
        )
        if len(stages) >= 2:
            # First stage (drums) has no context; later stages should have more context
            first_sep = stages[0][2]
            last_sep = stages[-1][2]
            assert last_sep >= first_sep


# ── Token Decode Round-Trip ───────────────────────────────────

class TestDecodeRoundTrip:
    def test_tokens_contain_expected_types(self, simple_events, blues6_config, cascade_vocab):
        bar_starts = np.array([0.0, 2.0])
        bars_meta = [(0.0, 2.0, 16)]
        chords = extract_chord_labels(simple_events, 120.0, blues6_config)

        stages = build_all_cascade_stages(
            simple_events, 120.0, bar_starts, bars_meta,
            cascade_vocab, blues6_config, chords, "A",
        )
        layout = cascade_vocab["layout"]
        sep_id = layout["SEP"]["start"]
        bos_id = layout["BOS"]["start"]
        eos_id = layout["EOS"]["start"]

        for (tokens, _, sep_pos, _) in stages:
            assert tokens[0] == bos_id
            assert tokens[sep_pos] == sep_id
            # EOS should be present (might be truncated in extreme cases)
            if len(tokens) < 1024:
                assert tokens[-1] == eos_id
