"""Tests for InstrumentConfig, presets, and aux_dim calculation."""

import pytest

from training.pre import (
    InstrumentConfig,
    make_instrument_config,
    compute_aux_layout,
    INSTRUMENT_PRESETS,
)


class TestInstrumentPresets:
    def test_blues6_names(self):
        config = make_instrument_config(INSTRUMENT_PRESETS["blues6"])
        assert config.names == ["voxlead", "voxharm", "guitar", "other", "bass", "drums"]
        assert config.num_instruments == 6

    def test_chorale4_names(self):
        config = make_instrument_config(INSTRUMENT_PRESETS["chorale4"])
        assert config.names == ["soprano", "alto", "tenor", "bassvox"]
        assert config.num_instruments == 4

    def test_blues6_drum_idx(self):
        config = make_instrument_config(INSTRUMENT_PRESETS["blues6"])
        assert config.drum_idx == 5
        assert config.has_drums()

    def test_chorale4_no_drums(self):
        config = make_instrument_config(INSTRUMENT_PRESETS["chorale4"])
        assert config.drum_idx is None
        assert not config.has_drums()

    def test_blues6_role_indices(self):
        config = make_instrument_config(INSTRUMENT_PRESETS["blues6"])
        assert config.guitar_idx == 2
        assert config.other_idx == 3
        assert config.bass_idx == 4
        assert config.voxlead_idx == 0
        assert config.voxharm_idx == 1

    def test_chorale4_no_guitar_other(self):
        config = make_instrument_config(INSTRUMENT_PRESETS["chorale4"])
        assert config.guitar_idx is None
        assert config.other_idx is None
        assert config.bass_idx is None

    def test_chorale4_voice_ranges(self):
        config = make_instrument_config(INSTRUMENT_PRESETS["chorale4"])
        assert "soprano" in config.voice_ranges
        assert config.voice_ranges["soprano"] == (57, 84)
        assert config.voice_ranges["bassvox"] == (33, 69)

    def test_chorale4_augmentation_smaller_range(self):
        config = make_instrument_config(INSTRUMENT_PRESETS["chorale4"])
        assert max(config.aug_transposes) <= 3
        assert min(config.aug_transposes) >= -3

    def test_chorale4_no_velocity_augmentation(self):
        config = make_instrument_config(INSTRUMENT_PRESETS["chorale4"])
        assert config.aug_vel_deltas == []


class TestAuxLayout:
    def test_blues6_aux_dim(self):
        config = make_instrument_config(INSTRUMENT_PRESETS["blues6"])
        layout = compute_aux_layout(config)
        # 6+6+6 + 4 + 12 + 1 + 1 = 36
        assert layout["aux_dim"] == 36
        assert layout["has_chords"]
        assert layout["has_swing_blues"]

    def test_chorale4_aux_dim(self):
        config = make_instrument_config(INSTRUMENT_PRESETS["chorale4"])
        layout = compute_aux_layout(config)
        # 4+4+4 + 12 = 24
        assert layout["aux_dim"] == 24
        assert not layout["has_chords"]
        assert not layout["has_swing_blues"]

    def test_custom_3_inst_with_drums(self):
        config = make_instrument_config(["guitar", "bass", "drums"])
        layout = compute_aux_layout(config)
        # 3+3+3 + 12 + 1 + 1 = 23  (no guitar+other pair means no chords)
        # Actually guitar is present but other is not, so has_chords = False
        assert not layout["has_chords"]
        assert layout["has_swing_blues"]
        assert layout["aux_dim"] == 3 + 3 + 3 + 12 + 1 + 1

    def test_custom_with_guitar_and_other(self):
        config = make_instrument_config(["guitar", "other", "drums"])
        layout = compute_aux_layout(config)
        assert layout["has_chords"]
        assert layout["has_swing_blues"]
        assert layout["aux_dim"] == 3 + 3 + 3 + 4 + 12 + 1 + 1


class TestMakeInstrumentConfig:
    def test_unknown_names_no_roles(self):
        config = make_instrument_config(["voice1", "voice2"])
        assert config.drum_idx is None
        assert config.guitar_idx is None
        assert config.other_idx is None
        assert config.num_instruments == 2
