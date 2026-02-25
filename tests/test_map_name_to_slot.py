"""Unit tests for map_name_to_slot() and _slot_from_gm_program()."""

import pytest
import pretty_midi

from training.pre import (
    INSTRUMENT_NAMES,
    DRUM_IDX,
    OTHER_IDX,
    GUITAR_IDX,
    BASS_IDX,
    VOXLEAD_IDX,
    VOXHARM_IDX,
    make_instrument_config,
    INSTRUMENT_PRESETS,
    map_name_to_slot,
    _slot_from_gm_program,
)


def _make_inst(name: str = "", program: int = 0, is_drum: bool = False) -> pretty_midi.Instrument:
    inst = pretty_midi.Instrument(program=program, is_drum=is_drum, name=name)
    return inst


# ── Default blues6 config (backward compat globals) ──

def test_legacy_globals():
    """Legacy globals still point to blues6."""
    assert INSTRUMENT_NAMES == ["voxlead", "voxharm", "guitar", "other", "bass", "drums"]
    assert DRUM_IDX == 5
    assert GUITAR_IDX == 2
    assert OTHER_IDX == 3
    assert BASS_IDX == 4
    assert VOXLEAD_IDX == 0
    assert VOXHARM_IDX == 1


# ── Drum detection ──

def test_drum_by_is_drum_flag():
    inst = _make_inst(name="", program=0, is_drum=True)
    assert map_name_to_slot(inst) == DRUM_IDX


def test_drum_by_name():
    for name in ("Drums", "kick", "HiHat", "snare_track", "Percussion"):
        inst = _make_inst(name=name, program=0, is_drum=False)
        assert map_name_to_slot(inst) == DRUM_IDX, f"Failed for name={name!r}"


# ── Name-based instrument matching ──

def test_guitar_by_name():
    inst = _make_inst(name="Electric Guitar", program=0)
    assert map_name_to_slot(inst) == GUITAR_IDX


def test_bass_by_name():
    inst = _make_inst(name="Bass", program=0)
    assert map_name_to_slot(inst) == BASS_IDX


def test_voxlead_by_name():
    inst = _make_inst(name="VoxLead", program=0)
    assert map_name_to_slot(inst) == VOXLEAD_IDX


def test_voxharm_by_name():
    inst = _make_inst(name="VoxHarm", program=0)
    assert map_name_to_slot(inst) == VOXHARM_IDX


# ── GM program fallback (empty names) ──

def test_empty_name_guitar_prog():
    inst = _make_inst(name="", program=25)  # Acoustic Guitar (steel)
    assert map_name_to_slot(inst) == GUITAR_IDX


def test_empty_name_bass_prog():
    inst = _make_inst(name="", program=33)  # Electric Bass (finger)
    assert map_name_to_slot(inst) == BASS_IDX


def test_empty_name_piano_prog():
    inst = _make_inst(name="", program=0)   # Acoustic Grand Piano
    assert map_name_to_slot(inst) == OTHER_IDX


def test_empty_name_voxlead_prog():
    inst = _make_inst(name="", program=52)  # Choir Aahs → voxlead
    assert map_name_to_slot(inst) == VOXLEAD_IDX


def test_empty_name_voxharm_prog():
    inst = _make_inst(name="", program=53)  # Voice Oohs → voxharm
    assert map_name_to_slot(inst) == VOXHARM_IDX


# ── Priority: name beats program ──

def test_name_takes_priority_over_prog():
    """If name says 'guitar' but prog says bass (33), name wins."""
    inst = _make_inst(name="Guitar Solo", program=33)
    assert map_name_to_slot(inst) == GUITAR_IDX


# ── Unknown program falls to other ──

def test_unknown_prog_falls_to_other():
    inst = _make_inst(name="", program=110)  # not in any GM range we map
    assert map_name_to_slot(inst) == OTHER_IDX


# ── _slot_from_gm_program direct tests ──

def test_gm_prog_guitar_range():
    for prog in range(24, 32):
        assert _slot_from_gm_program(prog) == GUITAR_IDX, f"prog={prog}"


def test_gm_prog_bass_range():
    for prog in range(32, 40):
        assert _slot_from_gm_program(prog) == BASS_IDX, f"prog={prog}"


def test_gm_prog_returns_none_for_unmapped():
    # prog 100 is not in any explicit range
    assert _slot_from_gm_program(100) is None


# ── Chorale4 config: exact name matching ──

class TestChorale4NameMapping:
    @pytest.fixture
    def config(self):
        return make_instrument_config(INSTRUMENT_PRESETS["chorale4"])

    def test_soprano_exact_match(self, config):
        inst = _make_inst(name="soprano")
        assert map_name_to_slot(inst, config) == 0

    def test_bassvox_exact_match(self, config):
        inst = _make_inst(name="bassvox")
        assert map_name_to_slot(inst, config) == 3

    def test_unknown_name_raises(self, config):
        inst = _make_inst(name="guitar")
        with pytest.raises(ValueError, match="not in config.names"):
            map_name_to_slot(inst, config)

    def test_case_insensitive(self, config):
        inst = _make_inst(name="Soprano")
        assert map_name_to_slot(inst, config) == 0
