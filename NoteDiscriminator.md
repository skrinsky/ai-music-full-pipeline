# Note Discriminator

## Problem Statement

The audio-to-MIDI pipeline (via basic-pitch) over-generates notes, particularly for guitar, bass, and synth ("other") stems. Polyphonic transients, string noise, pick clicks, and harmonic artifacts are frequently detected as separate note events when they are not musically intended. The current approach is rule-based (velocity / duration thresholds), which is brittle across different recording styles and timbres.

## Why a Learned Discriminator

A learned discriminator can exploit richer context than scalar thresholds: it can see whether a candidate note is consistent with the surrounding polyphony, whether its duration is typical for this stem, and whether the detection confidence (amplitude) is high relative to other simultaneous notes. This generalizes better than hard-coded cutoffs while remaining cheap to run (12 scalar features, a 3-layer MLP).

## Data Generation Approach

Because labelled real-recording data is scarce, we synthesize paired data:

1. **Source MIDIs** — GigaMIDI blues MIDIs from `data/blues_midi/` provide ground-truth note events.
2. **GM program sweep** — Each stem is re-programmed across 8 representative GM programs (e.g. guitar: nylon → harmonics, bass: acoustic → synth bass). This gives timbral diversity from a single MIDI corpus.
3. **FluidSynth rendering** — The per-stem MIDI is rendered to 44 kHz WAV via FluidSynth + FluidR3_GM.sf2.
4. **Augmentation** — 8 variants per render:
   - `clean` — unprocessed
   - `dist_light` (6 dB gain, soft-clip) — light overdrive
   - `dist_crunch` (18 dB) — mid-gain crunch
   - `dist_heavy` (35 dB) — heavy fuzz; **distortion smears onsets and creates false-positive detections**, so explicit coverage here is important
   - `reverb_room` (RT60=0.3 s) — tight room
   - `reverb_hall` (RT60=1.2 s) — large hall
   - `dist_light+reverb_room` — combined
   - `dist_heavy+reverb_hall` — worst-case combined
5. **basic-pitch detection** — `predict()` on each augmented WAV produces candidate note events.
6. **Greedy GT alignment** — Each detected note is matched to the closest ground-truth note within ±1 semitone and ±50 ms onset; matched → TP (label 1), unmatched → FP (label 0).
7. **Feature extraction** — 12 scalar features are computed and written to `runs/discriminator_data/notes.h5`.

## Model Architecture

- **Input**: 12 scalar features per note (no audio, no spectrogram)
- **Architecture**: LayerNorm → Linear(12→64) → ReLU → Dropout(0.3) → Linear(64→32) → ReLU → Linear(32→1)
- **Loss**: BCEWithLogitsLoss with `pos_weight` to handle TP/FP imbalance
- **Decision threshold**: 0.35 (kept lower than 0.5 to favour recall; tune on your data)
- **Output at inference**: `P(TP)` via sigmoid; events with `P(TP) < threshold` are dropped

### Features

| # | Name | Description |
|---|------|-------------|
| 0 | `amplitude` | basic-pitch confidence proxy (0–1); velocity/127 when derived from events |
| 1 | `duration_s` | note duration in seconds |
| 2 | `pitch` | MIDI pitch number (0–127) |
| 3 | `stem_id` | local stem index: 0=guitar, 1=bass, 2=other |
| 4 | `polyphony` | number of notes sounding at this note's onset |
| 5 | `density_100ms` | notes starting within ±50 ms of this onset |
| 6 | `octave_rank` | rank among simultaneously sounding notes (0=lowest) |
| 7 | `duration_zscore` | (duration − stem_mean_dur) / stem_std_dur |
| 8 | `pitch_rel` | (pitch − stem_mean_pitch) / stem_std_pitch |
| 9 | `hi_conf_flag` | 1 if amplitude > 0.7 |
| 10 | `short_flag` | 1 if duration < 0.05 s |
| 11 | `hi_poly_flag` | 1 if polyphony > 4 |

## Known Limitation: Distribution Shift

The discriminator is trained on FluidSynth renders of MIDI and applied to real recordings. Real guitar recordings contain pick noise, fret buzz, and room reflections not present in synthesis. The augmentation suite (especially heavy distortion and reverb variants) partially bridges this gap by training on timbres that share spectral properties with real instruments, and by using features that are more timbre-invariant than raw audio (duration statistics, polyphony context, pitch rank).

This shift is mitigated but not eliminated. Expect some degradation when applied to recordings with unusual timbres not covered by the GM program sweep.

## Future Work

Integrate a spectrogram patch from the Demucs stem audio alongside the 12 scalar features. A small convolutional encoder on the 50 ms window centred at each detected onset would provide richer timbral evidence (e.g. detecting whether a transient looks like pick noise vs a genuine note). This would require running the discriminator at the audio-to-MIDI conversion stage (inside `vendor/all-in-one-ai-midi-pipeline/`) rather than at `pre.py` time, because the stem audio is not available by the time `pre.py` runs.

## How to Run

### 1. Build training data

```bash
# Requires: fluidsynth on PATH, FluidR3_GM.sf2, blues MIDIs in data/blues_midi/
python scripts/build_discriminator_data.py \
    --midi_dir data/blues_midi \
    --out runs/discriminator_data/notes.h5 \
    --n_midis 50 \
    --workers 4
```

Override SF2 path if auto-detection fails:
```bash
python scripts/build_discriminator_data.py --sf2 /path/to/FluidR3_GM.sf2 ...
```

### 2. Train the discriminator

```bash
python -m training.train_discriminator \
    --data runs/discriminator_data/notes.h5 \
    --out  runs/discriminator/model.pt \
    --epochs 60 \
    --threshold 0.35
```

### 3. Use during preprocessing

```bash
python training/pre.py \
    --midi_folder data/blues_midi \
    --data_folder runs/blues_events \
    --discriminator runs/discriminator/model.pt \
    --disc_threshold 0.35
```

Pre.py will print a summary of how many notes were filtered per song at the end.
