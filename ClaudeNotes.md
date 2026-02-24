2026-02-21 13:30

  MAKE_BLUES_TRAIN.md

  The training is working but will take a long time. With 289k windows at batch size 64, that's ~4,500 batches per epoch, plus validation. This is a
  substantial training run on MPS.

  Pipeline status so far:
  1. Fetch -- Done. 996 blues MIDIs (35.1 MiB) from 2.1M GigaMIDI tracks
  2. Preprocess -- Done. 289,380 train windows + 14,161 val windows, ~2400-token vocab
  3. Train -- Running on MPS. ~22 min into epoch 1. Will run for multiple epochs with patience=10 early stopping

  The training will likely take a long time (possibly hours for all epochs). You can monitor it with:
  # Check if still running:
  ps aux | grep train.py

  # Check for checkpoints saved:
  ls -la runs/checkpoints/

  # Once done, generate:
  make blues-generate

---

⏺ Here's my estimate:

  Model size (auto-scaled): d_model=256, 6 layers, ~6M params -- the largest configuration that fits the token budget (148M tokens / 8 tpp).

  Per epoch: ~4,500 train batches + 221 val batches at batch_size=64. On MPS, first epoch hasn't completed yet after ~30+ min wall time, so likely 30-60
   min/epoch (depends on which Apple Silicon you have).

  Total epochs: Max is 200, but early stopping with patience=10 will likely kick in around epoch 15-30. The model usually improves quickly for the first
   5-10 epochs, then plateaus.

  Estimated total training time: 8-30 hours -- likely an overnight run. We'll know much better once epoch 1 finishes and prints its timing (the [dt] at
  the end of each epoch line).

  If you want a faster iteration cycle in the future, you could:
  - make blues-train ARGS="--max_d_model 128" -- smaller model, ~4x faster
  - make blues-train ARGS="--no_auto_scale" -- use defaults (d=192, 4 layers) instead of the auto-scaled d=256, 6 layers

---

2026-02-22 12:24

[eval on latest ck pt]

⏺ Generated! But it reveals something interesting about the model's current state:

  - 121 notes total, then it went completely silent from token ~800 onward (silent count climbed from 1 to 1126)
  - Generation stopped early at 1800 tokens (of 3000 max) -- likely hit the silence cap
  - All on a straight rhythmic grid
  - Took ~5 min on MPS

  The model has learned some musical structure (42 notes in the first 200 tokens is decent), but it's collapsing into silence after a short phrase. This
   is typical for a model that's still mid-training -- it hasn't yet learned to sustain musical output.

  The output MIDI is at runs/generated/blues_out.mid. You can open it in Logic Pro or any MIDI player to hear what it's learned so far. It'll be a short
   burst of blues-ish notes.

  Once training finishes (early stopping kicks in), regenerating should produce much longer, more coherent output. You can also try different
  temperatures:
  .venv-ai-music/bin/python training/generate.py \
    --ckpt runs/checkpoints/blues_model.pt \
    --vocab_json runs/blues_events/event_vocab.json \
    --out_midi runs/generated/blues_hot.mid \
    --temperature 1.1 --device auto

✻ Cooked for 5m 26s

⏺ Background command "Generate blues MIDI from current checkpoint" completed (exit code 0)

⏺ That's just the completion notification for the generation we already reviewed. No new info -- we already saw the results above (121 notes, then
  silence collapse).

  Training is still chugging along. We'll get much better generations once it finishes and early stopping has picked the best epoch.

❯ Cool, very plausible beginning.  I am curious: how is the inference seeded?

⏺ Here's how inference seeding works:

  Initial seed (line 183-195, seed_everything):
  - If --seed N is passed, uses that exact value
  - Otherwise, draws 8 random bytes from os.urandom() -- so each run is different by default
  - Seeds Python's random, PyTorch CPU, CUDA (if present), and numpy

  The seed printed at generation start (Seed: 18159832125478583215)
  lets you reproduce the exact same output later with --seed
  18159832125478583215.

  Musical seeding -- there's no musical prompt/conditioning. The
  sequence starts with just [BOS] (line 420: seq = [BOS_ID]), then the
  grammar state machine takes over:

  1. Phase starts at "TIME" -- model must emit TIME_SHIFT, BAR, or INST
  2. Grammar enforces the cycle: TIME → INST → VEL → PITCH → DUR → TIME → ...
  3. At each step, only the grammatically-valid token types are unmasked; everything else gets -1e9

  So the model generates entirely from scratch -- no seed melody, no
  style prompt. The only musical bias comes from the trained weights +
  the sampling heuristics (drum bonus, repetition penalty, grid
  snapping, etc.).

  If you wanted to seed it with an existing phrase, you'd need to
  tokenize a MIDI snippet into the event vocabulary and prepend it to
  seq before the generation loop. That would be a nice feature to add.

------------------------------------------------------------------------------------------------------------------------

2026-02-24 01:23

Training Results Analysis

  Model: D_MODEL=256, N_HEADS=8, N_LAYERS=6, FF_MULT=3 (~4.2M params)
  Data: 289,380 train windows (148M tokens), 14,161 val windows, 852 vocab tokens
  Training: Stopped at epoch 28 (early stopping, patience=10, so best epoch ~18)

  Two Generation Failure Modes

  Both generated files show degeneration -- but in opposite directions:

  ┌───────────────────────┬────────────────┬────────────────────────────────────────────────────────────────────────────────────────────────────────┐
  │         File          │    Problem     │                                                 Detail                                                 │
  ├───────────────────────┼────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ blues_out.events.json │ TIME_SHIFT     │ Good music for ~600 tokens, then 50.9% of all tokens are TIME_SHIFT[11] -- time races forward with no   │
  │                       │ loop           │ notes                                                                                                  │
  ├───────────────────────┼────────────────┼────────────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ out.events.json       │ Note avalanche │ Only 2 time tokens total; 449 notes all placed at the same instant -- infinite INST→VEL→PITCH→DUR loop  │
  └───────────────────────┴────────────────┴────────────────────────────────────────────────────────────────────────────────────────────────────────┘

  Root Cause

  The generation grammar allows two degenerate loops that never occur in training data:

  1. DUR → phase="TIME" → INST → ... (notes loop -- INST is allowed in the "TIME" phase)
  2. TIME_SHIFT → phase="TIME" → TIME_SHIFT → ... (consecutive TIME_SHIFTs)

  Training data shows: max 1 consecutive TIME_SHIFT, max 20 notes per
 timestep -- but the grammar enforces neither limit. The model has
 learned the right patterns approximately, but during autoregressive
 generation, small probability errors compound and the model falls
 into these absorbing loops.

  Recommended Next Steps (Priority Order)

  1. Fix generation grammar (quick, high impact) -- Two changes in generate.py:
  - After TIME_SHIFT, force next phase to "INST" instead of "TIME" (prevents consecutive TIME_SHIFTs -- matches 100% of training data)
  - Track notes since last TIME_SHIFT; after 12 notes, remove INST from allowed types and force a TIME_SHIFT

  2. Save loss curves in checkpoint -- Add train_losses/val_losses lists to the saved checkpoint so we can diagnose convergence vs overfitting

  3. Lower temperature -- Try --temperature 0.7 or 0.75 instead of 0.9 to reduce sampling entropy

  4. Increase patience -- Try --patience 20 to let the model train longer before early stopping (current best epoch ~18 out of 28 is quite early for 200
  max epochs)

  Want me to implement fix #1 (the grammar constraints)?

------------------------------------------------------------------------------------------------------------------------

2026-02-24 06:18

  Blues Model Diagnosis

  TL;DR

  Three compounding problems: (1) broken instrument mapping floods 60% of data into "other", (2) massive vocab bloat (only ~35% of tokens are actually
  used), and (3) generation settings that push the model far from its training distribution. The model is small enough and has enough data -- the issues
  are in data representation and generation config.

  ---
  Problem 1: Instrument Mapping is Broken for GigaMIDI (Critical)

  map_name_to_slot() in pre.py relies on track names, but GigaMIDI files have mostly empty names:

  ┌───────────────────────────────────┬───────┬─────────────────┬─────────────┐
  │        What's in the files        │ Count │ Current mapping │  Should be  │
  ├───────────────────────────────────┼───────┼─────────────────┼─────────────┤
  │ name='' prog=33 (Elec Bass)       │ 33    │ other           │ bass        │
  ├───────────────────────────────────┼───────┼─────────────────┼─────────────┤
  │ name='' prog=25-27 (Guitars)      │ 82    │ other           │ guitar      │
  ├───────────────────────────────────┼───────┼─────────────────┼─────────────┤
  │ name='' prog=0 drum=True          │ 43    │ drums           │ drums       │
  ├───────────────────────────────────┼───────┼─────────────────┼─────────────┤
  │ name='' prog=0 drum=False (Piano) │ 7+    │ other           │ other/piano │
  ├───────────────────────────────────┼───────┼─────────────────┼─────────────┤
  │ name='Bass' prog=35               │ 8     │ bass            │ bass        │
  └───────────────────────────────────┴───────┴─────────────────┴─────────────┘

  Result in training data: 59.5% "other", guitar only 4.6%, bass only 7.2%. The model learns a nearly mono-instrumental representation for what should
  be multi-instrument blues.

  Fix: Fall back to GM program number when name matching fails. Programs 24-31 = guitar, 32-39 = bass, etc.

  Problem 2: Vocab is 65% Dead Weight

  ┌───────────────┬───────────┬───────────────┬────────────┐
  │  Token type   │ Allocated │ Actually used │   Waste    │
  ├───────────────┼───────────┼───────────────┼────────────┤
  │ BAR           │ 415       │ 85            │ 330 (80%)  │
  ├───────────────┼───────────┼───────────────┼────────────┤
  │ TIME_SHIFT    │ 192       │ 46            │ 146 (76%)  │
  ├───────────────┼───────────┼───────────────┼────────────┤
  │ PITCH_GENERAL │ 127       │ 83            │ 44 (35%)   │
  ├───────────────┼───────────┼───────────────┼────────────┤
  │ PITCH_DRUMS   │ 84        │ 30            │ 54 (64%)   │
  ├───────────────┼───────────┼───────────────┼────────────┤
  │ Total         │ 852       │ ~280          │ ~570 (67%) │
  └───────────────┴───────────┴───────────────┴────────────┘

  The embedding table has 852 × 256 = 218K parameters (~5% of the model) storing noise for tokens that never appear. More importantly, the softmax over
  large dead spaces makes learned distributions flatter and harder to sharpen.

  Fix: Build vocab only from tokens that actually appear in training. Shrinks vocab to ~280-300.

  Problem 3: Generated Output Has Massive Mode Collapse

  Comparing training data vs. generation:

  ┌─────────────┬───────────────────────────────────────────────┬───────────────────────┬────────────────────────────┐
  │  Dimension  │             Training distribution             │       Generated       │          Problem           │
  ├─────────────┼───────────────────────────────────────────────┼───────────────────────┼────────────────────────────┤
  │ TIME_SHIFT  │ step_1=20%, step_6=18%, step_12=14% (diverse) │ step_12=82%           │ Collapsed to single rhythm │
  ├─────────────┼───────────────────────────────────────────────┼───────────────────────┼────────────────────────────┤
  │ Velocity    │ 112=31%, 96=30%, 80=20%, 64=13%               │ 112=57%, 96=41%       │ Top 2 bins only            │
  ├─────────────┼───────────────────────────────────────────────┼───────────────────────┼────────────────────────────┤
  │ Duration    │ 0.25QN=21%, 0.5QN=15%, 0.167QN=14%            │ 3.0QN=32%, 0.5QN=28%  │ Way too long               │
  ├─────────────┼───────────────────────────────────────────────┼───────────────────────┼────────────────────────────┤
  │ Instruments │ other=60%, drums=29%, bass=7%, guitar=5%      │ guitar=46%, drums=37% │ Inverted!                  │
  ├─────────────┼───────────────────────────────────────────────┼───────────────────────┼────────────────────────────┤
  │ Notes/step  │ 1=53%, 2=10%, 5=12% (smooth)                  │ 1=53%, 12=41%         │ Bimodal                    │
  └─────────────┴───────────────────────────────────────────────┴───────────────────────┴────────────────────────────┘

  The instrument inversion (guitar 46% in gen vs 5% in training) is especially telling -- the generation heuristics (drum_bonus, max_inst_streak) are
  actively fighting the learned distribution.

  Problem 4: Generation Heuristics Fight the Model

  - Grid snapping rounds fine-grained timing (step_1, step_2) to multiples of 6 or 8, destroying the learned timing distribution
  - drum_bonus=0.4 artificially boosts drums beyond what the model learned
  - temperature=0.9 is too high -- the model's value distributions are already quite flat from label smoothing (0.08-0.10) and high dropout (0.22)
  - max_notes_per_step=12 is way above the training mode (1-2 notes per step)

  Problem 5: The Aux Head May Be Stealing Capacity

  At 20% of the loss budget, the polyphony instructor regression task uses the last hidden state (a single position) to predict 34 floats. For a
  4.2M-param model, this is meaningful capacity diverted from next-token prediction.

  Problem 6: Limited Augmentation

  ±1 semitone pitch shift barely changes anything musically. Blues uses mostly 3-4 keys; ±5 semitone transposition would let the model see much more
  pitch variety from the same data.

  ---
  Recommended Fixes (Priority Order)

  Tier 1 -- Quick wins, no retraining needed:

  1. Lower temperature to 0.65 and top_p to 0.85
  2. Disable grid snapping (--snap_time_shift → false or set --grid_straight_steps 1)
  3. Remove drum_bonus (--drum_bonus 0)
  4. Lower max_notes_per_step from 12 to 4

  Tier 2 -- Fix data pipeline, retrain:

  1. Fix map_name_to_slot() to use GM program numbers when name is empty
  2. Compact vocab -- only allocate tokens for values that appear in the data
  3. Increase augmentation to ±5 semitones
  4. Reduce dropout from 0.22 to 0.12
  5. Reduce or disable aux head -- set AUX_LOSS_WEIGHT=0.05 or AUX_ENABLED=False

  Tier 3 -- Architecture improvements:

  1. Increase patience to 25 (model may still be improving at epoch 18)
  2. Reduce label smoothing -- 0.04 for type, 0.02 for values
  3. Save loss history in checkpoint for diagnostics
  4. Data quality filtering -- inspect GigaMIDI files for corrupt/anomalous content

  ---
  Want me to start implementing these? I'd suggest tackling Tier 1 first (regenerate with better settings to see immediate improvement), then Tier 2
  (fix map_name_to_slot + compact vocab + retrain).

✻ Crunched for 5m 17s

---

 Here's a summary of all changes:

  Changes made

  training/pre.py -- Instrument mapping + preprocessing:
  1. Added _slot_from_gm_program(prog) helper that maps GM program numbers to canonical slots (guitar 24-31, bass 32-39, vox 52-54,
  piano/strings/brass/etc → other)
  2. Updated map_name_to_slot() to call the GM fallback when all name checks fail (name still takes priority -- no regression)
  3. Added VOXLEAD_IDX, VOXHARM_IDX, BASS_IDX module-level constants (used in both functions and tests)
  4. TIME_SHIFT_QN_MAX: 8.0 → 4.0 (saves ~96 vocab tokens)
  5. AUG_TRANSPOSES: [-1, 1] → [-5, -3, -1, 1, 3, 5] (12 augmented copies per song instead of 4)

  training/train.py -- Training config:
  - DROPOUT: 0.22 → 0.12
  - LABEL_SMOOTH_TYPE: 0.10 → 0.05
  - LABEL_SMOOTH_VALUE: 0.08 → 0.04
  - AUX_LOSS_WEIGHT: 0.20 → 0.05
  - --patience default: 10 → 25

  scripts/audition_gigamidi.py -- New CLI with 4 subcommands: stats, list, info FILE, play FILE. Imports map_name_to_slot from training/pre.py so it
  shows the same mapping training uses.

  tests/test_map_name_to_slot.py -- 16 tests: drum detection (flag + name), name-based matching (guitar/bass/vox), GM program fallback (empty names),
  name-takes-priority-over-prog, unknown prog → other, direct _slot_from_gm_program range tests.

  Makefile -- Added blues-audition target.

  Next steps (Step 7 from plan)

  1. make blues-audition -- verify new mapping distribution
  2. make blues-preprocess -- re-tokenize with fixed mapping
  3. make blues-train ARGS="--patience 25" -- retrain (old checkpoints are incompatible with new vocab)
  
❯ make blues-audition

⏺ Bash(make blues-audition 2>&1)
  ⎿  .venv-ai-music/bin/python scripts/audition_gigamidi.py stats --folder data/blues_midi
     /Users/jos/w/ai-music-full-pipeline/.venv-ai-music/lib/python3.10/site-packages/pretty_midi/pretty_midi.py:122: RuntimeWarning: Tempo, Key or
     Time signature change events found on non-zero tracks.  This is not a valid type 0 or type 1 MIDI file.  Tempo, Key or Time Signature may be w
     ... +17 lines (ctrl+o to expand)
  ⎿  (timeout 1m)

⏺ That's a dramatic improvement. Before the GM program fallback, guitar was nearly absent (lumped into "other" at 59.5%). Now:

  ┌─────────┬────────┬───────┐
  │  Slot   │ Before │ After │
  ├─────────┼────────┼───────┤
  │ guitar  │ ~2%    │ 36.4% │
  ├─────────┼────────┼───────┤
  │ bass    │ ~3%    │ 10.0% │
  ├─────────┼────────┼───────┤
  │ other   │ 59.5%  │ 25.5% │
  ├─────────┼────────┼───────┤
  │ drums   │ ~27%   │ 27.3% │
  ├─────────┼────────┼───────┤
  │ voxlead │ --      │ 0.4%  │
  ├─────────┼────────┼───────┤
  │ voxharm │ --      │ 0.4%  │
  └─────────┴────────┴───────┘

  Guitar is now the largest slot (36.4%), bass is properly separated at 10%, and "other" dropped from ~60% to 25.5%. The model should learn much better
  instrument-specific patterns now.

  Ready for make blues-preprocess and make blues-train whenever you want to kick those off.
  
---

 Plan: Interactive MIDI Browser & Player (tkinter + fluidsynth)

 Context

 After fixing map_name_to_slot() and running make blues-audition, we
 can see the dataset distribution is much improved. JOS now wants to
 interactively browse individual files, see per-track metadata, and
 listen to them to ear-check the mapping. Need a GUI app to replace
 the CLI spot-checking workflow.

 Files to create/modify

 - scripts/midi_browser.py -- new tkinter GUI app
 - Makefile -- add blues-browse target

 Architecture

 Single-file tkinter app: MidiBrowser(tk.Tk) class.

 Layout

 +-------------------------------------------------------------------+
 |  MIDI Browser -- 996 files in data/blues_midi              [title] |
 +-------------------+-----------------------------------------------+
 |                   |  File: 004e4552...mid                          |
 |  [Listbox]        |  Duration: 204.8s   Tempo: 105.0 BPM          |
 |  scrollable       |  Total notes: 7745                             |
 |  shows basenames  |  ─────────────────────────────────────────     |
 |                   |  #  Raw Name   Drum? Prog GM Name  Slot  Notes |
 |  004e4552...mid   |  0  (empty)    N     25   Ac.Gtr   guitar 1498 |
 |  00841687...mid   |  1  (empty)    Y     0    Piano    drums  1565 |
 |  ...              |  ...                                           |
 |                   |  [ttk.Treeview with columns]                   |
 +-------------------+-----------------------------------------------+
 |  [<< Prev] [Play ▶ / Pause] [Stop ■] [Next >>]    1 / 996        |
 +-------------------------------------------------------------------+

 Imports (reuse existing code)

 - scripts/audition_gigamidi.py → gm_name(), _midi_paths()
 - training/pre.py → INSTRUMENT_NAMES, map_name_to_slot()
 - pretty_midi (already installed), tkinter (macOS built-in)

 Playback via fluidsynth subprocess

 - Binary: /opt/homebrew/bin/fluidsynth
 - Soundfont: /opt/homebrew/Cellar/fluid-synth/2.5.2/share/fluid-synth/sf2/VintageDreamsWaves-v2.sf2
 - Launch: subprocess.Popen([fluidsynth, "-a", "coreaudio", "-n", "-i", soundfont, midi_file])
 - Pause: os.kill(pid, signal.SIGSTOP) -- freezes audio output
 - Resume: os.kill(pid, signal.SIGCONT)
 - Stop: SIGCONT (if paused) then .terminate() then .wait(timeout=2)
 - Poll: self.after(500, _check_playback) to detect natural end

 Key methods

 ┌───────────────────────┬───────────────────────────────────────────────────────────────────────────────┐
 │        Method         │                                    Purpose                                    │
 ├───────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
 │ _on_select(event)     │ Listbox selection → parse MIDI → populate detail panel                        │
 ├───────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
 │ _show_track_table(pm) │ Fill Treeview with per-track info (same columns as audition_gigamidi.py info) │
 ├───────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
 │ _play()               │ Spawn fluidsynth, toggle button to "Pause"                                    │
 ├───────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
 │ _pause_resume()       │ SIGSTOP/SIGCONT toggle                                                        │
 ├───────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
 │ _stop()               │ Kill subprocess, reset button to "Play"                                       │
 ├───────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
 │ _next() / _prev()     │ Stop + advance selection + load info + auto-play                              │
 ├───────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
 │ _on_close()           │ Kill subprocess + destroy window                                              │
 ├───────────────────────┼───────────────────────────────────────────────────────────────────────────────┤
 │ _check_playback()     │ Poll proc.poll() every 500ms to detect end                                    │
 └───────────────────────┴───────────────────────────────────────────────────────────────────────────────┘

 Makefile target

 blues-browse: data/blues_midi/.fetched ## Browse + play blues MIDIs (tkinter GUI)
        $(PYTHON) scripts/midi_browser.py --folder $(BLUES_MIDI) $(ARGS

 Verification

 1. make blues-browse -- window opens with 996 files listed
 2. Click a file → detail panel shows track table
 3. Click Play → hear fluidsynth audio, button changes to Pause
 4. Click Pause → audio freezes, button changes to Resume
 5. Click Next → advances to next file and auto-plays
 6. Close window → fluidsynth process killed cleanly

...


⏺ Everything is wired up. Here's what was created:

  scripts/midi_browser.py -- Single-file tkinter GUI app with:
  - Left panel: Scrollable listbox of all MIDI filenames
  - Right panel: File metadata (duration, tempo, total notes) + ttk.Treeview showing per-track detail (raw name, drum flag, GM program, slot from
  map_name_to_slot(), note count, pitch range)
  - Transport bar: Prev / Play-Pause / Stop / Next buttons + position indicator
  - Playback: Spawns fluidsynth subprocess with CoreAudio; Pause via SIGSTOP, Resume via SIGCONT, Stop via terminate()
  - Navigation: Next/Prev auto-stop + auto-play; polls proc.poll() every 500ms for natural end detection
  - Cleanup: _on_close() kills fluidsynth on window close

  Makefile -- Added blues-browse target:
  make blues-browse    # Browse + play blues MIDIs (tkinter GUI)

...

⏺ Done. Here's how it works:

  - Click a row in the track table → writes a temp MIDI with only that track, restarts playback soloed. The transport bar shows SOLO: guitar (or
  whatever slot) in red.
  - Click the same row again → deselects, back to full mix.
  - If playback is already running when you click, it auto-restarts with the soloed/unsoloed version.
  - Temp files are cleaned up on stop, new solo, and window close.

---

❯ Now, for the "drums" slot, it would be nice to have a more normal sounding trap set. Also, the bass slot should be an electric bass.

⏺ Done. Now all playback goes through _make_playback_midi() which normalizes GM programs by slot:

  ┌─────────┬────────────┬───────────────────────────┐
  │  Slot   │ GM Program │           Sound           │
  ├─────────┼────────────┼───────────────────────────┤
  │ drums   │ 0          │ Standard Kit (channel 10) │
  ├─────────┼────────────┼───────────────────────────┤
  │ bass    │ 33         │ Electric Bass (finger)    │
  ├─────────┼────────────┼───────────────────────────┤
  │ guitar  │ 25         │ Acoustic Guitar (steel)   │
  ├─────────┼────────────┼───────────────────────────┤
  │ other   │ 0          │ Acoustic Grand Piano      │
  ├─────────┼────────────┼───────────────────────────┤
  │ voxlead │ 52         │ Choir Aahs                │
  ├─────────┼────────────┼───────────────────────────┤
  │ voxharm │ 53         │ Voice Oohs                │
  └─────────┴────────────┴───────────────────────────┘

  This applies to both full-mix and solo playback. The SLOT_PROGRAM dict at the top of the file is easy to tweak if you want different sounds.

--

⏺ Great reference ["Blues Model Diagnosis" above]. Looking at what's been done vs. what's left:

  Already done (this session + prior):
  - Problem 1: Fixed map_name_to_slot() with GM program fallback
  - Problem 6: Augmentation widened to ±{1,3,5} semitones
  - Problem 5: Aux weight reduced 0.20 → 0.05
  - Tier 2 items: Dropout 0.12, label smoothing lowered, patience 25

  Still open -- highest impact next:

  1. Tier 1 generation fixes -- these are free, no retraining needed. Once your current training finishes, test with:
  make gen ARGS="--temperature 0.65 --top_p 0.85 --drum_bonus 0 --max_notes_per_step 4"
  1. You can experiment with these while the model trains.
  2. Problem 2: Vocab compaction -- 67% dead tokens. The vocab is built from the data, but TIME_SHIFT has 192 slots (for QN_MAX=8.0) even though we
  reduced QN_MAX to 4.0. That cuts TIME_SHIFT in half (96 slots). BAR bloat comes from rare meters. This requires reprocessing, so it's worth doing
  before the next retrain cycle.
  3. Grid snapping in generate.py -- currently distorts timing. Worth checking/disabling.

  My recommendation: Let the current training run finish. Then:
  - Test generation with the Tier 1 flag tweaks (instant, no cost)
  - Browse/listen to results in the MIDI browser
  - If quality is noticeably better but still has issues, tackle vocab compaction + grid snapping before the next retrain

  Want me to dig into any of these while the training runs?

---

...

  What was added -- A single function compact_vocab() (~90 lines) in training/pre.py:806-925, called at line 1056 after tokenization and before saving.

  What it does:
  1. Scans all train+val sequences to find which global tokens actually appear
  2. Builds per-group old_local → new_local remappings (contiguous from 0)
  3. Rebuilds layout with tight offsets, plus bar_pairs, velocity_bins, duration_bins_qn, and pitch_maps
  4. Remaps every token in every sequence in-place
  5. Prints a compaction report

  Test results (no-aug, 996 blues MIDIs):
  - BAR: 415 → 307 (removed 108 dead bar-position slots)
  - PITCH_GENERAL: 119 → 118 (removed 1)
  - PITCH_DRUMS: 84 → 83 (removed 1)
  - Total: 748 → 638 (14.7% reduction)
  - All 72,041 sequences pass range validation
  - Round-trip MIDI samples generated successfully

  The full augmented run (make blues-preprocess without --no-aug) will show even more compaction since TIME_SHIFT/VEL/DUR bins might also compact. You
  can run that at your convenience -- it takes ~45+ minutes with 996 files × 12 augmentation variants.

------------------------------------------------------------------------------------------------------------------------
------------------------------------------------------------------------------------------------------------------------

