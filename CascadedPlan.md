# Cascaded Voice Models for Bach Chorales

## Motivation

The current pipeline interleaves all 4 voices into a single event stream
(~17 tokens per timestep, ~4000–5000 tokens per chorale). A 512-token
context window sees only ~30 timesteps — never a full phrase.

By factoring into per-voice models, each voice needs ~300–400 tokens for
a full chorale. The context window problem disappears.

## Architecture: Cascade with conditioning

```
Stage 1:  Soprano model   →  melody (no conditioning)
Stage 2:  Bass model      →  bass line, conditioned on soprano
Stage 3:  Alto model      →  alto, conditioned on soprano + bass
Stage 4:  Tenor model     →  tenor, conditioned on soprano + bass + alto
```

This mirrors compositional practice: outer voices define the harmonic
framework, inner voices fill according to voice-leading rules.

## Conditioning strategy

The central design question: how does voice model N "see" the
already-generated voices 1..N-1?

### Option A: Prefix token blocks

Concatenate completed voices as a token prefix, separated by voice
boundary tokens:

```
[BOS] [VOX_S] s₁ s₂ s₃ ... [VOX_B] b₁ b₂ b₃ ... [VOX_A] a₁ a₂ ... → predict next a
       ────── soprano ──────  ────── bass ─────────  ── alto so far ──
```

Each sᵢ and bᵢ are just the voice's own tokens (TIME_SHIFT + PITCH,
possibly VEL + DUR).  The voice-boundary tokens (`VOX_S`, `VOX_B`,
`VOX_A`) tell the model which voice follows.

**Time alignment is implicit.** The model must learn that soprano's
3rd TIME_SHIFT and bass's 3rd TIME_SHIFT refer to the same musical
moment.  Self-attention can discover this — the TIME_SHIFT values
encode elapsed time, so cumulative sums must match at simultaneous
events — but it's an extra burden on the model.

**Token budget for the alto model (3 conditioning voices):**
~300 (S) + ~300 (B) + ~300 (A so far) = ~900 tokens.  Fits in 1024.

**Pros:** Simple, uses the existing encoder-only transformer unchanged.
**Cons:** Time alignment is learned, not structural.  Prefix grows
linearly with number of conditioning voices.

### Option B: Chronological merge with time-shifts

Merge all conditioning voices into a single chronological event stream
ordered by onset time, using time-shifts (deltas) between events:

```
Context (soprano + bass merged by onset time):
  Δ0 S:C5  Δ0 B:C3  Δ3 S:D5  Δ1 B:G2  Δ2 S:E5  Δ0 B:C3 ...

Target (alto, autoregressive):
  [SEP] Δ0 A:E4  Δ2 A:F4  Δ4 A:... → predict next
```

Events from all conditioning voices appear in a single timeline.
Time-shifts encode the delta to the next event, regardless of which
voice it belongs to.  This is compact — voices with long held notes
contribute no tokens during the hold.

**The alignment problem:** the target voice's timeline and the
context's timeline must be synchronized.  If the alto model has
generated Δ2+Δ4=6 beats into the piece, it needs to attend to context
events near cumulative time = 6.  Three sub-approaches:

**B1. Learned alignment.** Trust the model to learn cumulative
time-shift tracking through self-attention.  Transformers can learn
this, but it's an implicit burden — the model must compute running
sums to discover which context events are "now."

**B2. Absolute time annotations.** Add absolute beat-position tokens
alongside time-shifts:

```
Δ0 @0 S:C5   Δ0 @0 B:C3   Δ3 @3 S:D5   Δ1 @4 B:G2 ...
   [SEP]
Δ0 @0 A:E4   Δ2 @2 A:F4   Δ4 @6 A:...
```

The `@N` tokens are redundant (computable from cumulative Δ) but give
the model explicit anchors.  When generating alto at `@6`, attention
can directly match `@6` in the context.  Costs ~1 extra token per
event.

**B3. Musical time embedding (recommended).** Instead of discrete
time tokens, add a continuous sinusoidal embedding indexed by
cumulative musical time — analogous to positional encoding but on a
musical clock rather than token position:

```python
# Standard positional encoding: token index i
pe[i] = sin(i / 10000^(2k/d))

# Musical time encoding: cumulative beat position t
te[i] = sin(t_i / 10000^(2k/d))
```

Each token gets both embeddings:
```python
token_repr = token_embedding + token_position_encoding + musical_time_encoding
```

Events at the same musical time get similar time embeddings regardless
of where they sit in the token sequence.  No extra tokens, no grid
assumption.  The `musical_time` values are precomputed by accumulating
time-shifts during preprocessing and stored alongside the token IDs.

**Why B3 is the best alignment mechanism:**
- Zero token overhead (embedded, not tokenized)
- Works for arbitrary rhythms and rubato, not just fixed grids
- Alignment is structural (same beat → similar embedding) not learned
- Naturally extends to cross-attention (Option C): encoder and decoder
  share the same musical time embedding space, so cross-attention
  weights can align by beat position without any explicit annotation

**Token budget:** compact because only note onsets appear — a voice
sustaining a half note contributes 0 tokens during the hold.  Typical
alto model context: ~150 (S events) + ~150 (B events) + ~200 (A so
far) = ~500 tokens.  Fits in 512 or 1024.

**Pros:** Compact (no rest/sustain tokens), handles irregular rhythms,
musical time embedding gives structural alignment for free.
**Cons:** Slightly more complex preprocessing (accumulate times, merge
and sort events across voices).

### Option C: Cross-attention into conditioning voices

Separate the architecture into an encoder for conditioning voices and
a decoder for the target voice.  The decoder cross-attends into the
encoder's output.  Both sides use the same musical time embedding
from B3, so cross-attention weights naturally align by beat position.

```
┌─────────────────────────────────────────────────┐
│  Conditioning encoder (shared across stages)    │
│                                                 │
│  Input: chronological merge of completed voices │
│    Δ0 S:C5  Δ0 B:C3  Δ3 S:D5  Δ1 B:G2 ...   │
│  Embeddings: token + token_position + mus_time  │
│                                                 │
│  Output: hidden states H_cond (T_c, D)         │
└──────────────────────┬──────────────────────────┘
                       │ cross-attention
┌──────────────────────▼──────────────────────────┐
│  Voice decoder (one per voice, or shared+adapter)│
│                                                 │
│  Self-attention over target voice tokens so far │
│  Embeddings: token + token_position + mus_time  │
│  Cross-attention into H_cond at each layer      │
│                                                 │
│  Output: next token prediction for target voice │
└─────────────────────────────────────────────────┘
```

**How multiple conditioning voices are handled:**

The conditioning encoder takes ALL completed voices as a chronological
merge (same format as Option B), producing a single sequence of hidden
states.  The decoder doesn't need to know how many conditioning voices
there are — it just cross-attends into whatever the encoder produced.

This means the same decoder architecture works for all stages:
- Stage 2 (bass): encoder sees soprano only
- Stage 3 (alto): encoder sees soprano + bass merged
- Stage 4 (tenor): encoder sees soprano + bass + alto merged

The encoder grows by ~150 tokens per added voice.  The decoder stays
the same size (~200–300 tokens for the full target voice).

**Why cross-attention suits this problem:**

1. **Separation of concerns.** The encoder builds a rich representation
   of the harmonic context.  The decoder focuses on voice-leading for
   its specific voice.  Neither needs to parse the other's token format.

2. **Temporal alignment via musical time embedding.** Both encoder and
   decoder use the same sinusoidal musical time embedding.  When the
   decoder generates a note at beat 6, its musical time embedding is
   similar to encoder tokens near beat 6.  Cross-attention weights
   align by beat position automatically — no explicit time grid needed.

3. **Scalable to N voices.** Adding a 5th conditioning voice just adds
   ~150 tokens to the encoder input.  The decoder is unchanged.

4. **Shared encoder weights.** The same encoder can process any subset
   of voices.  Only the decoder (or a small adapter layer) needs to be
   voice-specific.

**Pros:** Cleanest architecture, naturally handles variable numbers of
conditioning voices, good separation of context vs generation.
**Cons:** Encoder-decoder is more complex than encoder-only.  Requires
modifying the current `FactorizedESModel` architecture (which is
encoder-only).

### Recommendation

**Start with Option B + B3 (chronological merge with musical time
embedding), plan for C.**

Option B reuses the existing encoder-only transformer — just changes
the tokenization and adds a musical time embedding.  The B3 embedding
provides structural alignment without a fixed grid, so the approach
works for both chorales and free-rhythm music from the start.

If we later want cleaner separation of conditioning from generation,
we migrate to cross-attention (Option C).  The chronological merge
format and musical time embedding carry over unchanged — only the
model architecture changes (encoder-only → encoder-decoder).

## Per-voice token vocabulary

Each voice model needs a much smaller vocabulary than the full pipeline:

| Token type   | Size | Notes                                 |
|--------------|------|---------------------------------------|
| PAD          | 1    |                                       |
| BOS          | 1    |                                       |
| EOS          | 1    |                                       |
| SEP          | 1    | Boundary between context and target   |
| BAR          | 1    | Bar boundary marker                   |
| TIME_SHIFT   | 48   | 1/24-QN steps, max 2 QN (half note)   |
| PITCH        | 52   | Union range MIDI 33–84 (see below)    |
| REST         | 1    | Explicit silence event                |
| VOX_S/B/A/T  | 4    | Voice tags in conditioning context    |
| **Total**    | ~110 | Shared across all voice models        |

TIME_SHIFT tokens encode deltas (time to next event), not grid
positions.  The 1/24-QN resolution matches the existing pipeline and
handles both straight and triplet rhythms.

Voice-specific pitch ranges (from existing `pre.py` config):

| Voice    | MIDI range | Semitones | Pitch tokens |
|----------|------------|-----------|--------------|
| Soprano  | 57–84      | 27        | 28           |
| Alto     | 50–77      | 27        | 28           |
| Tenor    | 43–72      | 29        | 30           |
| Bassvox  | 33–69      | 36        | 37           |

The pitch space is a union of all voice ranges (MIDI 33–84 = 52
semitones) so that shared weights can handle any voice.  Per-voice
range constraints are enforced at generation time by masking logits
outside the voice's range, not by restricting the vocabulary.

VEL and DUR tokens can likely be omitted for chorales — velocity is
uniform and duration is implicit (note-on to next note-on or rest in
the same voice).

### Musical time embedding

In addition to the token vocabulary, each token carries a precomputed
musical time value (cumulative beat position as a float).  This is
encoded as a sinusoidal embedding added to the token representation:

```python
token_repr = token_embedding[tok_id]
           + position_encoding[tok_pos]     # token sequence position
           + musical_time_encoding[mus_t]   # cumulative beat position
```

The `musical_time_encoding` uses the same sinusoidal formula as
standard positional encoding but indexed by continuous musical time
(in quarter notes) rather than discrete token index.  This provides
structural temporal alignment across context and target voice streams
without any extra tokens or grid assumptions.

**Base frequency:** Standard positional encoding uses base=10000,
tuned for sequences of thousands of tokens.  Musical time spans
~0–60 QN for a typical chorale.  A base of **100** gives good
frequency resolution across this range.  Start there and tune if
attention patterns look misaligned.

## Model hyperparameters

Starting point for the shared-weight model:

| Parameter    | Value | Notes                                  |
|--------------|-------|----------------------------------------|
| d_model      | 128   | Small vocab → smaller embedding        |
| n_heads      | 4     | head_dim = 32                          |
| n_layers     | 4     | Same depth as current pipeline         |
| ff_mult      | 3     | FFN = 384                              |
| dropout      | 0.12  | Match current pipeline                 |
| seq_len      | 1024  | Fits tenor stage (~600 tokens worst case) |
| batch_size   | 64    | Same as current pipeline               |
| lr           | 2e-4  | AdamW, cosine schedule                 |
| epochs       | 200   | With early stopping (patience=25)      |

Estimated parameters: ~300K.  Small because the vocab is ~110 tokens
(vs ~2400 in the interleaved pipeline) and sequences are short.

The voice-ID embedding is a learned vector (dim = d_model) added to
every target-voice token.  Four embeddings total (S, B, A, T).  This
tells the shared model which voice it is generating.

## Training procedure

**Data preparation:**
1. Parse each chorale MIDI into 4 separate voice sequences
2. Encode each voice as: `Δt PITCH Δt PITCH ... [EOS]`
   (time-shifts between consecutive note onsets in that voice)
3. Compute cumulative musical time for each token
4. For voice N, build conditioning context by chronologically merging
   voices 1..N-1 with voice tags: `Δ0 [S] PITCH Δ0 [B] PITCH Δ3 [S] PITCH ...`
5. Concatenate: `[BOS] context [SEP] target_voice [EOS]`
6. Augment: transpose ±3 semitones (7x), checking voice ranges

**Training:**
- Teacher forcing: conditioning voices are ground truth during training
- Each stage can be trained independently and in parallel
- Loss: cross-entropy on target voice tokens only (exclude prefix)
- Shared-weight variant: one model with a voice-ID embedding, trained
  on all (context, target) pairs from all 4 stages simultaneously

**Scheduled sampling (optional, combats error cascade):**
During training, occasionally replace ground-truth conditioning tokens
with model-generated tokens (from an earlier training snapshot).
Teaches the model to be robust to imperfect conditioning inputs.

## Generation procedure

```
1. Generate soprano (no conditioning):
   model([BOS] [SEP]) → Δt₁ P₁ Δt₂ P₂ ... [EOS]
   Compute cumulative musical times for soprano tokens.

2. Build context from soprano:
   ctx = chronological_merge([soprano], voice_tags=[[S]])

3. Generate bass (conditioned on soprano):
   model([BOS] ctx [SEP]) → Δt₁ P₁ Δt₂ P₂ ... [EOS]
   Musical time embedding keeps bass aligned with soprano context.

4. Build context from soprano + bass:
   ctx = chronological_merge([soprano, bass], voice_tags=[[S],[B]])

5. Generate alto (conditioned on soprano + bass):
   model([BOS] ctx [SEP]) → Δt₁ P₁ ... [EOS]

6. Build context from soprano + bass + alto:
   ctx = chronological_merge([soprano, bass, alto], voice_tags=[[S],[B],[A]])

7. Generate tenor (conditioned on all three):
   model([BOS] ctx [SEP]) → Δt₁ P₁ ... [EOS]

8. Convert each voice's time-shift + pitch sequence to absolute
   note events, merge into multi-track MIDI.
```

Total generation: 4 sequential forward passes.  Each is fast (small
model, short sequences).  Musical time embedding is recomputed for
the context at each stage (cumulative sum of the merged time-shifts).

## Shared vs independent weights

**Option I: Four independent models**
- Each specializes fully for its voice
- 4× the total parameters, but each is small (~50K?)
- Train in parallel

**Option II: Shared body + voice adapter**
- One transformer body shared across all 4 stages
- Small per-voice adapter (learned voice embedding + output head)
- ~1/3 the parameters of Option I
- Trained on all stages jointly: each batch mixes soprano-only,
  soprano→bass, S+B→alto, S+B+A→tenor examples
- The model learns general voice-leading; adapters specialize

Recommendation: **Start with Option II (shared weights).**  The small
dataset size (~350 chorales × 7 augmentations = ~2450 examples) favors
parameter sharing.  If voices prove too different, split later.

## Relationship to existing pipeline

This cascaded approach is a **separate model architecture**, not a
modification of the current interleaved event-stream model.  It would
live alongside the existing pipeline:

```
training/
  train.py              ← existing interleaved model
  train_cascade.py      ← new cascaded voice model
  pre.py                ← reuse MIDI parsing, add per-voice serialization
  generate.py           ← existing interleaved generation
  generate_cascade.py   ← new cascaded generation
```

The Makefile would get new targets:
```
make chorale-cascade-preprocess
make chorale-cascade-train
make chorale-cascade-generate
```

## Evaluation: how to compare with TonicNet

To fairly compare the cascaded pipeline with TonicNet:

1. **Token-level accuracy** — predict held-out chorale voices given
   ground-truth conditioning (matches TonicNet's 90% metric)
2. **Harmonicity** — % of simultaneities that form valid triads/7ths
3. **Voice-leading violations** — parallel 5ths/8ves, voice crossing,
   range violations (standard chorale metrics)
4. **Listening test** — blind A/B comparison of generated chorales

## Implementation order

A fresh session should implement in this order:

### Phase 1: Preprocessing (`training/pre_cascade.py`)
1. Parse chorale MIDIs into 4 separate voice note-lists (reuse
   existing MIDI parsing from `pre.py`)
2. Encode each voice as a token sequence: `Δt PITCH Δt PITCH ...`
3. Compute cumulative musical time for each token
4. Build training examples for all 4 cascade stages:
   - Stage 1 (soprano): `[BOS] [SEP] soprano_tokens [EOS]`
   - Stage 2 (bass): `[BOS] merged(S) [SEP] bass_tokens [EOS]`
   - Stage 3 (alto): `[BOS] merged(S,B) [SEP] alto_tokens [EOS]`
   - Stage 4 (tenor): `[BOS] merged(S,B,A) [SEP] tenor_tokens [EOS]`
5. Apply transposition augmentation (±3 semitones, 7x)
6. Save as pickle: token_ids, musical_times, voice_ids, stage_ids
7. Save vocab JSON (shared across all stages)
8. Unit tests for preprocessing

### Phase 2: Model (`training/model_cascade.py`)
1. Musical time embedding (sinusoidal, base=100)
2. Shared transformer encoder with triple embedding
   (token + position + musical_time)
3. Voice-ID embedding (4 learned vectors)
4. Output head: pitch logits + time-shift logits (simple, not
   factored — vocab is small enough for a single softmax)
5. Causal mask + loss only on target tokens (after [SEP])
6. Unit tests for model forward pass

### Phase 3: Training (`training/train_cascade.py`)
1. DataLoader that mixes all 4 stages in each batch
2. Training loop (AdamW, cosine schedule, early stopping)
3. Checkpoint saving with config metadata
4. Makefile targets: `chorale-cascade-preprocess`, `chorale-cascade-train`

### Phase 4: Generation (`training/generate_cascade.py`)
1. 4-stage sequential generation with context building
2. Voice-range masking on pitch logits
3. Temperature + nucleus sampling
4. Merge voice sequences → multi-track MIDI
5. Makefile target: `chorale-cascade-generate`

### Phase 5: Evaluation
1. Token-level accuracy on held-out chorales (teacher-forced)
2. Harmonicity and voice-leading metrics
3. Compare with existing interleaved model and TonicNet

## Resolved design decisions

- **Simultaneous onset ordering:** When merging voices chronologically,
  events at the same musical time are ordered S before B before A
  before T.  This is consistent during training and generation.

- **Musical time embedding base frequency:** Start with base=100
  (good resolution for 0–60 QN range).  Tune if needed.

- **Single softmax vs factored heads:** The cascade vocab (~110 tokens)
  is small enough for a single output softmax.  No need for the
  type+value factored heads used in the interleaved pipeline (~2400
  tokens).

## Open questions

1. **Soprano source.** Should the soprano model generate freely, or
   should we seed it with a known hymn tune (as Bach did)?  Seeding
   makes evaluation more comparable to TonicNet's harmonization task.

2. **Chord conditioning.** Should we add an explicit chord stage before
   the soprano?  This creates a 5-stage cascade: Chords → S → B → A → T.
   The chord model could be very simple (HMM or small transformer).

3. **Error propagation.** How badly do soprano mistakes degrade the
   inner voices?  Scheduled sampling during training should help, but
   we'll need to measure this empirically.
