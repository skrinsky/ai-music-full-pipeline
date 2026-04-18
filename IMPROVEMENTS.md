# Generation & Training Improvements

## In Progress

### Entropy ceiling (`generate.py`)
Stop placing notes when the model's PITCH distribution is too flat (high entropy = guessing).
Instead of emitting a chaotic note, force a TIME_SHIFT and let the model re-orient.
- Add `--entropy_ceiling` float arg (e.g. 3.5); 0 = disabled
- Apply at PITCH sampling step only (that's where musical collapse shows up most)
- **Status:** done — `--entropy_ceiling 3.5` (try between 3.0–4.0)

---

## Queued

### Prompt seeding from real MIDI (`generate_v2.py`)
Feed the first N bars of a real song as context so the model starts in-distribution,
but write **only the generated tail** to the output MIDI — not the prompt.

Args: `--seed_midi PATH`, `--seed_bars N`
- `seed_offset` tracked so decode_to_midi receives `[BOS] + seq[seed_offset:]` only
- **Status:** done in `generate_v2.py`

### Longer training context (512 → 1024 tokens)
Re-run `pre.py --seq_len 1024` then retrain. Doubles the coherent horizon from ~1 bar to ~2-3 bars.
Cost: 4× compute per step (attention is quadratic), but manageable on CUDA.
- **Dependency:** entropy ceiling first — want to know baseline quality before retraining

### Drum density / pattern second pass
Drum pitch vocab is already constrained to 6 sounds (PITCH_DRUMS), so wrong-pitch hits aren't the issue.
The problem is rhythmic density — too many or too sparse hits, no consistent groove pattern.
Options:
- Post-process: remove simultaneous drum hits above N per time step
- Post-process: thin drum hits that don't fall on the snapping grid
- Train a tiny drum-only RNN on the drum tracks of summer_midi to rewrite drum layer

### KNN-LM guided generation
At each token step during generation, look up K nearest neighbors in the training set
by the model's current hidden state, compute a distribution over what token came next
in those neighbors, and interpolate with the model's own prediction. Pulls generation
toward patterns it's actually seen — implements "snap to known distribution" without
any additional training.

Implementation steps:
1. **Index build** (`training/build_knn_index.py`): run the model over all training
   windows, save the hidden state at each position + the next token → FAISS index
2. **Generation change** (`training/generate.py`): at each PITCH (and optionally INST)
   step, query FAISS for top-K neighbors, mix their next-token distribution with the
   model's logits via interpolation weight `--knn_lambda` (0=model only, 1=KNN only)
3. **Tunable knobs:** `--knn_k` (neighborhood size, try 8–32), `--knn_lambda` (mix
   weight, try 0.2–0.4), apply only at PITCH steps to avoid slowing down structural tokens

Why this works: the transformer hidden state already encodes musical context — the KNN
lookup is essentially asking "when the model was in a similar musical moment during
training, what note did the real data play next?"

Dependencies: `faiss-cpu` (or `faiss-gpu`)
- **Status:** done in `generate_v2.py` + `training/build_knn_index.py`
  - Build index: `python training/build_knn_index.py --ckpt ... --vocab_json ... --train_pkl ... --out runs/knn/pitch_general`
  - Use: add `--knn_index runs/knn/pitch_general --knn_k 16 --knn_lambda 0.3` to generate_v2.py

### N-best reranking
Generate N candidates per snippet, re-score each by model perplexity, keep the best.
Slow but no retraining needed. Practical for idea-snapshot workflow.
- `--n_candidates N` flag in generate.py

### Generation one-liner (batch snippets)
Already documented — copy-paste for generating 20 snapshots at once:
```
for i in $(seq 1 20); do python training/generate.py --ckpt runs/checkpoints/es_model.pt --vocab_json runs/events/event_vocab.json --out_midi runs/generated/snap_$i.mid --max_tokens 800 --temperature 0.75 --top_p 0.75 --device auto; done
```

---

## Ruled Out / Parked

### MMT (Multitrack Music Transformer) fine-tuning
MMT skips drum tracks entirely — not suitable for this style.
Scripts remain in `finetune/mmt_*.py` if useful for melodic-only experiments.

### LoRA fine-tune on HuggingFace model
`Natooz/Multitrack-Music-Transformer` doesn't exist on HuggingFace — the finetune was
training from random init the whole time. Parked in `finetune/` for reference.

### Full 12-key pitch augmentation
`pre.py` already does ±1, ±3, ±5 semitone shifts (6 transpositions). Going to all 12 keys
gives diminishing returns and transposes into unnatural keys for the source material.
