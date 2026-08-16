# HumanWriter

A custom ~40M-parameter decoder-only Transformer, written and trained from scratch in
PyTorch, for natural-language transformation and writing analysis. Requirements:
[`docs/PRD.txt`](docs/PRD.txt), [`docs/SRD.txt`](docs/SRD.txt).

No commercial LLM API is used for generation, evaluation or the baseline comparison.
This is a writing-quality and meaning-preservation tool, **not** an AI-detector evader
(PRD 6.2, PRD 16).

## Setup

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Pipeline

```bash
# 0. fetch the corpus: 70% Project Gutenberg (public domain), 30% Wikipedia (CC BY-SA).
#    Books teach natural prose; Wikipedia keeps the vocabulary from sounding Victorian.
#    Resumable -- re-run after an interruption and it picks up at the next shard.
.venv/bin/python -m data.fetch --target-tokens 800000000

# 1. corpus -> cleaned, deduplicated, split
.venv/bin/python -m data.prepare clean

# 2. train the BPE tokenizer on the train split only
.venv/bin/python -m tokenizer.train_tokenizer --vocab 16000

# 3. encode all splits to uint16 memmaps
.venv/bin/python -m data.prepare encode

# 4. pretrain
.venv/bin/python -m training.pretrain --steps 48000

# 5. build rewrite pairs, and add real paraphrases
.venv/bin/python -m data.make_pairs --pairs 150000
.venv/bin/python -m data.fetch_paraphrases --limit 40000

# 6. fine-tune for rewriting
.venv/bin/python -m training.finetune --epochs 2

# 7. use it
.venv/bin/streamlit run app/streamlit_app.py
```

Checks:

```bash
.venv/bin/python -m tests.test_smoke     # mask, shapes, params, checkpoint, overfit
.venv/bin/python -m tests.test_app       # SRD 12 input matrix
.venv/bin/python -m tests.test_rewrite   # the rewriter is not a copier
```

## Model

`Config()` in [`model/config.py`](model/config.py) — **39.8M trainable parameters**:
vocab 16k, d_model 512, 10 layers, 8 heads, d_ff 2048, context 256, tied embeddings.

Vocab is 16k rather than 32k on purpose: at 32k the embedding matrix is ~39% of the
parameter budget, at 16k it is ~21%, which pays for two extra Transformer blocks at the
same headline size. Weight tying saves a further 8M.

## Measured (PERF-02)

M2 Mac, MPS, 39.8M config, 16,384 tokens per optimiser step: **~4,600 tokens/s**.
Pretraining ran 48,000 steps over 787.7M tokens and finished at **validation loss
3.4615, perplexity 31.9**.

Two performance findings worth keeping:

- Batches padded to the widest example ran fine-tuning at **15.7 s/step**, decaying
  to 37. Every new tensor shape makes Metal recompile its kernels. Padding to a
  fixed width gives **1.79 s/step — 8.8× faster**.
- MPS holds freed blocks; `torch.mps.empty_cache()` every 50 steps stops the machine
  swapping on a long run.

## The rewriter, and why loss cannot select it

Fine-tuning taught the model to rewrite flattened text back into human cadence.
The trap: **93.6% of every target is copied verbatim from its source**, so echoing
the input scores about 91% correct. Two consequences, both measured rather than
assumed:

1. An unweighted loss produces a **pure copier** — 100% word overlap, no feature
   changed — at a validation perplexity of 1.14 that looks like success.
2. Weighting alone is not enough either. At `--changed-weight 6` the model rewrote
   properly at step 200 (44% overlap) and had **reverted to copying by step 2000**
   (93%), while validation loss improved the entire time, 0.46 → 0.27.

So the trainer generates from fixed probes at each eval and keeps only checkpoints
whose rewrite overlap stays inside a healthy band — above 85% it is echoing, below
35% it has stopped preserving the meaning. **Validation loss is reported but never
used to choose a checkpoint.**

Training data mixes two sets that teach different halves of the job:

| set | changed tokens | teaches |
|---|---|---|
| `make_pairs.py` (synthetic, 149k) | 6.4% | punctuation, sentence joins, rhythm |
| PAWS (real, 21.6k ×2) | 15.6% | word choice, clause order |

## Status

- [x] Model, tokenizer, data pipeline, training loop, inference — verified end to end
- [x] Corpus: 787.7M tokens, Gutenberg + Wikipedia, licences in `data/SOURCES.md`
- [x] Pretraining — 48,000 steps, val loss 3.4615, perplexity 31.9
- [x] Writing analysis + human-vs-AI study ([evaluation/human_vs_ai.py](evaluation/human_vs_ai.py))
- [x] Evaluation: perplexity ([evaluate_lm.py](evaluation/evaluate_lm.py)),
      semantic similarity ([evaluate_semantics.py](evaluation/evaluate_semantics.py))
- [x] Streamlit app with the reference-band profile ([app/](app/streamlit_app.py))
- [x] Rewrite pairs — synthetic + PAWS, and the copier guard
- [ ] Fine-tuning run to completion, then the before/after numbers

### What the human-vs-AI study found

Three features separate the classes; the other four sit within ±9% and are noise at
44 documents. Human text has **+46% punctuation, +37% sentence length, +26%
burstiness**. Those are the targets the rewriter aims at, and the app's reference
bands come from these measurements.

Classifier accuracy is **0.79 ± 0.06** with grouped folds, not the 0.83–0.95 the
source paper reports — every ensemble overlaps every other, so no classifier is
meaningfully better than another at this sample size.

### Known limitations

- **Semantic similarity cannot detect contradiction.** "Water boils at 90°C" scores
  0.969 against the 100°C original — higher than a faithful rewrite at 0.921. It
  catches drift, not falsehood, so FR-06 cannot rest on it alone.
- The rewriter learns to invert *our* corruption, a proxy for machine cadence rather
  than a sample of it. Validate on real AI text in `data/human_ai`, not held-out
  pairs alone.
- 70% of the pretraining corpus predates 1929, so the base model reads archaic.
  Register is set by fine-tuning, not pretraining.
- This is a writing-quality tool. It is **not** a detector-evasion system, and is not
  evaluated as one (PRD 6.2, 16).

## Layout

Follows SRD 14, with `model/attention.py` and `model/transformer.py` merged into
[`model/model.py`](model/model.py) — the whole architecture is ~130 lines and splitting it
across three files makes it harder to read, not easier.
