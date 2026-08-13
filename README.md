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
.venv/bin/python -m training.pretrain --steps 20000

# 5. generate
.venv/bin/python -m inference.generate --prompt "It was a bright cold day"
```

Checks (SRD 12: causal mask, shapes, param count, checkpoint round-trip, overfit):

```bash
.venv/bin/python -m tests.test_smoke
```

## Model

`Config()` in [`model/config.py`](model/config.py) — **39.8M trainable parameters**:
vocab 16k, d_model 512, 10 layers, 8 heads, d_ff 2048, context 256, tied embeddings.

Vocab is 16k rather than 32k on purpose: at 32k the embedding matrix is ~39% of the
parameter budget, at 16k it is ~21%, which pays for two extra Transformer blocks at the
same headline size. Weight tying saves a further 8M.

## Measured (PERF-02)

M2 Mac, MPS, full 39.8M config at `--batch-size 8 --grad-accum 8 --block-size 256`
(16,384 tokens per optimiser step): **~4,600 tokens/s** training throughput.

Chinchilla-optimal for 40M params is ~800M tokens, i.e. **~48 h of continuous M2
training**. Budget for that, or cut the token target and say so in the report.

## Status

- [x] Model, tokenizer, data pipeline, training loop, inference — all verified end to end
- [x] Corpus fetcher with licence manifest ([data/fetch.py](data/fetch.py))
- [ ] Full corpus pull + pretraining run
- [ ] Human-vs-AI dataset (supplied separately) — wire into fine-tuning or analysis
      depending on whether its texts are paired
- [ ] Fine-tuning on paraphrase pairs (`training/finetune.py`)
- [ ] Evaluation: perplexity, semantic similarity, readability, lexical diversity (`evaluation/`)
- [ ] Streamlit app (`app/`)

## Layout

Follows SRD 14, with `model/attention.py` and `model/transformer.py` merged into
[`model/model.py`](model/model.py) — the whole architecture is ~130 lines and splitting it
across three files makes it harder to read, not easier.
