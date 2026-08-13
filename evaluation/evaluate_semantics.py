"""Semantic preservation: did the rewrite keep the meaning? (FR-06, SYS-F13)

    python -m evaluation.evaluate_semantics --demo
    python -m evaluation.evaluate_semantics --csv pairs.csv --source Generated --rewrite Humanized

Embeds both texts with a local sentence encoder and reports cosine similarity.
The encoder is downloaded once and then runs offline; it only ever measures, and
is never in the generation path, so the "no commercial LLM API" rule holds.

Why a separate encoder rather than our own model: HumanWriter is trained for
next-token prediction, and its hidden states are not a sentence representation.
Scoring a rewrite with the model that produced it would also be marking your own
homework -- the metric has to be independent of the thing it judges.
"""
import argparse
from functools import lru_cache

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Below this, a rewrite has usually dropped or contradicted something in the
# source rather than rephrasing it. Calibrated on the pairs in --demo; treat as
# a starting threshold to tune on real output, not a constant of nature.
PRESERVED = 0.80


@lru_cache(maxsize=1)
def _encoder(name=MODEL_NAME):
    """Loaded once, lazily -- importing this module must not pull a model."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise SystemExit(
            "sentence-transformers is not installed.\n"
            "  .venv/bin/pip install sentence-transformers")
    return SentenceTransformer(name)


def embed(texts, name=MODEL_NAME):
    return _encoder(name).encode(list(texts), normalize_embeddings=True,
                                 show_progress_bar=False)


def similarity(sources, rewrites, name=MODEL_NAME):
    """Cosine similarity for each (source, rewrite) pair, in order."""
    sources, rewrites = list(sources), list(rewrites)
    assert len(sources) == len(rewrites), "pair counts differ"
    a, b = embed(sources, name), embed(rewrites, name)
    return (a * b).sum(axis=1)          # both are unit vectors


def report(sources, rewrites, label="pairs", name=MODEL_NAME):
    import numpy as np
    s = similarity(sources, rewrites, name)
    print(f"\n{label}: {len(s)} pairs")
    print(f"  mean   {s.mean():.4f}")
    print(f"  median {np.median(s):.4f}")
    print(f"  min    {s.min():.4f}   max {s.max():.4f}")
    print(f"  >= {PRESERVED:.2f}: {(s >= PRESERVED).sum()}/{len(s)} "
          f"({100 * (s >= PRESERVED).mean():.0f}%)")
    return s


DEMO = [
    # (source, rewrite, what it is)
    ("The committee approved the budget on Tuesday after a long debate.",
     "After lengthy discussion, the committee gave the budget its approval on Tuesday.",
     "faithful rewrite"),
    ("The committee approved the budget on Tuesday after a long debate.",
     "The committee rejected the budget on Tuesday after a long debate.",
     "meaning inverted"),
    ("The committee approved the budget on Tuesday after a long debate.",
     "Rainfall in the valley has been unusually heavy this spring.",
     "unrelated"),
    ("Water boils at 100 degrees Celsius at sea level.",
     "At sea level, water reaches its boiling point at 100 degrees Celsius.",
     "faithful rewrite"),
    ("Water boils at 100 degrees Celsius at sea level.",
     "Water boils at 90 degrees Celsius at sea level.",
     "fact altered"),
]


def demo():
    """The metric is only useful if it separates these cases -- so check that."""
    srcs = [d[0] for d in DEMO]
    rws = [d[1] for d in DEMO]
    s = similarity(srcs, rws)
    print(f"encoder: {MODEL_NAME}\n")
    for (src, rw, kind), score in zip(DEMO, s):
        flag = "keeps meaning" if score >= PRESERVED else "flagged"
        print(f"  {score:.4f}  {flag:14s} {kind}")
        print(f"          {rw[:72]}")
    faithful = [sc for (_, _, k), sc in zip(DEMO, s) if k == "faithful rewrite"]
    unrelated = [sc for (_, _, k), sc in zip(DEMO, s) if k == "unrelated"]
    assert min(faithful) > max(unrelated), "encoder cannot tell a rewrite from a non sequitur"
    print("\nfaithful rewrites score above unrelated text -- metric is discriminating")
    print("Note: 'meaning inverted' and 'fact altered' score high. Embedding "
          "similarity measures topic overlap,\nnot factual agreement -- it catches "
          "drift, not contradiction. Pair it with human review (PRD 12).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--csv")
    ap.add_argument("--source", default="Generated")
    ap.add_argument("--rewrite", default="Humanized")
    ap.add_argument("--model", default=MODEL_NAME)
    a = ap.parse_args()

    if a.demo or not a.csv:
        demo()
        return
    import pandas as pd
    df = pd.read_csv(a.csv)
    report(df[a.source], df[a.rewrite], f"{a.source} -> {a.rewrite}", a.model)


if __name__ == "__main__":
    main()
