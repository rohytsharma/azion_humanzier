"""End-to-end rewrite evaluation on real AI text (PRD 12, FR-05/FR-06).

    python -m evaluation.evaluate_rewrite --n 12

Everything else in evaluation/ scores one property. This answers the project's
actual question: given text a language model wrote, does the rewriter move it
toward human writing without changing what it says?

Held-out in the way that matters -- the model never saw this data in any form.
Its fine-tuning pairs were built from Gutenberg and Wikipedia paragraphs; these
inputs are ChatGPT, Gemini, Grok and DeepSeek output.
"""
import argparse
import difflib
import statistics as st
from pathlib import Path

import pandas as pd

from evaluation.features import extract, sentences
from inference.generate import load, rewrite

# Measured in evaluation/human_vs_ai.py over 20 human and 24 AI documents.
AI_MEAN = {"punct_ratio": 0.1449, "sent_len_words": 18.4346, "burstiness": 0.3444}
HUMAN_MEAN = {"punct_ratio": 0.2121, "sent_len_words": 25.2293, "burstiness": 0.4356}


def toward_human(before, after):
    """Share of the AI->human gap closed, per feature. 0 = no move, 1 = arrived."""
    out = {}
    for k, human in HUMAN_MEAN.items():
        ai = AI_MEAN[k]
        gap = human - ai
        out[k] = (after[k] - before[k]) / gap if gap else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="training/checkpoints/finetuned.pt")
    ap.add_argument("--data", default="data/human_ai/AI_Generated.csv")
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--temperature", type=float, default=0.85)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    a = ap.parse_args()

    if not Path(a.data).exists():
        raise SystemExit(f"missing {a.data}")
    df = pd.read_csv(a.data).head(a.n)
    model, tok, device = load(a.ckpt, device="cpu")

    rows, sims = [], None
    srcs, outs = [], []
    for _, r in df.iterrows():
        # first few sentences: enough to carry rhythm, short enough to stay quick
        src = " ".join(sentences(r["Generated"])[:6])
        out = rewrite(model, tok, device, src, temperature=a.temperature,
                      top_p=a.top_p, repetition_penalty=a.repetition_penalty)
        srcs.append(src)
        outs.append(out)
        before, after = extract(src), extract(out)
        rows.append({
            "model": r["Model"],
            "overlap": difflib.SequenceMatcher(None, src.split(), out.split()).ratio(),
            **{f"{k}_before": before[k] for k in HUMAN_MEAN},
            **{f"{k}_after": after[k] for k in HUMAN_MEAN},
            **{f"{k}_closed": v for k, v in toward_human(before, after).items()},
        })

    try:
        from evaluation.evaluate_semantics import similarity
        sims = similarity(srcs, outs)
    except SystemExit:
        pass

    t = pd.DataFrame(rows)
    print(f"\n{len(t)} AI passages rewritten "
          f"(decoding: temp {a.temperature}, top-p {a.top_p}, rep {a.repetition_penalty})\n")
    print(f"{'feature':18s} {'AI in':>9s} {'rewritten':>10s} {'human ref':>10s} {'gap closed':>11s}")
    for k in HUMAN_MEAN:
        print(f"{k:18s} {t[k + '_before'].mean():9.4f} {t[k + '_after'].mean():10.4f} "
              f"{HUMAN_MEAN[k]:10.4f} {t[k + '_closed'].mean():10.0%}")

    print(f"\nword overlap with input   {t.overlap.mean():.1%} "
          f"(median {t.overlap.median():.1%})")
    if sims is not None:
        print(f"semantic similarity       {st.mean(sims):.4f} "
              f"(min {min(sims):.3f}, >= 0.80 in {sum(s >= 0.80 for s in sims)}/{len(sims)})")

    print("\nA positive gap closed means the rewrite moved that feature toward the "
          "human mean.\nSemantic similarity measures topic overlap and cannot detect a "
          "reversed fact:\nread it with the samples, not instead of them.")

    print("\n--- sample ---")
    print(f"IN : {srcs[0][:240]}")
    print(f"OUT: {outs[0][:240]}")

    Path("evaluation/figures").mkdir(parents=True, exist_ok=True)
    t.to_csv("evaluation/figures/rewrite_eval.csv", index=False)
    print("\nwrote evaluation/figures/rewrite_eval.csv")


if __name__ == "__main__":
    main()
