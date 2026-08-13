"""Build rewrite pairs for fine-tuning (PRD 5, SYS-F10).

    python -m data.make_pairs --pairs 200000

The problem with public paraphrase corpora is that they are sentence-level, so
a model trained on them never learns to vary sentence length *across* a
paragraph -- and cross-sentence variation (burstiness) is the single feature
that separated human from machine writing most clearly in our own measurements.

So the pairs are built the other way round. Take real human paragraphs from the
pretraining corpus and mechanically *flatten* them into machine cadence:
split long sentences at their joins, drop the rich punctuation, thin the commas.
The pair is (flattened, original) and the model learns to invert it.

Three properties this buys:
  - paragraph-level, so sentence-length variation is learnable
  - meaning preservation is structural: flattening only ever moves punctuation
    and sentence boundaries, never content words, so the target says exactly
    what the source says
  - unlimited quantity from a corpus we already have, no new licence

The known limitation, worth stating in the report: the model learns to invert
*this* corruption, which is a proxy for machine cadence rather than a sample of
it. Validate on real AI text (data/human_ai) rather than on held-out pairs alone.
"""
import argparse
import json
import random
import re
from pathlib import Path

from evaluation.features import extract, sentences

OUT = Path("data/pairs")

# Joins where a long sentence can be cut into two without touching content words.
_SPLIT = re.compile(r",\s+(and|but|yet|so|while|whereas|although|though|because|since)\s+",
                    re.IGNORECASE)
_PARENS = re.compile(r"\s*\(([^)]{1,120})\)")
_DASH = re.compile(r"\s*[—–]\s*")
_SPACE = re.compile(r"\s{2,}")


def flatten(text, rng, comma_drop=0.45):
    """Human prose -> machine cadence. Punctuation and boundaries only."""
    text = _PARENS.sub(r", \1", text)          # unwrap rather than delete: keep the content
    text = _DASH.sub(". ", text)
    text = text.replace(";", ".").replace(":", ".")

    # cut at conjunction joins, capitalising the new sentence
    def cut(m):
        return ". " + m.group(1).capitalize() + " " if rng.random() < 0.75 else m.group(0)
    text = _SPLIT.sub(cut, text)

    # thin the commas: machine text carries noticeably fewer per word
    out = []
    for ch in text:
        if ch == "," and rng.random() < comma_drop:
            continue
        out.append(ch)
    text = _SPACE.sub(" ", "".join(out)).strip()

    # Regularise toward a uniform length. Splitting alone makes text *more*
    # bursty, not less -- irregular cuts add variance -- so long sentences are
    # cut and short ones absorbed, converging on machine-like even cadence.
    target = 17
    fixed = []
    for s in sentences(text):
        words = s.split()
        while len(words) > target * 1.6:
            cut = target
            fixed.append(" ".join(words[:cut]).rstrip(",;:") + ".")
            words = words[cut:]
            words[0] = words[0].capitalize()
        fixed.append(" ".join(words))

    merged = []
    for s in fixed:
        if merged and len(s.split()) < target * 0.6 and len(merged[-1].split()) < target * 1.2:
            merged[-1] = merged[-1].rstrip(".") + ", " + s[0].lower() + s[1:]
        else:
            merged.append(s)
    return " ".join(merged)


_DIGIT = re.compile(r"\d")


def is_prose(text):
    """Reject citations, indexes and catalogues -- they make useless targets.

    A public-domain corpus carries a lot of non-prose, and a pair built from a
    table of statutes teaches the model to emit tables of statutes.
    """
    words = text.split()
    if not words:
        return False
    if sum(bool(_DIGIT.search(w)) for w in words) / len(words) > 0.06:
        return False
    lens = [len(s.split()) for s in sentences(text)]
    if not lens or min(lens) < 4 or max(lens) > 90:
        return False
    return 9 <= sum(lens) / len(lens) <= 45


def spans(path, min_sents=3, max_sents=5, min_words=35, max_words=85):
    """Multi-sentence spans: long enough to carry rhythm, short enough that
    *both* halves of the pair fit a 256-token context together.

    85 words is roughly 115 tokens, so source + target + specials lands near
    240. Earlier limits looked generous and silently dropped a third of the
    pairs at encode time.
    """
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        sents = sentences(line)
        i = 0
        while i < len(sents):
            take = min_sents
            span = " ".join(sents[i:i + take])
            while take < max_sents and len(span.split()) < min_words and i + take < len(sents):
                take += 1
                span = " ".join(sents[i:i + take])
            n = len(span.split())
            if min_words <= n <= max_words:
                yield span
            i += take


def build(split, limit, seed=0):
    src = Path(f"data/processed/{split}.txt")
    if not src.exists():
        raise SystemExit(f"missing {src} -- run `python -m data.prepare clean` first")
    rng = random.Random(seed)
    OUT.mkdir(parents=True, exist_ok=True)
    kept = 0
    with (OUT / f"{split}.jsonl").open("w", encoding="utf-8") as out:
        for span in spans(src):
            if not is_prose(span):
                continue
            flat = flatten(span, rng)
            # a pair only teaches something if flattening actually changed the shape
            if flat == span or len(flat.split()) < 20:
                continue
            out.write(json.dumps({"src": flat, "tgt": span}) + "\n")
            kept += 1
            if kept >= limit:
                break
    print(f"{split:5s} {kept:>8,} pairs -> data/pairs/{split}.jsonl")
    return kept


def verify(split="train", n=400):
    """The corruption must move the three measured features the right way."""
    import statistics as st
    rows = [json.loads(l) for l in (OUT / f"{split}.jsonl").open()][:n]
    f_src = [extract(r["src"]) for r in rows]
    f_tgt = [extract(r["tgt"]) for r in rows]

    def mean(fs, k):
        return st.mean(f[k] for f in fs)

    print(f"\n{'feature':18s} {'flattened':>10s} {'natural':>10s} {'delta':>9s}")
    ok = True
    for k in ("sent_len_words", "punct_ratio", "burstiness"):
        a, b = mean(f_src, k), mean(f_tgt, k)
        print(f"{k:18s} {a:10.4f} {b:10.4f} {100 * (b / a - 1):+8.1f}%")
        ok &= b > a          # the natural side must be higher on all three
    assert ok, "flattening did not move all three target features"
    print("\nflattening inverts the measured human/machine gap on every target feature")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs", type=int, default=200_000)
    ap.add_argument("--val-pairs", type=int, default=2_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    build("train", a.pairs, a.seed)
    build("val", a.val_pairs, a.seed + 1)
    verify()
