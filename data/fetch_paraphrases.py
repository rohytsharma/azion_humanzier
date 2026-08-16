"""Add real paraphrase pairs to the fine-tuning set (PRD 11, SYS-F10).

    python -m data.fetch_paraphrases --limit 40000

The synthetic pairs in make_pairs.py only move punctuation and sentence
boundaries, so just 6.4% of target tokens differ from the source and the model
keeps rediscovering that copying is nearly optimal. PAWS pairs are genuine
rewrites -- words substituted, clauses reordered -- so a much larger share of
each target has to be produced rather than echoed.

The two sets teach different halves of the job and are meant to be mixed:
  make_pairs      punctuation, sentence joins, rhythm  (the style direction)
  paraphrases     word choice and clause order          (real rewriting)

PAWS is Wikipedia-derived and human-annotated. Only label == 1 rows are true
paraphrases; the label == 0 rows are adversarial near-misses that mean something
different, and training on those would teach the model to change the meaning.
"""
import argparse
import json
import re
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

REPO = "google-research-datasets/paws"
CONFIG = "labeled_final"
OUT = Path("data/pairs/paraphrases.jsonl")
LICENCE = "PAWS, Google Research — free for research/commercial use with attribution"

# PAWS ships tokenised: "In Paris , in October 1560 , he met ..."
_SPACE_BEFORE = re.compile(r"\s+([,.;:!?%)\]])")
_SPACE_AFTER = re.compile(r"([(\[])\s+")
_CONTRACTION = re.compile(r"\s+('(?:s|t|re|ve|ll|d|m))\b")


def detokenise(s):
    s = _SPACE_BEFORE.sub(r"\1", s)
    s = _SPACE_AFTER.sub(r"\1", s)
    s = _CONTRACTION.sub(r"\1", s)
    return re.sub(r"\s{2,}", " ", s).strip()


def shard_urls():
    api = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{CONFIG}"
    with urllib.request.urlopen(api, timeout=60) as r:
        tree = json.load(r)
    names = sorted(x["path"] for x in tree
                   if x["path"].endswith(".parquet") and "train" in x["path"])
    return [f"https://huggingface.co/datasets/{REPO}/resolve/main/{n}" for n in names]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=40000)
    ap.add_argument("--min-words", type=int, default=8)
    a = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path("data/.paws.parquet")
    kept = dropped = 0

    with OUT.open("w", encoding="utf-8") as out:
        for url in shard_urls():
            if kept >= a.limit:
                break
            print(f"shard {url.rsplit('/', 1)[1]} …", end=" ", flush=True)
            urllib.request.urlretrieve(url, tmp)
            table = pq.read_table(tmp, columns=["sentence1", "sentence2", "label"])
            for s1, s2, lab in zip(table.column(0).to_pylist(),
                                   table.column(1).to_pylist(),
                                   table.column(2).to_pylist()):
                if lab != 1:                       # 0 = adversarial non-paraphrase
                    dropped += 1
                    continue
                src, tgt = detokenise(s1), detokenise(s2)
                if len(src.split()) < a.min_words or len(tgt.split()) < a.min_words:
                    dropped += 1
                    continue
                if src == tgt:
                    dropped += 1
                    continue
                out.write(json.dumps({"src": src, "tgt": tgt}) + "\n")
                kept += 1
                if kept >= a.limit:
                    break
            tmp.unlink()
            print(f"{kept:,} kept")

    print(f"\n{kept:,} paraphrase pairs -> {OUT}  ({dropped:,} rejected)")
    src_file = Path("data/SOURCES.md")
    if src_file.exists() and "PAWS" not in src_file.read_text():
        with src_file.open("a") as fh:
            fh.write(f"\n## paraphrase pairs\n- Source: https://huggingface.co/datasets/{REPO}"
                     f"\n- Licence: {LICENCE}\n- Kept: {kept:,} pairs (label == 1 only)\n")
        print("recorded in data/SOURCES.md")


if __name__ == "__main__":
    main()
