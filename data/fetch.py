"""Fetch the pretraining corpus (PRD 11: legally usable, sources recorded).

    python -m data.fetch --target-tokens 800000000       # ~48h of M2 training
    python -m data.fetch --target-tokens 5000000         # a quick end-to-end trial

Parquet shards are pulled straight over HTTPS and read in batches, so peak RAM
stays around one batch rather than one shard. Each shard is deleted once its
text has been extracted; re-running skips sources already at target.

Sources and licences are written to data/SOURCES.md as they are fetched.
"""
import argparse
import json
import random
import urllib.request
from pathlib import Path

import pyarrow.parquet as pq

RAW = Path("data/raw")
CHARS_PER_TOKEN = 4  # rough BPE ratio for English; only used to decide when to stop

SOURCES = {
    "gutenberg": dict(
        repo="sedthh/gutenberg_english",
        prefix="data/train-{i:05d}-of-00037",
        n_shards=37,
        column="TEXT",
        licence="Public domain (Project Gutenberg); dataset packaging MIT",
        url="https://huggingface.co/datasets/sedthh/gutenberg_english",
    ),
    "wikipedia": dict(
        repo="wikimedia/wikipedia",
        prefix="20231101.en/train-{i:05d}-of-00041",
        n_shards=41,
        column="text",
        licence="CC BY-SA 3.0 / GFDL",
        url="https://huggingface.co/datasets/wikimedia/wikipedia",
    ),
}

# Gutenberg texts still carry the transcriber's header/footer around the work itself.
PG_START, PG_END = "*** START OF", "*** END OF"


def shard_urls(src, seed=1):
    """HF stores shard filenames with a content hash suffix, so list the tree.

    Shards are shuffled: Gutenberg shard order tracks catalogue ID, so taking
    the first N sequentially yields a corpus of Bibles and legal texts. Fixed
    seed keeps a partial fetch reproducible.
    """
    repo = src["repo"]
    api = f"https://huggingface.co/api/datasets/{repo}/tree/main/{src['prefix'].split('/')[0]}"
    with urllib.request.urlopen(api, timeout=60) as r:
        tree = json.load(r)
    names = sorted(x["path"] for x in tree if x["path"].endswith(".parquet"))
    random.Random(seed).shuffle(names)
    return [f"https://huggingface.co/datasets/{repo}/resolve/main/{n}" for n in names]


def _state(update=None):
    """Remember which shards are already consumed so an interrupted fetch resumes."""
    p = Path("data/.fetch_state.json")
    s = json.loads(p.read_text()) if p.exists() else {}
    if update:
        s.update(update)
        p.write_text(json.dumps(s))
    return s


def strip_boilerplate(text):
    if PG_START in text:
        text = text.split(PG_START, 1)[1].split("\n", 1)[-1]
    if PG_END in text:
        text = text.split(PG_END, 1)[0]
    return text.replace("\r\n", "\n").strip()


def fetch(name, target_tokens, seed=1):
    src = SOURCES[name]
    RAW.mkdir(parents=True, exist_ok=True)
    out_path = RAW / f"{name}.txt"
    target_chars = target_tokens * CHARS_PER_TOKEN
    chars = out_path.stat().st_size if out_path.exists() else 0
    done = _state().get(name, 0)
    if chars >= target_chars:
        print(f"{name}: already at {chars/1e9:.2f} GB, skipping")
        return chars

    tmp = Path(f"data/.{name}.parquet")
    with out_path.open("a", encoding="utf-8") as out:
        for i, url in enumerate(shard_urls(src, seed)):
            if chars >= target_chars:
                break
            if i < done:      # already consumed by an earlier run
                continue
            print(f"{name}: shard {i} ...", end=" ", flush=True)
            urllib.request.urlretrieve(url, tmp)
            for batch in pq.ParquetFile(tmp).iter_batches(batch_size=200, columns=[src["column"]]):
                for doc in batch.column(0).to_pylist():
                    if not doc:
                        continue
                    doc = strip_boilerplate(doc) if name == "gutenberg" else doc.strip()
                    out.write(doc + "\n\n")
                    chars += len(doc) + 2
                    if chars >= target_chars:  # per-document: one book is ~0.5 MB
                        break
                if chars >= target_chars:
                    break
            tmp.unlink()
            out.flush()
            _state({name: i + 1})
            print(f"{chars/1e9:.2f} GB / {target_chars/1e9:.2f} GB")
    return chars


def write_manifest(got):
    lines = ["# Corpus sources (PRD 11)", ""]
    for name, chars in got.items():
        s = SOURCES[name]
        lines += [f"## {name}", f"- Source: {s['url']}", f"- Licence: {s['licence']}",
                  f"- Fetched: {chars/1e9:.2f} GB (~{chars//CHARS_PER_TOKEN/1e6:.0f}M tokens)", ""]
    Path("data/SOURCES.md").write_text("\n".join(lines))
    print("wrote data/SOURCES.md")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-tokens", type=int, default=800_000_000)
    ap.add_argument("--gutenberg-frac", type=float, default=0.7,
                    help="books teach natural prose, wikipedia keeps the vocabulary modern")
    ap.add_argument("--only", choices=list(SOURCES), default=None)
    ap.add_argument("--seed", type=int, default=1,
                    help="shard shuffle order; seed 0 happens to start on the catalogue's "
                         "opening texts (Bibles, statutes), which skews a small trial fetch")
    a = ap.parse_args()

    split = {"gutenberg": a.gutenberg_frac, "wikipedia": 1 - a.gutenberg_frac}
    got = {n: fetch(n, int(a.target_tokens * f), a.seed)
           for n, f in split.items() if a.only in (None, n)}
    write_manifest(got)
    print(f"\ntotal ~{sum(got.values())//CHARS_PER_TOKEN/1e6:.0f}M tokens in data/raw/")
