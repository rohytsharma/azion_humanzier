"""Corpus pipeline (SYS-F01/F02/F04, PRD 11).

    python -m data.prepare clean            # data/raw/*.txt -> data/processed/{train,val,test}.txt
    python -m data.prepare encode           # ... -> data/splits/*.bin (uint16 token ids)

Documents are split on blank lines, normalised, deduplicated by hash and
assigned to a split by a hash of their content, so the same document always
lands in the same split and re-running cannot leak test data into train.
"""
import argparse
import hashlib
import re
import unicodedata
from pathlib import Path

RAW, PROC, SPLITS = Path("data/raw"), Path("data/processed"), Path("data/splits")
VAL_PCT, TEST_PCT = 1, 1  # percent; raise for small corpora

_ws = re.compile(r"[ \t]+")
_nl = re.compile(r"\n{3,}")


def normalise(doc: str) -> str:
    doc = unicodedata.normalize("NFKC", doc)
    doc = doc.replace("\r\n", "\n").replace(" ", " ")
    doc = _ws.sub(" ", doc)
    return _nl.sub("\n\n", doc).strip()


def bucket(doc: str, val_pct=VAL_PCT, test_pct=TEST_PCT) -> str:
    """Stable 0-99 bucket from content hash -> deterministic, leak-proof splits."""
    h = int(hashlib.sha1(doc.encode()).hexdigest()[:8], 16) % 100
    if h < val_pct:
        return "val"
    if h < val_pct + test_pct:
        return "test"
    return "train"


def clean(min_chars=200, val_pct=VAL_PCT, test_pct=TEST_PCT):
    files = sorted(RAW.rglob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt files under {RAW}/ -- add corpus files first")

    seen, out = set(), {"train": [], "val": [], "test": []}
    kept = dropped = 0
    for f in files:
        for doc in f.read_text(encoding="utf-8", errors="ignore").split("\n\n"):
            doc = normalise(doc)
            key = hashlib.sha1(doc.lower().encode()).digest()
            if len(doc) < min_chars or key in seen:
                dropped += 1
                continue
            seen.add(key)
            out[bucket(doc, val_pct, test_pct)].append(doc)
            kept += 1

    PROC.mkdir(parents=True, exist_ok=True)
    for split, docs in out.items():
        p = PROC / f"{split}.txt"
        p.write_text("\n\n".join(docs) + "\n", encoding="utf-8")
        print(f"{split:5s} {len(docs):>8,} docs  {p.stat().st_size/1e6:>8.1f} MB")
    print(f"kept {kept:,} / dropped {dropped:,} (short or duplicate) from {len(files)} file(s)")


def encode(tokenizer_path: str):
    import numpy as np
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(tokenizer_path)
    assert tok.get_vocab_size() <= 65535, "uint16 storage needs vocab <= 65535"
    eos = tok.token_to_id("<eos>")

    SPLITS.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        src = PROC / f"{split}.txt"
        if not src.exists():
            continue
        ids = []
        for doc in src.read_text(encoding="utf-8").split("\n\n"):
            if doc.strip():
                ids.extend(tok.encode(doc).ids + [eos])
        arr = np.array(ids, dtype=np.uint16)
        arr.tofile(SPLITS / f"{split}.bin")
        print(f"{split:5s} {len(arr):>12,} tokens -> data/splits/{split}.bin")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["clean", "encode"])
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer_files/bpe.json")
    ap.add_argument("--min-chars", type=int, default=200, help="drop documents shorter than this")
    ap.add_argument("--val-pct", type=int, default=VAL_PCT)
    ap.add_argument("--test-pct", type=int, default=TEST_PCT)
    a = ap.parse_args()
    clean(a.min_chars, a.val_pct, a.test_pct) if a.cmd == "clean" else encode(a.tokenizer)
