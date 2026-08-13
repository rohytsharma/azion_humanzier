"""Corpus pipeline (SYS-F01/F02/F04, PRD 11).

    python -m data.prepare clean            # data/raw/*.txt -> data/processed/{train,val,test}.txt
    python -m data.prepare encode           # ... -> data/splits/*.bin (uint16 token ids)

Everything streams line by line, so a multi-GB corpus never lands in RAM.

A "document" here is a block of consecutive paragraphs of at least --min-chars.
With a 256-token context the model cannot learn coherence beyond roughly that
span anyway, and grouping keeps short dialogue paragraphs -- which carry most of
the conversational style we want -- instead of filtering them out.

Split assignment is a hash of the document text, so re-running after adding more
corpus can never move a test document into train.
"""
import argparse
import hashlib
import re
import unicodedata
from pathlib import Path

RAW, PROC, SPLITS = Path("data/raw"), Path("data/processed"), Path("data/splits")
VAL_PCT, TEST_PCT = 1, 1  # percent; raise for small corpora

WRAP_WIDTH = 50            # lines shorter than this end a paragraph naturally
_ENDINGS = '.!?:;"”’)]}»'
_ws = re.compile(r"[ \t]+")


def normalise_line(line: str) -> str:
    line = unicodedata.normalize("NFKC", line).replace(" ", " ")
    return _ws.sub(" ", line).strip()


def _finished(line: str) -> bool:
    """True if the line ends a paragraph rather than being a wrap point."""
    return len(line) < WRAP_WIDTH or line[-1] in _ENDINGS


def iter_paragraphs(path: Path):
    """Stream paragraphs, rejoining hard-wrapped lines.

    Many Project Gutenberg texts wrap at ~79 chars and some put a blank line
    between every wrapped line; read naively, each fragment looks like its own
    document and the length filter deletes most of the corpus.

    ponytail: line-level heuristic, so verse and tables flatten into prose.
    Fine for LM pretraining; revisit if poetry becomes a target style.
    """
    cur = ""
    with path.open(encoding="utf-8", errors="ignore") as fh:
        for raw in fh:
            line = normalise_line(raw)
            if not line:
                if cur and _finished(cur):
                    yield cur
                    cur = ""
                continue                        # else: a wrap artefact, drop it
            if cur and not _finished(cur):
                cur += " " + line
            else:
                if cur:
                    yield cur
                cur = line
    if cur:
        yield cur


def iter_docs(path: Path, min_chars: int):
    """Group consecutive paragraphs into documents of at least min_chars."""
    block, size = [], 0
    for para in iter_paragraphs(path):
        block.append(para)
        size += len(para) + 1
        if size >= min_chars:
            yield "\n".join(block)
            block, size = [], 0
    if block:
        yield "\n".join(block)


def bucket(doc: str, val_pct=VAL_PCT, test_pct=TEST_PCT) -> str:
    h = int(hashlib.sha1(doc.encode()).hexdigest()[:8], 16) % 100
    if h < val_pct:
        return "val"
    if h < val_pct + test_pct:
        return "test"
    return "train"


def clean(min_chars=200, val_pct=VAL_PCT, test_pct=TEST_PCT):
    files = sorted(RAW.rglob("*.txt"))
    if not files:
        raise SystemExit(f"no .txt files under {RAW}/ -- run `python -m data.fetch` first")

    PROC.mkdir(parents=True, exist_ok=True)
    handles = {s: (PROC / f"{s}.txt").open("w", encoding="utf-8") for s in ("train", "val", "test")}
    seen, counts = set(), dict(train=0, val=0, test=0)
    kept = dropped = 0

    for f in files:
        for doc in iter_docs(f, min_chars):
            key = hashlib.sha1(doc.lower().encode()).digest()
            if len(doc) < min_chars or key in seen:
                dropped += 1
                continue
            seen.add(key)
            split = bucket(doc, val_pct, test_pct)
            handles[split].write(doc + "\n\n")
            counts[split] += 1
            kept += 1

    for s, h in handles.items():
        h.close()
        print(f"{s:5s} {counts[s]:>9,} docs  {(PROC / f'{s}.txt').stat().st_size/1e6:>8.1f} MB")
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
        total = 0
        with (SPLITS / f"{split}.bin").open("wb") as out:
            batch = []
            for doc in _read_docs(src):
                batch.append(doc)
                if len(batch) >= 1000:
                    total += _flush(tok, batch, eos, out, np)
                    batch = []
            total += _flush(tok, batch, eos, out, np)
        print(f"{split:5s} {total:>12,} tokens -> data/splits/{split}.bin")


def _read_docs(path: Path):
    """Stream blank-line-separated documents back out of a processed split."""
    lines = []
    with path.open(encoding="utf-8") as fh:
        for raw in fh:
            if raw.strip():
                lines.append(raw.rstrip("\n"))
            elif lines:
                yield "\n".join(lines)
                lines = []
    if lines:
        yield "\n".join(lines)


def _flush(tok, batch, eos, out, np):
    if not batch:
        return 0
    ids = []
    for enc in tok.encode_batch(batch):
        ids.extend(enc.ids)
        ids.append(eos)
    np.array(ids, dtype=np.uint16).tofile(out)
    return len(ids)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["clean", "encode"])
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer_files/bpe.json")
    ap.add_argument("--min-chars", type=int, default=200, help="minimum document length")
    ap.add_argument("--val-pct", type=int, default=VAL_PCT)
    ap.add_argument("--test-pct", type=int, default=TEST_PCT)
    a = ap.parse_args()
    clean(a.min_chars, a.val_pct, a.test_pct) if a.cmd == "clean" else encode(a.tokenizer)
