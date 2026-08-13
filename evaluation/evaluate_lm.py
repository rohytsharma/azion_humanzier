"""Language-model quality: held-out loss and perplexity (PRD 12, SYS-F13).

    python -m evaluation.evaluate_lm --split test
    python -m evaluation.evaluate_lm --ckpt training/checkpoints/last.pt --device cpu

Uses strided evaluation: a fixed-context model scoring back-to-back windows
grades the first tokens of every window on almost no context, which inflates
perplexity. Each window here is scored only on the tokens the stride advanced
past, so every token is predicted with a full context behind it.

The training run holds the GPU, so --device cpu is the polite way to check a
checkpoint mid-run.
"""
import argparse
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch

from model.config import Config
from model.model import HumanWriterLM


def load_checkpoint(path, device):
    """Copy first: the trainer rewrites best.pt periodically, and reading it
    mid-write gives a truncated file rather than a clear error."""
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        shutil.copyfile(path, tmp.name)
        ck = torch.load(tmp.name, map_location=device, weights_only=True)
    Path(tmp.name).unlink()
    cfg = Config(**ck["config"])
    model = HumanWriterLM(cfg).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, cfg, ck.get("step", -1)


@torch.no_grad()
def perplexity(model, cfg, data, device, stride=None, max_tokens=None, batch_size=4):
    """Token-level NLL over `data`, scoring each token once with full context."""
    stride = stride or cfg.block_size // 2
    n = len(data) if max_tokens is None else min(len(data), max_tokens)

    starts, prev_end = [], 0
    for begin in range(0, n - 1, stride):
        end = min(begin + cfg.block_size, n - 1)
        if end - prev_end <= 0:
            continue
        starts.append((begin, end, end - prev_end))
        prev_end = end
        if end >= n - 1:
            break

    total_nll, total_tok = 0.0, 0
    for i in range(0, len(starts), batch_size):
        chunk = starts[i:i + batch_size]
        width = max(e - b for b, e, _ in chunk)
        xs, ys = [], []
        for b, e, trg in chunk:
            x = torch.from_numpy(data[b:b + width].astype(np.int64))
            y = torch.from_numpy(data[b + 1:b + 1 + width].astype(np.int64))
            y = y.clone()
            y[: (e - b) - trg] = -100          # already scored by an earlier window
            if (e - b) < width:
                y[(e - b):] = -100             # past the end of this window
            xs.append(x)
            ys.append(y)
        x = torch.stack(xs).to(device)
        y = torch.stack(ys).to(device)
        loss = model(x, y)[1]
        n_tok = int((y != -100).sum())
        total_nll += loss.item() * n_tok
        total_tok += n_tok

    mean_nll = total_nll / total_tok
    return mean_nll, math.exp(min(mean_nll, 20)), total_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="training/checkpoints/best.pt")
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    ap.add_argument("--stride", type=int, default=None, help="default: half the context")
    ap.add_argument("--max-tokens", type=int, default=2_000_000)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    if not Path(a.ckpt).exists():
        raise SystemExit(f"no checkpoint at {a.ckpt} -- train one first")

    device = torch.device(a.device)
    model, cfg, step = load_checkpoint(a.ckpt, device)
    data = np.memmap(f"data/splits/{a.split}.bin", dtype=np.uint16, mode="r")

    nll, ppl, n_tok = perplexity(model, cfg, data, device, a.stride,
                                 a.max_tokens, a.batch_size)
    print(f"checkpoint  {a.ckpt} (step {step})")
    print(f"split       {a.split}, {n_tok:,} tokens scored of {len(data):,}")
    print(f"loss        {nll:.4f}")
    print(f"perplexity  {ppl:,.2f}")


if __name__ == "__main__":
    main()
