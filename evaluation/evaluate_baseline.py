"""Baseline comparison against a larger pretrained model (PRD objective 10, PRD 12).

    python -m evaluation.evaluate_baseline --max-bytes 400000

GPT-2 (124M) runs locally from downloaded weights. No inference API is involved,
which is the constraint the PRD places on this comparison.

Perplexity cannot be compared across models with different tokenizers: it is
per-token, and a model whose tokenizer packs more characters into each token
scores a lower number for free. Ours has a 16k vocabulary, GPT-2 has 50k, so its
tokens cover more text and its perplexity would look better on that alone.

**Bits per byte** removes the tokenizer from the comparison. Total negative
log-likelihood over a passage is a property of the model's distribution over
*text*; dividing by the byte count of that same text gives a number both models
can be scored on honestly.

    bits/byte = (sum of token NLL in nats / ln 2) / bytes of text

GPT-2 is scored twice: once at our 256-token context so the comparison is
like-for-like, and once at its native 1024, which is its best case. Reporting
only the handicapped number would flatter us.
"""
import argparse
import math
from pathlib import Path

import torch

from evaluation.evaluate_lm import load_checkpoint


def read_text(path, max_bytes):
    raw = Path(path).read_text(encoding="utf-8")[:max_bytes]
    # whole paragraphs only, so neither model is charged for a severed clause
    return raw[:raw.rfind("\n\n")] if "\n\n" in raw else raw


@torch.no_grad()
def bits_per_byte(score_window, ids, n_bytes, context, stride=None):
    """Strided NLL over token ids, converted to bits per byte of source text."""
    stride = stride or context // 2
    total_nll, prev_end = 0.0, 0
    for begin in range(0, len(ids) - 1, stride):
        end = min(begin + context, len(ids) - 1)
        n_new = end - prev_end
        if n_new <= 0:
            continue
        total_nll += score_window(begin, end, n_new)
        prev_end = end
        if end >= len(ids) - 1:
            break
    return (total_nll / math.log(2)) / n_bytes


def score_ours(ckpt, text, device):
    from tokenizers import Tokenizer
    model, cfg, step = load_checkpoint(ckpt, device)
    tok = Tokenizer.from_file("tokenizer/tokenizer_files/bpe.json")
    ids = tok.encode(text).ids

    def window(begin, end, n_new):
        x = torch.tensor([ids[begin:end]], device=device)
        y = torch.tensor([ids[begin + 1:end + 1]], device=device)
        y[:, : (end - begin) - n_new] = -100
        loss = model(x, y)[1]
        return loss.item() * int((y != -100).sum())

    n_bytes = len(text.encode("utf-8"))
    return bits_per_byte(window, ids, n_bytes, cfg.block_size), len(ids), step


def score_gpt2(name, text, device, context):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name).to(device).eval()
    ids = tok(text)["input_ids"]

    def window(begin, end, n_new):
        x = torch.tensor([ids[begin:end]], device=device)
        y = x.clone()
        y[:, : (end - begin) - n_new] = -100
        out = model(x, labels=y)
        n_scored = int((y[:, 1:] != -100).sum())
        return out.loss.item() * n_scored

    n_bytes = len(text.encode("utf-8"))
    n_params = sum(p.numel() for p in model.parameters())
    return bits_per_byte(window, ids, n_bytes, context), len(ids), n_params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="training/checkpoints/best.pt")
    ap.add_argument("--split", default="data/processed/test.txt")
    ap.add_argument("--baseline", default="gpt2")
    ap.add_argument("--max-bytes", type=int, default=400_000)
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args()

    device = torch.device(a.device)
    text = read_text(a.split, a.max_bytes)
    n_bytes = len(text.encode("utf-8"))
    print(f"held-out test text: {n_bytes:,} bytes\n")

    ours_bpb, ours_tokens, step = score_ours(a.ckpt, text, device)
    base_matched, base_tokens, base_params = score_gpt2(a.baseline, text, device, 256)
    base_native, _, _ = score_gpt2(a.baseline, text, device, 1024)

    print(f"{'model':28s} {'params':>9s} {'context':>8s} {'tokens':>9s} {'bits/byte':>10s}")
    print(f"{'HumanWriter (ours)':28s} {'39.8M':>9s} {256:>8} {ours_tokens:>9,} {ours_bpb:>10.4f}")
    print(f"{a.baseline + ' (matched ctx)':28s} {base_params / 1e6:>8.1f}M {256:>8} "
          f"{base_tokens:>9,} {base_matched:>10.4f}")
    print(f"{a.baseline + ' (native ctx)':28s} {base_params / 1e6:>8.1f}M {1024:>8} "
          f"{base_tokens:>9,} {base_native:>10.4f}")

    ratio = ours_bpb / base_native
    print(f"\nOurs uses {39.8 / (base_params / 1e6):.2f}x the parameters and needs "
          f"{ratio:.2f}x the bits per byte of the baseline at its best.")
    print("Lower is better. Bits per byte is tokenizer-independent; per-token "
          "perplexity is not\nand would have flattered whichever model has the "
          "coarser vocabulary.")
    print("\nThe two corpora differ: ours saw Gutenberg and Wikipedia, GPT-2 saw "
          "WebText, and this\ntest split is from ours -- an advantage to us that "
          "the gap has to be read against.")


if __name__ == "__main__":
    main()
