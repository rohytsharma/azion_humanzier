"""Supervised fine-tuning for rewriting (SYS-F10, FR-05).

    python -m training.finetune --epochs 2

Run this after pretraining finishes -- it starts from that checkpoint and would
otherwise compete with it for the GPU.

Each example is packed as

    <bos> <src> flattened text <tgt> natural text <eos>

and the loss is masked over everything up to and including <tgt>. The model is
therefore only ever graded on what it produces, never on copying the prompt back
-- without the mask, most of the gradient signal is the trivial task of echoing
the source, and the rewrite quality suffers for it.
"""
import argparse
import json
import math
import time
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

from model.config import Config
from model.model import HumanWriterLM

CKPT_DIR = Path("training/checkpoints")
PAIRS = Path("data/pairs")
TOKENIZER = "tokenizer/tokenizer_files/bpe.json"


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def encode_pairs(path, tok, block_size, limit=None):
    """-> list of (ids, n_prompt, changed). Examples longer than the context are
    dropped rather than truncated: a truncated target teaches the model to stop
    mid-clause.

    `changed` marks the target tokens that are NOT already present in the source
    at the aligned position. Source and target overlap by ~91% of words -- the
    rewrite is mostly a copy with insertions -- so an unweighted loss is
    minimised by echoing the input, which is exactly what the first run learned.
    Those flags let the trainer put the gradient where the work is.
    """
    bos, eos = tok.token_to_id("<bos>"), tok.token_to_id("<eos>")
    src_t, tgt_t = tok.token_to_id("<src>"), tok.token_to_id("<tgt>")

    rows = []
    with path.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if limit and i >= limit:
                break
            rows.append(json.loads(line))

    out = []
    srcs = tok.encode_batch([r["src"] for r in rows])
    tgts = tok.encode_batch([r["tgt"] for r in rows])
    changed_frac = []
    for s, t in zip(srcs, tgts):
        ids = [bos, src_t] + s.ids + [tgt_t] + t.ids + [eos]
        if len(ids) > block_size:
            continue

        # which target tokens are new relative to the source
        changed = np.ones(len(t.ids) + 1, dtype=np.uint8)      # +1 for <eos>
        for blk in SequenceMatcher(None, s.ids, t.ids, autojunk=False).get_matching_blocks():
            changed[blk.b:blk.b + blk.size] = 0
        changed_frac.append(float(changed[:-1].mean()) if len(t.ids) else 0.0)

        out.append((np.array(ids, dtype=np.uint16), 2 + len(s.ids) + 1, changed))

    print(f"  {path.name}: {len(out):,} examples fit in {block_size} tokens "
          f"({len(rows) - len(out):,} too long, dropped); "
          f"{np.mean(changed_frac):.1%} of target tokens differ from the source")
    return out


def batches(data, batch_size, pad_id, device, shuffle=True, seed=0,
            changed_weight=1.0, fixed_width=256):
    order = np.arange(len(data))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for i in range(0, len(order) - batch_size + 1, batch_size):
        chunk = [data[j] for j in order[i:i + batch_size]]
        # Fixed width, not the batch maximum: a new tensor shape makes Metal
        # recompile its kernels, and variable-width batches ran 8x slower than
        # pretraining did on the same token count.
        width = fixed_width
        x = np.full((len(chunk), width - 1), pad_id, dtype=np.int64)
        y = np.full((len(chunk), width - 1), -100, dtype=np.int64)
        w = np.zeros((len(chunk), width - 1), dtype=np.float32)
        for r, (ids, n_prompt, changed) in enumerate(chunk):
            seq = ids.astype(np.int64)
            x[r, :len(seq) - 1] = seq[:-1]
            tgt = seq[1:].copy()
            tgt[:n_prompt - 1] = -100          # the prompt half is not scored
            y[r, :len(tgt)] = tgt
            row = np.where(changed.astype(bool), changed_weight, 1.0).astype(np.float32)
            w[r, n_prompt - 1:n_prompt - 1 + len(row)] = row
        yield (torch.from_numpy(x).to(device), torch.from_numpy(y).to(device),
               torch.from_numpy(w).to(device))


def weighted_loss(model, x, y, w):
    """Cross-entropy with per-token weights, so inserted punctuation and
    connectives carry more gradient than the words being copied through."""
    logits = model(x)[0]
    flat = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)), y.reshape(-1),
        ignore_index=-100, reduction="none")
    wf = w.reshape(-1) * (y.reshape(-1) != -100)
    total = wf.sum()
    return (flat * wf).sum() / torch.clamp(total, min=1.0)


@torch.no_grad()
def evaluate(model, data, args, pad_id, device, max_batches=40):
    model.eval()
    losses = []
    for i, (x, y, w) in enumerate(batches(data, args.batch_size, pad_id, device,
                                          shuffle=False, changed_weight=args.changed_weight,
                                          fixed_width=args.block_width)):
        losses.append(weighted_loss(model, x, y, w).item())
        if i + 1 >= max_batches:
            break
    model.train()
    return sum(losses) / max(1, len(losses))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", default=str(CKPT_DIR / "best.pt"))
    ap.add_argument("--out", default=str(CKPT_DIR / "finetuned.pt"))
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=8e-5, help="well below pretraining: "
                    "a high rate here erases what pretraining learned")
    ap.add_argument("--min-lr", type=float, default=8e-6)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--weight-decay", type=float, default=0.05)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=200)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--changed-weight", type=float, default=6.0,
                    help="gradient multiplier on target tokens absent from the source; "
                         "at 1.0 the model minimises loss by copying the input verbatim")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    if not Path(args.init).exists():
        raise SystemExit(f"no pretrained checkpoint at {args.init} -- run training.pretrain first")

    torch.manual_seed(args.seed)
    device = pick_device(args.device)
    tok = Tokenizer.from_file(TOKENIZER)
    pad_id = tok.token_to_id("<pad>")

    ck = torch.load(args.init, map_location="cpu", weights_only=True)
    cfg = Config(**ck["config"])
    model = HumanWriterLM(cfg).to(device)
    model.load_state_dict(ck["model"])
    print(f"device={device}  init={args.init} (pretrain step {ck.get('step', '?')})")

    train = encode_pairs(PAIRS / "train.jsonl", tok, cfg.block_size, args.limit)
    val = encode_pairs(PAIRS / "val.jsonl", tok, cfg.block_size)
    args.block_width = cfg.block_size
    if not train:
        raise SystemExit("no usable pairs -- run `python -m data.make_pairs` first")

    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}], lr=args.lr, betas=(0.9, 0.95))

    steps_per_epoch = len(train) // (args.batch_size * args.grad_accum)
    total = steps_per_epoch * args.epochs
    print(f"{steps_per_epoch:,} steps/epoch x {args.epochs} = {total:,} steps")

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        r = (s - args.warmup) / max(1, total - args.warmup)
        return args.min_lr + 0.5 * (1 + math.cos(math.pi * min(r, 1.0))) * (args.lr - args.min_lr)

    best = float("inf")
    step, t0 = 0, time.time()
    model.train()
    for epoch in range(args.epochs):
        it = batches(train, args.batch_size, pad_id, device, seed=args.seed + epoch,
                     changed_weight=args.changed_weight, fixed_width=args.block_width)
        done = False
        while not done:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            for _ in range(args.grad_accum):
                try:
                    x, y, w = next(it)
                except StopIteration:
                    done = True
                    break
                loss = weighted_loss(model, x, y, w) / args.grad_accum
                loss.backward()
            if done:
                break

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

            # MPS holds on to freed blocks and the machine starts swapping; an
            # unweighted first run decayed from 1.7 s/step to 37 s/step this way.
            if device.type == "mps" and step % 50 == 0:
                torch.mps.empty_cache()

            print(f"\r  epoch {epoch + 1}/{args.epochs}  step {step}/{total}  "
                  f"loss {loss.item() * args.grad_accum:.3f}  "
                  f"{(time.time() - t0) / max(step, 1):.2f}s/step   ", end="", flush=True)

            if step % args.eval_interval == 0:
                vl = evaluate(model, val, args, pad_id, device)
                print(f"\r  step {step:>6}  val {vl:.4f}  ppl {math.exp(min(vl, 20)):.2f}"
                      f"{'  <- best' if vl < best else '':>10}", flush=True)
                if vl < best:
                    best = vl
                    torch.save({"config": cfg.to_dict(), "model": model.state_dict(),
                                "step": step, "best_val": best, "args": vars(args),
                                "finetuned": True}, args.out)

    vl = evaluate(model, val, args, pad_id, device)
    if vl < best:
        best = vl
        torch.save({"config": cfg.to_dict(), "model": model.state_dict(), "step": step,
                    "best_val": best, "args": vars(args), "finetuned": True}, args.out)
    print(f"\ndone. best val {best:.4f} (ppl {math.exp(min(best, 20)):.2f}) -> {args.out}")


if __name__ == "__main__":
    main()
