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
    """-> list of (ids, n_prompt). Examples longer than the context are dropped
    rather than truncated: a truncated target teaches the model to stop mid-clause."""
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
    for s, t in zip(srcs, tgts):
        ids = [bos, src_t] + s.ids + [tgt_t] + t.ids + [eos]
        if len(ids) > block_size:
            continue
        out.append((np.array(ids, dtype=np.uint16), 2 + len(s.ids) + 1))
    print(f"  {path.name}: {len(out):,} examples fit in {block_size} tokens "
          f"({len(rows) - len(out):,} too long, dropped)")
    return out


def batches(data, batch_size, pad_id, device, shuffle=True, seed=0):
    order = np.arange(len(data))
    if shuffle:
        np.random.default_rng(seed).shuffle(order)
    for i in range(0, len(order) - batch_size + 1, batch_size):
        chunk = [data[j] for j in order[i:i + batch_size]]
        width = max(len(ids) for ids, _ in chunk)
        x = np.full((len(chunk), width - 1), pad_id, dtype=np.int64)
        y = np.full((len(chunk), width - 1), -100, dtype=np.int64)
        for r, (ids, n_prompt) in enumerate(chunk):
            seq = ids.astype(np.int64)
            x[r, :len(seq) - 1] = seq[:-1]
            tgt = seq[1:].copy()
            tgt[:n_prompt - 1] = -100          # the prompt half is not scored
            y[r, :len(tgt)] = tgt
        yield torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


@torch.no_grad()
def evaluate(model, data, args, pad_id, device, max_batches=40):
    model.eval()
    losses = []
    for i, (x, y) in enumerate(batches(data, args.batch_size, pad_id, device, shuffle=False)):
        losses.append(model(x, y)[1].item())
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
        it = batches(train, args.batch_size, pad_id, device, seed=args.seed + epoch)
        done = False
        while not done:
            for g in opt.param_groups:
                g["lr"] = lr_at(step)
            for _ in range(args.grad_accum):
                try:
                    x, y = next(it)
                except StopIteration:
                    done = True
                    break
                loss = model(x, y)[1] / args.grad_accum
                loss.backward()
            if done:
                break

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

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
