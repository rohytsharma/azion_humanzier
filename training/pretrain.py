"""Causal-LM pretraining (SYS-F07/F08/F09, SRD 8).

    python -m training.pretrain --steps 5000
    python -m training.pretrain --resume training/checkpoints/last.pt

Everything the SRD asks to be configurable is a flag, and every flag is written
into the checkpoint and the CSV log so a run can be reproduced from either.
"""
import argparse
import csv
import math
import time
from pathlib import Path

import numpy as np
import torch

from model.config import Config
from model.model import HumanWriterLM

CKPT_DIR = Path("training/checkpoints")


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i:i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1:i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device, non_blocking=True), y.to(device, non_blocking=True)


@torch.no_grad()
def estimate_loss(model, splits, args, device, iters=20):
    model.eval()
    out = {}
    for name, data in splits.items():
        losses = torch.zeros(iters)
        for k in range(iters):
            x, y = get_batch(data, args.batch_size, args.block_size, device)
            losses[k] = model(x, y)[1].item()
        out[name] = losses.mean().item()
    model.train()
    return out


def lr_at(step, args):
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    if step > args.steps:
        return args.min_lr
    r = (step - args.warmup) / max(1, args.steps - args.warmup)
    return args.min_lr + 0.5 * (1 + math.cos(math.pi * r)) * (args.lr - args.min_lr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=8, help="effective batch = batch_size * grad_accum")
    ap.add_argument("--block-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=6e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=250)
    ap.add_argument("--eval-iters", type=int, default=20)
    ap.add_argument("--ckpt-interval", type=int, default=500)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--amp", action="store_true", help="bf16 autocast (CUDA)")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--run-name", default=time.strftime("run-%Y%m%d-%H%M%S"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)

    splits = {
        s: np.memmap(f"data/splits/{s}.bin", dtype=np.uint16, mode="r")
        for s in ("train", "val")
    }
    print(f"device={device}  train={len(splits['train']):,} tokens  val={len(splits['val']):,}")

    cfg = Config(block_size=args.block_size)
    start_step, best_val = 0, float("inf")
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=True)
        cfg = Config(**ck["config"])
        model = HumanWriterLM(cfg).to(device)
        model.load_state_dict(ck["model"])
        start_step, best_val = ck["step"], ck.get("best_val", float("inf"))
    else:
        model = HumanWriterLM(cfg).to(device)
    print(f"{model.n_params()/1e6:.1f}M trainable params  |  "
          f"effective batch {args.batch_size * args.grad_accum * args.block_size:,} tokens/step")

    # no weight decay on 1-D params (norms, biases) -- decaying them just fights the norm
    decay = [p for p in model.parameters() if p.dim() >= 2]
    nodecay = [p for p in model.parameters() if p.dim() < 2]
    opt = torch.optim.AdamW(
        [{"params": decay, "weight_decay": args.weight_decay},
         {"params": nodecay, "weight_decay": 0.0}],
        lr=args.lr, betas=(0.9, 0.95),
    )
    if args.resume:
        opt.load_state_dict(ck["optimizer"])
    if args.compile:
        model = torch.compile(model)

    amp = args.amp and device.type == "cuda"
    ctx = torch.autocast("cuda", dtype=torch.bfloat16) if amp else torch.autocast("cpu", enabled=False)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    Path("experiments").mkdir(exist_ok=True)
    log_path = Path("experiments") / f"{args.run_name}.csv"
    new_log = not log_path.exists()
    log_file = log_path.open("a", newline="")
    log = csv.writer(log_file)
    if new_log:
        log.writerow(["step", "train_loss", "val_loss", "ppl", "lr", "tok_per_s", "peak_mem_gb"])
        (Path("experiments") / f"{args.run_name}.json").write_text(
            __import__("json").dumps({"args": vars(args), "config": cfg.to_dict()}, indent=2))

    def save(step, val, path):
        torch.save({"config": cfg.to_dict(), "model": getattr(model, "_orig_mod", model).state_dict(),
                    "optimizer": opt.state_dict(), "step": step, "best_val": val,
                    "args": vars(args)}, path)

    t0, tokens = time.time(), 0
    for step in range(start_step, args.steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(step, args)

        for micro in range(args.grad_accum):
            x, y = get_batch(splits["train"], args.batch_size, args.block_size, device)
            with ctx:
                loss = model(x, y)[1] / args.grad_accum
            loss.backward()
            tokens += x.numel()

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)

        if step % args.eval_interval == 0 or step == args.steps - 1:
            dt = time.time() - t0  # measured before eval so tok/s is training-only
            l = estimate_loss(model, splits, args, device, args.eval_iters)
            mem = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else 0.0
            print(f"step {step:>6}  train {l['train']:.4f}  val {l['val']:.4f}  "
                  f"ppl {math.exp(min(l['val'], 20)):>8.1f}  {tokens/dt:>7,.0f} tok/s", flush=True)
            log.writerow([step, f"{l['train']:.4f}", f"{l['val']:.4f}",
                          f"{math.exp(min(l['val'], 20)):.2f}", f"{lr_at(step, args):.2e}",
                          f"{tokens/dt:.0f}", f"{mem:.2f}"])
            log_file.flush()
            if l["val"] < best_val:
                best_val = l["val"]
                save(step, best_val, CKPT_DIR / "best.pt")
            t0, tokens = time.time(), 0

        if step and step % args.ckpt_interval == 0:
            save(step, best_val, CKPT_DIR / "last.pt")

    save(args.steps, best_val, CKPT_DIR / "last.pt")
    print(f"done. best val loss {best_val:.4f} (ppl {math.exp(min(best_val, 20)):.1f})")


if __name__ == "__main__":
    main()
