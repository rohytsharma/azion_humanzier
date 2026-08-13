"""Local inference (SYS-F11, FR-08). No external API anywhere in this path.

    python -m inference.generate --prompt "It was a bright cold day" --tokens 80
"""
import argparse
from pathlib import Path

import torch
from tokenizers import Tokenizer

from model.config import Config
from model.model import HumanWriterLM

CKPT = "training/checkpoints/best.pt"
TOKENIZER = "tokenizer/tokenizer_files/bpe.json"


def load(ckpt=CKPT, tokenizer=TOKENIZER, device=None):
    """Returns (model, tokenizer, device). FR-11: say plainly what is missing."""
    for p in (ckpt, tokenizer):
        if not Path(p).exists():
            raise FileNotFoundError(f"missing {p} -- train it first (see README)")
    if device is None:
        device = ("cuda" if torch.cuda.is_available()
                  else "mps" if torch.backends.mps.is_available() else "cpu")
    ck = torch.load(ckpt, map_location=device, weights_only=True)
    model = HumanWriterLM(Config(**ck["config"])).to(device).eval()
    model.load_state_dict(ck["model"])
    return model, Tokenizer.from_file(tokenizer), device


def generate(model, tok, device, prompt, max_new_tokens=100, **sampling):
    ids = torch.tensor([[tok.token_to_id("<bos>")] + tok.encode(prompt).ids], device=device)
    out = model.generate(ids, max_new_tokens, eos_id=tok.token_to_id("<eos>"), **sampling)
    return tok.decode(out[0].tolist())


def rewrite_span(model, tok, device, text, max_new_tokens=None, **sampling):
    """One span through the fine-tuned rewriter.

    Prompt format must match training exactly: <bos> <src> ... <tgt>, then the
    model continues. Only the continuation is returned.
    """
    bos, src_t, tgt_t = (tok.token_to_id(t) for t in ("<bos>", "<src>", "<tgt>"))
    eos = tok.token_to_id("<eos>")
    src = tok.encode(text).ids
    prompt = [bos, src_t] + src + [tgt_t]

    budget = model.cfg.block_size - len(prompt) - 1
    if budget < 8:
        return text                       # nothing to work with; pass it through
    want = min(max_new_tokens or int(len(src) * 1.6) + 12, budget)

    ids = torch.tensor([prompt], device=device)
    out = model.generate(ids, want, eos_id=eos, **sampling)
    return tok.decode(out[0, len(prompt):].tolist()).strip()


def rewrite(model, tok, device, text, span_sentences=4, **sampling):
    """Rewrite a whole passage span by span.

    Spans are several sentences long rather than one, so the model can vary
    sentence length across them -- the feature that most distinguishes human
    writing, and one that a sentence-at-a-time rewriter cannot produce.
    """
    from evaluation.features import sentences

    sents = sentences(text)
    if not sents:
        return ""
    out = []
    for i in range(0, len(sents), span_sentences):
        span = " ".join(sents[i:i + span_sentences])
        try:
            piece = rewrite_span(model, tok, device, span, **sampling)
        except Exception:
            piece = span                  # never lose the user's text to a bad span
        out.append(piece if len(piece.split()) >= 3 else span)
    return " ".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", default="")
    ap.add_argument("--tokens", type=int, default=100)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--ckpt", default=CKPT)
    a = ap.parse_args()

    model, tok, device = load(a.ckpt)
    print(generate(model, tok, device, a.prompt, a.tokens, temperature=a.temperature,
                   top_p=a.top_p, repetition_penalty=a.repetition_penalty))
