"""Decoder-only Transformer, implemented from scratch (SRD 7).

Pre-LN blocks, learned positional embeddings, multi-head causal self-attention
written out explicitly rather than via nn.MultiheadAttention/SDPA so the mask
behaviour is inspectable and testable (SYS-F06).
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Config


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.d_head = cfg.d_model // cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        self.attn_drop = nn.Dropout(cfg.dropout)
        self.resid_drop = nn.Dropout(cfg.dropout)
        mask = torch.tril(torch.ones(cfg.block_size, cfg.block_size)).bool()
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, n_head, T, d_head)
        q, k, v = (t.view(B, T, self.n_head, self.d_head).transpose(1, 2) for t in (q, k, v))

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(~self.mask[:T, :T], float("-inf"))  # no attending to the future
        att = self.attn_drop(att.softmax(dim=-1))

        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_drop(self.proj(y))


class FeedForward(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.ff = FeedForward(cfg)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        return x + self.ff(self.ln2(x))


class HumanWriterLM(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layer))
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight  # saves ~8M params at vocab 16k
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def n_params(self, trainable_only=True):
        ps = self.parameters()
        return sum(p.numel() for p in ps if p.requires_grad or not trainable_only)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence of {T} exceeds block_size {self.cfg.block_size}"
        pos = torch.arange(T, device=idx.device)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))
        for blk in self.blocks:
            x = blk(x)
        logits = self.head(self.ln_f(x))

        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.reshape(-1), ignore_index=-100
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.9, top_p=0.9,
                 repetition_penalty=1.1, eos_id=None):
        """Nucleus sampling. Decoding params are a logged experiment variable, not
        constants -- at 40M they move naturalness more than the weights do."""
        self.eval()
        for _ in range(max_new_tokens):
            ctx = idx[:, -self.cfg.block_size:]
            logits = self(ctx)[0][:, -1, :]

            if repetition_penalty != 1.0:
                for b in range(idx.size(0)):
                    seen = torch.unique(idx[b])
                    s = logits[b, seen]
                    logits[b, seen] = torch.where(s < 0, s * repetition_penalty, s / repetition_penalty)

            logits = logits / max(temperature, 1e-5)
            if top_p < 1.0:
                srt, si = logits.sort(descending=True, dim=-1)
                cum = srt.softmax(dim=-1).cumsum(dim=-1)
                drop = cum - srt.softmax(dim=-1) > top_p  # keep the first token past the threshold
                logits = logits.masked_fill(drop.scatter(1, si, drop), float("-inf"))

            nxt = torch.multinomial(logits.softmax(dim=-1), num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
            if eos_id is not None and (nxt == eos_id).all():
                break
        return idx
