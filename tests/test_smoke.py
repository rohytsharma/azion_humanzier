"""End-to-end pipeline check (SRD 12: mask, shapes, param count, checkpoint, overfit).

Run: .venv/bin/python -m tests.test_smoke
"""
import tempfile, os
import torch

from model.config import Config, TINY
from model.model import HumanWriterLM


def test_shapes():
    m = HumanWriterLM(TINY)
    idx = torch.randint(0, TINY.vocab_size, (2, 16))
    logits, loss = m(idx, targets=idx)
    assert logits.shape == (2, 16, TINY.vocab_size), logits.shape
    assert loss.ndim == 0 and loss.item() > 0
    # untrained loss should sit near ln(vocab)
    assert abs(loss.item() - torch.log(torch.tensor(float(TINY.vocab_size)))) < 1.0


def test_causal_mask():
    """SYS-F06: changing token t must not alter logits at positions < t."""
    m = HumanWriterLM(TINY).eval()
    a = torch.randint(0, TINY.vocab_size, (1, 12))
    b = a.clone()
    b[0, 7] = (b[0, 7] + 1) % TINY.vocab_size
    with torch.no_grad():
        la, lb = m(a)[0], m(b)[0]
    assert torch.allclose(la[:, :7], lb[:, :7], atol=1e-6), "past leaked from the future"
    assert not torch.allclose(la[:, 7:], lb[:, 7:]), "change had no effect at all -- suspicious"


def test_param_count():
    n = HumanWriterLM(Config()).n_params()
    print(f"  full config: {n/1e6:.1f}M trainable params")
    assert 35e6 < n < 45e6, f"{n} outside the ~40M target"


def test_checkpoint_roundtrip():
    m = HumanWriterLM(TINY).eval()
    idx = torch.randint(0, TINY.vocab_size, (1, 8))
    with torch.no_grad():
        before = m(idx)[0]
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "ckpt.pt")
        torch.save({"config": TINY.to_dict(), "model": m.state_dict()}, p)
        ck = torch.load(p, weights_only=True)
        m2 = HumanWriterLM(Config(**ck["config"])).eval()
        m2.load_state_dict(ck["model"])
    with torch.no_grad():
        assert torch.allclose(before, m2(idx)[0], atol=1e-6)


def test_overfit_tiny():
    """The pipeline works iff the model can memorise one batch."""
    torch.manual_seed(0)
    m = HumanWriterLM(TINY)
    x = torch.randint(0, TINY.vocab_size, (4, 32))
    y = torch.roll(x, -1, dims=1)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4)
    first = None
    for _ in range(200):
        loss = m(x, y)[1]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        first = first if first is not None else loss.item()
    print(f"  overfit loss {first:.3f} -> {loss.item():.3f}")
    assert loss.item() < 0.5, f"failed to overfit 4x32 tokens (loss {loss.item():.3f})"


def test_generate():
    m = HumanWriterLM(TINY)
    out = m.generate(torch.zeros((2, 1), dtype=torch.long), max_new_tokens=10, top_p=0.9)
    assert out.shape == (2, 11)
    assert out.min() >= 0 and out.max() < TINY.vocab_size
    # greedy + top_p must be deterministic and equal to argmax
    idx = torch.randint(0, TINY.vocab_size, (1, 5))
    g = m.generate(idx, 1, temperature=1e-6, top_p=1.0, repetition_penalty=1.0)
    with torch.no_grad():
        assert g[0, -1].item() == m(idx)[0][0, -1].argmax().item()


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            print(name)
            fn()
    print("\nall good")
