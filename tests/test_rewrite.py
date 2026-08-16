"""Guard against the rewriter collapsing into a copier.

    .venv/bin/python -m tests.test_rewrite [checkpoint]

Source and target in the fine-tuning pairs share ~91% of their words, so an
unweighted loss is minimised by echoing the input. The first fine-tuning run did
exactly that: 100% word overlap, every style feature unchanged, val perplexity
1.14 that looked like success. This check is what that run should have had to
pass, so measure overlap, never just loss.
"""
import difflib
import sys
from pathlib import Path

CKPT = sys.argv[1] if len(sys.argv) > 1 else "training/checkpoints/finetuned.pt"

FLAT = [
    "The system processes data efficiently. It stores results in a database. "
    "The results are retrieved later. This approach improves performance.",
    "Machine learning is a field of study. It focuses on algorithms. These "
    "algorithms learn from data. They improve with experience.",
    "The report was published. It contained several findings. The findings were "
    "reviewed. The committee accepted them.",
]

# A rewrite that changes nothing is a copier; one that keeps nothing has lost the
# meaning. Both ends are failures, so the check is a band, not a floor.
# The pairs themselves are 91% similar: only punctuation and sentence joins
# change, so a faithful structural rewrite scores high on overlap too. Only a
# near-total match is copying, and the real signal is whether rhythm moved.
MAX_OVERLAP = 0.97
MIN_OVERLAP = 0.55
MIN_BURST_GAIN = 0.02


def main():
    if not Path(CKPT).exists():
        raise SystemExit(f"no checkpoint at {CKPT} -- fine-tune one first")

    from evaluation.features import extract
    from inference.generate import load, rewrite_span

    model, tok, device = load(CKPT, device="cpu")
    overlaps, bursts = [], []
    for text in FLAT:
        out = rewrite_span(model, tok, device, text, temperature=1e-6,
                           top_p=1.0, repetition_penalty=1.0)   # deterministic
        ov = difflib.SequenceMatcher(None, text.split(), out.split()).ratio()
        a, b = extract(text), extract(out)
        overlaps.append(ov)
        bursts.append(b["burstiness"] - a["burstiness"])
        print(f"  overlap {ov:5.0%}   sent_len {a['sent_len_words']:5.1f} -> "
              f"{b['sent_len_words']:5.1f}   punct {a['punct_ratio']:.3f} -> {b['punct_ratio']:.3f}")
        print(f"    {out[:110]}")

    mean_ov = sum(overlaps) / len(overlaps)
    mean_burst = sum(bursts) / len(bursts)
    print(f"\nmean word overlap {mean_ov:.1%}   mean burstiness gain {mean_burst:+.3f}")
    assert mean_burst >= MIN_BURST_GAIN, (
        f"rhythm did not move ({mean_burst:+.3f}). Overlap alone cannot tell a "
        f"structural rewrite from a copy -- this is the check that can.")
    assert mean_ov <= MAX_OVERLAP, (
        f"the model is copying its input ({mean_ov:.0%} overlap). Raise "
        f"--changed-weight so insertions carry more gradient than the copy path.")
    assert mean_ov >= MIN_OVERLAP, (
        f"only {mean_ov:.0%} of words survive -- the rewrite is not preserving content.")
    print("rewriter changes its input without discarding it")


if __name__ == "__main__":
    main()
