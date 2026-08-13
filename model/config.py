"""Model configuration. SRD 7 (parameter target ~40M), SRD 8 (recorded config)."""
from dataclasses import dataclass, asdict


@dataclass
class Config:
    # vocab 16k over 32k: keeps embeddings at ~21% of params instead of ~39%,
    # buying two extra layers for the same 40M budget. See docs/PRD.txt 10.
    vocab_size: int = 16000
    n_layer: int = 10
    n_head: int = 8
    d_model: int = 512
    d_ff: int = 2048
    block_size: int = 256   # PRD 10: start 256, scale to 512 if VRAM permits
    dropout: float = 0.1
    tie_weights: bool = True

    def to_dict(self):
        return asdict(self)


# ~1.2M params: for smoke tests and the overfit check (SRD 12).
TINY = Config(vocab_size=1000, n_layer=2, n_head=2, d_model=128, d_ff=512, block_size=64)
