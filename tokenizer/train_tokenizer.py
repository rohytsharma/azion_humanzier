"""Train the custom BPE tokenizer (SYS-F03).

    python -m tokenizer.train_tokenizer --input data/processed/train.txt --vocab 16000

Specials are declared up front because the vocabulary is frozen once trained --
adding <src>/<tgt> later would mean retraining and re-encoding everything.
"""
import argparse
from pathlib import Path

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders

SPECIALS = ["<pad>", "<bos>", "<eos>", "<src>", "<tgt>"]
OUT_DIR = Path("tokenizer/tokenizer_files")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", nargs="+", default=["data/processed/train.txt"])
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--min-frequency", type=int, default=2)
    ap.add_argument("--out", default=str(OUT_DIR / "bpe.json"))
    a = ap.parse_args()

    tok = Tokenizer(models.BPE(unk_token=None))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=True)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=a.vocab,
        min_frequency=a.min_frequency,
        special_tokens=SPECIALS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),  # byte-level => never an UNK
        show_progress=True,
    )
    tok.train(a.input, trainer)

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    tok.save(a.out)
    print(f"saved {a.out}  vocab={tok.get_vocab_size()}")

    # SRD 12: encode/decode round-trip on representative text
    sample = "The quick brown fox -- it jumped over 3 lazy dogs, didn't it?"
    ids = tok.encode(sample).ids
    back = tok.decode(ids)
    print(f"round-trip {'OK' if back.strip() == sample else 'MISMATCH'}: {back!r} ({len(ids)} tokens)")


if __name__ == "__main__":
    main()
