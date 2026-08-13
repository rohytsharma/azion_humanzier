"""Linguistic and readability features (SYS-F12, PRD 12).

Shared by the human-vs-AI study and, later, the app's writing-analysis panel.
No NLTK: its tokenizers need a runtime download, and a regex sentence splitter
plus sklearn's built-in stopword list cover what these features need.

    from evaluation.features import extract
    extract("Some text.")  -> dict of named features
"""
import re
import statistics
from collections import Counter

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Coordinating, subordinating and correlative conjunctions -- the connective
# tissue whose density separates flat machine prose from varied writing.
CONJUNCTIONS = frozenset("""
and but or nor for yet so although though because since unless until while
whereas whether if once when whenever where wherever after before as than
that provided supposing considering given lest
""".split())

_WORD = re.compile(r"[A-Za-z']+")
_PUNCT = re.compile(r"[.,;:!?\"'()\[\]{}\-—–…]")

# Candidate boundary: sentence punctuation, optional closers, then a gap or dash.
_BOUND = re.compile(r'(?<=[.!?])["\')\]]*(?:\s+|\s*[—–]\s*)')
_LAST_WORD = re.compile(r"([A-Za-z0-9']+)[.!?]+[\"')\]]*\s*$")

# Periods that end a token without ending a sentence. Legal and historical texts
# in a public-domain corpus are dense with these ("9 Hen. 5.—3 Hen. 8, c. 11."),
# and splitting naively turns one citation into a dozen four-word "sentences".
ABBREV = frozenset("""
mr mrs ms dr prof rev hon st jr sr vs etc eg ie cf viz al fig figs no nos vol
vols ch chap sec art p pp ed eds inc ltd co corp dept est approx min max
jan feb mar apr jun jul aug sept sep oct nov dec mon tue wed thu fri sat sun
hen geo wm edw eliz ric jas chas anne geo phil
""".split())


def sentences(text):
    """Split into sentences, refusing boundaries that are only abbreviations.

    A candidate break is rejected when the token before it is a known
    abbreviation, a single initial, or a bare number, or when what follows does
    not begin like a new sentence.
    """
    text = text.strip()
    if not text:
        return []

    parts, prev = [], 0
    for m in _BOUND.finditer(text):
        parts.append(text[prev:m.end()])
        prev = m.end()
    parts.append(text[prev:])

    out = []
    for part in parts:
        if not part.strip():
            continue
        if out and _merge(out[-1], part):
            out[-1] = out[-1].rstrip() + " " + part.lstrip()
        else:
            out.append(part)
    return [s.strip() for s in out if s.strip()]


def _merge(prev, cur):
    head = cur.lstrip()
    if not head:
        return True
    m = _LAST_WORD.search(prev)
    if m:
        word = m.group(1)
        if word.lower() in ABBREV:
            return True
        if len(word) == 1 and word.isalpha():       # an initial: "J. Smith"
            return True
        if word.isdigit():                          # enumerations: "5.—3 Hen."
            return True
    return not (head[0].isupper() or head[0] in "\"“'([")


def words(text):
    return _WORD.findall(text.lower())


def extract(text):
    """Return the feature dictionary for one text. Empty text yields zeros."""
    sents = sentences(text)
    ws = words(text)
    n_w, n_s = len(ws), len(sents)
    if n_w == 0 or n_s == 0:
        return {k: 0.0 for k in FEATURES}

    counts = Counter(ws)
    sent_words = [len(words(s)) for s in sents]
    sent_chars = [len(s) for s in sents]
    word_lens = [len(w) for w in ws]
    n_stop = sum(c for w, c in counts.items() if w in ENGLISH_STOP_WORDS)

    def sd(xs):
        return statistics.pstdev(xs) if len(xs) > 1 else 0.0

    return {
        # --- volume
        "characters": float(len(text)),
        "words": float(n_w),
        "sentences": float(n_s),
        "unique_words": float(len(counts)),
        "stopwords": float(n_stop),
        "punctuation": float(len(_PUNCT.findall(text))),
        "conjunctions": float(sum(c for w, c in counts.items() if w in CONJUNCTIONS)),
        # --- averages
        "sent_len_words": statistics.mean(sent_words),
        "sent_len_chars": statistics.mean(sent_chars),
        "word_len_chars": statistics.mean(word_lens),
        # --- ratios, the length-independent signal
        "type_token_ratio": len(counts) / n_w,
        "stopword_ratio": n_stop / n_w,
        "punct_ratio": len(_PUNCT.findall(text)) / n_w,
        "conj_ratio": sum(c for w, c in counts.items() if w in CONJUNCTIONS) / n_w,
        # --- variation: how much sentence length moves about, which is the
        # feature most often cited as separating human from machine prose
        "sent_len_sd": sd(sent_words),
        "burstiness": sd(sent_words) / statistics.mean(sent_words) if sent_words else 0.0,
    }


FEATURES = [
    "characters", "words", "sentences", "unique_words", "stopwords",
    "punctuation", "conjunctions", "sent_len_words", "sent_len_chars",
    "word_len_chars", "type_token_ratio", "stopword_ratio", "punct_ratio",
    "conj_ratio", "sent_len_sd", "burstiness",
]

# Ratios and shapes only -- safe to compare across texts of different lengths.
# The volume features scale with document size, so a classifier given them
# learns "how long is it", not "who wrote it".
SCALE_FREE = [
    "sent_len_words", "sent_len_chars", "word_len_chars", "type_token_ratio",
    "stopword_ratio", "punct_ratio", "conj_ratio", "sent_len_sd", "burstiness",
]


def demo():
    flat = ("The system processes data. The system stores data. The system "
            "returns data. The system logs data.")
    varied = ("It processes the data, stores it, and hands back a result — all in "
              "one pass. Then it logs what happened. Why? Because when something "
              "breaks at three in the morning, that log is the only witness you have.")
    f, v = extract(flat), extract(varied)
    assert f["sentences"] == 4, f["sentences"]
    assert v["burstiness"] > f["burstiness"], "varied prose must be burstier"
    assert v["type_token_ratio"] > f["type_token_ratio"], "repetitive text must have lower TTR"
    assert extract("")["words"] == 0
    assert abs(extract("Hi there.")["sent_len_words"] - 2) < 1e-9

    # abbreviations must not manufacture sentence boundaries
    assert len(sentences("Dr. Smith met Mr. Jones in St. Albans. They talked.")) == 2
    assert len(sentences("See 9 Hen. 5.—3 Hen. 8, c. 11.—5 Hen. 8, c. 6.")) == 1
    assert len(sentences("J. R. R. Tolkien wrote it. Others followed.")) == 2
    assert len(sentences("He left. She stayed. They waited.")) == 3
    assert len(sentences("Ready?—Then go. Now!")) == 3   # a dash can end one too
    print(f"flat   burstiness {f['burstiness']:.3f}  ttr {f['type_token_ratio']:.3f}")
    print(f"varied burstiness {v['burstiness']:.3f}  ttr {v['type_token_ratio']:.3f}")
    print("ok")


if __name__ == "__main__":
    demo()
