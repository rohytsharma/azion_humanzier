"""HumanWriter — local writing analysis and transformation (SRD 9).

    .venv/bin/streamlit run app/streamlit_app.py

Nothing leaves this machine. The model and tokenizer load from disk; the
semantic encoder runs locally after its first download.
"""
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.features import extract  # noqa: E402

st.set_page_config(page_title="HumanWriter", page_icon="✎", layout="wide")

# --- Reference bands -------------------------------------------------------
# Measured, not guessed: means over 20 human documents and 24 model-generated
# documents (evaluation/figures/feature_table.csv). (ai_mean, human_mean).
BANDS = {
    "punct_ratio":      (0.1449, 0.2121, "Punctuation density", "marks per word"),
    "sent_len_words":   (18.4346, 25.2293, "Sentence length", "words per sentence"),
    "burstiness":       (0.3444, 0.4356, "Burstiness", "variation in sentence length"),
}
# Measured but within ±9% between classes — shown separately because at 44
# documents that gap is not distinguishable from noise.
WEAK = {
    "type_token_ratio": (0.5176, 0.4923, "Type–token ratio", "vocabulary variety"),
    "stopword_ratio":   (0.4277, 0.4404, "Stopword ratio", "function-word share"),
    "conj_ratio":       (0.0891, 0.0812, "Conjunction ratio", "connectives per word"),
    "word_len_chars":   (5.4261, 5.3895, "Word length", "characters per word"),
}

MAX_CHARS = 20000

CSS = """
<style>
  .stApp { background: #E9EDF1; }
  #MainMenu, footer, header { visibility: hidden; }
  .block-container { padding: 2.2rem 3rem 5rem; max-width: 1400px; }

  .hw-mast { border-bottom: 2px solid #121A23; padding-bottom: 18px; margin-bottom: 26px; }
  .hw-eyebrow { font-family: ui-monospace, Menlo, monospace; font-size: 10.5px;
                letter-spacing: .16em; text-transform: uppercase; color: #58687A; margin-bottom: 8px; }
  /* Streamlit ships its own h1/h2 rules; these must outrank them. */
  .hw-mast h1.hw-title { font-family: "Iowan Old Style", Palatino, Georgia, serif;
              font-size: 42px; line-height: 1; font-weight: 600; letter-spacing: -.02em;
              color: #121A23; margin: 0; padding: 0; }
  .hw-sub { color: #58687A; font-size: 14.5px; margin-top: 10px; }

  .hw-card { background: #fff; border: 1px solid #C4CFDA; padding: 20px 22px; margin-bottom: 16px; }
  .hw-h { font-family: "Iowan Old Style", Palatino, Georgia, serif; font-size: 19px;
          font-weight: 600; color: #121A23; margin: 0 0 4px; }
  .hw-note { color: #58687A; font-size: 12.5px; margin: 0 0 16px; line-height: 1.5; }

  /* the index */
  .hw-index { display: flex; align-items: baseline; gap: 12px; margin-bottom: 2px; }
  .hw-index b { font-family: ui-monospace, Menlo, monospace; font-size: 46px;
                font-weight: 500; letter-spacing: -.03em; color: #27528F; line-height: 1; }
  .hw-index span { font-size: 13px; color: #58687A; }

  /* per-feature track */
  .hw-row { margin: 0 0 18px; }
  .hw-lab { display: flex; justify-content: space-between; align-items: baseline;
            font-size: 12.5px; color: #121A23; margin-bottom: 7px; }
  .hw-lab em { font-style: normal; color: #58687A; font-size: 11px; }
  .hw-val { font-family: ui-monospace, Menlo, monospace; font-size: 12.5px;
            font-variant-numeric: tabular-nums; color: #121A23; }
  .hw-track { position: relative; height: 26px; }
  .hw-rail { position: absolute; top: 11px; left: 0; right: 0; height: 3px; background: #DFE5EB; }
  .hw-span { position: absolute; top: 11px; height: 3px; background: #C7D8EE; }
  .hw-tick { position: absolute; top: 4px; width: 1px; height: 17px; background: #93A3B4; }
  .hw-tick i { position: absolute; top: -12px; left: 3px; font-style: normal;
               font-family: ui-monospace, Menlo, monospace; font-size: 8.5px;
               letter-spacing: .08em; color: #7A8B9C; white-space: nowrap; }
  .hw-dot { position: absolute; top: 6px; width: 13px; height: 13px; margin-left: -6.5px;
            border-radius: 50%; background: #27528F; border: 2.5px solid #fff;
            box-shadow: 0 0 0 1px #27528F; }

  .hw-pill { display: inline-block; font-family: ui-monospace, Menlo, monospace;
             font-size: 9.5px; letter-spacing: .09em; text-transform: uppercase;
             padding: 3px 8px; border: 1px solid currentColor; }
  .ok   { color: #1F6B45; background: #E2F0E8; }
  .warn { color: #8A5A12; background: #F5E9D6; }
  .off  { color: #58687A; background: #EEF1F4; }

  .hw-kv { display: flex; justify-content: space-between; font-size: 12.5px;
           padding: 7px 0; border-bottom: 1px solid #EEF1F4; }
  .hw-kv b { font-family: ui-monospace, Menlo, monospace; font-weight: 500;
             font-variant-numeric: tabular-nums; }
  .stTextArea textarea { font-size: 14px; line-height: 1.6; border-radius: 0 !important; }
  .stButton button { border-radius: 0; border: 1px solid #27528F; background: #27528F;
                     color: #fff; font-size: 13.5px; padding: 8px 20px; }
  .stButton button:hover { background: #1D3F70; border-color: #1D3F70; color: #fff; }
</style>
"""


def track(value, ai, human, name, unit):
    """One feature against its reference band. Direction-aware: `human` may be
    below `ai`, and the drawing must not assume otherwise."""
    lo_ref, hi_ref = min(ai, human), max(ai, human)
    pad = (hi_ref - lo_ref) * 0.9 or abs(hi_ref) * 0.25 or 1
    lo, hi = lo_ref - pad, hi_ref + pad

    def pos(v):
        return max(0.0, min(1.0, (v - lo) / (hi - lo))) * 100

    toward = (value - ai) / (human - ai) if human != ai else 0
    tone = "ok" if toward >= 0.6 else "warn" if toward >= 0.2 else "off"
    verdict = "human range" if toward >= 0.6 else "between" if toward >= 0.2 else "machine range"

    return f"""
    <div class="hw-row">
      <div class="hw-lab"><span>{name} <em>· {unit}</em></span>
        <span class="hw-val">{value:.4g} <span class="hw-pill {tone}">{verdict}</span></span></div>
      <div class="hw-track">
        <div class="hw-rail"></div>
        <div class="hw-span" style="left:{pos(lo_ref)}%;width:{pos(hi_ref) - pos(lo_ref)}%"></div>
        <div class="hw-tick" style="left:{pos(ai)}%"><i>AI {ai:.3g}</i></div>
        <div class="hw-tick" style="left:{pos(human)}%"><i>HUMAN {human:.3g}</i></div>
        <div class="hw-dot" style="left:{pos(value)}%"></div>
      </div>
    </div>"""


def index_score(f):
    """Mean progress from the AI mean toward the human mean, over the three
    features that actually separated the classes. 0 = machine-typical,
    100 = at or past the human mean."""
    vals = []
    for k, (ai, human, _, _) in BANDS.items():
        vals.append(max(0.0, min(1.0, (f[k] - ai) / (human - ai))))
    return 100 * sum(vals) / len(vals)


@st.cache_resource(show_spinner=False)
def get_model(ckpt):
    from inference.generate import load
    return load(ckpt, device="cpu")      # training owns the GPU


@st.cache_resource(show_spinner=False)
def get_encoder():
    from evaluation.evaluate_semantics import _encoder
    return _encoder()


st.markdown(CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="hw-mast"><div class="hw-eyebrow">Local · no text leaves this machine</div>'
    '<h1 class="hw-title">HumanWriter</h1>'  # styled via .hw-mast h1.hw-title
    '<div class="hw-sub">Measure how a piece of writing sits against human reference ranges, '
    'then rewrite it with the custom 40M-parameter model.</div></div>',
    unsafe_allow_html=True)

# The fine-tuned checkpoint is the one that actually rewrites; the pretrained
# one only continues text. Prefer the former, and say plainly which is loaded.
finetuned_path = Path("training/checkpoints/finetuned.pt")
pretrained_path = Path("training/checkpoints/best.pt")
tok_path = Path("tokenizer/tokenizer_files/bpe.json")

is_finetuned = finetuned_path.exists()
ckpt_path = finetuned_path if is_finetuned else pretrained_path
ready = ckpt_path.exists() and tok_path.exists()

left, right = st.columns([1, 1], gap="large")

with left:
    st.markdown('<div class="hw-h">Your text</div>'
                '<div class="hw-note">Paste anything — an abstract, an essay, a paragraph '
                'you suspect reads flat.</div>', unsafe_allow_html=True)
    text = st.text_area("Input", height=300, label_visibility="collapsed",
                        placeholder="Paste text here…")

    if len(text) > MAX_CHARS:
        st.warning(f"That's {len(text):,} characters. Only the first {MAX_CHARS:,} "
                   f"will be analysed.")
        text = text[:MAX_CHARS]

    c1, c2 = st.columns([1, 2])
    analyse = c1.button("Analyse", use_container_width=True)
    transform = c2.button("Analyse + rewrite", use_container_width=True, disabled=not ready)

    if is_finetuned:
        pill, note = "ok", "fine-tuned rewriter loaded"
    elif ready:
        pill, note = "warn", "pretrained only — continues text, does not rewrite yet"
    else:
        pill, note = "off", "no checkpoint yet — train a model first"
    st.markdown(f'<div style="margin-top:12px">'
                f'<span class="hw-pill {pill}">{"model ready" if ready else "no model"}</span> '
                f'<span style="font-size:11.5px;color:#58687A">{note}</span></div>',
                unsafe_allow_html=True)

    with st.expander("Rewrite settings"):
        temp = st.slider("Temperature", 0.05, 1.3, 0.10, 0.05,
                         help="Near zero keeps the meaning; raising it degrades this "
                              "model's output quickly.")
        top_p = st.slider("Top-p", 0.5, 1.0, 1.00, 0.05)
        rep = st.slider("Repetition penalty", 1.0, 1.4, 1.00, 0.05)
        st.caption("Measured on 12 real AI passages: near-greedy scores 0.958 semantic "
                   "similarity, temperature 0.85 only 0.894 and produces word salad. At "
                   "40M these settings move quality more than the weights do. Logged "
                   "with every exported report.")

with right:
    if not (analyse or transform) or not text.strip():
        st.markdown('<div class="hw-card"><div class="hw-h">Writing profile</div>'
                    '<div class="hw-note">Reference bands come from 20 human and 24 '
                    'model-generated documents measured for this project. Three features '
                    'separated the two classes; the rest are shown lower down because at '
                    'that sample size their gaps are not distinguishable from noise.</div>'
                    '</div>', unsafe_allow_html=True)
    else:
        f = extract(text)
        if f["words"] < 25:
            st.error("Too short to profile — give it at least a couple of sentences.")
            st.stop()

        idx = index_score(f)
        rows = "".join(track(f[k], *v) for k, v in BANDS.items())
        weak = "".join(track(f[k], *v) for k, v in WEAK.items())

        st.markdown(
            f'<div class="hw-card"><div class="hw-h">Writing profile</div>'
            f'<div class="hw-note">Distance from the machine-typical mean toward the '
            f'human mean, on the three features that separated the classes.</div>'
            f'<div class="hw-index"><b>{idx:.0f}</b><span>of 100 · human-range index</span></div>'
            f'<div style="height:14px"></div>{rows}</div>', unsafe_allow_html=True)

        with st.expander("Features with little separation in the reference data"):
            st.markdown(f'<div style="padding-top:6px">{weak}</div>', unsafe_allow_html=True)
            st.caption("Kept visible for completeness. Human and AI means differ by under "
                       "9% on these, which 44 documents cannot resolve.")

        counts = ["characters", "words", "sentences", "unique_words", "stopwords",
                  "punctuation", "conjunctions"]
        kv = "".join(f'<div class="hw-kv"><span>{k.replace("_", " ").title()}</span>'
                     f'<b>{f[k]:,.0f}</b></div>' for k in counts)
        with st.expander("Raw counts"):
            st.markdown(kv, unsafe_allow_html=True)

        report = [f"HumanWriter analysis", f"human-range index: {idx:.0f}/100", ""]
        report += [f"{k}: {f[k]:.4f}" for k in f]

        if transform:
            head = ('<div class="hw-note">Rewritten span by span, several sentences at a '
                    'time, so sentence length can vary across the passage.</div>'
                    if is_finetuned else
                    '<div class="hw-note">The loaded checkpoint is pretrained only, so it '
                    'continues the text rather than rewriting it. Shown so the path is '
                    'verifiable end to end before fine-tuning lands.</div>')
            st.markdown(f'<div class="hw-card"><div class="hw-h">Rewrite</div>{head}</div>',
                        unsafe_allow_html=True)
            try:
                with st.spinner("Generating locally…"):
                    from inference.generate import generate, rewrite
                    model, tok, device = get_model(str(ckpt_path))
                    sampling = dict(temperature=temp, top_p=top_p, repetition_penalty=rep)
                    if is_finetuned:
                        out = rewrite(model, tok, device, text, **sampling)
                    else:
                        out = generate(model, tok, device, text[:600],
                                       max_new_tokens=120, **sampling)
                st.text_area("Output", out, height=200, label_visibility="collapsed")
                report += ["", f"decoding: {sampling}", "rewrite:", out]

                if is_finetuned and out.strip():
                    g = extract(out)
                    moved = "".join(track(g[k], *v) for k, v in BANDS.items())
                    st.markdown(
                        f'<div class="hw-card"><div class="hw-h">Where the rewrite landed</div>'
                        f'<div class="hw-note">Same three bands, measured on the output. '
                        f'Index {index_score(g):.0f} versus {idx:.0f} before.</div>'
                        f'{moved}</div>', unsafe_allow_html=True)
                    report += ["", f"index before: {idx:.0f}", f"index after: {index_score(g):.0f}"]

                with st.spinner("Scoring meaning…"):
                    from evaluation.evaluate_semantics import similarity, PRESERVED
                    sim = float(similarity([text], [out])[0])
                tone = "ok" if sim >= PRESERVED else "warn"
                st.markdown(
                    f'<div class="hw-card"><div class="hw-h">Meaning preservation</div>'
                    f'<div class="hw-index"><b>{sim:.3f}</b>'
                    f'<span>cosine similarity · <span class="hw-pill {tone}">'
                    f'{"within range" if sim >= PRESERVED else "below threshold"}</span></span></div>'
                    f'<div class="hw-note" style="margin-top:12px">Embedding similarity '
                    f'measures topic overlap, not factual agreement: it catches drift, and '
                    f'cannot catch a reversed or altered fact. Read it alongside the text, '
                    f'not instead of it.</div></div>', unsafe_allow_html=True)
                report += ["", f"semantic similarity: {sim:.4f}"]
            except FileNotFoundError as e:
                st.error(str(e))
            except Exception:
                st.error("Generation failed. Check that the checkpoint and tokenizer "
                         "are readable and were produced by the same configuration.")

        st.download_button("Export report", "\n".join(report),
                           file_name="humanwriter-analysis.txt", mime="text/plain")
