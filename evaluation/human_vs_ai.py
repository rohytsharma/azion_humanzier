"""Human vs AI writing study: feature comparison and classifier benchmark.

    python -m evaluation.human_vs_ai --data ~/Downloads/archive

Produces in evaluation/figures/ :
    features_by_class.png    per-feature comparison, Human vs each AI model
    variant_shift.png        what Paraphrase / Translate / Humanize change
    classifier_accuracy.png  five ensembles, with and without length features
    feature_table.csv        the numbers behind all three

Two methodological choices worth knowing about, both of which lower the
headline accuracy and both of which are the reason to trust it:

1. Chunks are grouped by source document, so no fold ever trains and tests on
   two pieces of the same text. Without this, accuracy is measuring memorisation.
2. The primary classifier uses scale-free features only. Document length is an
   artefact of how this corpus was assembled, not a property of machine writing;
   a model given raw counts learns "how long is it". The second panel shows what
   including them buys, which is precisely the size of that leak.
"""
import argparse
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier,
                              ExtraTreesClassifier, GradientBoostingClassifier,
                              RandomForestClassifier)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score

from evaluation.features import FEATURES, SCALE_FREE, extract, sentences

OUT = Path("evaluation/figures")

# dataviz reference palette, categorical slots 1-5 (validated light mode)
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#d8d8d4", "#fcfcfb"

VARIANTS = ["Generated", "Paraphrased", "Translated", "Humanized"]
CLASSIFIERS = {
    "Random Forest": lambda: RandomForestClassifier(n_estimators=300, random_state=0),
    "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=0),
    "AdaBoost": lambda: AdaBoostClassifier(random_state=0),
    "Bagging": lambda: BaggingClassifier(n_estimators=100, random_state=0),
    "Extra Trees": lambda: ExtraTreesClassifier(n_estimators=300, random_state=0),
}


# ---------------------------------------------------------------- data

def load(data_dir):
    """One row per text: class label, variant, source document id, the text."""
    d = Path(data_dir).expanduser()
    h = pd.read_csv(d / "Human-Written.csv")
    a = pd.read_csv(d / "AI_Generated.csv")

    rows = [{"label": "Human", "variant": "Written", "doc": f"h{i}", "text": t}
            for i, t in enumerate(h.Text)]
    for i, r in a.iterrows():
        for v in VARIANTS:
            rows.append({"label": r.Model, "variant": v, "doc": f"a{i}", "text": r[v]})
    return pd.DataFrame(rows)


def chunk(text, per_chunk=6, min_words=40):
    """Sentence windows, so a 2,600-character document yields several samples."""
    sents = sentences(text)
    out = []
    for i in range(0, len(sents), per_chunk):
        piece = " ".join(sents[i:i + per_chunk])
        if len(piece.split()) >= min_words:
            out.append(piece)
    return out


def feature_frame(df, chunked=False):
    rows = []
    for _, r in df.iterrows():
        pieces = chunk(r.text) if chunked else [r.text]
        for p in pieces:
            rows.append({"label": r.label, "variant": r.variant, "doc": r.doc, **extract(p)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- plot helpers

def style(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.savefig(OUT / name, dpi=160, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"  wrote evaluation/figures/{name}")


# ---------------------------------------------------------------- figure 1

PANELS = [
    ("sentences", "Sentences"), ("words", "Words"), ("characters", "Characters"),
    ("unique_words", "Unique words"), ("stopwords", "Stopwords"),
    ("punctuation", "Punctuation"), ("conjunctions", "Conjunctions"),
    ("sent_len_words", "Sentence length\n(words)"),
    ("word_len_chars", "Word length\n(characters)"),
    ("type_token_ratio", "Type–token ratio"),
    ("stopword_ratio", "Stopword ratio"),
    ("burstiness", "Burstiness\n(sentence-length variation)"),
]


def fig_features_by_class(feats):
    """Small multiples: one panel per feature, so each keeps its own scale.

    A single shared axis would put character counts (~2,600) beside sentence
    counts (~12) and flatten everything but the largest feature into the floor.
    """
    sub = feats[(feats.label == "Human") | (feats.variant == "Generated")]
    classes = ["Human"] + sorted(c for c in sub.label.unique() if c != "Human")
    colors = dict(zip(classes, PALETTE))

    fig, axes = plt.subplots(3, 4, figsize=(13.5, 8.6))
    for ax, (key, title) in zip(axes.flat, PANELS):
        means = [sub[sub.label == c][key].mean() for c in classes]
        bars = ax.bar(range(len(classes)), means,
                      color=[colors[c] for c in classes], width=0.66)
        style(ax)
        ax.set_title(title, fontsize=9.5, color=INK, pad=8)
        ax.set_xticks(range(len(classes)))
        ax.set_xticklabels([c[:9] for c in classes], rotation=35, ha="right", fontsize=7.5)
        top = max(means) if max(means) else 1
        ax.set_ylim(0, top * 1.22)
        for b, m in zip(bars, means):                      # relief: visible labels
            ax.text(b.get_x() + b.get_width() / 2, m + top * 0.04,
                    f"{m:,.2f}".rstrip("0").rstrip(".") if m < 100 else f"{m:,.0f}",
                    ha="center", fontsize=7, color=MUTED)

    fig.suptitle("Human writing against four models, feature by feature",
                 fontsize=14, color=INK, x=0.5, y=0.99)
    fig.text(0.5, 0.955, "AI texts are the Generated variant. Each panel keeps its own "
             "scale — the features differ by three orders of magnitude.",
             ha="center", fontsize=9, color=MUTED)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save(fig, "features_by_class.png")


# ---------------------------------------------------------------- figure 2

def fig_variant_shift(feats):
    """What each transformation does to the writing, relative to Generated."""
    keys = ["sent_len_words", "type_token_ratio", "stopword_ratio",
            "punct_ratio", "conj_ratio", "burstiness"]
    names = ["Sentence length", "Type–token ratio", "Stopword ratio",
             "Punctuation ratio", "Conjunction ratio", "Burstiness"]
    ai = feats[feats.label != "Human"]
    human = feats[feats.label == "Human"]

    base = ai[ai.variant == "Generated"][keys].mean()
    shifts = {v: (ai[ai.variant == v][keys].mean() / base - 1) * 100 for v in VARIANTS[1:]}
    shifts["Human (reference)"] = (human[keys].mean() / base - 1) * 100

    fig, ax = plt.subplots(figsize=(11, 4.6))
    series = list(shifts)
    w = 0.8 / len(series)
    x = np.arange(len(keys))
    for i, name in enumerate(series):
        vals = shifts[name].values
        off = (i - (len(series) - 1) / 2) * w
        bars = ax.bar(x + off, vals, width=w * 0.88, color=PALETTE[i + 1], label=name)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + (2.5 if v >= 0 else -6),
                    f"{v:+.0f}%", ha="center", fontsize=6.8, color=MUTED)

    style(ax)
    ax.axhline(0, color=INK, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=9)
    ax.set_ylabel("change vs the model's own Generated text", fontsize=9, color=MUTED)
    fig.suptitle("What paraphrasing, translating and “humanizing” actually change",
                 fontsize=13, color=INK, x=0.012, ha="left", y=1.13)
    leg = fig.legend(*ax.get_legend_handles_labels(), frameon=False, fontsize=8.5,
                     ncol=4, loc="upper left", bbox_to_anchor=(0.01, 1.055))
    for t in leg.get_texts():
        t.set_color(MUTED)
    fig.tight_layout()
    save(fig, "variant_shift.png")


# ---------------------------------------------------------------- figure 3

def benchmark(chunks, feature_set):
    """Grouped CV so chunks of one document never straddle a fold."""
    out = {}
    models = sorted(c for c in chunks.label.unique() if c != "Human")
    for model in models:
        sub = chunks[chunks.label.isin(["Human", model])]
        X = sub[feature_set].values
        y = (sub.label == model).astype(int).values
        groups = sub.doc.values
        for clf_name, make in CLASSIFIERS.items():
            scores = []
            for seed in range(5):
                cv = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=seed)
                scores += list(cross_val_score(make(), X, y, groups=groups,
                                               cv=cv, scoring="accuracy"))
            out[(model, clf_name)] = (statistics.mean(scores), statistics.pstdev(scores))
    return out


def fig_accuracy(chunks):
    sets = [("Scale-free features only", SCALE_FREE), ("All features, length included", FEATURES)]
    results = [(title, benchmark(chunks, fs)) for title, fs in sets]
    models = sorted(c for c in chunks.label.unique() if c != "Human")
    names = list(CLASSIFIERS)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), sharey=True)
    for ax, (title, res) in zip(axes, results):
        w = 0.8 / len(models)
        x = np.arange(len(names))
        for i, model in enumerate(models):
            means = [res[(model, n)][0] for n in names]
            errs = [res[(model, n)][1] for n in names]
            off = (i - (len(models) - 1) / 2) * w
            ax.bar(x + off, means, width=w * 0.86, color=PALETTE[i],
                   label=f"Human vs {model}",
                   yerr=errs, error_kw=dict(ecolor=MUTED, elinewidth=0.8, capsize=1.5))
        style(ax)
        ax.axhline(0.5, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
        ax.text(len(names) - 0.4, 0.515, "chance", fontsize=7.5, color=MUTED, ha="right")
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=8.5, rotation=18, ha="right")
        ax.set_ylim(0, 1.12)
        ax.set_title(title, fontsize=10.5, color=INK, pad=10)
    axes[0].set_ylabel("accuracy, grouped 3-fold CV × 5 seeds", fontsize=9, color=MUTED)

    fig.suptitle("Can an ensemble tell this apart? Only 44 source documents, so read the "
                 "error bars, not the peaks.", fontsize=12.5, color=INK, y=1.10)
    leg = fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=8.5,
                     ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.045))
    for t in leg.get_texts():          # identity lives in the swatch, not the words
        t.set_color(MUTED)
    fig.tight_layout()

    # relief for the low-contrast slots, and the numbers behind the bars
    pd.DataFrame([{"features": title, "comparison": f"Human vs {m}", "classifier": c,
                   "accuracy": round(v[0], 4), "sd": round(v[1], 4)}
                  for title, res in results for (m, c), v in res.items()]
                 ).to_csv(OUT / "accuracy_table.csv", index=False)
    print("  wrote evaluation/figures/accuracy_table.csv")
    save(fig, "classifier_accuracy.png")
    return results


# ---------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="~/Downloads/archive")
    a = ap.parse_args()

    df = load(a.data)
    print(f"loaded {len(df)} texts from {df.doc.nunique()} source documents "
          f"({(df.label == 'Human').sum()} human, {(df.label != 'Human').sum()} AI)")

    feats = feature_frame(df)
    chunks = feature_frame(df[df.variant.isin(["Written", "Generated"])], chunked=True)
    print(f"chunked into {len(chunks)} samples for classification "
          f"({(chunks.label == 'Human').sum()} human / {(chunks.label != 'Human').sum()} AI)")

    OUT.mkdir(parents=True, exist_ok=True)
    feats.groupby(["label", "variant"])[FEATURES].mean().round(4) \
         .to_csv(OUT / "feature_table.csv")
    print("  wrote evaluation/figures/feature_table.csv")

    fig_features_by_class(feats)
    fig_variant_shift(feats)
    results = fig_accuracy(chunks)

    print("\naccuracy, mean over 15 grouped folds:")
    for title, res in results:
        best = max(res.items(), key=lambda kv: kv[1][0])
        allm = [v[0] for v in res.values()]
        print(f"  {title:32s} mean {statistics.mean(allm):.3f}  "
              f"best {best[0][1]} on {best[0][0]} {best[1][0]:.3f} ±{best[1][1]:.3f}")


if __name__ == "__main__":
    main()
