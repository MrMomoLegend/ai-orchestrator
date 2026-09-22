"""
make_figures_v3.py - Chapter 5 figures (5.1-5.5) from the new open-corpus results (21 Sep 2026).

Reads only results/*.csv at the repository root (new corpus, GPU). Frozen parameters:
threshold 0.53, k=10. Figure 5.6 (WER) is corpus-independent and is not regenerated here.
Terminology: an answer to an out-of-corpus question is an "unsupported answer", not a
hallucination - several ungrounded answers are factually correct.
"""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)
THRESHOLD = 0.53

NAVY, AMBER, GREY, GREEN, LIGHT = "#1f3864", "#c0762a", "#8a8f99", "#1a7f52", "#c9ced6"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})


def num(s):
    return int(str(s).split("/")[0])


def style(ax, ylabel=None, title=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def save(fig, name):
    fig.tight_layout()
    fig.savefig(FIGS / name, dpi=300)
    plt.close(fig)
    print(f"  -> {name}")


def read(name):
    return pd.read_csv(RESULTS / name)


def is_true(v):
    return str(v).strip().lower() in ("true", "1", "1.0")


def labelled(b, ax, texts):
    for rect, t in zip(b, texts):
        ax.annotate(t, (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8.5)


# ---------------------------------------------------------------- Figure 5.1
def fig_rag():
    s = read("rag_rag_summary.csv").set_index("condition")
    r = read("rag_rag_responses.csv")
    ans = r[r.category == "answerable"]
    off = ans[ans.condition == "RAG off"].set_index("qid")
    on = ans[ans.condition == "RAG on"].set_index("qid")
    both = [q for q in off.index if off.loc[q, "outcome"] == "answered" and on.loc[q, "outcome"] == "answered"]
    paired = {c: sum(is_true(d.loc[q, "answer_correct"]) for q in both) for c, d in (("RAG off", off), ("RAG on", on))}
    n_pair = len(both)

    conds = ["RAG off", "RAG on"]
    unsup = [int(s.loc[c, "answered_on_unanswerable"]) + int(s.loc[c, "hedged_on_unanswerable"]) for c in conds]
    answered = [num(s.loc[c, "answered_when_it_should"]) for c in conds]
    correct = [paired[c] for c in conds]

    fig, ax = plt.subplots(figsize=(7.6, 4.3))
    x = np.arange(2)
    w = 0.26
    series = [(unsup, 15, "Unsupported answer\n(unanswerable, n=15)", AMBER),
              (answered, 15, "Answered\n(answerable, n=15)", GREY),
              (correct, n_pair, f"Factually correct\n(both answered, n={n_pair})", NAVY)]
    for i, (vals, n, lab, col) in enumerate(series):
        pct = [100 * v / n for v in vals]
        b = ax.bar(x + (i - 1) * w, pct, w, label=lab, color=col)
        labelled(b, ax, [f"{v}/{n}" for v in vals])
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylim(0, 125)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=3)
    style(ax, "Percentage of questions", "Retrieval grounding against ungrounded generation (n=30)")
    save(fig, "fig5_1_rag_on_off.png")
    return unsup, answered, correct, n_pair


# ---------------------------------------------------------------- Figure 5.2
def fig_distances():
    d = read("threshold_distances.csv")
    groups = [("answerable", "Answerable", NAVY),
              ("related_unanswerable", "Related but\nunanswerable", GREY),
              ("out_of_corpus", "Out of corpus", AMBER)]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    rng = np.random.default_rng(0)
    for i, (key, label, col) in enumerate(groups):
        v = d[d.category == key]["nearest_distance"]
        ax.scatter(rng.normal(i, 0.06, len(v)), v, s=44, color=col, alpha=0.9,
                   edgecolor="white", linewidth=0.8, zorder=3,
                   label=f"{label.replace(chr(10), ' ')} (n={len(v)})")
    a_max = d[d.category == "answerable"]["nearest_distance"].max()
    b_min = d[d.category == "out_of_corpus"]["nearest_distance"].min()
    b_id = d.loc[d[d.category == "out_of_corpus"]["nearest_distance"].idxmin(), "qid"]
    ax.axhline(THRESHOLD, color=GREEN, ls="--", lw=1.6, zorder=2)
    ax.text(-0.45, THRESHOLD + 0.006, f"frozen threshold {THRESHOLD:.2f}", color=GREEN,
            fontsize=9, va="bottom", ha="left", fontweight="bold")
    ax.annotate(f"{b_id} {b_min:.3f}", (2, b_min), xytext=(2.12, b_min + 0.05),
                fontsize=8.5, arrowprops=dict(arrowstyle="-", color="#555", lw=0.8))
    ax.annotate(f"answerable max {a_max:.3f}", (0, a_max), xytext=(0.18, a_max + 0.04),
                fontsize=8.5, arrowprops=dict(arrowstyle="-", color="#555", lw=0.8))
    ax.text(1.6, THRESHOLD + 0.008, "above: refused by gate", ha="right", fontsize=8, color="#555", style="italic", va="bottom")
    ax.text(1.6, THRESHOLD - 0.008, "below: passed to the model", ha="right", fontsize=8, color="#555", style="italic", va="top")
    ax.set_xticks(range(3))
    ax.set_xticklabels([g[1] for g in groups], fontsize=9)
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.1, 0.87)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(ax, "Cosine distance to nearest chunk",
          "Nearest-chunk distance by question category (threshold not re-tuned)")
    save(fig, "fig5_2_threshold_distances.png")
    return a_max, b_min, b_id


# ---------------------------------------------------------------- Figure 5.3
def fig_threshold():
    s = read("rag_threshold_summary.csv")
    conds = list(s.condition)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.6, 4.3), gridspec_kw={"width_ratios": [1.5, 1]})
    x = np.arange(len(conds))
    w = 0.26
    series = [([int(v) for v in s.answered_on_unanswerable + s.hedged_on_unanswerable], "Unsupported answers", AMBER),
              ([num(v) for v in s.answered_when_it_should], "Answered", NAVY),
              ([num(v) for v in s.false_refusals_on_answerable], "False refusals", GREY)]
    for i, (vals, lab, col) in enumerate(series):
        b = ax1.bar(x + (i - 1) * w, vals, w, label=lab, color=col)
        labelled(b, ax1, [f"{v}/15" for v in vals])
    ax1.set_xticks(x)
    ax1.set_xticklabels(conds)
    ax1.set_ylim(0, 18.5)
    ax1.legend(frameon=False, fontsize=8, loc="upper center", ncol=3)
    style(ax1, "Questions (of 15 per category)", "Outcomes")

    gate = list(s.threshold_refusals.astype(int))
    model = list(s.model_refusals.astype(int))
    b1 = ax2.bar(x, gate, 0.5, color=GREEN, label="Refused by gate (~10 ms)")
    b2 = ax2.bar(x, model, 0.5, bottom=gate, color=LIGHT, label="Refused by model (~1.5 s)")
    for i in range(len(conds)):
        if gate[i]:
            ax2.text(i, gate[i] / 2, str(gate[i]), ha="center", va="center", color="white", fontsize=9)
        ax2.text(i, gate[i] + model[i] / 2, str(model[i]), ha="center", va="center", fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xticklabels(conds)
    ax2.set_ylim(0, 22)
    ax2.legend(frameon=False, fontsize=8, loc="upper center")
    style(ax2, "Refusals", "Where refusals happen")
    fig.suptitle("Retrieval-confidence threshold on against off", fontsize=11)
    save(fig, "fig5_3_threshold_ablation.png")
    return s


# ---------------------------------------------------------------- Figure 5.4
def fig_chunking():
    s = read("rag_chunking_summary.csv")
    conds = list(s.condition)
    fig, ax = plt.subplots(figsize=(7, 4.3))
    x = np.arange(len(conds))
    w = 0.2
    series = [([int(v) for v in s.answered_on_unanswerable + s.hedged_on_unanswerable], "Unsupported answers", AMBER),
              ([num(v) for v in s.answered_when_it_should], "Answered", NAVY),
              ([num(v) for v in s.false_refusals_on_answerable], "False refusals", GREY),
              ([int(v) for v in s.hedged_on_answerable], "Hedged", LIGHT)]
    for i, (vals, lab, col) in enumerate(series):
        b = ax.bar(x + (i - 1.5) * w, vals, w, label=lab, color=col)
        labelled(b, ax, [f"{v}/15" for v in vals])
    ax.set_xticks(x)
    ax.set_xticklabels(conds)
    ax.set_ylim(0, 18.5)
    ax.legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center")
    style(ax, "Questions (of 15 per category)", "Fixed-size against sentence-aware chunking")
    save(fig, "fig5_4_chunking_ablation.png")
    return s


# ---------------------------------------------------------------- Figure 5.5
def fig_topk():
    s = read("rag_topk_summary.csv")
    ks = [int(str(c).split("=")[1]) for c in s.condition]
    answered = [num(v) for v in s.answered_when_it_should]
    fr = [num(v) for v in s.false_refusals_on_answerable]
    unsup = [int(v) for v in s.answered_on_unanswerable + s.hedged_on_unanswerable]
    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(ks, answered, "o-", color=NAVY, lw=2, label="Answered (answerable)")
    ax.plot(ks, fr, "s--", color=GREY, lw=2, label="False refusals (answerable)")
    ax.plot(ks, unsup, "^:", color=AMBER, lw=2, label="Unsupported answers (unanswerable)")
    for xi, yi in zip(ks, answered):
        ax.annotate(f"{yi}/15", (xi, yi), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8.5, color=NAVY)
    for xi, yi in zip(ks, fr):
        ax.annotate(f"{yi}/15", (xi, yi), textcoords="offset points", xytext=(7, 5),
                    ha="left", fontsize=8.5, color="#555")
    ax.set_xticks(ks)
    ax.set_xlim(0.4, 11.2)
    ax.set_xlabel("k (passages retrieved)")
    ax.set_ylim(-1, 17)
    ax.legend(frameon=False, fontsize=8.5, loc="center right")
    style(ax, "Questions (of 15 per category)", "Effect of k on answer coverage and refusal behaviour")
    save(fig, "fig5_5_topk_sweep.png")
    return s


if __name__ == "__main__":
    print("Building Figures 5.1-5.5 from new-corpus results...")
    r1 = fig_rag()
    r2 = fig_distances()
    fig_threshold()
    fig_chunking()
    fig_topk()
    print("\nCHECK VALUES")
    print("5.1 unsupported off/on:", r1[0], " answered:", r1[1], " paired correct:", r1[2], "of", r1[3])
    print("5.2 answerable max %.4f  out-of-corpus min %.4f (%s)  margin to threshold %.4f" % (r2[0], r2[1], r2[2], r2[1] - THRESHOLD))
