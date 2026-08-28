"""
make_figures_v2.py — Chapter 5 figures from the rescored (corrected-label) results.

Run after rescore.py. Every figure is regenerated from the CSVs, so a change
to the data always propagates to the report.
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

NAVY, AMBER, GREY, GREEN = "#1f3864", "#c0762a", "#8a8f99", "#1a7f52"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})


def frac(s):
    a, b = str(s).split("/")
    return 100 * int(a) / max(int(b), 1)


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
    print(f"  -> {FIGS / name}")


def read(name):
    p = RESULTS / name
    return pd.read_csv(p) if p.exists() else None


def fig_rag():
    df = read("rescored_rag_summary.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.3))
    x = np.arange(len(df))
    w = 0.38
    hall = df["hallucination_out_of_corpus"] * 100
    corr = [frac(v) for v in df["answered_when_it_should"]]
    b1 = ax.bar(x - w / 2, hall, w, label="Hallucination rate\n(unanswerable, n=15)", color=AMBER)
    b2 = ax.bar(x + w / 2, corr, w, label="Answered correctly\n(answerable, n=15)", color=NAVY)
    ax.bar_label(b1, fmt="%.0f%%", padding=3, fontsize=9)
    ax.bar_label(b2, fmt="%.0f%%", padding=3, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(df["condition"])
    ax.set_ylim(0, 118)
    ax.legend(frameon=False, fontsize=8.5, loc="upper center", ncol=2)
    style(ax, "Percentage of questions", "Retrieval grounding versus ungrounded generation (n=30)")
    save(fig, "fig5_2_rag_on_off.png")


def fig_two_bar(csv, title, fname):
    df = read(csv)
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.3))
    x = np.arange(len(df))
    w = 0.27
    hall = df["hallucination_out_of_corpus"] * 100
    fr = [frac(v) for v in df["false_refusals"]]
    corr = [frac(v) for v in df["answered_when_it_should"]]
    b1 = ax.bar(x - w, hall, w, label="Hallucination rate", color=AMBER)
    b2 = ax.bar(x, fr, w, label="False refusals", color=GREY)
    b3 = ax.bar(x + w, corr, w, label="Answered correctly", color=NAVY)
    for b in (b1, b2, b3):
        ax.bar_label(b, fmt="%.0f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(df["condition"])
    ax.set_ylim(0, 108)
    ax.legend(frameon=False, fontsize=8.5, ncol=3, loc="upper center")
    style(ax, "Percentage of questions", title)
    save(fig, fname)


def fig_topk():
    df = read("rescored_topk_summary.csv")
    if df is None:
        return
    ks = [int(str(c).split("=")[1]) for c in df["condition"]]
    corr = [frac(v) for v in df["answered_when_it_should"]]
    fr = [frac(v) for v in df["false_refusals"]]
    hall = df["hallucination_out_of_corpus"] * 100

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    ax.plot(ks, corr, "o-", color=NAVY, lw=2, label="Answered correctly")
    ax.plot(ks, fr, "s--", color=GREY, lw=2, label="False refusals")
    ax.plot(ks, hall, "^:", color=AMBER, lw=2, label="Hallucination rate")
    for xi, yi in zip(ks, corr):
        ax.annotate(f"{yi:.0f}%", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8.5, color=NAVY)
    ax2 = ax.twinx()
    ax2.plot(ks, df["median_s"], "d-.", color=GREEN, lw=1.4, label="Median latency")
    ax2.set_ylabel("Median latency (s)", color=GREEN)
    ax2.tick_params(axis="y", labelcolor=GREEN)
    ax2.spines[["top"]].set_visible(False)

    ax.set_xticks(ks)
    ax.set_xlabel("k (passages retrieved)")
    ax.set_ylim(0, 108)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, fontsize=8.5, loc="center right")
    style(ax, "Percentage of questions", "Effect of k on answer coverage and refusal behaviour")
    save(fig, "fig5_5_topk_sweep.png")


def fig_distances():
    p = RESULTS / "rescored_rag_responses.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    on = d[d.condition == "RAG on"]
    groups = [("answerable", "Answerable", NAVY),
              ("related_unanswerable", "Related but\nunanswerable", GREY),
              ("out_of_corpus", "Out of corpus", AMBER)]

    fig, ax = plt.subplots(figsize=(7.2, 4.3))
    for i, (key, label, colour) in enumerate(groups):
        vals = on[on.category == key]["nearest_distance"].dropna()
        ax.scatter(np.random.default_rng(0).normal(i, 0.055, len(vals)), vals,
                   s=42, color=colour, alpha=0.85, edgecolor="white", linewidth=0.8,
                   label=f"{label.replace(chr(10), ' ')} (n={len(vals)})", zorder=3)

    a = on[on.expected == "answer"]["nearest_distance"].max()
    b = on[on.category == "out_of_corpus"]["nearest_distance"].min()
    ax.axhspan(a, b, color=GREEN, alpha=0.12, zorder=1)
    ax.axhline((a + b) / 2, color=GREEN, ls="--", lw=1.5, zorder=2)
    ax.text(2.42, (a + b) / 2, f" threshold {((a + b) / 2):.3f}", color=GREEN,
            fontsize=9, va="bottom", ha="right", fontweight="bold")
    ax.text(2.42, a + 0.004, f" separating margin {b - a:.3f}", color="#2f6b4f",
            fontsize=8, va="bottom", ha="right")

    ax.set_xticks(range(3))
    ax.set_xticklabels([g[1] for g in groups], fontsize=9)
    ax.set_xlim(-0.5, 2.5)
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    style(ax, "Cosine distance to nearest chunk",
          "Retrieval distance separates in-corpus from out-of-corpus queries")
    save(fig, "fig5_3_threshold_distances.png")


if __name__ == "__main__":
    print("Building figures from rescored results...")
    fig_rag()
    fig_distances()
    fig_two_bar("rescored_threshold_summary.csv",
                "Retrieval-confidence threshold on versus off",
                "fig5_3_threshold_ablation.png")
    fig_two_bar("rescored_chunking_summary.csv",
                "Fixed-size versus sentence-aware chunking",
                "fig5_4_chunking_ablation.png")
    fig_topk()
    print(f"\nDone — {FIGS}")
