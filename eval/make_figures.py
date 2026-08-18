"""
make_figures.py — every Chapter 5 figure, from the results CSVs.

Rerun this after any experiment reruns. No figure in the report should be
built by hand: when a number changes, the figure has to change with it, and
a chart pasted from a spreadsheet three days ago is how inconsistencies get
into a report.

Usage
-----
    python make_figures.py
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RESULTS = Path(__file__).resolve().parent.parent / "results"
FIGS = RESULTS / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

NAVY, AMBER, GREY = "#1f3864", "#c0762a", "#8a8f99"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11})


def style(ax, ylabel=None, title=None):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", alpha=0.25)
    if ylabel:
        ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)


def save(fig, name):
    path = FIGS / name
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)
    print(f"  -> {path}")


def read(name):
    p = RESULTS / name
    if not p.exists():
        print(f"  (skipped: {name} not found)")
        return None
    return pd.read_csv(p)


# --------------------------------------------------------------------------
def fig_rag():
    df = read("rag_rag_summary.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    vals = df["hallucination_rate"] * 100
    bars = ax.bar(df["condition"], vals, width=0.5,
                  color=[AMBER if v > 50 else NAVY for v in vals])
    ax.bar_label(bars, fmt="%.1f%%", padding=3)
    ax.set_ylim(0, 108)
    style(ax, "Hallucination rate (%)",
          "Unsupported answers on out-of-corpus questions (n=18)")
    save(fig, "fig_rag_on_off.png")


def fig_ablation(name, csv, title, fname):
    df = read(csv)
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(len(df))
    w = 0.38

    hall = df["hallucination_rate"] * 100
    false_ref = [
        100 * int(str(v).split("/")[0]) / max(int(str(v).split("/")[1]), 1)
        for v in df["false_refusals_on_answerable"]
    ]

    b1 = ax.bar(x - w / 2, hall, w, label="Hallucination rate", color=AMBER)
    b2 = ax.bar(x + w / 2, false_ref, w, label="False refusals", color=NAVY)
    ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=8, padding=2)

    ax.set_xticks(x)
    ax.set_xticklabels(df["condition"])
    ax.set_ylim(0, max(max(hall.max(), max(false_ref)) * 1.35, 20))
    ax.legend(frameon=False)
    style(ax, "Percentage of questions", title)
    save(fig, fname)


def fig_topk():
    df = read("rag_topk_summary.csv")
    if df is None:
        return
    ks = [int(str(c).split("=")[1]) for c in df["condition"]]
    hall = df["hallucination_rate"] * 100
    prec = read("retrieval_precision.csv")

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.plot(ks, hall, "o-", color=AMBER, label="Hallucination rate (%)")
    if prec is not None:
        ax.plot(prec["k"], prec["precision_at_k"] * 100, "s--", color=NAVY,
                label="Precision@k (%)")
        if prec["recall_at_k"].notna().any():
            ax.plot(prec["k"], prec["recall_at_k"] * 100, "^:", color=GREY,
                    label="Recall@k (%)")
    ax.set_xticks(ks)
    ax.set_xlabel("k (passages retrieved)")
    ax.legend(frameon=False)
    style(ax, "Percentage", "Effect of k on retrieval and on answers")
    save(fig, "fig_topk_sweep.png")


def fig_threshold_sweep():
    df = read("threshold_sweep.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(df["threshold"], df["refuse_out_of_corpus"] * 100, "-", color=NAVY,
            label="Refused: out of corpus (want 100%)")
    ax.plot(df["threshold"], df["refuse_related_unanswerable"] * 100, "--", color="#4a6ea8",
            label="Refused: related but unanswerable (want 100%)")
    ax.plot(df["threshold"], df["refuse_answerable"] * 100, "-", color=AMBER,
            label="Refused: answerable (want 0%)")

    best = df.loc[df["balanced_accuracy"].idxmax()]
    ax.axvline(best["threshold"], color=GREY, ls=":", lw=1.4)
    ax.annotate(f"chosen: {best['threshold']:.2f}",
                xy=(best["threshold"], 50), xytext=(6, 0),
                textcoords="offset points", fontsize=9, color="#444")

    ax.set_xlabel("Cosine distance threshold")
    ax.legend(frameon=False, fontsize=8.5)
    style(ax, "Questions refused (%)", "Choosing the retrieval-confidence threshold")
    save(fig, "fig_threshold_sweep.png")


def fig_latency():
    df = read("latency_summary.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    x = np.arange(len(df))
    w = 0.36

    b1 = ax.bar(x - w / 2, df["steady_mean_s"], w, label="Steady state", color=NAVY,
                yerr=df["steady_sd_s"], capsize=3, error_kw={"lw": 1, "ecolor": "#888"})
    b2 = ax.bar(x + w / 2, df["cold_start_s"], w, label="Cold start", color=AMBER)
    ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=3)
    ax.bar_label(b2, fmt="%.1f", fontsize=8, padding=3)

    ax.axhline(10, color="#b3261e", ls="--", lw=1.3)
    ax.text(0.012, 10, " NFR target: 10s", transform=ax.get_yaxis_transform(),
            va="bottom", fontsize=9, color="#b3261e", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(":", "\n") for m in df["model"]], fontsize=8.5)
    ax.legend(frameon=False)
    style(ax, "Seconds", "End-to-end latency by model")
    save(fig, "fig_latency.png")


def fig_squad():
    df = read("squad_summary.csv")
    if df is None:
        return
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    x = np.arange(len(df))
    w = 0.38
    b1 = ax.bar(x - w / 2, df["exact_match"], w, label="Exact Match", color=NAVY)
    b2 = ax.bar(x + w / 2, df["f1"], w, label="F1", color=AMBER)
    ax.bar_label(b1, fmt="%.1f", fontsize=8, padding=2)
    ax.bar_label(b2, fmt="%.1f", fontsize=8, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s}\n(n={n})" for s, n in zip(df["subset"], df["n"])])
    ax.set_ylim(0, 108)
    ax.legend(frameon=False)
    style(ax, "Score (%)", "SQuAD 2.0 performance")
    save(fig, "fig_squad.png")


def fig_sus():
    """Built from results/sus_scores.csv once Saturday's study is scored."""
    df = read("sus_scores.csv")
    if df is None:
        return
    scores = df["sus_score"].dropna()
    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    ax.bar(range(1, len(scores) + 1), sorted(scores), color=NAVY, width=0.6)
    ax.axhline(68, color="#b3261e", ls="--", lw=1.3)
    ax.annotate("industry average: 68", xy=(len(scores), 68), xytext=(0, 4),
                textcoords="offset points", ha="right", fontsize=9, color="#b3261e")
    ax.axhline(scores.mean(), color=AMBER, ls="-", lw=1.5)
    ax.annotate(f"mean: {scores.mean():.1f}", xy=(0.6, scores.mean()), xytext=(0, 5),
                textcoords="offset points", fontsize=9, color=AMBER)
    ax.set_xticks(range(1, len(scores) + 1))
    ax.set_xlabel("Participant (sorted)")
    ax.set_ylim(0, 105)
    style(ax, "SUS score (0-100)", f"System Usability Scale (n={len(scores)})")
    save(fig, "fig_sus.png")


def main():
    if not RESULTS.exists():
        sys.exit(f"ERROR: {RESULTS} not found. Run the experiments first.")

    print("Building figures...")
    fig_rag()
    fig_ablation("threshold", "rag_threshold_summary.csv",
                 "Does the retrieval threshold work?", "fig_threshold_ablation.png")
    fig_ablation("chunking", "rag_chunking_summary.csv",
                 "Fixed-size versus sentence-aware chunking", "fig_chunking_ablation.png")
    fig_topk()
    fig_threshold_sweep()
    fig_latency()
    fig_squad()
    fig_sus()
    print(f"\nAll figures in {FIGS}")
    print("Paste them straight into Chapter 5 — they are 300 dpi.")


if __name__ == "__main__":
    main()
