"""
rescore.py — re-apply corrected ground-truth labels to responses already collected.

Why this exists
---------------
The category labels in questions_rag.csv are claims about the CORPUS, not about
the system. Three questions originally labelled "related but unanswerable" turned
out to be answerable: TF-IDF, cosine similarity and PPMI are all present in the
Week 8 readings, whose filenames list section numbers that understate what the
PDFs actually contain. Labelling from filenames was the mistake; every label is
now verified by full-text search over the extracted corpus.

Correcting a label is not correcting a result. The model's answers are unchanged
and are not re-generated — only the ground truth they are scored against moves.
That is why this rescores from the saved responses instead of re-running an hour
of generation, and it is the honest way round: re-running after seeing the scores
would invite the question of what else was tuned.

Usage
-----
    python rescore.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, classify  # noqa: E402

HERE = Path(__file__).resolve().parent
QUESTIONS = HERE / "questions_rag.csv"

EXPERIMENTS = ["rag", "threshold", "chunking", "topk"]

# Questions that were relabelled keep their responses but change identity.
# Mapping them rather than dropping them means the corrected scoring uses
# every answer that was actually generated, and nothing is quietly lost.
ALIASES = {
    "C01": "A15",   # TF-IDF          — present in 4 files
    "C02": "A16",   # cosine similarity — present in 5 files
    "C06": "A17",   # PPMI            — present in 7 files
}


def summarise(df):
    out = []
    for label, sub in df.groupby("condition", sort=False):
        unans = sub[sub.expected == "refuse"]
        ans = sub[sub.expected == "answer"]
        leaked = unans[unans.outcome != "refused"]

        # Reported separately because the two unanswerable categories behave
        # very differently: out-of-corpus questions retrieve nothing close,
        # while topically-related ones retrieve the right section that simply
        # lacks the fact. Averaging them hides the only interesting part.
        ooc = sub[sub.category == "out_of_corpus"]
        rel = sub[sub.category == "related_unanswerable"]

        out.append({
            "condition": label,
            "n": len(sub),
            "hallucination_rate_all": round(len(leaked) / len(unans), 4) if len(unans) else None,
            "hallucination_out_of_corpus": round(
                (ooc.outcome != "refused").mean(), 4) if len(ooc) else None,
            "hallucination_related": round(
                (rel.outcome != "refused").mean(), 4) if len(rel) else None,
            "clean_refusals": f"{int((unans.outcome == 'refused').sum())}/{len(unans)}",
            "hedged_on_unanswerable": int((unans.outcome == "hedged").sum()),
            "answered_when_it_should": f"{int((ans.outcome == 'answered').sum())}/{len(ans)}",
            "false_refusals": f"{int((ans.outcome == 'refused').sum())}/{len(ans)}",
            "hedged_on_answerable": int((ans.outcome == "hedged").sum()),
            "threshold_refusals": int((sub.refusal_reason == "retrieval_distance").sum()),
            "model_refusals": int((sub.refusal_reason == "model_judgement").sum()),
            "mean_s": round(sub["total_s"].dropna().mean(), 2),
            "median_s": round(sub["total_s"].dropna().median(), 2),
        })
    return pd.DataFrame(out)


def main():
    truth = pd.read_csv(QUESTIONS)[["qid", "category", "expected"]]
    print(f"Ground truth: {len(truth)} questions")
    print(truth["category"].value_counts().to_string(), "\n")

    for exp in EXPERIMENTS:
        path = RESULTS / f"rag_{exp}_responses.csv"
        if not path.exists():
            print(f"  (skipped {exp}: no responses file)")
            continue

        df = pd.read_csv(path)
        before = len(df)
        df["qid"] = df["qid"].replace(ALIASES)
        df = df.drop(columns=["category", "expected"], errors="ignore")
        df = df.merge(truth, on="qid", how="inner")
        dropped = before - len(df)

        # Re-derive the outcome from the answer text so that responses saved
        # before the hedged/refused distinction existed are scored the same way.
        df["outcome"] = df.apply(
            lambda r: classify({
                "answer": r.get("answer"),
                "refusal_reason": r.get("refusal_reason"),
            }), axis=1)
        df["refused"] = df.outcome == "refused"
        df["behaved_correctly"] = df.refused == (df.expected == "refuse")

        df.to_csv(RESULTS / f"rescored_{exp}_responses.csv", index=False)
        s = summarise(df)
        s.to_csv(RESULTS / f"rescored_{exp}_summary.csv", index=False)

        print("=" * 78)
        print(f"{exp.upper()}   ({len(df)} scored"
              + (f", {dropped} dropped as relabelled/removed)" if dropped else ")"))
        print("=" * 78)
        cols = ["condition", "hallucination_out_of_corpus", "hallucination_related",
                "clean_refusals", "answered_when_it_should", "false_refusals",
                "hedged_on_answerable", "median_s"]
        print(s[cols].to_string(index=False))
        print()

    # ---- separating margin on the corrected labels ----------------------
    rag = RESULTS / "rescored_rag_responses.csv"
    if rag.exists():
        d = pd.read_csv(rag)
        on = d[d.condition == "RAG on"]
        a = on[on.expected == "answer"]["nearest_distance"].dropna()
        b = on[on.category == "out_of_corpus"]["nearest_distance"].dropna()
        if len(a) and len(b):
            print("RETRIEVAL DISTANCE, corrected labels")
            print(on.groupby("category")["nearest_distance"]
                  .agg(["count", "min", "mean", "max"]).round(4).to_string())
            print()
            print(f"  answerable max    {a.max():.4f}")
            print(f"  out-of-corpus min {b.min():.4f}")
            print(f"  separating margin {b.min() - a.max():.4f}")
            print(f"  max-margin choice {(a.max() + b.min()) / 2:.3f}")

    print(f"\nWrote rescored_*.csv to {RESULTS}")


if __name__ == "__main__":
    main()
