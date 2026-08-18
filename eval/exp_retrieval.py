"""
exp_retrieval.py — Section 5.7, retrieval precision@k.

Measures the retrieval step on its own, with no language model involved.
This matters because Section 2.4 establishes that answer quality in a
retrieval-augmented system is bounded by retrieval quality: if this number
is poor, no amount of prompting fixes the answers.

Relevance labelling
-------------------
Run once with --label to generate a worksheet listing, for each query, the
top 10 retrieved chunks. Mark each 1 (relevant) or 0 (not), then rerun
without --label to score.

The labelling is done by one person — the author — who also wrote the
queries. That is a real limitation and Section 5.7 should name it rather
than hope nobody asks.

Usage
-----
    python exp_retrieval.py --label      # step 1: build the worksheet
    # ... fill in the 'relevant' column in results/retrieval_labels.csv ...
    python exp_retrieval.py              # step 2: score it
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import API, RESULTS, check_backend, progress, save

LABELS = RESULTS / "retrieval_labels.csv"
MAX_K = 10
KS = (1, 3, 5, 10)

# Queries with a known answer location in the NLP lecture corpus.
#
# Each one targets a distinct section, so a relevant passage exists and is
# findable. Queries whose answer is spread thinly across the whole corpus
# would make precision@k measure the question rather than the retriever.
QUERIES = [
    "What is stemming?",
    "What is lemmatization?",
    "What is the Markov assumption?",
    "How is TF-IDF calculated?",
    "What is cosine similarity?",
    "What is perplexity?",
    "What is add-one smoothing?",
    "What are stopwords?",
    "What is the difference between types and tokens?",
    "What is the parallelogram model for analogies?",
    "What is the conditional independence assumption in Naive Bayes?",
    "What is a confusion matrix?",
]


def retrieve(query, k=MAX_K):
    """
    Retrieval only — threshold=0.0 forces a refusal before the LLM is called,
    so the API returns the retrieved chunks and distances in milliseconds.
    """
    r = requests.post(f"{API}/ask", json={
        "question": query, "use_rag": True, "top_k": k,
        "use_threshold": True, "threshold": 0.0, "collection": "sentence",
    }, timeout=120)
    r.raise_for_status()
    return r.json()


def build_worksheet():
    print(f"Retrieving top {MAX_K} for {len(QUERIES)} queries ...")
    rows = []
    for i, q in enumerate(QUERIES, 1):
        data = retrieve(q)
        chunks = data.get("retrieved_chunks") or []
        dists = data.get("distances") or []
        for rank, chunk in enumerate(chunks, 1):
            rows.append({
                "query": q,
                "rank": rank,
                "distance": dists[rank - 1] if rank <= len(dists) else None,
                "relevant": "",          # <- you fill this in: 1 or 0
                "chunk": chunk.replace("\n", " ")[:400],
            })
        progress(i, len(QUERIES), q[:36])

    save("retrieval_labels.csv", rows)
    print(f"\nNow open {LABELS} and put 1 or 0 in the 'relevant' column")
    print("for every row, then rerun this script without --label.")
    print("\nJudge relevance as: could this passage alone support an answer")
    print("to the query? Apply the same rule to every row — drifting standards")
    print("between the top and bottom of the file is the main risk here.")


def score():
    if not LABELS.exists():
        sys.exit(f"ERROR: {LABELS} not found. Run with --label first.")

    df = pd.read_csv(LABELS)
    unlabelled = df["relevant"].isna().sum() + (df["relevant"].astype(str).str.strip() == "").sum()
    if unlabelled:
        sys.exit(f"ERROR: {unlabelled} rows still have no relevance label.")

    df["relevant"] = df["relevant"].astype(int)

    rows = []
    for k in KS:
        topk = df[df["rank"] <= k]
        per_query = topk.groupby("query")["relevant"].agg(["sum", "count"])
        per_query["precision"] = per_query["sum"] / per_query["count"]

        total_relevant = df.groupby("query")["relevant"].sum()
        recall = (per_query["sum"] / total_relevant.replace(0, pd.NA)).dropna()

        rows.append({
            "k": k,
            "precision_at_k": round(per_query["precision"].mean(), 4),
            "recall_at_k": round(recall.mean(), 4) if len(recall) else None,
            "queries_with_a_hit": int((per_query["sum"] > 0).sum()),
            "n_queries": len(per_query),
        })

    out = pd.DataFrame(rows)
    save("retrieval_precision.csv", out)

    print("\n" + "=" * 52)
    print("RETRIEVAL PRECISION@K")
    print("=" * 52)
    print(out.to_string(index=False))
    print("\nPrecision falling as k rises is expected and not a problem in")
    print("itself — what matters is whether k=3 captures the relevant passage")
    print("often enough. Read this against the top-k sweep in Section 5.5:")
    print("precision here, answer quality there.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", action="store_true", help="build the labelling worksheet")
    args = ap.parse_args()

    check_backend()
    build_worksheet() if args.label else score()


if __name__ == "__main__":
    main()
