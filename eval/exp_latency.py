"""
exp_latency.py — Section 5.8, end-to-end latency across model variants.

The preliminary report set a sub-10-second non-functional requirement and
measured ~15s steady-state on the prototype. This is where that NFR is
either met or revised with measured justification. The project risk table
already anticipated this, so a revision here is executed contingency rather
than failure — but it has to be evidenced.

Cold start and steady state are reported separately because they are
different user experiences: the first question of a session pays for the
model load, every later one does not.

Usage
-----
    python exp_latency.py
    python exp_latency.py --models llama3.1:8b,llama3.2:3b
    python exp_latency.py --n 20
"""

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import API, RESULTS, ask, check_backend, progress, save

DEFAULT_MODELS = ["llama3.1:8b", "llama3.2:3b", "llama3.1:8b-instruct-q4_K_M"]

# Twenty questions the corpus can answer, so every run exercises the full
# retrieve-and-generate path. Questions that trigger a refusal would skip
# the language model entirely and report a latency of near zero, which
# would quietly flatter every mean in this table.
QUESTIONS = [
    "What is stemming in text normalisation?",
    "What is lemmatization?",
    "Explain the Markov assumption.",
    "How does TF-IDF weight terms?",
    "What is cosine similarity used for?",
    "What is perplexity?",
    "What is add-one smoothing?",
    "What are stopwords?",
    "What is the difference between types and tokens?",
    "Explain the parallelogram model for analogies.",
    "What assumption makes Naive Bayes naive?",
    "Define precision and recall.",
    "What is a confusion matrix?",
    "What is overfitting?",
    "Explain cross-validation.",
    "What are the three categories of machine learning?",
    "What is a context-free grammar?",
    "What is the CKY algorithm?",
    "What is a text corpus?",
    "Why are natural languages harder to parse than formal languages?",
]


def unload_models():
    """
    Stop any running Ollama model so the next call genuinely pays cold start.

    Without this, "cold start" is measured against an already-warm model and
    the number is meaningless. Best effort — if the CLI is unavailable the
    script says so rather than reporting a figure it cannot stand behind.
    """
    try:
        subprocess.run(["ollama", "stop", "--all"], capture_output=True, timeout=30)
        return True
    except Exception:
        try:
            out = subprocess.run(["ollama", "ps"], capture_output=True, text=True, timeout=20)
            for line in out.stdout.splitlines()[1:]:
                name = line.split()[0] if line.split() else None
                if name:
                    subprocess.run(["ollama", "stop", name], capture_output=True, timeout=30)
            return True
        except Exception:
            return False


def set_model(model):
    """
    Point the backend at a different model.

    LLM_MODEL is read at import time, so the backend has to be restarted
    between models. The script cannot do that for you — it prompts instead,
    which is honest about what it controls.
    """
    health = requests.get(f"{API}/health", timeout=10).json()
    if health["llm"] == model:
        return True
    print(f"\n  Backend is running {health['llm']}, this block needs {model}.")
    print(f"  In the backend terminal: Ctrl+C, then")
    print(f"      LLM_MODEL={model} uvicorn main:app")
    print("  (Windows PowerShell:  $env:LLM_MODEL='%s'; uvicorn main:app)" % model)
    input("  Press Enter once it has restarted... ")
    health = requests.get(f"{API}/health", timeout=10).json()
    if health["llm"] != model:
        print(f"  Backend still reports {health['llm']} — skipping {model}.")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--nfr", type=float, default=10.0, help="the latency NFR in seconds")
    args = ap.parse_args()

    check_backend()
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    questions = QUESTIONS[: args.n]

    rows = []
    for model in models:
        if not set_model(model):
            continue

        print(f"\n--- {model} ---")
        cold_ok = unload_models()
        if not cold_ok:
            print("  (could not unload; cold-start figure will be unreliable)")
        time.sleep(2)

        for i, q in enumerate(questions, 1):
            r = ask(q, use_rag=True, use_threshold=False, collection="sentence")
            rows.append({
                "model": model,
                "index": i,
                "run_type": "cold" if i == 1 else "steady",
                "question": q,
                "retrieve_s": r.get("retrieve_s"),
                "generate_s": r.get("generate_s"),
                "total_s": r.get("total_s"),
                "wall_s": r.get("wall_s"),
                "cold_start_reliable": cold_ok if i == 1 else "",
                "error": r.get("error", ""),
            })
            progress(i, len(questions), f"{r.get('total_s')}s")

    if not rows:
        sys.exit("No measurements taken.")

    df = pd.DataFrame(rows)
    save("latency_raw.csv", df)

    summary = []
    for model, sub in df.groupby("model", sort=False):
        steady = sub[sub.run_type == "steady"]["total_s"].dropna()
        cold = sub[sub.run_type == "cold"]["total_s"].dropna()
        if steady.empty:
            continue
        summary.append({
            "model": model,
            "n_steady": len(steady),
            "cold_start_s": round(cold.iloc[0], 2) if len(cold) else None,
            "steady_mean_s": round(steady.mean(), 2),
            "steady_median_s": round(steady.median(), 2),
            "steady_p95_s": round(steady.quantile(0.95), 2),
            "steady_min_s": round(steady.min(), 2),
            "steady_max_s": round(steady.max(), 2),
            "steady_sd_s": round(statistics.stdev(steady), 2) if len(steady) > 1 else 0.0,
            "retrieve_mean_s": round(sub["retrieve_s"].dropna().mean(), 3),
            "meets_nfr": bool(steady.mean() < args.nfr),
            "pct_under_nfr": round((steady < args.nfr).mean() * 100, 1),
        })

    s = pd.DataFrame(summary)
    save("latency_summary.csv", s)

    print("\n" + "=" * 64)
    print(f"LATENCY SUMMARY (NFR target: under {args.nfr}s)")
    print("=" * 64)
    print(s.to_string(index=False))

    print("\nReport cold start and steady state separately — they are different")
    print("user experiences. Retrieval is the small number here; if the mean")
    print("misses the NFR, the honest move is to revise the NFR with these")
    print("figures and state the quality trade-off of the model you switch to.")


if __name__ == "__main__":
    main()
