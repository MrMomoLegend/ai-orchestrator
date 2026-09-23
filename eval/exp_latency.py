"""
exp_latency.py — Section 5.9, end-to-end latency.

The preliminary report set a sub-10-second non-functional requirement and
measured ~15 s steady state on the prototype. This is where that NFR is met
or revised with measured justification.

Why this script is not a simple timing loop
-------------------------------------------
Running the Section 5.3-5.6 ablations under deterministic decoding produced
three separate runs of one identical configuration. Every correctness
outcome agreed exactly across all three. The median latencies were
**40.64 s, 3.11 s and 3.25 s** — a thirteen-fold spread on a pipeline that
was, by its own outputs, doing precisely the same work.

Latency on a single laptop is therefore not a property of the
configuration alone. It depends on machine state: whether the model is
resident in memory, whether its weights are already in the OS file cache,
and how much sustained load preceded the measurement. A single pass over a question list
measures one arbitrary point on that surface and reports it as a fact.

Four things follow, and this script implements all four.

1.  **Warm-up is discarded, not averaged in.** Cold start is measured
    deliberately, once, after unloading the model. The requests immediately
    after it are thrown away rather than folded into the steady-state mean.

2.  **Every condition is measured in repeated blocks.** Block-level medians
    are reported alongside the pooled figure, so drift across a run is
    visible in the output instead of hiding inside it. If block 3 is twice
    block 1, the pooled mean is not a number worth printing and the script
    says so.

3.  **Retrieved context size is recorded per request.** Generation time on
    CPU is dominated by prompt processing, so context length is the
    explanatory variable behind the k=3 versus k=10 gap. Recording it turns
    Section 5.9 from a table of durations into a statement about what drives
    them, and lets `--k` produce the latency-versus-k curve that Section 5.9
    needs in order to state the true cost of raising k.

4.  **The threshold is disabled throughout.** A refused query returns in
    ~0.03 s without reaching the model. Including refusals would pull every
    median toward zero in proportion to how many questions the corpus cannot
    answer, which is a property of the question set, not of the system.

Usage
-----
    python exp_latency.py                      # shipped model, 3 blocks
    python exp_latency.py --repeats 5
    python exp_latency.py --k 1,3,5,10         # latency as a function of k
    python exp_latency.py --models llama3.1:8b,llama3.2:3b
    python exp_latency.py --cooldown 20        # pause between blocks
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
from common import API, RESULTS, ask, check_backend, progress, save  # noqa: E402

DEFAULT_MODELS = ["llama3.1:8b"]

# Twenty questions the corpus can answer, so every request exercises the full
# retrieve-and-generate path.
QUESTIONS = [
    "What does the Viterbi algorithm compute?",
    "What is Chomsky normal form?",
    "Which researchers is the CYK algorithm named after?",
    "What are the two Word2Vec architectures?",
    "What is negative sampling?",
    "What is hierarchical softmax?",
    "How is tf-idf calculated?",
    "What does GloVe learn from?",
    "What is masked language modelling in BERT?",
    "How does fastText represent words?",
    "Why does the transformer need positional encoding?",
    "What is multi-head attention?",
    "How is speech recognition accuracy measured?",
    "What is a hidden Markov model?",
    "What is part-of-speech tagging?",
    "What is named-entity recognition?",
    "What is an n-gram?",
    "When was GPT-4 released?",
    "What is retrieval-augmented generation?",
    "What is hallucination in large language models?",
]

# Above this ratio between the slowest and fastest block median, the pooled
# summary is not describing a steady state. Two-fold is generous; the drift
# actually observed in the ablation runs was thirteen-fold.
DRIFT_LIMIT = 2.0


# --------------------------------------------------------------------------
# Machine state
# --------------------------------------------------------------------------
def ollama_ps():
    """What Ollama currently holds in memory. Recorded, not acted on."""
    try:
        out = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=20
        )
        lines = [l for l in out.stdout.splitlines()[1:] if l.strip()]
        return "; ".join(l.split()[0] for l in lines) or "none"
    except Exception:
        return "unknown"


def installed_models():
    try:
        out = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=20
        )
        return {l.split()[0] for l in out.stdout.splitlines()[1:] if l.strip()}
    except Exception:
        return set()


def unload_models(wait_s=20):
    """
    Stop any running model so the next request genuinely pays cold start.

    Without this, "cold start" is measured against an already-resident model
    and the figure is meaningless.

    `ollama stop` returns before eviction has finished, so a residency check
    made immediately afterwards still sees the model and the unload is
    wrongly reported as failed. The first run of this script produced one
    reliable cold start and three marked unreliable for exactly that reason.
    Poll until Ollama reports nothing resident, or give up and say so.
    """
    try:
        out = subprocess.run(
            ["ollama", "ps"], capture_output=True, text=True, timeout=20
        )
        for line in out.stdout.splitlines()[1:]:
            parts = line.split()
            if parts:
                subprocess.run(["ollama", "stop", parts[0]], capture_output=True, timeout=30)
    except Exception:
        return False

    deadline = time.time() + wait_s
    while time.time() < deadline:
        state = ollama_ps()
        if state == "none":
            return True
        if state == "unknown":
            return False        # no CLI — cannot verify, so do not claim it
        time.sleep(1)
    return False


def set_model(model):
    """
    Point the backend at a different model.

    LLM_MODEL is read at import, so the backend must be restarted between
    models. The script cannot do that for you — it prompts, which is honest
    about what it controls.
    """
    health = requests.get(f"{API}/health", timeout=10).json()
    if health["llm"] == model:
        return True
    print(f"\n  Backend is running {health['llm']}, this block needs {model}.")
    print("  In the backend terminal: Ctrl+C, then")
    print(f"      $env:LLM_MODEL='{model}'; uvicorn main:app")
    input("  Press Enter once it has restarted... ")
    health = requests.get(f"{API}/health", timeout=10).json()
    if health["llm"] != model:
        print(f"  Backend still reports {health['llm']} — skipping {model}.")
        return False
    return True


# --------------------------------------------------------------------------
# Measurement
# --------------------------------------------------------------------------
def measure(question, k, t_start):
    """One request, with everything needed to explain the number it returns."""
    r = ask(question, use_rag=True, use_threshold=False, collection="sentence",
            top_k=k)
    chunks = r.get("retrieved_chunks") or []
    return {
        "question": question,
        "k_requested": k,
        "n_chunks": len(chunks),
        "context_chars": sum(len(c) for c in chunks),
        "retrieve_s": r.get("retrieve_s"),
        "generate_s": r.get("generate_s"),
        "total_s": r.get("total_s"),
        "wall_s": r.get("wall_s"),
        "elapsed_since_start_s": round(time.time() - t_start, 1),
        "error": r.get("error", ""),
    }


def run_model(model, questions, ks, repeats, warmup, cooldown):
    rows = []
    t_start = time.time()

    for k in ks:
        label = f"{model} k={k}"
        print(f"\n--- {label} ---")

        # ---- cold start: one measured request against an unloaded model ----
        cold_ok = unload_models()
        if not cold_ok:
            print("  (could not unload — cold-start figure is unreliable)")
        time.sleep(2)

        cold = measure(questions[0], k, t_start)
        cold.update(model=model, block=0, phase="cold",
                    cold_start_reliable=cold_ok, resident=ollama_ps())
        rows.append(cold)
        print(f"  cold start: {cold['total_s']}s"
              + ("" if cold_ok else "  [unreliable]"))

        # ---- warm-up: discarded, because it is neither cold nor steady ----
        for i in range(warmup):
            w = measure(questions[(i + 1) % len(questions)], k, t_start)
            w.update(model=model, block=0, phase="warmup",
                     cold_start_reliable="", resident=ollama_ps())
            rows.append(w)
        print(f"  warm-up: {warmup} requests discarded")

        # ---- steady state: repeated blocks over the full question set ----
        for block in range(1, repeats + 1):
            resident = ollama_ps()
            for i, q in enumerate(questions, 1):
                row = measure(q, k, t_start)
                row.update(model=model, block=block, phase="steady",
                           cold_start_reliable="", resident=resident)
                rows.append(row)
                progress(i, len(questions), f"block {block}  {row['total_s']}s")
            if cooldown and block < repeats:
                print(f"  cooling down {cooldown}s")
                time.sleep(cooldown)

    return rows


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------
def summarise(df, nfr):
    out = []
    steady = df[df.phase == "steady"]

    for (model, k), sub in steady.groupby(["model", "k_requested"], sort=False):
        tot = sub["total_s"].dropna()
        if tot.empty:
            continue

        block_medians = sub.groupby("block")["total_s"].median().round(2)
        drift = (block_medians.max() / block_medians.min()) if block_medians.min() else None

        cold_rows = df[(df.model == model) & (df.k_requested == k) & (df.phase == "cold")]
        cold_s = round(cold_rows["total_s"].iloc[0], 2) if len(cold_rows) else None

        out.append({
            "model": model,
            "k": k,
            "n_steady": len(tot),
            "blocks": len(block_medians),
            "cold_start_s": cold_s,
            "steady_median_s": round(tot.median(), 2),
            "steady_mean_s": round(tot.mean(), 2),
            "steady_p95_s": round(tot.quantile(0.95), 2),
            "steady_sd_s": round(statistics.stdev(tot), 2) if len(tot) > 1 else 0.0,
            "block_medians": " / ".join(str(v) for v in block_medians.tolist()),
            "drift_ratio": round(drift, 2) if drift else None,
            "stable": bool(drift and drift <= DRIFT_LIMIT),
            "mean_context_chars": int(sub["context_chars"].dropna().mean()),
            "retrieve_median_s": round(sub["retrieve_s"].dropna().median(), 3),
            "generate_median_s": round(sub["generate_s"].dropna().median(), 2),
            "meets_nfr": bool(tot.median() < nfr),
            "pct_under_nfr": round((tot < nfr).mean() * 100, 1),
        })
    return pd.DataFrame(out)


def _ols(xs, ys):
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx == 0:
        return None
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    intercept = my - slope * mx

    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    syy = sum((y - my) ** 2 for y in ys)
    r = sxy / (sxx * syy) ** 0.5 if syy else 0.0

    return {
        "n": n,
        "slope_s_per_1000_chars": round(slope * 1000, 3),
        "intercept_s": round(intercept, 2),
        "r_squared": round(r * r, 3),
    }


def context_fit(df):
    """
    Relate generation time to retrieved context length — the mechanism behind
    the k comparison. On CPU, prompt processing is roughly linear in context,
    so raising k should cost in proportion to the characters it adds.

    Two fits are reported, because they answer two different questions and
    only one of them is about k.

    **Per condition** (one point per k, using medians) isolates the effect of
    context size, because every other source of variation has been averaged
    out. This is the fit that supports a claim about the cost of k.

    **Per request** pools all measurements and will have a much weaker fit.
    That is expected and worth reporting rather than hiding: within a single
    k, context length is nearly constant, so the remaining scatter is driven
    by how long an answer the model chose to generate. Output length, not
    input length, dominates request-to-request variance. Quoting only the
    strong per-condition fit would overstate how predictable any individual
    request is.
    """
    steady = df[df.phase == "steady"].dropna(subset=["generate_s", "context_chars"])
    if len(steady) < 10 or steady["context_chars"].nunique() < 2:
        return None, None

    per_request = _ols(
        steady["context_chars"].astype(float).tolist(),
        steady["generate_s"].astype(float).tolist(),
    )

    grouped = steady.groupby("k_requested").agg(
        context=("context_chars", "median"), gen=("generate_s", "median")
    )
    per_condition = None
    if len(grouped) >= 3:
        per_condition = _ols(
            grouped["context"].astype(float).tolist(),
            grouped["gen"].astype(float).tolist(),
        )
    return per_condition, per_request


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--k", default="", help="comma-separated k values; default = shipped k")
    ap.add_argument("--n", type=int, default=20, help="questions per block")
    ap.add_argument("--repeats", type=int, default=3, help="steady-state blocks")
    ap.add_argument("--warmup", type=int, default=2, help="requests discarded after cold start")
    ap.add_argument("--cooldown", type=int, default=0, help="seconds between blocks")
    ap.add_argument("--nfr", type=float, default=10.0)
    args = ap.parse_args()

    health = check_backend()
    print(f"Decoding: temperature {health.get('temperature')}, seed {health.get('seed')}")

    ks = [int(x) for x in args.k.split(",") if x.strip()] or [health.get("top_k")]
    questions = QUESTIONS[: args.n]

    requested = [m.strip() for m in args.models.split(",") if m.strip()]
    have = installed_models()
    models = []
    for m in requested:
        if have and m not in have:
            print(f"  skipping {m} — not installed (ollama pull {m})")
            continue
        models.append(m)
    if not models:
        sys.exit("No requested model is installed.")

    total = len(models) * len(ks) * (1 + args.warmup + args.repeats * len(questions))
    print(f"\n{len(models)} model(s) x {len(ks)} k value(s) x "
          f"{args.repeats} blocks of {len(questions)} = {total} requests")

    rows = []
    for model in models:
        if not set_model(model):
            continue
        rows += run_model(model, questions, ks, args.repeats, args.warmup, args.cooldown)

    if not rows:
        sys.exit("No measurements taken.")

    df = pd.DataFrame(rows)

    # --repeats 0 re-measures cold start only. Steady state takes over an
    # hour; a failed unload does not justify repeating it.
    if args.repeats == 0:
        cold = df[df.phase == "cold"][
            ["model", "k_requested", "total_s", "cold_start_reliable", "resident"]
        ]
        save("latency_cold.csv", cold)
        print("\nCOLD START ONLY")
        print(cold.to_string(index=False))
        if not cold["cold_start_reliable"].all():
            print("\nSome unloads could not be verified. Those rows are not "
                  "cold starts and must not be reported as such.")
        return

    save("latency_raw.csv", df)

    s = summarise(df, args.nfr)
    save("latency_summary.csv", s)

    print("\n" + "=" * 76)
    print(f"LATENCY SUMMARY  (NFR target: median under {args.nfr}s)")
    print("=" * 76)
    print(s.to_string(index=False))

    per_condition, per_request = context_fit(df)
    if per_condition:
        print(f"\nGeneration time vs retrieved context — per condition "
              f"(n={per_condition['n']} k values, medians):")
        print(f"  {per_condition['slope_s_per_1000_chars']} s per 1000 characters, "
              f"intercept {per_condition['intercept_s']} s, "
              f"r2 = {per_condition['r_squared']}")
        print("  This is the cost of k, as a rate rather than an anecdote.")
    if per_request:
        print(f"\nSame fit per request (n={per_request['n']}): "
              f"r2 = {per_request['r_squared']}")
        print("  Much weaker, and that is the honest picture: within one k the")
        print("  context barely varies, so request-to-request scatter is driven by")
        print("  how long an answer the model generated, not by how much it read.")

    unstable = s[~s["stable"].astype(bool)]
    if len(unstable):
        print("\n" + "!" * 76)
        print("DRIFT DETECTED — the pooled figures above are not a steady state.")
        print("!" * 76)
        for _, r in unstable.iterrows():
            print(f"  {r['model']} k={r['k']}: block medians {r['block_medians']} "
                  f"(ratio {r['drift_ratio']})")
        print("\nThe machine did not return to the same state between blocks. Likely")
        print("causes, in order: the model was evicted and reloaded mid-run; thermal")
        print("throttling under sustained CPU load; another process competing for")
        print("memory. Re-run with --cooldown 30 and nothing else running before")
        print("quoting any of these numbers in Section 5.9.")
    else:
        print(f"\nAll conditions stable across blocks (drift ratio <= {DRIFT_LIMIT}).")

    print("\nReport cold start and steady state separately — they are different user")
    print("experiences. If the median misses the NFR, revise the NFR with these")
    print("figures and state the quality trade-off of any model you switch to.")


if __name__ == "__main__":
    main()
