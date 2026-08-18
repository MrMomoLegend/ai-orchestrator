"""
common.py — shared helpers for every Chapter 5 experiment.

One HTTP client, one refusal detector, one results writer, so that every
experiment measures the same running system through the same door. Nothing
here reimplements the pipeline; the scripts drive the live API exactly as a
user would, which is what makes the numbers describe the deployed system
rather than a test harness that resembles it.
"""

import json
import sys
import time
from pathlib import Path

import requests

API = "http://127.0.0.1:8000"
RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

REFUSAL_MARKERS = (
    "do not contain",
    "does not contain",
    "doesn't contain",
    "not in the context",
    "no information",
    "cannot answer",
    "can't answer",
    "unable to answer",
    "not provided in the context",
    "insufficient",
    "i don't know",
    "i do not know",
)


# Below this many chunks, retrieval distances are dominated by how little
# there is to match against rather than by relevance, and any threshold
# tuned on them is fitting noise. A realistic course-notes corpus produces
# hundreds of chunks; single digits means ingestion did not pick up the
# documents you think it did.
MIN_USABLE_CHUNKS = 30


def check_backend(require_chunks=True):
    """Fail early and legibly rather than 30 questions into a run."""
    try:
        health = requests.get(f"{API}/health", timeout=10).json()
    except Exception:
        sys.exit(
            "ERROR: the backend is not responding at "
            f"{API}\nStart it with:  uvicorn main:app --reload"
        )

    print(f"Backend OK — {health['llm']}, collections: {health.get('collections')}")

    if not require_chunks:
        return health

    n = health.get("chunks_in_corpus", 0)
    if not n:
        sys.exit("ERROR: the corpus is empty. Run:  python ingest.py --reset")

    if n < MIN_USABLE_CHUNKS:
        print()
        print("=" * 68)
        print(f"WARNING: only {n} chunks in the corpus.")
        print("=" * 68)
        print("That is roughly a page of text. Retrieval distances measured")
        print("against a corpus this small reflect how little there is to match")
        print("against, not relevance, and a threshold tuned on them will not")
        print("hold once real documents are loaded.")
        print()
        print("Check what actually got ingested:")
        print(f"    {API}/documents")
        print("    ls docs/")
        print()
        print("Then re-ingest your real course materials:")
        print("    python ingest.py --reset")
        print()
        try:
            docs = requests.get(f"{API}/documents", timeout=10).json()
            print(f"Currently ingested: {docs.get('documents')}")
        except Exception:
            pass
        print()
        if input("Continue anyway? [y/N] ").strip().lower() != "y":
            sys.exit("Stopped. Fix the corpus first.")

    return health


def ask(question, **kwargs):
    """
    POST /ask with any experiment overrides.

    Accepts use_rag, top_k, use_threshold, threshold, collection.
    Retries once, because a cold Ollama model load can exceed the timeout on
    the first request of a run and that is not a failure worth losing.
    """
    payload = {"question": question}
    payload.update({k: v for k, v in kwargs.items() if v is not None})

    for attempt in (1, 2):
        try:
            t0 = time.time()
            r = requests.post(f"{API}/ask", json=payload, timeout=300)
            r.raise_for_status()
            data = r.json()
            data["wall_s"] = round(time.time() - t0, 3)
            return data
        except Exception as exc:
            if attempt == 2:
                return {
                    "question": question,
                    "answer": "",
                    "error": str(exc),
                    "refused": False,
                    "wall_s": None,
                }
            time.sleep(3)


def is_refusal(result):
    """
    Did the system decline to answer?

    Two ways this can happen, and the distinction is the whole point of
    Section 5.3. An explicit refusal from the retrieval threshold is a
    control-flow branch and is reported by the API. A refusal from the
    language model's own judgement has to be read out of the text.
    """
    if result.get("refused"):
        return True
    answer = (result.get("answer") or "").lower()
    return any(m in answer for m in REFUSAL_MARKERS)


def save(name, rows, index=False):
    """Write a results CSV and say where it went."""
    import pandas as pd

    path = RESULTS / name
    pd.DataFrame(rows).to_csv(path, index=index)
    print(f"  -> {path}")
    return path


def save_json(name, obj):
    path = RESULTS / name
    path.write_text(json.dumps(obj, indent=2))
    print(f"  -> {path}")
    return path


def progress(i, total, label=""):
    bar = "#" * int(24 * i / total)
    sys.stdout.write(f"\r  [{bar:<24}] {i}/{total} {label[:40]:<40}")
    sys.stdout.flush()
    if i == total:
        print()
