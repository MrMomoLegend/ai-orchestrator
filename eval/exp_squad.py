"""
exp_squad.py — Section 5.9, SQuAD 2.0 Exact Match and F1.

SQuAD 2.0 is the citable benchmark. Its unanswerable subset is the part that
speaks to this project's thesis: those questions have a plausible-looking
passage attached and no answer in it, which is exactly the trap that produces
hallucination.

How this works
--------------
Each SQuAD question comes with its own context paragraph, so the paragraph
is ingested as a temporary single-document corpus and the question is asked
against it. That measures the full retrieve-and-generate pipeline on
standard data, rather than measuring the language model alone.

EM and F1 use the official SQuAD normalisation (lower-case, strip articles,
strip punctuation), so the numbers are comparable to published results.

Usage
-----
    python exp_squad.py --download        # fetch dev-v2.0.json (~4 MB)
    python exp_squad.py --n 200           # run 200 questions (~50 min)
    python exp_squad.py --n 40            # quick sanity run first
"""

import argparse
import json
import re
import string
import sys
import urllib.request
from collections import Counter
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import RESULTS, ask, check_backend, is_refusal, progress, save

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SQUAD_FILE = DATA / "dev-v2.0.json"
SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"


# ---- official SQuAD scoring -------------------------------------------
def normalise(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def exact_match(pred, golds):
    return float(any(normalise(pred) == normalise(g) for g in golds))


def f1(pred, golds):
    best = 0.0
    p_tokens = normalise(pred).split()
    for g in golds:
        g_tokens = normalise(g).split()
        if not p_tokens or not g_tokens:
            best = max(best, float(p_tokens == g_tokens))
            continue
        common = Counter(p_tokens) & Counter(g_tokens)
        same = sum(common.values())
        if same == 0:
            continue
        prec = same / len(p_tokens)
        rec = same / len(g_tokens)
        best = max(best, 2 * prec * rec / (prec + rec))
    return best


def download():
    DATA.mkdir(exist_ok=True)
    if SQUAD_FILE.exists():
        print(f"Already have {SQUAD_FILE}")
        return
    print(f"Downloading SQuAD 2.0 dev set ...")
    urllib.request.urlretrieve(SQUAD_URL, SQUAD_FILE)
    print(f"  -> {SQUAD_FILE} ({SQUAD_FILE.stat().st_size / 1e6:.1f} MB)")


def load_sample(n, seed=42):
    """Balanced sample: half answerable, half unanswerable."""
    if not SQUAD_FILE.exists():
        sys.exit("ERROR: dev-v2.0.json not found. Run with --download first.")

    data = json.loads(SQUAD_FILE.read_text())
    answerable, unanswerable = [], []
    for article in data["data"]:
        for para in article["paragraphs"]:
            for qa in para["qas"]:
                item = {
                    "qid": qa["id"],
                    "question": qa["question"],
                    "context": para["context"],
                    "is_impossible": qa.get("is_impossible", False),
                    "answers": [a["text"] for a in qa.get("answers", [])],
                }
                (unanswerable if item["is_impossible"] else answerable).append(item)

    import random
    random.seed(seed)
    half = n // 2
    sample = random.sample(answerable, min(half, len(answerable)))
    sample += random.sample(unanswerable, min(n - half, len(unanswerable)))
    random.shuffle(sample)
    print(f"Sampled {len(sample)} questions "
          f"({sum(not s['is_impossible'] for s in sample)} answerable, "
          f"{sum(s['is_impossible'] for s in sample)} unanswerable)")
    return sample


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    if args.download:
        download()
        return

    check_backend(require_chunks=False)
    sample = load_sample(args.n, args.seed)

    import chromadb
    import ollama
    from chromadb.utils import embedding_functions
    sys.path.insert(0, str(HERE.parent))
    from main import (DB_FOLDER, DISTANCE_THRESHOLD, EMBED_MODEL, MODEL_NAME,
                      REFUSAL, TOP_K, USE_THRESHOLD, chunk_sentence)

    client = chromadb.PersistentClient(path=DB_FOLDER)
    embed = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
    TEMP = "squad_temp"

    rows = []
    for i, item in enumerate(sample, 1):
        # Fresh single-paragraph corpus for each question.
        try:
            client.delete_collection(TEMP)
        except Exception:
            pass
        coll = client.create_collection(
            name=TEMP, embedding_function=embed, metadata={"hnsw:space": "cosine"}
        )
        chunks = chunk_sentence(item["context"]) or [item["context"]]
        coll.add(
            documents=chunks,
            metadatas=[{"source": "squad"} for _ in chunks],
            ids=[f"s{j}" for j in range(len(chunks))],
        )

        # The temporary collection is not one of the two named collections the
        # API exposes, so this reproduces answer_question()'s logic against it
        # directly — same prompt, same top-k, same threshold as the live system.
        res = coll.query(query_texts=[item["question"]], n_results=TOP_K)
        chunks = res["documents"][0]
        dists = [float(d) for d in res["distances"][0]]

        if USE_THRESHOLD and (not dists or dists[0] > DISTANCE_THRESHOLD):
            pred, refused, reason = REFUSAL, True, "retrieval_distance"
        else:
            prompt = (
                "Answer the question using ONLY the context below. If the answer "
                f"is not in the context, say '{REFUSAL}'"
                f"\n\nContext:\n{chr(10).join(chunks)}\n\nQuestion: {item['question']}"
            )
            out = ollama.chat(
                model=MODEL_NAME, messages=[{"role": "user", "content": prompt}]
            )
            pred = out["message"]["content"].strip()
            refused = is_refusal({"answer": pred})
            reason = "model_judgement" if refused else None

        if item["is_impossible"]:
            em = float(refused)
            f = float(refused)
        else:
            em = 0.0 if refused else exact_match(pred, item["answers"])
            f = 0.0 if refused else f1(pred, item["answers"])

        rows.append({
            "qid": item["qid"],
            "is_impossible": item["is_impossible"],
            "question": item["question"],
            "gold": " | ".join(item["answers"]),
            "prediction": pred.replace("\n", " ")[:300],
            "refused": refused,
            "em": em,
            "f1": round(f, 4),
        })
        progress(i, len(sample), item["question"][:34])

    try:
        client.delete_collection(TEMP)
    except Exception:
        pass

    df = pd.DataFrame(rows)
    save("squad_responses.csv", df)

    summary = []
    for label, sub in (("overall", df),
                       ("answerable", df[~df.is_impossible]),
                       ("unanswerable", df[df.is_impossible])):
        if sub.empty:
            continue
        summary.append({
            "subset": label,
            "n": len(sub),
            "exact_match": round(sub["em"].mean() * 100, 2),
            "f1": round(sub["f1"].mean() * 100, 2),
            "refusal_rate": round(sub["refused"].mean() * 100, 2),
        })
    s = pd.DataFrame(summary)
    save("squad_summary.csv", s)

    print("\n" + "=" * 52)
    print("SQuAD 2.0 RESULTS")
    print("=" * 52)
    print(s.to_string(index=False))
    print("\nReport the unanswerable subset separately — it is the part that")
    print("speaks to the thesis. Note in Section 5.9 that SQuAD paragraphs are")
    print("short and clean relative to real course PDFs, so these figures are")
    print("an upper bound on performance over the project's own corpus.")


if __name__ == "__main__":
    main()
