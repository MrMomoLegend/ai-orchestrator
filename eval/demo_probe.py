"""
demo_probe.py - try candidate demo questions against the NEW public corpus,
RAG off vs RAG on, before filming. Needs the backend running (uvicorn main:app).

    python eval/demo_probe.py

Prints each pair and writes results/demo_probe.csv. Pick the shots from this,
not from the Chapter 5 results, which were measured on the old corpus.
"""
import csv, json, urllib.request
from pathlib import Path

API = "http://127.0.0.1:8000"
QUESTIONS = [
    "How many parameters does GPT-4 have?",
    "Who first described the dynamic programming parser behind the CYK algorithm?",
    "How can word vectors be used to solve analogies such as man is to woman as king is to queen?",
    "What is retrieval-augmented generation and why does it reduce hallucination?",
    "What does the Viterbi algorithm compute in a hidden Markov model?",
    "How is TF-IDF weighting calculated?",
    "What is the capital of Brazil?",
    "Who won the 2018 FIFA World Cup?",
    "Which Python library should I use to implement a CYK parser?",
]

def post(path, body):
    req = urllib.request.Request(API + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        return json.load(r)

def main():
    with urllib.request.urlopen(API + "/health", timeout=30) as r:
        h = json.load(r)
    print(f"health: chunks={h['collections']} temp={h['temperature']} seed={h['seed']}\n")
    if not h["chunks_in_corpus"]:
        raise SystemExit("Corpus is empty - run: python ingest.py --reset")
    post("/ask", {"question": "warm up", "use_rag": False})   # absorb cold start
    rows = []
    for q in QUESTIONS:
        off = post("/ask", {"question": q, "use_rag": False})
        on = post("/ask", {"question": q, "use_rag": True})
        d = min(on["distances"]) if on.get("distances") else None
        print("=" * 90 + f"\nQ: {q}\n--- RAG OFF ({off['total_s']}s):\n{off['answer'][:600]}")
        print(f"--- RAG ON ({on['total_s']}s, refused={on['refused']} "
              f"reason={on['refusal_reason']} nearest={d} sources={on['sources']}):\n{on['answer'][:600]}\n")
        rows.append({"question": q, "rag_off": off["answer"], "rag_on": on["answer"],
                     "refused": on["refused"], "reason": on["refusal_reason"],
                     "nearest_dist": d, "sources": ";".join(map(str, on["sources"])),
                     "on_total_s": on["total_s"]})
    out = Path("results/demo_probe.csv")
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"saved {out}")

if __name__ == "__main__":
    main()
