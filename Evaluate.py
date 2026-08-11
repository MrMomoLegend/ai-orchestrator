import csv
import requests

API = "http://127.0.0.1:8000/ask"

# 10 questions tagged by category, matched to the actual document corpus:
#   AI.txt, University of London Final Project.txt, Machine Learning.txt, Space Exploration.txt
test_queries = [
    # Category A — answerable directly from your documents (4)
    ("A", "What percentage is the Final Report worth?"),
    ("A", "Which architecture did Google introduce in 2017?"),
    ("A", "What metrics are used to evaluate machine learning model performance?"),
    ("A", "Who was the first person to walk on the Moon?"),

    # Category B — clearly NOT in your documents (4) — tests hallucination resistance
    ("B", "What is the capital of Brazil?"),
    ("B", "Who wrote the play Hamlet?"),
    ("B", "What is the boiling point of water in Fahrenheit?"),
    ("B", "Who is the current president of France?"),

    # Category C — same topics as your docs, but the specific answer is NOT stated (2)
    ("C", "What programming language is recommended for the CM3070 project?"),
    ("C", "How many layers does a typical neural network have?"),
]

rows = []
for category, question in test_queries:
    for use_rag in (True, False):
        try:
            resp = requests.post(API, json={"question": question, "use_rag": use_rag})
            data = resp.json()
            answer = data.get("answer", "").replace("\n", " ").strip()
            sources = "; ".join(data.get("sources", []))
        except Exception as e:
            answer = f"ERROR: {e}"
            sources = ""
        rows.append({
            "category": category,
            "question": question,
            "rag": "on" if use_rag else "off",
            "answer": answer,
            "sources": sources,
        })
        print(f"[{category}] rag={'on' if use_rag else 'off'}: {question[:50]}")

with open("evaluation_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["category", "question", "rag", "answer", "sources"])
    writer.writeheader()
    writer.writerows(rows)

print("\nDone. Open evaluation_results.csv in Excel and score each row.")