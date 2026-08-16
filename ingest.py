"""
ingest.py — build the vector store from a folder of documents.

Creates TWO collections from the same source files:

    documents_sentence   sentence-aware chunking   (the shipped default)
    documents_fixed      fixed-size chunking       (the prototype's strategy)

Both are needed for the chunking ablation in Section 5.4. Building them from
one pass over the same files guarantees the only difference between them is
the chunking.

Both collections use COSINE distance, set explicitly. This matters: Chroma
defaults to squared L2, and a threshold value tuned against L2 distances is
not comparable to one tuned against cosine distances. The distance you see
in the API response is the distance the threshold in main.py compares
against, and it is cosine in the range 0-2.

Usage
-----
    python ingest.py                  # ingest ./docs
    python ingest.py --docs mydocs    # ingest a different folder
    python ingest.py --reset          # wipe and rebuild from scratch
"""

import argparse
import os
import sys
import uuid
from pathlib import Path

import chromadb
from chromadb.utils import embedding_functions

sys.path.insert(0, str(Path(__file__).resolve().parent))
from main import CHUNKERS, COLLECTIONS, DB_FOLDER, EMBED_MODEL, extract_text  # noqa: E402

SUPPORTED = {".txt", ".md", ".markdown", ".pdf"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs", default="docs", help="folder of source documents")
    ap.add_argument("--reset", action="store_true", help="delete collections first")
    args = ap.parse_args()

    docs_dir = Path(args.docs)
    if not docs_dir.is_dir():
        sys.exit(f"ERROR: '{docs_dir}' is not a folder. Put your source documents there.")

    files = sorted(p for p in docs_dir.iterdir() if p.suffix.lower() in SUPPORTED)
    if not files:
        sys.exit(f"ERROR: no .txt, .md or .pdf files found in '{docs_dir}'.")

    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )
    client = chromadb.PersistentClient(path=DB_FOLDER)

    print(f"Ingesting {len(files)} file(s) from {docs_dir}/\n")

    for key, name in COLLECTIONS.items():
        if args.reset:
            try:
                client.delete_collection(name)
                print(f"[{key}] deleted existing collection")
            except Exception:
                pass

        collection = client.get_or_create_collection(
            name=name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},   # <- not the default; see docstring
        )

        chunker = CHUNKERS[key]
        total = 0
        for path in files:
            text = extract_text(str(path), path.name)
            chunks = chunker(text)
            if not chunks:
                print(f"  [{key}] {path.name}: no text extracted, skipped")
                continue
            batch = uuid.uuid4().hex[:8]
            collection.add(
                documents=chunks,
                metadatas=[{"source": path.name} for _ in chunks],
                ids=[f"{batch}-{i}" for i in range(len(chunks))],
            )
            total += len(chunks)
            print(f"  [{key}] {path.name}: {len(chunks)} chunks")

        print(f"[{key}] collection '{name}' now holds {collection.count()} chunks\n")

    print("Done. Both collections built with cosine distance.")
    print("Sanity check:  curl http://127.0.0.1:8000/health")


if __name__ == "__main__":
    main()