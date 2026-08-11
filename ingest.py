import os
import chromadb
from chromadb.utils import embedding_functions

# Folder containing your .txt documents
DOCS_FOLDER = "documents"

# Where ChromaDB saves its data on disk (persists between runs)
DB_FOLDER = "chroma_db"

# The embedding model (downloaded automatically the first time)
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Connect to a persistent ChromaDB stored in DB_FOLDER
client = chromadb.PersistentClient(path=DB_FOLDER)

# Create (or reset) a collection to hold our document chunks
# Delete any existing collection so re-running gives a clean state
try:
    client.delete_collection("documents")
except Exception:
    pass
collection = client.create_collection(
    name="documents",
    embedding_function=embedding_fn,
)


def chunk_text(text, chunk_size=500, overlap=50):
    # Split text into overlapping chunks of roughly chunk_size characters.
    # Overlap keeps sentences from being cut awkwardly between chunks.
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# Read every .txt file, chunk it, and collect everything
all_chunks = []
all_ids = []
all_sources = []

for filename in os.listdir(DOCS_FOLDER):
    if not filename.endswith(".txt"):
        continue
    path = os.path.join(DOCS_FOLDER, filename)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chunks = chunk_text(text)
    for i, chunk in enumerate(chunks):
        all_chunks.append(chunk)
        all_ids.append(f"{filename}-{i}")
        all_sources.append(filename)

if not all_chunks:
    print("No .txt files found in the 'documents' folder. Add some and re-run.")
else:
    # Store everything in ChromaDB
    collection.add(
        documents=all_chunks,
        ids=all_ids,
        metadatas=[{"source": s} for s in all_sources],
    )
    print(f"Ingested {len(all_chunks)} chunks from {len(set(all_sources))} file(s).")