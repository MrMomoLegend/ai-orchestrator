from fastapi import FastAPI
from pydantic import BaseModel
import ollama
import chromadb
from chromadb.utils import embedding_functions

MODEL_NAME = "llama3.1:8b"   # change to "llama3.2:3b" if you used the smaller one
DB_FOLDER = "chroma_db"
TOP_K = 3   # how many document chunks to retrieve per question

app = FastAPI(title="AI Orchestrator - Prototype with RAG")

# Connect to the ChromaDB we built with ingest.py
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
client = chromadb.PersistentClient(path=DB_FOLDER)
collection = client.get_collection(name="documents", embedding_function=embedding_fn)


class Query(BaseModel):
    question: str
    use_rag: bool = True   # lets us turn RAG on/off for the evaluation


@app.get("/")
def health_check():
    return {"status": "running", "model": MODEL_NAME}


@app.post("/ask")
def ask(query: Query):
    if query.use_rag:
        # 1. Retrieve the most relevant chunks for this question
        results = collection.query(
            query_texts=[query.question],
            n_results=TOP_K,
        )
        retrieved_chunks = results["documents"][0]
        sources = [m["source"] for m in results["metadatas"][0]]

        # 2. Build a prompt that tells Llama to answer ONLY from the context
        context = "\n\n".join(retrieved_chunks)
        prompt = (
            "Answer the question using ONLY the context below. "
            "If the answer is not in the context, say "
            "'The provided documents do not contain this information.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query.question}"
        )
    else:
        # RAG off: ask the model directly, no documents
        prompt = query.question
        retrieved_chunks = []
        sources = []

    # 3. Send to Llama 3
    response = ollama.chat(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response["message"]["content"]

    return {
        "question": query.question,
        "use_rag": query.use_rag,
        "answer": answer,
        "sources": sources,            # which files the context came from
        "retrieved_chunks": retrieved_chunks,  # the actual text used
    }