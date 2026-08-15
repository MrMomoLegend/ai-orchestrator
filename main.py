"""
main.py — Locally-Deployed AI Orchestration System
FastAPI orchestration backend.

Day 2 (Wed 12 Aug): adds Whisper speech-to-text alongside the existing
text + RAG path. Everything still runs locally with no external API calls.

Run with:   uvicorn main:app --reload
Docs at:    http://127.0.0.1:8000/docs
"""

import os
import time
import tempfile
from typing import Any, Dict, List, Optional

import chromadb
import ollama
from chromadb.utils import embedding_functions
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
MODEL_NAME = os.getenv("LLM_MODEL", "llama3.1:8b")
EMBED_MODEL = "all-MiniLM-L6-v2"
DB_FOLDER = "chroma_db"
COLLECTION = "documents"
TOP_K = 3

# base.en is the starting point. Move to small.en only if Section 5.6 shows
# word error rate demands it AND Section 5.8 shows latency still permits it.
WHISPER_SIZE = os.getenv("WHISPER_SIZE", "base.en")

# beam_size is kept identical here and in evaluate_asr.py on purpose: the
# measured word error rate must describe the system that actually ships.
WHISPER_BEAM = 5

REFUSAL = "The provided documents do not contain this information."

# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
app = FastAPI(title="AI Orchestrator — Local RAG + Speech")

# CORS. Without this the React dev server cannot talk to this backend at all,
# and the browser error does not say so clearly. Vite serves on 5173.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------
# Retrieval
# --------------------------------------------------------------------------
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)
chroma_client = chromadb.PersistentClient(path=DB_FOLDER)
collection = chroma_client.get_collection(
    name=COLLECTION, embedding_function=embedding_fn
)

# --------------------------------------------------------------------------
# Whisper — loaded lazily
#
# Loading the model at import time would add ~10-20s to every server start,
# including every hot reload while developing. Loading on first use means the
# text-only path never pays for the speech model at all. The first spoken
# question is slower than the rest; that cold-start cost is reported
# separately in Section 5.8.
# --------------------------------------------------------------------------
_whisper_model = None


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        print(f"[whisper] loading {WHISPER_SIZE} (first use only)...")
        t0 = time.time()
        # int8 quantisation: roughly 2x faster on CPU with negligible WER cost,
        # which matters on a 16 GB laptop with no GPU headroom.
        _whisper_model = WhisperModel(
            WHISPER_SIZE, device="cpu", compute_type="int8"
        )
        print(f"[whisper] loaded in {time.time() - t0:.1f}s")
    return _whisper_model


def transcribe_path(path: str) -> Dict[str, Any]:
    """Transcribe an audio file that already exists on disk."""
    t0 = time.time()
    segments, info = get_whisper().transcribe(
        path, beam_size=WHISPER_BEAM, language="en"
    )
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return {
        "transcript": text,
        "duration_s": round(info.duration, 2),
        "transcribe_s": round(time.time() - t0, 2),
    }


async def save_upload(upload: UploadFile) -> str:
    """Write an uploaded file to a temp path and return that path."""
    suffix = os.path.splitext(upload.filename or "")[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(await upload.read())
    return path


# --------------------------------------------------------------------------
# Core orchestration — one function, used by every entry point
#
# Both the text endpoint and the voice endpoint call this. That is the whole
# point of the design: speech is a front door onto an unchanged pipeline, not
# a second pipeline. It also means Section 5.2's RAG comparison measures the
# same code path a user actually exercises.
# --------------------------------------------------------------------------
def answer_question(question: str, use_rag: bool = True) -> Dict[str, Any]:
    t0 = time.time()
    retrieved: List[str] = []
    sources: List[str] = []
    distances: List[float] = []

    if use_rag:
        results = collection.query(query_texts=[question], n_results=TOP_K)
        retrieved = results["documents"][0]
        sources = [m.get("source", "unknown") for m in results["metadatas"][0]]
        distances = [round(float(d), 4) for d in results["distances"][0]]

        # DAY 4 (Fri 14 Aug): the retrieval-confidence threshold goes here.
        # Refuse before calling the LLM when distances[0] exceeds the tuned
        # threshold, so that refusal becomes a property of retrieval rather
        # than of the prompt. Keep this behaviour behind a flag so the
        # ablation in Section 5.3 can run both ways.

        context = "\n\n".join(retrieved)
        prompt = (
            "Answer the question using ONLY the context below. If the answer "
            f"is not in the context, say '{REFUSAL}'"
            f"\n\nContext:\n{context}\n\nQuestion: {question}"
        )
    else:
        prompt = question

    response = ollama.chat(
        model=MODEL_NAME, messages=[{"role": "user", "content": prompt}]
    )

    return {
        "question": question,
        "use_rag": use_rag,
        "answer": response["message"]["content"],
        "sources": sources,
        "retrieved_chunks": retrieved,
        "distances": distances,
        "generate_s": round(time.time() - t0, 2),
    }


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
class Query(BaseModel):
    question: str
    use_rag: bool = True


@app.get("/health")
def health():
    """Cheap check that the backend, the corpus, and Ollama are all alive."""
    return {
        "status": "ok",
        "llm": MODEL_NAME,
        "whisper": WHISPER_SIZE,
        "whisper_loaded": _whisper_model is not None,
        "chunks_in_corpus": collection.count(),
    }


@app.post("/ask")
def ask(query: Query):
    """Text question in, grounded answer out. Unchanged from the prototype."""
    if not query.question.strip():
        raise HTTPException(400, "Question is empty.")
    return answer_question(query.question, query.use_rag)


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    """Audio in, transcript out. Speech only — no retrieval, no generation."""
    path = await save_upload(audio)
    try:
        return transcribe_path(path)
    finally:
        os.remove(path)


@app.post("/ask/audio")
async def ask_audio(
    audio: UploadFile = File(...),
    use_rag: bool = Form(True),
):
    """
    Spoken question in, grounded answer out.

    The transcript is returned alongside the answer. This is a deliberate
    transparency feature, not debug output: when the assistant answers a
    question the user did not ask, they can see immediately that the fault
    was transcription rather than retrieval. A voice interface that hides
    what it heard gives the user no way to tell those two failures apart.
    """
    path = await save_upload(audio)
    try:
        asr = transcribe_path(path)
    finally:
        os.remove(path)

    if not asr["transcript"]:
        raise HTTPException(422, "No speech detected in the audio.")

    result = answer_question(asr["transcript"], use_rag)
    result["transcript"] = asr["transcript"]
    result["audio_duration_s"] = asr["duration_s"]
    result["transcribe_s"] = asr["transcribe_s"]
    return result