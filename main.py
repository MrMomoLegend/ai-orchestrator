"""
main.py — Locally-Deployed AI Orchestration System
FastAPI orchestration backend.

Day 4 (Fri 14 Aug): the two limitations identified in the preliminary
report are fixed, and both are switchable per request so the ablations in
Sections 5.3 and 5.4 measure one code path rather than two forks.

  1. Refusal now triggers on retrieval confidence, before the LLM is called,
     rather than relying on the model to notice the context is insufficient.
  2. Chunking is sentence-aware. The old fixed-size strategy is retained as
     a second collection so the difference can be measured.

Run with:   uvicorn main:app --reload
Docs at:    http://127.0.0.1:8000/docs
"""

import os
import re
import time
import tempfile
import uuid
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
TOP_K = 3

WHISPER_SIZE = os.getenv("WHISPER_SIZE", "base.en")
WHISPER_BEAM = 5

# Two collections, same documents, different chunking. Section 5.4 compares
# them. "sentence" is the shipped default; "fixed" is the prototype's
# behaviour, kept so the improvement can be measured rather than asserted.
COLLECTIONS = {"sentence": "documents_sentence", "fixed": "documents_fixed"}
DEFAULT_COLLECTION = os.getenv("COLLECTION", "sentence")

# Retrieval-confidence threshold.
#
# Distances are COSINE (set explicitly in ingest.py), so the range is 0-2:
# 0 is identical, 1 is orthogonal. If the nearest chunk is further than this,
# no chunk is relevant enough to answer from and the system refuses without
# calling the LLM at all. Tune with:  python eval/exp_rag.py --sweep
DISTANCE_THRESHOLD = float(os.getenv("DISTANCE_THRESHOLD", "0.6"))
USE_THRESHOLD = os.getenv("USE_THRESHOLD", "1") not in ("0", "false", "False")

REFUSAL = "The provided documents do not contain this information."

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --------------------------------------------------------------------------
# App
# --------------------------------------------------------------------------
app = FastAPI(title="AI Orchestrator — Local RAG + Speech")

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


def get_collection(key: str = None):
    """Fetch one of the two chunking collections by key."""
    key = key or DEFAULT_COLLECTION
    name = COLLECTIONS.get(key)
    if name is None:
        raise HTTPException(400, f"Unknown collection '{key}'. Use: {list(COLLECTIONS)}")
    try:
        return chroma_client.get_collection(name=name, embedding_function=embedding_fn)
    except Exception:
        # Created on demand so a fresh clone works before ingest.py is run.
        return chroma_client.get_or_create_collection(
            name=name,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )


# --------------------------------------------------------------------------
# Whisper — loaded lazily
# --------------------------------------------------------------------------
_whisper_model = None


def get_whisper():
    global _whisper_model
    if _whisper_model is None:
        from faster_whisper import WhisperModel

        print(f"[whisper] loading {WHISPER_SIZE} (first use only)...")
        t0 = time.time()
        _whisper_model = WhisperModel(WHISPER_SIZE, device="cpu", compute_type="int8")
        print(f"[whisper] loaded in {time.time() - t0:.1f}s")
    return _whisper_model


def transcribe_path(path: str) -> Dict[str, Any]:
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
    suffix = os.path.splitext(upload.filename or "")[1] or ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "wb") as f:
        f.write(await upload.read())
    return path


# --------------------------------------------------------------------------
# Chunking — the Section 5.4 ablation
# --------------------------------------------------------------------------
def chunk_fixed(text: str) -> List[str]:
    """
    The prototype's strategy: fixed-size character windows with overlap.

    This is what split a sentence mid-clause at
    ", while the written examination contributes 20%."
    Retained so Section 5.4 has a real baseline rather than a remembered one.
    """
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [
        c for c in (text[i : i + CHUNK_SIZE] for i in range(0, len(text), step))
        if c.strip()
    ]


def chunk_sentence(text: str) -> List[str]:
    """
    Sentence-aware splitting. Prefers to break at a paragraph boundary, then
    a line break, then a sentence end, and only falls back to a word boundary
    when a single sentence exceeds the chunk size.

    Uses LangChain's RecursiveCharacterTextSplitter when available, since
    that is the implementation the report cites. The pure-Python fallback
    below follows the same separator hierarchy so the system still runs
    without the dependency.
    """
    text = text.strip()
    if not text:
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
        except ImportError:
            RecursiveCharacterTextSplitter = None

    if RecursiveCharacterTextSplitter is not None:
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return [c for c in splitter.split_text(text) if c.strip()]

    if not _FALLBACK_WARNED:
        print("[chunking] langchain-text-splitters not installed; using the "
              "built-in fallback. It follows the same separator hierarchy but "
              "does NOT add overlap. Install it before running the Section 5.4 "
              "ablation:  pip install langchain-text-splitters")
        globals()["_FALLBACK_WARNED"] = True
    return _recursive_split(text, ["\n\n", "\n", ". ", " "])


_FALLBACK_WARNED = False


def _split_keeping_punctuation(text: str, sep: str) -> List[str]:
    """
    Split on a separator, leaving sentence-ending punctuation attached to the
    sentence it belongs to. Splitting naively on ". " throws the full stop
    away, which is precisely the kind of mid-clause damage this chunker
    exists to avoid.
    """
    parts = text.split(sep)
    if len(parts) == 1:
        return parts
    tail = sep.rstrip()          # "." from ". ", "" from "\n\n" and " "
    return [p + tail if i < len(parts) - 1 else p for i, p in enumerate(parts)]


def _recursive_split(text: str, seps: List[str]) -> List[str]:
    """Fallback with the same separator hierarchy as LangChain's splitter."""
    text = text.strip()
    if len(text) <= CHUNK_SIZE:
        return [text] if text else []
    if not seps:
        return [text[i : i + CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]

    sep, rest = seps[0], seps[1:]
    pieces = [p for p in _split_keeping_punctuation(text, sep) if p.strip()]
    if len(pieces) <= 1:
        return _recursive_split(text, rest)

    chunks, buf = [], ""
    for piece in pieces:
        candidate = f"{buf} {piece}".strip() if buf else piece
        if len(candidate) <= CHUNK_SIZE:
            buf = candidate
            continue
        if buf:
            chunks.append(buf)
            buf = ""
        if len(piece) > CHUNK_SIZE:
            chunks.extend(_recursive_split(piece, rest))
        else:
            buf = piece
    if buf:
        chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]


CHUNKERS = {"sentence": chunk_sentence, "fixed": chunk_fixed}


# --------------------------------------------------------------------------
# Document text extraction
# --------------------------------------------------------------------------
TEXT_EXTS = {".txt", ".md", ".markdown"}


def extract_text(path: str, filename: str) -> str:
    ext = os.path.splitext(filename)[1].lower()

    if ext in TEXT_EXTS:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    if ext == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError:
            raise HTTPException(500, "PDF support needs pypdf. Run: pip install pypdf")
        pages = [(p.extract_text() or "") for p in PdfReader(path).pages]
        text = "\n\n".join(pages).strip()
        if not text:
            raise HTTPException(
                422,
                "No text found in that PDF. It is probably a scan of images "
                "rather than a text document.",
            )
        return text

    raise HTTPException(415, f"Cannot read '{ext}' files. Supported: .txt, .md, .pdf")


# --------------------------------------------------------------------------
# Core orchestration
# --------------------------------------------------------------------------
def looks_like_refusal(answer: str) -> bool:
    """
    Did the language model decline to answer?

    Needed for scoring the threshold-off condition in Section 5.3, where
    refusal is the model's judgement rather than an explicit control-flow
    branch and therefore has to be detected from the text.
    """
    a = (answer or "").lower()
    return any(
        p in a
        for p in (
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
        )
    )


def answer_question(
    question: str,
    use_rag: bool = True,
    top_k: int = None,
    use_threshold: bool = None,
    threshold: float = None,
    collection: str = None,
) -> Dict[str, Any]:
    """
    One function, every entry point. Text and voice share it, and every
    experiment in Chapter 5 exercises it with different parameters rather
    than a different implementation.
    """
    t0 = time.time()
    top_k = TOP_K if top_k is None else top_k
    use_threshold = USE_THRESHOLD if use_threshold is None else use_threshold
    threshold = DISTANCE_THRESHOLD if threshold is None else threshold

    retrieved: List[str] = []
    sources: List[str] = []
    distances: List[float] = []
    retrieve_s = 0.0

    if not use_rag:
        prompt = question
    else:
        tr = time.time()
        coll = get_collection(collection)
        results = coll.query(query_texts=[question], n_results=top_k)
        retrieve_s = round(time.time() - tr, 3)

        retrieved = results["documents"][0]
        sources = [m.get("source", "unknown") for m in results["metadatas"][0]]
        distances = [round(float(d), 4) for d in results["distances"][0]]

        # ---- THE FIX (Section 4.5) -----------------------------------
        # Refusal is now a property of retrieval, not of the prompt. If the
        # nearest chunk is further away than the threshold, nothing in the
        # corpus is relevant and the language model is never consulted. The
        # preliminary report identified prompt-driven refusal as fragile
        # precisely because retrieval can return weakly-related chunks and
        # the model may then answer from them.
        if use_threshold and (not distances or distances[0] > threshold):
            return {
                "question": question,
                "use_rag": True,
                "answer": REFUSAL,
                "refused": True,
                "refusal_reason": "retrieval_distance",
                "sources": [],
                "retrieved_chunks": retrieved,
                "distances": distances,
                "top_k": top_k,
                "threshold": threshold,
                "collection": collection or DEFAULT_COLLECTION,
                "retrieve_s": retrieve_s,
                "generate_s": 0.0,
                "total_s": round(time.time() - t0, 3),
            }

        context = "\n\n".join(retrieved)
        prompt = (
            "Answer the question using ONLY the context below. If the answer "
            f"is not in the context, say '{REFUSAL}'"
            f"\n\nContext:\n{context}\n\nQuestion: {question}"
        )

    tg = time.time()
    response = ollama.chat(
        model=MODEL_NAME, messages=[{"role": "user", "content": prompt}]
    )
    answer = response["message"]["content"]
    refused = use_rag and looks_like_refusal(answer)

    return {
        "question": question,
        "use_rag": use_rag,
        "answer": answer,
        "refused": refused,
        "refusal_reason": "model_judgement" if refused else None,
        "sources": sources,
        "retrieved_chunks": retrieved,
        "distances": distances,
        "top_k": top_k,
        "threshold": threshold if use_threshold else None,
        "collection": (collection or DEFAULT_COLLECTION) if use_rag else None,
        "retrieve_s": retrieve_s,
        "generate_s": round(time.time() - tg, 3),
        "total_s": round(time.time() - t0, 3),
    }


# --------------------------------------------------------------------------
# Endpoints
# --------------------------------------------------------------------------
class Query(BaseModel):
    question: str
    use_rag: bool = True
    # Experiment overrides. Absent in normal use; set by the Chapter 5 scripts.
    top_k: Optional[int] = None
    use_threshold: Optional[bool] = None
    threshold: Optional[float] = None
    collection: Optional[str] = None


@app.get("/health")
def health():
    counts = {}
    for key in COLLECTIONS:
        try:
            counts[key] = get_collection(key).count()
        except Exception:
            counts[key] = 0
    return {
        "status": "ok",
        "llm": MODEL_NAME,
        "whisper": WHISPER_SIZE,
        "whisper_loaded": _whisper_model is not None,
        "default_collection": DEFAULT_COLLECTION,
        "threshold_enabled": USE_THRESHOLD,
        "distance_threshold": DISTANCE_THRESHOLD,
        "top_k": TOP_K,
        "chunks_in_corpus": counts.get(DEFAULT_COLLECTION, 0),
        "collections": counts,
    }


@app.post("/ask")
def ask(query: Query):
    if not query.question.strip():
        raise HTTPException(400, "Question is empty.")
    return answer_question(
        query.question,
        use_rag=query.use_rag,
        top_k=query.top_k,
        use_threshold=query.use_threshold,
        threshold=query.threshold,
        collection=query.collection,
    )


@app.post("/upload")
async def upload(document: UploadFile = File(...)):
    """
    Add a document to the corpus (FR3).

    Ingests into BOTH collections so the chunking ablation stays valid for
    anything uploaded through the interface, not just the seed corpus.
    """
    filename = document.filename or "untitled"
    path = await save_upload(document)
    try:
        text = extract_text(path, filename)
    finally:
        os.remove(path)

    batch = uuid.uuid4().hex[:8]
    added = {}
    for key, chunker in CHUNKERS.items():
        chunks = chunker(text)
        if not chunks:
            continue
        get_collection(key).add(
            documents=chunks,
            metadatas=[{"source": filename} for _ in chunks],
            ids=[f"{batch}-{key}-{i}" for i in range(len(chunks))],
        )
        added[key] = len(chunks)

    if not added:
        raise HTTPException(422, "That document appears to be empty.")

    return {
        "filename": filename,
        "characters": len(text),
        "chunks_added": added.get(DEFAULT_COLLECTION, 0),
        "chunks_added_by_strategy": added,
        "chunks_in_corpus": get_collection().count(),
    }


@app.get("/documents")
def documents():
    try:
        got = get_collection().get(include=["metadatas"])
        names = sorted({m.get("source", "unknown") for m in got["metadatas"]})
    except Exception:
        names = []
    return {"documents": names, "chunks_in_corpus": get_collection().count()}


@app.post("/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    path = await save_upload(audio)
    try:
        return transcribe_path(path)
    finally:
        os.remove(path)


@app.post("/ask/audio")
async def ask_audio(audio: UploadFile = File(...), use_rag: bool = Form(True)):
    """
    Spoken question in, grounded answer out.

    The transcript is returned alongside the answer. This is a transparency
    feature, not debug output: it is what lets the user tell a transcription
    failure apart from a retrieval failure — a distinction that matters most
    on technical vocabulary, where Section 5.6 measured the highest word
    error rate.
    """
    path = await save_upload(audio)
    try:
        asr = transcribe_path(path)
    finally:
        os.remove(path)

    if not asr["transcript"]:
        raise HTTPException(422, "No speech detected in the audio.")

    result = answer_question(asr["transcript"], use_rag=use_rag)
    result["transcript"] = asr["transcript"]
    result["audio_duration_s"] = asr["duration_s"]
    result["transcribe_s"] = asr["transcribe_s"]
    return result