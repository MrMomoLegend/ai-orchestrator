"""
Shared fixtures for the test suite.

Two principles hold across every test here:

1. **No language model is called.** `ollama.chat` is replaced by a spy in
   every test that reaches the generation step. This is not only about
   speed — several tests assert that the model was *not* consulted, which
   is only meaningful if calls are observable.

2. **No real vector store is queried.** The threshold tests stub the
   collection so retrieval distances are inputs to the test rather than
   properties of whatever happens to be in `chroma_db/` on the day. The
   distance-metric tests are the exception: they build a real ephemeral
   Chroma collection, because the thing under test is Chroma's own
   behaviour.

Importing `main` constructs the embedding function and a PersistentClient
at module scope. Both are cheap on a machine where the model is already
cached, and neither is exercised by the tests below.
"""

import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import main  # noqa: E402


class FakeCollection:
    """
    Minimal stand-in for a Chroma collection.

    Returns whatever distances the test asks for, and records the
    `n_results` it was called with so tests can assert that top_k was
    resolved correctly before it reached retrieval.
    """

    def __init__(self, distances, documents=None, sources=None):
        self.distances = list(distances)
        n = len(self.distances)
        self.documents = list(documents) if documents else [f"chunk {i}" for i in range(n)]
        self.sources = list(sources) if sources else ["doc.txt"] * n
        self.calls = []
        self.added = []

    def query(self, query_texts, n_results):
        self.calls.append({"query_texts": query_texts, "n_results": n_results})
        return {
            "documents": [self.documents],
            "metadatas": [[{"source": s} for s in self.sources]],
            "distances": [self.distances],
        }

    def add(self, documents, metadatas, ids):
        """Record an ingestion. Upload writes into both collections."""
        self.added.append({"documents": list(documents), "ids": list(ids)})
        self.documents.extend(documents)
        self.sources.extend(m.get("source", "unknown") for m in metadatas)

    def get(self, include=None):
        """Only /documents calls this, and only for metadatas."""
        return {"metadatas": [{"source": s} for s in self.sources]}

    def count(self):
        return len(self.documents)


class ChatSpy:
    """Replacement for `ollama.chat` that records every call it receives."""

    def __init__(self, reply="A generated answer about perplexity."):
        self.reply = reply
        self.calls = []

    def __call__(self, model, messages, **kwargs):
        # kwargs is flattened in rather than nested, so a test can assert on
        # `options` without knowing how the call site chose to pass it.
        self.calls.append({"model": model, "messages": messages, **kwargs})
        return {"message": {"content": self.reply}}

    @property
    def called(self):
        return bool(self.calls)

    @property
    def last_prompt(self):
        return self.calls[-1]["messages"][0]["content"]


@pytest.fixture
def chat_spy(monkeypatch):
    """Install a chat spy and hand it to the test."""
    spy = ChatSpy()
    monkeypatch.setattr(main.ollama, "chat", spy)
    return spy


@pytest.fixture
def collection_at(monkeypatch):
    """
    Factory: `collection_at([0.31, 0.44])` makes `get_collection` return a
    fake collection whose nearest chunk sits at distance 0.31.
    """

    def _install(distances, documents=None, sources=None):
        fake = FakeCollection(distances, documents, sources)
        monkeypatch.setattr(main, "get_collection", lambda key=None: fake)
        return fake

    return _install


# ==========================================================================
# Part 2 — HTTP layer, speech, and error paths
# ==========================================================================
#
# Part 1 tested the orchestration function directly. These fixtures add the
# transport in front of it, because several of the guarantees Chapter 4
# claims are properties of the *endpoint*, not of `answer_question`: that an
# empty question is rejected before retrieval, that a bad collection name is
# a 400 rather than a 500, that the temporary file behind an upload is
# always removed, and that a spoken question returns its transcript.
#
# The same two principles still hold — no language model is called, and no
# real vector store is queried.

from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client():
    """The FastAPI app under test, with no server process involved."""
    return TestClient(main.app)


class FakeSegment:
    """One faster-whisper segment. Only `.text` is read by transcribe_path."""

    def __init__(self, text):
        self.text = text


class FakeInfo:
    """faster-whisper's info object. Only `.duration` is read."""

    def __init__(self, duration=3.5):
        self.duration = duration


class WhisperSpy:
    """
    Replacement for the lazily-loaded Whisper model.

    Records the path and decoding options it was handed. The path matters:
    two tests assert that the temporary file exists while transcription is
    running and is gone afterwards, which is the only externally visible
    evidence that the `finally: os.remove(path)` branch works.
    """

    def __init__(self, segments=("Hello", "world"), duration=3.5):
        self._segments = [FakeSegment(s) for s in segments]
        self._duration = duration
        self.calls = []

    def transcribe(self, path, **kwargs):
        self.calls.append(
            {"path": path, "existed_during_call": os.path.exists(path), **kwargs}
        )
        return iter(self._segments), FakeInfo(self._duration)

    @property
    def called(self):
        return bool(self.calls)


@pytest.fixture
def whisper_spy(monkeypatch):
    """
    Factory: `whisper_spy()` installs a default two-word transcript,
    `whisper_spy(segments=())` installs silence.
    """

    def _install(segments=("Hello", "world"), duration=3.5):
        spy = WhisperSpy(segments, duration)
        monkeypatch.setattr(main, "get_whisper", lambda: spy)
        return spy

    return _install


@pytest.fixture
def collections_by_key(monkeypatch):
    """
    A separate fake collection per chunking strategy.

    `collection_at` returns one collection for every key, which is right for
    the retrieval tests. Upload is different: it must write into *both*
    collections, or the Section 5.5 chunking ablation quietly stops being a
    like-for-like comparison for anything added through the interface. That
    can only be asserted if the two are distinguishable.
    """
    fakes = {}

    def _get(key=None):
        return fakes.setdefault(key or main.DEFAULT_COLLECTION, FakeCollection([]))

    monkeypatch.setattr(main, "get_collection", _get)
    return fakes
