"""
Tests for the distance metric (HANDOVER.md §8, first entry).

ChromaDB defaults to **squared L2**, not cosine. Nothing warns you. The
numbers that come back are plausible, monotonic and correctly ordered —
they are simply on a different scale, and a threshold of 0.53 tuned
against cosine distances means something else entirely against squared L2.

For unit-length vectors the two are related by

    squared_L2 = 2 * (1 - cos_sim) = 2 * cosine_distance

so the failure mode is a silent doubling. Every test below is built around
a pair of orthogonal vectors, where cosine distance is 1.0 and squared L2
is 2.0 — far enough apart that no rounding can disguise which metric ran.

These tests use explicit embeddings rather than the sentence-transformer
model, so they are deterministic, offline, and test Chroma's behaviour
rather than the embedding model's.
"""

import itertools
from pathlib import Path

import chromadb
import pytest

import ingest  # noqa: F401  — imported to assert it is wired to main's config
import main

ROOT = Path(__file__).resolve().parent.parent

# Two orthogonal unit vectors. cosine distance = 1.0, squared L2 = 2.0.
NEAR = [1.0, 0.0, 0.0]
FAR = [0.0, 1.0, 0.0]

COSINE_APART = 1.0
SQUARED_L2_APART = 2.0


# `chromadb.EphemeralClient()` is memoised on its settings, so every call in
# this module returns the *same* in-memory client. Collections therefore
# outlive the function that made them, and a fixed name collides on the
# second use. Each probe gets its own name instead.
_probe_names = itertools.count()


def _build(metadata):
    client = chromadb.EphemeralClient()
    kwargs = {"name": f"probe_{next(_probe_names)}"}
    if metadata is not None:
        kwargs["metadata"] = metadata
    coll = client.create_collection(**kwargs)
    coll.add(
        ids=["near", "far"],
        embeddings=[NEAR, FAR],
        documents=["the near chunk", "the far chunk"],
        metadatas=[{"source": "a.txt"}, {"source": "b.txt"}],
    )
    return coll


def _distances(coll):
    res = coll.query(query_embeddings=[NEAR], n_results=2)
    return [round(float(d), 4) for d in res["distances"][0]]


def test_cosine_space_puts_orthogonal_vectors_at_distance_one():
    """The metric the threshold was tuned against."""
    d = _distances(_build({"hnsw:space": "cosine"}))

    assert d[0] == pytest.approx(0.0, abs=1e-4)
    assert d[1] == pytest.approx(COSINE_APART, abs=1e-3)


def test_chroma_defaults_to_squared_l2_not_cosine():
    """
    Not a test of our code — a test of the assumption our code exists to
    defend against. If Chroma ever changed its default to cosine, this test
    would fail and the explicit metadata in `ingest.py` could be documented
    as belt-and-braces rather than load-bearing. Until then it stays
    load-bearing, and this test is the proof.
    """
    d = _distances(_build(None))

    assert d[1] == pytest.approx(SQUARED_L2_APART, abs=1e-3), (
        "Chroma's default distance is no longer squared L2 — re-read "
        "HANDOVER.md §8 before trusting this suite's premise."
    )
    assert d[1] != pytest.approx(COSINE_APART, abs=1e-2)


def test_the_two_metrics_are_distinguishable_by_a_factor_of_two():
    """
    Pins the relationship the report relies on when explaining why a
    threshold tuned under the wrong metric is not merely imprecise but
    meaningless: it would sit at roughly half the scale it was fitted to.
    """
    cosine = _distances(_build({"hnsw:space": "cosine"}))
    default = _distances(_build(None))

    assert default[1] == pytest.approx(2 * cosine[1], rel=1e-2)


def test_threshold_would_misclassify_under_the_wrong_metric():
    """
    The concrete consequence, stated as a test rather than a comment.

    Under cosine, an orthogonal chunk sits at 1.0 — comfortably outside the
    0.53 threshold, so the system refuses either way. The danger is the
    other direction: a *relevant* chunk at cosine distance 0.35 is well
    inside the threshold and answerable, but the same chunk under squared
    L2 sits at 0.70 and is refused. A false refusal, on a question the
    corpus can answer, with no error anywhere.
    """
    coll = _build({"hnsw:space": "cosine"})

    # cos_sim 0.65 with NEAR, 0.0 with FAR — the remainder is placed on the
    # third axis so the two documents are not in competition. Putting it on
    # FAR's axis instead would make FAR the nearer chunk and quietly measure
    # the wrong document.
    res = coll.query(query_embeddings=[[0.65, 0.0, 0.76]], n_results=1)
    assert res["ids"][0][0] == "near"
    nearest = float(res["distances"][0][0])
    assert nearest == pytest.approx(0.35, abs=1e-3)

    assert nearest < main.DISTANCE_THRESHOLD, (
        f"a chunk at cosine distance {nearest:.4f} should be answerable "
        f"under the {main.DISTANCE_THRESHOLD} threshold"
    )
    assert nearest * 2 > main.DISTANCE_THRESHOLD, (
        "the same chunk under squared L2 would be refused — which is the "
        "silent failure this test exists to catch"
    )


def test_collections_created_on_demand_request_cosine(monkeypatch):
    """
    `main.get_collection` creates a collection when one is missing, so a
    fresh clone works before `ingest.py` has run. That creation path must
    set cosine too — otherwise the metric depends on whether the collection
    happened to exist, which is the least debuggable kind of difference.
    """
    seen = {}

    class FakeClient:
        def get_collection(self, *a, **kw):
            raise RuntimeError("does not exist yet")

        def get_or_create_collection(self, **kw):
            seen.update(kw)
            return object()

    monkeypatch.setattr(main, "chroma_client", FakeClient())

    main.get_collection("sentence")

    assert seen["metadata"] == {"hnsw:space": "cosine"}
    assert seen["name"] == main.COLLECTIONS["sentence"]


def test_ingest_builds_both_collections_from_mains_configuration():
    """
    Section 5.5 is only a fair ablation if both collections were built from
    the same source pass with the same embedding model, differing solely in
    the chunker. `ingest.py` imports its configuration from `main.py` for
    exactly this reason; this guards against the two drifting apart.
    """
    assert ingest.COLLECTIONS is main.COLLECTIONS
    assert ingest.EMBED_MODEL == main.EMBED_MODEL
    assert set(ingest.CHUNKERS) == set(main.COLLECTIONS)


@pytest.mark.skipif(
    not (ROOT / "chroma_db").is_dir(),
    reason="no local chroma_db — run `python ingest.py` first",
)
@pytest.mark.parametrize("key", ["sentence", "fixed"])
def test_the_live_collections_on_disk_are_cosine(key):
    """
    The one test that inspects real state. Everything above proves the code
    *asks* for cosine; this proves the store that Chapter 5's numbers were
    measured against actually *is* cosine.
    """
    client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
    try:
        coll = client.get_collection(name=main.COLLECTIONS[key])
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"collection '{key}' not built: {exc}")

    metadata = coll.metadata or {}
    assert metadata.get("hnsw:space") == "cosine", (
        f"collection '{key}' was built with "
        f"{metadata.get('hnsw:space', 'the Chroma default (squared L2)')}. "
        "Every distance in Chapter 5 measured against it is on the wrong "
        "scale. Rebuild with: python ingest.py --reset"
    )
