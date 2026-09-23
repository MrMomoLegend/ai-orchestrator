"""
The HTTP surface: /health, /ask and /documents.

Part 1 tested `answer_question` directly. These tests go through the
transport, because the guarantees below live in the endpoint rather than in
the orchestration function, and because an examiner cloning the repository
meets the API before they meet the code.

Nothing here calls a language model or queries a real vector store.
"""

import main


# --------------------------------------------------------------------------
# /health — the configuration the system advertises about itself
# --------------------------------------------------------------------------
def test_health_reports_the_shipped_configuration(client, collection_at, monkeypatch):
    """
    /health is the reproducibility claim made machine-readable.

    Chapter 5 states its results were produced at k=10, threshold 0.53,
    temperature 0 and seed 42. If the running server reports anything else,
    every number in the chapter describes a different system than the one an
    examiner is holding. This test fails loudly if the two ever diverge.
    """
    collection_at([0.31])
    monkeypatch.setattr(main, "_whisper_model", None)

    body = client.get("/health").json()

    assert body["status"] == "ok"
    assert body["top_k"] == 10
    assert body["distance_threshold"] == 0.53
    assert body["threshold_enabled"] is True
    assert body["temperature"] == 0
    assert body["seed"] == 42
    assert body["default_collection"] == "sentence"


def test_health_shows_whisper_unloaded_until_first_use(client, collection_at, monkeypatch):
    """
    Lazy Whisper loading (Section 4.4) is a deliberate trade: the text path
    never pays the 10-20 s model load, and the cost reappears as a cold start
    on the first spoken question. That decision is only defensible if it is
    observable, so /health reports residency rather than hiding it.
    """
    collection_at([0.31])
    monkeypatch.setattr(main, "_whisper_model", None)

    assert client.get("/health").json()["whisper_loaded"] is False


def test_health_survives_an_unreachable_vector_store(client, monkeypatch):
    """
    A fresh clone has no chroma_db/ until ingest.py runs. /health must still
    answer -- it is what the frontend polls to decide whether the backend is
    up, so a 500 here presents as "the whole system is down" when in fact
    only the corpus is missing.
    """
    def _explode(key=None):
        raise RuntimeError("no such collection")

    monkeypatch.setattr(main, "get_collection", _explode)

    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks_in_corpus"] == 0


# --------------------------------------------------------------------------
# /ask — rejection before work is done
# --------------------------------------------------------------------------
def test_empty_question_is_rejected_before_retrieval(client, chat_spy, collection_at):
    """An empty question must cost nothing: no retrieval, no generation."""
    fake = collection_at([0.31])

    r = client.post("/ask", json={"question": ""})

    assert r.status_code == 400
    assert fake.calls == []
    assert chat_spy.called is False


def test_whitespace_only_question_is_rejected(client, chat_spy, collection_at):
    """
    The guard is `.strip()`, not truthiness. A question of three spaces is
    empty to a user and truthy to Python, and the frontend sends exactly that
    when someone hits enter on an untouched box.
    """
    fake = collection_at([0.31])

    r = client.post("/ask", json={"question": "   \n\t "})

    assert r.status_code == 400
    assert fake.calls == []
    assert chat_spy.called is False


def test_missing_question_field_is_a_validation_error(client, chat_spy):
    """A malformed body is Pydantic's 422, distinct from the 400 above."""
    r = client.post("/ask", json={"use_rag": True})

    assert r.status_code == 422
    assert chat_spy.called is False


# --------------------------------------------------------------------------
# /ask — the response contract
# --------------------------------------------------------------------------
EXPECTED_KEYS = {
    "question", "use_rag", "answer", "refused", "refusal_reason",
    "sources", "retrieved_chunks", "distances", "top_k", "threshold",
    "collection", "retrieve_s", "generate_s", "total_s",
}


def test_grounded_answer_returns_the_documented_response_shape(
    client, chat_spy, collection_at
):
    """
    Every field here is consumed by something: the frontend renders `sources`
    and `distances` in the panel, and the Chapter 5 scripts read `refused`,
    `refusal_reason` and the timings straight into the results CSVs. A
    silently renamed key breaks the experiments, not just the UI.
    """
    collection_at([0.31, 0.44], sources=["week8.txt", "week8.txt"])

    body = client.post("/ask", json={"question": "What is perplexity?"}).json()

    assert set(body) == EXPECTED_KEYS
    assert body["refused"] is False
    assert body["refusal_reason"] is None
    assert body["sources"] == ["week8.txt", "week8.txt"]
    assert body["distances"] == [0.31, 0.44]
    assert body["collection"] == "sentence"
    assert chat_spy.called is True


def test_rag_disabled_sends_the_bare_question_and_reports_no_sources(
    client, chat_spy, collection_at
):
    """
    The ungrounded condition of Section 5.3. The model must receive the
    question with no context wrapper at all -- if any instruction leaked into
    the prompt, the 15/15 fabrication result would be measuring a weakened
    version of the baseline rather than the baseline.
    """
    fake = collection_at([0.31])

    body = client.post(
        "/ask", json={"question": "Who invented the CKY algorithm?", "use_rag": False}
    ).json()

    assert fake.calls == []
    assert chat_spy.last_prompt == "Who invented the CKY algorithm?"
    assert body["sources"] == []
    assert body["retrieved_chunks"] == []
    assert body["collection"] is None


def test_threshold_refusal_never_reaches_the_model_over_http(
    client, chat_spy, collection_at
):
    """
    Section 4.5's claim, asserted at the transport layer: a refusal above the
    threshold is a 200 carrying a refusal, not an error, and the model is not
    consulted. Returning 4xx here would be wrong -- the system answered
    correctly by declining.
    """
    collection_at([0.61, 0.72])

    r = client.post("/ask", json={"question": "What is the capital of Peru?"})
    body = r.json()

    assert r.status_code == 200
    assert body["refused"] is True
    assert body["refusal_reason"] == "retrieval_distance"
    assert body["answer"] == main.REFUSAL
    assert body["sources"] == []
    assert chat_spy.called is False


def test_unknown_collection_is_a_400_not_a_500(client, chat_spy):
    """
    Swagger's "Try it out" pre-fills `collection: "string"`. That reaches
    get_collection as an unknown key, and it must surface as a client error
    naming the valid options -- an unhandled 500 here reads as a broken
    endpoint and cost real debugging time once already.
    """
    r = client.post("/ask", json={"question": "What is perplexity?", "collection": "string"})

    assert r.status_code == 400
    assert "sentence" in r.json()["detail"]
    assert chat_spy.called is False


def test_placeholder_zero_top_k_falls_back_to_the_configured_default(
    client, chat_spy, collection_at
):
    """
    top_k=0 retrieves nothing, so a client sending Swagger's placeholder zero
    would get a confident answer from an empty context rather than an error.
    It is coerced to the configured default instead.

    Note the deliberate asymmetry with threshold=0.0, which is *not* coerced
    because `exp_rag.py --sweep` sends it on purpose to harvest retrieval
    distances without paying for 30 generations. That case is covered in
    test_threshold.py.
    """
    fake = collection_at([0.31])

    body = client.post("/ask", json={"question": "What is perplexity?", "top_k": 0}).json()

    assert fake.calls[0]["n_results"] == main.TOP_K == 10
    assert body["top_k"] == 10


def test_explicit_top_k_is_passed_through_to_retrieval(client, chat_spy, collection_at):
    """The Section 5.6 sweep is a request parameter; k must arrive unchanged."""
    fake = collection_at([0.31] * 3)

    client.post("/ask", json={"question": "What is perplexity?", "top_k": 3})

    assert fake.calls[0]["n_results"] == 3


# --------------------------------------------------------------------------
# /documents
# --------------------------------------------------------------------------
def test_documents_lists_each_source_once(client, collection_at):
    """
    The corpus is thousands of chunks drawn from a handful of files. The
    endpoint backs a file list in the UI, so it must report documents, not
    chunks.
    """
    collection_at([0.1] * 4, sources=["b.txt", "a.txt", "b.txt", "a.txt"])

    body = client.get("/documents").json()

    assert body["documents"] == ["a.txt", "b.txt"]
    assert body["chunks_in_corpus"] == 4
