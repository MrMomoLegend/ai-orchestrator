"""
Tests for the retrieval-confidence threshold (Section 4.5 / Section 5.3).

Why this branch is worth testing rather than eyeballing: when it is wrong
it does not raise. A threshold that never fires produces fluent answers,
and a threshold that always fires produces polite refusals. Both look like
a working system from the outside, and both invalidate every number in
Section 5.3. The only externally visible signal is *whether the language
model was consulted at all*, which is what most of these tests assert.
"""

import pytest

import main


# ---------------------------------------------------------------------------
# The shipped configuration matches the reported configuration
# ---------------------------------------------------------------------------
def test_shipped_threshold_is_the_value_reported_in_chapter_5():
    """
    0.53 is the midpoint of the separating margin found in Section 5.2
    (answerable max 0.4799, out-of-corpus min 0.5915). If this drifts, every
    refusal figure in the report describes a system that is no longer the
    one in the repository.
    """
    assert main.DISTANCE_THRESHOLD == pytest.approx(0.53), (
        "DISTANCE_THRESHOLD does not match the reported value. If an env "
        "var is set in this shell, unset it before running the suite."
    )


def test_shipped_top_k_is_ten():
    """Section 5.5 selected k=10 over the inherited k=3. Guard against drift."""
    assert main.TOP_K == 10


def test_threshold_is_enabled_by_default():
    assert main.USE_THRESHOLD is True


# ---------------------------------------------------------------------------
# The branch itself
# ---------------------------------------------------------------------------
def test_refuses_without_calling_the_model_when_nearest_chunk_is_too_far(
    collection_at, chat_spy
):
    """
    The load-bearing claim of Section 4.5: refusal is a property of
    retrieval, not of the model's judgement. If the model is consulted at
    all, the claim is false regardless of what it answers.
    """
    collection_at([0.72, 0.81, 0.90])

    result = main.answer_question("Who invented the transistor?")

    assert result["refused"] is True
    assert result["refusal_reason"] == "retrieval_distance"
    assert result["answer"] == main.REFUSAL
    assert chat_spy.called is False, "the language model was consulted on a refusal"


def test_answers_when_nearest_chunk_is_within_threshold(collection_at, chat_spy):
    collection_at([0.31, 0.44, 0.58])

    result = main.answer_question("What is perplexity?")

    assert result["refused"] is False
    assert result["refusal_reason"] is None
    assert chat_spy.called is True


def test_boundary_exactly_at_the_threshold_does_not_refuse(collection_at, chat_spy):
    """
    The comparison is strictly greater-than, so a distance equal to the
    threshold is inside it. Stated explicitly because an off-by-one here
    would shift borderline questions between buckets in Section 5.3
    without producing any error.
    """
    collection_at([0.53])

    result = main.answer_question("A borderline question")

    assert result["refused"] is False
    assert chat_spy.called is True


def test_boundary_just_beyond_the_threshold_refuses(collection_at, chat_spy):
    collection_at([0.5301])

    result = main.answer_question("A borderline question")

    assert result["refused"] is True
    assert chat_spy.called is False


def test_only_the_nearest_distance_decides(collection_at, chat_spy):
    """
    The decision reads distances[0], relying on Chroma returning results in
    ascending distance order. This test pins that assumption: given an
    unsorted list whose *minimum* is well inside the threshold but whose
    *first* element is outside it, the system refuses. If Chroma ever
    stopped sorting, this test would be the thing that noticed.
    """
    collection_at([0.60, 0.10, 0.12])

    result = main.answer_question("A question")

    assert result["refused"] is True
    assert chat_spy.called is False


def test_empty_retrieval_refuses(collection_at, chat_spy):
    """An empty corpus must refuse, not fall through to an ungrounded answer."""
    collection_at([])

    result = main.answer_question("A question against an empty corpus")

    assert result["refused"] is True
    assert result["refusal_reason"] == "retrieval_distance"
    assert chat_spy.called is False


def test_disabling_the_threshold_reaches_the_model_even_when_retrieval_is_poor(
    collection_at, chat_spy
):
    """
    This is the 'threshold off' arm of the Section 5.3 ablation. It must be
    the *same* code path with one parameter changed, or the ablation is
    comparing two implementations rather than one variable.
    """
    collection_at([0.95])

    result = main.answer_question("A question", use_threshold=False)

    assert chat_spy.called is True
    assert result["refusal_reason"] != "retrieval_distance"
    assert result["threshold"] is None


def test_a_threshold_refusal_costs_no_generation_time(collection_at, chat_spy):
    """
    Section 5.3's defensible benefit is cost, not accuracy: a refused query
    returns in ~0.029 s instead of ~1.8 s because generation never happens.
    """
    collection_at([0.88])

    result = main.answer_question("A question")

    assert result["generate_s"] == 0.0
    assert result["total_s"] >= 0.0
    assert "retrieve_s" in result


# ---------------------------------------------------------------------------
# Placeholder values from the API surface
# ---------------------------------------------------------------------------
def test_top_k_of_zero_falls_back_to_the_default(collection_at, chat_spy):
    """
    Swagger's 'Try it out' pre-fills optional integers with 0. top_k=0
    retrieves nothing, which would refuse every question and look like a
    broken corpus rather than a bad request.
    """
    fake = collection_at([0.20])

    result = main.answer_question("A question", top_k=0)

    assert fake.calls[0]["n_results"] == main.TOP_K
    assert result["top_k"] == main.TOP_K


def test_threshold_of_zero_is_honoured_rather_than_ignored(collection_at, chat_spy):
    """
    Deliberately asymmetric with top_k above, and the asymmetry is load-
    bearing rather than an oversight.

    top_k=0 is never a meaningful request. threshold=0.0 is: `exp_rag.py
    --sweep` sends it precisely to force an immediate refusal on every
    question, which returns the retrieval distances without ever paying for
    a language-model call. Coercing 0.0 to the default here would turn the
    threshold sweep into 30 full generations and silently change the data
    that Figure 5.2 is built from.
    """
    collection_at([0.01])

    result = main.answer_question("A question", threshold=0.0)

    assert result["refused"] is True
    assert result["refusal_reason"] == "retrieval_distance"
    assert result["distances"] == [0.01], "the sweep needs the distances back"
    assert chat_spy.called is False


# ---------------------------------------------------------------------------
# The other refusal path: the model's own judgement
# ---------------------------------------------------------------------------
def test_model_judgement_refusal_is_labelled_distinctly(collection_at, monkeypatch):
    """
    With the threshold off, refusal can only be detected from the answer
    text. Section 5.3 counts the two mechanisms separately, so they must be
    distinguishable in the response.
    """
    collection_at([0.95])
    monkeypatch.setattr(
        main.ollama,
        "chat",
        lambda model, messages, **kw: {
            "message": {"content": "The provided documents do not contain this information."}
        },
    )

    result = main.answer_question("A question", use_threshold=False)

    assert result["refused"] is True
    assert result["refusal_reason"] == "model_judgement"


def test_ungrounded_requests_never_retrieve(collection_at, chat_spy):
    """
    The 'RAG off' arm of Section 5.2. It must send the bare question — if
    any context leaked into the prompt, the 15/15 hallucination result
    would be measuring a weaker contrast than it claims.
    """
    fake = collection_at([0.10])

    result = main.answer_question("Who wrote the CKY algorithm?", use_rag=False)

    assert fake.calls == [], "retrieval ran on an ungrounded request"
    assert result["retrieved_chunks"] == []
    assert result["sources"] == []
    assert chat_spy.last_prompt == "Who wrote the CKY algorithm?"


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("The provided documents do not contain this information.", True),
        ("The context does not contain that.", True),
        ("I cannot answer that from the provided context.", True),
        ("The information is insufficient.", True),
        ("Perplexity is the inverse probability of the test set, normalised.", False),
        ("", False),
    ],
)
def test_refusal_detector(answer, expected):
    assert main.looks_like_refusal(answer) is expected
