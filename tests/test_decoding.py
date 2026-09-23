"""
Tests for deterministic decoding.

The reason this file exists is worth stating plainly, because it is the
strongest methodological point in the evaluation and it was found by
accident rather than by design.

The threshold experiment and the chunking experiment each contain a
condition with identical parameters — RAG on, threshold on, sentence
collection, default k. The same configuration was therefore measured
twice, in two separate runs, and the two runs disagreed:

    k=3    8/15 false refusals   vs   6/15 false refusals
    k=10   0/15 false refusals   vs   1/15 false refusals, and a
           disagreement on one of the three related-but-unanswerable
           questions

Two of fifteen. Most of the effects reported in Sections 5.4 and 5.5 are
of that size or smaller, so under stochastic decoding they were not
separable from run-to-run variance.

Nothing here asserts that llama3.1 is bit-for-bit reproducible — that is
a property of the runtime, not of this code. What these tests pin is that
the system *asks* for deterministic decoding on every generation path. A
call site that silently omits the options would reintroduce the variance
and produce no error at all: the answers would simply drift, and only a
replicate would ever reveal it.
"""

import main


def test_shipped_temperature_is_zero():
    assert main.LLM_TEMPERATURE == 0.0, (
        "Ollama's default is 0.8. A non-zero temperature makes every "
        "ablation a comparison between samples as well as configurations."
    )


def test_shipped_seed_is_fixed():
    assert isinstance(main.LLM_SEED, int)
    assert main.LLM_SEED == 42


def test_grounded_generation_requests_deterministic_decoding(collection_at, chat_spy):
    collection_at([0.20])

    main.answer_question("What is perplexity?")

    assert chat_spy.calls, "no generation call was made"
    options = chat_spy.calls[-1].get("options")
    assert options == {"temperature": 0.0, "seed": 42}


def test_ungrounded_generation_requests_deterministic_decoding(collection_at, chat_spy):
    """
    The RAG-off arm of Section 5.3 takes a different branch through
    `answer_question`. It reaches the same single call site, and this test
    is what keeps it that way.
    """
    main.answer_question("Who invented the transistor?", use_rag=False)

    options = chat_spy.calls[-1].get("options")
    assert options == {"temperature": 0.0, "seed": 42}


def test_refused_requests_make_no_generation_call_at_all(collection_at, chat_spy):
    """
    Determinism on the refusal path is structural rather than statistical:
    a threshold refusal returns a fixed string without consulting the model,
    so it cannot vary between runs. Stated here so the claim in Section 5.4
    that refusal became *deterministic and inspectable* has a test behind it
    rather than only an argument.
    """
    collection_at([0.91])

    result = main.answer_question("Who invented the transistor?")

    assert result["answer"] == main.REFUSAL
    assert chat_spy.calls == []
