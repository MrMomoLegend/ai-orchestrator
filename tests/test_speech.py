"""
The speech path: /transcribe and /ask/audio.

Section 5.7 measured 19.7% word error rate overall and 37.8% on technical
vocabulary -- the words that carry the retrieval signal. That result is the
reason the transcript is returned to the user rather than consumed silently,
and the reason the voice flow is two requests rather than one. Both of those
design decisions are asserted here.

No audio is decoded: the lazily-loaded Whisper model is replaced by a spy, so
these tests run in milliseconds and never touch the second speaker's clips.
"""

import os

import main


# --------------------------------------------------------------------------
# /transcribe
# --------------------------------------------------------------------------
def test_transcribe_returns_transcript_and_timings(client, whisper_spy):
    """Segments are joined and stripped; timings accompany the text."""
    whisper_spy(segments=("  What is ", " perplexity? "), duration=2.345)

    body = client.post(
        "/transcribe", files={"audio": ("q.wav", b"RIFFfake", "audio/wav")}
    ).json()

    assert body["transcript"] == "What is perplexity?"
    assert body["duration_s"] == 2.35
    assert "transcribe_s" in body


def test_transcribe_uses_the_decoding_settings_the_report_measured(client, whisper_spy):
    """
    The decisions log commits to base.en / int8 / beam 5 in both main.py and
    evaluate_asr.py, so that the measured word error rate describes the
    system that ships rather than a differently-tuned one. If the endpoint
    ever drifted from the evaluation script, Section 5.7 would be reporting
    somebody else's numbers.
    """
    spy = whisper_spy()

    client.post("/transcribe", files={"audio": ("q.wav", b"RIFFfake", "audio/wav")})

    assert spy.calls[0]["beam_size"] == main.WHISPER_BEAM == 5
    assert spy.calls[0]["language"] == "en"


def test_transcribe_deletes_the_temporary_file(client, whisper_spy):
    """
    Every upload is written to a temp file and removed in a `finally`. A leak
    here is invisible in normal use and unbounded across a usability session
    -- ten participants asking several spoken questions each would strand
    dozens of audio files on disk.
    """
    spy = whisper_spy()

    client.post("/transcribe", files={"audio": ("q.wav", b"RIFFfake", "audio/wav")})

    assert spy.calls[0]["existed_during_call"] is True
    assert not os.path.exists(spy.calls[0]["path"])


def test_transcribe_preserves_the_upload_extension(client, whisper_spy):
    """
    faster-whisper dispatches on the container format, which it reads from
    the filename. The Section 5.7 clips are .m4a; a temp file that silently
    became .wav would fail to decode for reasons nothing in the traceback
    would explain.
    """
    spy = whisper_spy()

    client.post("/transcribe", files={"audio": ("clip.m4a", b"fake", "audio/mp4")})

    assert spy.calls[0]["path"].endswith(".m4a")


def test_missing_audio_file_is_a_validation_error(client, whisper_spy):
    """A request with no file must not load the speech model."""
    spy = whisper_spy()

    r = client.post("/transcribe")

    assert r.status_code == 422
    assert spy.called is False


# --------------------------------------------------------------------------
# /ask/audio
# --------------------------------------------------------------------------
def test_spoken_question_returns_the_transcript_alongside_the_answer(
    client, whisper_spy, chat_spy, collection_at
):
    """
    The transparency guarantee of Section 4.4, and the single most important
    test in this file.

    At 37.8% word error rate on technical vocabulary, a mis-transcribed
    question retrieves nothing and the system refuses -- which is
    indistinguishable, from the user's side, from the corpus genuinely not
    covering the topic. Returning the transcript is what separates "I
    misheard you" from "I do not know". If this key ever disappears, the
    interface silently loses that distinction and Section 4.4's argument
    stops being true of the shipped system.
    """
    whisper_spy(segments=("What is perplexity?",), duration=1.8)
    collection_at([0.31], sources=["week8.txt"])

    body = client.post(
        "/ask/audio", files={"audio": ("q.wav", b"RIFFfake", "audio/wav")}
    ).json()

    assert body["transcript"] == "What is perplexity?"
    assert body["question"] == "What is perplexity?"
    assert body["audio_duration_s"] == 1.8
    assert "transcribe_s" in body
    assert body["answer"] == chat_spy.reply
    assert body["sources"] == ["week8.txt"]


def test_silent_audio_is_refused_before_the_model_is_called(
    client, whisper_spy, chat_spy, collection_at
):
    """
    An empty transcript must stop the pipeline. Passing "" through to
    retrieval would embed the empty string, return whichever chunks happen to
    be nearest the origin, and produce a confident answer to a question
    nobody asked.
    """
    whisper_spy(segments=())
    fake = collection_at([0.31])

    r = client.post("/ask/audio", files={"audio": ("silence.wav", b"RIFFfake", "audio/wav")})

    assert r.status_code == 422
    assert "No speech" in r.json()["detail"]
    assert fake.calls == []
    assert chat_spy.called is False


def test_spoken_question_honours_use_rag_false_from_the_form(
    client, whisper_spy, chat_spy, collection_at
):
    """
    The RAG toggle reaches this endpoint as multipart form data, not JSON, so
    it is parsed by a different code path than /ask. The live demonstration
    of the CKY fabrication flips this switch on a spoken question.
    """
    whisper_spy(segments=("Who invented the CKY algorithm?",))
    fake = collection_at([0.31])

    body = client.post(
        "/ask/audio",
        files={"audio": ("q.wav", b"RIFFfake", "audio/wav")},
        data={"use_rag": "false"},
    ).json()

    assert body["use_rag"] is False
    assert fake.calls == []
    assert chat_spy.last_prompt == "Who invented the CKY algorithm?"


def test_spoken_question_deletes_its_temporary_file(
    client, whisper_spy, chat_spy, collection_at
):
    """The two-request flow duplicates the temp-file handling; so does the test."""
    spy = whisper_spy(segments=("What is perplexity?",))
    collection_at([0.31])

    client.post("/ask/audio", files={"audio": ("q.wav", b"RIFFfake", "audio/wav")})

    assert not os.path.exists(spy.calls[0]["path"])
