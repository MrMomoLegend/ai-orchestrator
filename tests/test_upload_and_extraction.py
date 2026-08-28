"""
Document ingestion (FR3) and the error paths around it.

Upload is the one endpoint that mutates the corpus, so its failure modes
matter more than its happy path: a document that ingests into only one
collection quietly invalidates the Section 5.4 comparison for everything
added afterwards, and a temporary file that survives a rejected upload is a
leak nothing in the interface would reveal.
"""

import os

import pypdf

import main


# --------------------------------------------------------------------------
# The ablation guarantee
# --------------------------------------------------------------------------
def test_upload_ingests_into_both_chunking_collections(client, collections_by_key):
    """
    Section 5.4 compares fixed-size against sentence-aware chunking on the
    same documents. That is only a like-for-like comparison while both
    collections hold the same corpus -- so an upload through the interface
    has to write into both, not just the one being served.

    If this ever regressed, the ablation would still run and still produce
    numbers; they would simply be comparing two different corpora.
    """
    text = ("Perplexity measures how well a model predicts a sample. " * 40).encode()

    body = client.post("/upload", files={"document": ("notes.txt", text, "text/plain")}).json()

    assert set(collections_by_key) == {"sentence", "fixed"}
    assert collections_by_key["sentence"].added
    assert collections_by_key["fixed"].added
    assert set(body["chunks_added_by_strategy"]) == {"sentence", "fixed"}
    assert body["chunks_added"] == body["chunks_added_by_strategy"]["sentence"]


def test_uploaded_chunks_carry_the_filename_as_their_source(client, collections_by_key):
    """
    The source panel shows users which document an answer came from, and the
    Section 5.10 task scenarios ask participants to inspect it. Chunks with a
    missing or wrong source render as "unknown" and make that task
    unanswerable.
    """
    text = ("Smoothing redistributes probability mass. " * 40).encode()

    client.post("/upload", files={"document": ("week8.txt", text, "text/plain")})

    assert set(collections_by_key["sentence"].sources) == {"week8.txt"}


def test_uploaded_chunk_ids_are_unique_per_strategy(client, collections_by_key):
    """
    Chroma silently overwrites on a duplicate id. Both collections are
    written in the same request from the same batch, so the ids have to carry
    the strategy key or the second write would clobber the first.
    """
    text = ("Backoff falls back to a shorter context. " * 40).encode()

    client.post("/upload", files={"document": ("notes.txt", text, "text/plain")})

    sentence_ids = collections_by_key["sentence"].added[0]["ids"]
    fixed_ids = collections_by_key["fixed"].added[0]["ids"]

    assert len(set(sentence_ids)) == len(sentence_ids)
    assert not set(sentence_ids) & set(fixed_ids)


# --------------------------------------------------------------------------
# Error paths
# --------------------------------------------------------------------------
def test_unsupported_extension_is_rejected_as_415(client, collections_by_key):
    """
    A .docx or .csv dropped on the upload control must be refused by media
    type, naming what is accepted. Nothing may reach the corpus.
    """
    r = client.post(
        "/upload", files={"document": ("essay.docx", b"PK\x03\x04junk", "application/octet-stream")}
    )

    assert r.status_code == 415
    assert ".txt" in r.json()["detail"]
    assert collections_by_key == {}


def test_empty_document_is_rejected_as_422(client, collections_by_key):
    """
    A file that reads as zero chunks is accepted by extraction and produces
    nothing to ingest. Reporting success with `chunks_added: 0` would tell
    the user their document is searchable when it is not.
    """
    r = client.post("/upload", files={"document": ("blank.txt", b"   \n\n  ", "text/plain")})

    assert r.status_code == 422
    assert "empty" in r.json()["detail"].lower()


def test_temporary_file_is_removed_even_when_extraction_fails(client, monkeypatch):
    """
    The temp file is cleaned up in a `finally`, so it must survive the
    rejected path as well as the successful one. Every unsupported file a
    user tries would otherwise strand a copy of its contents on disk -- which
    matters here specifically, because the documents being uploaded are
    copyrighted textbook extracts.
    """
    seen = {}
    real_extract = main.extract_text

    def spy(path, filename):
        seen["path"] = path
        seen["existed"] = os.path.exists(path)
        return real_extract(path, filename)

    monkeypatch.setattr(main, "extract_text", spy)

    r = client.post(
        "/upload", files={"document": ("essay.docx", b"PK\x03\x04junk", "application/octet-stream")}
    )

    assert r.status_code == 415
    assert seen["existed"] is True
    assert not os.path.exists(seen["path"])


def test_scanned_pdf_is_422_with_an_actionable_message(client, monkeypatch, collections_by_key):
    """
    A PDF of page images extracts to an empty string. The corpus is textbook
    scans, so this is the likeliest upload failure a user will actually hit,
    and the message has to say *why* rather than reporting an empty document.
    """

    class _ImagePage:
        def extract_text(self):
            return ""

    class _ScannedReader:
        def __init__(self, path):
            self.pages = [_ImagePage(), _ImagePage()]

    monkeypatch.setattr(pypdf, "PdfReader", _ScannedReader)

    r = client.post("/upload", files={"document": ("scan.pdf", b"%PDF-1.4 fake", "application/pdf")})

    assert r.status_code == 422
    assert "scan" in r.json()["detail"].lower()
    assert collections_by_key == {}


def test_markdown_is_accepted_alongside_plain_text(client, collections_by_key):
    """`.md` is in TEXT_EXTS; the four tracked corpus files are plain text."""
    text = ("# Notes\n\nPerplexity is the exponentiated cross-entropy. " * 30).encode()

    r = client.post("/upload", files={"document": ("notes.md", text, "text/markdown")})

    assert r.status_code == 200
    assert r.json()["filename"] == "notes.md"
    assert collections_by_key["sentence"].added


def test_missing_document_field_is_a_validation_error(client, collections_by_key):
    """A request with no file must not touch the corpus."""
    r = client.post("/upload")

    assert r.status_code == 422
    assert collections_by_key == {}
