/*
 * App.jsx — Locally-Deployed AI Orchestration System
 * Single-page React client for the FastAPI orchestration backend.
 *
 * Six components map directly onto the functional requirements:
 *   DocumentPanel  -> FR3  upload documents into the corpus
 *   QuestionInput  -> FR1  text input
 *   RecordButton   -> FR2  voice input via MediaRecorder
 *   RagToggle      ->      drives the evaluation and the demonstration
 *   AnswerPanel    -> FR5  grounded answer
 *   SourcePanel    -> FR6  retrieved passages, the hallucination-mitigation
 *                          story made visible
 *
 * No router, no state library. The application is one screen with one
 * request in flight at a time; anything more would be scaffolding without
 * a load to carry.
 */

import { useCallback, useEffect, useRef, useState } from "react";
import "./App.css";

const API = "http://127.0.0.1:8000";

const STAGE_LABEL = {
  uploading: "Adding your document…",
  transcribing: "Transcribing what you said…",
  retrieving: "Searching your documents…",
  generating: "Generating an answer…",
};

/* Read FastAPI's error detail rather than showing "500". */
async function apiError(res) {
  let detail = `${res.status} ${res.statusText}`;
  try {
    const body = await res.json();
    if (body.detail) detail = body.detail;
  } catch {
    /* non-JSON error body; keep the status line */
  }
  return new Error(detail);
}

export default function App() {
  const [backendUp, setBackendUp] = useState(null); // null = still checking
  const [docs, setDocs] = useState([]);
  const [chunkCount, setChunkCount] = useState(0);

  const [question, setQuestion] = useState("");
  const [useRag, setUseRag] = useState(true);

  const [stage, setStage] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const busy = stage !== null;

  /* ---------------- corpus state ---------------- */
  const refreshDocs = useCallback(async () => {
    try {
      const res = await fetch(`${API}/documents`);
      if (!res.ok) throw await apiError(res);
      const data = await res.json();
      setDocs(data.documents || []);
      setChunkCount(data.chunks_in_corpus || 0);
      setBackendUp(true);
    } catch {
      setBackendUp(false);
    }
  }, []);

  useEffect(() => {
    refreshDocs();
  }, [refreshDocs]);

  /* ---------------- elapsed timer ----------------
   * This counter is measured, not simulated. It is the honest signal in the
   * interface: the stage labels tell the user which part of the pipeline is
   * running, and this tells them how long they have actually been waiting.
   */
  useEffect(() => {
    if (!busy) return undefined;
    const t0 = Date.now();
    setElapsed(0);
    const id = setInterval(() => setElapsed((Date.now() - t0) / 1000), 100);
    return () => clearInterval(id);
  }, [busy]);

  /* ---------------- ask ---------------- */
  async function askText(text, transcript = null) {
    const trimmed = text.trim();
    if (!trimmed) return;

    setError(null);
    setResult(transcript ? { transcript } : null);
    setStage("retrieving");

    // Retrieval completes in well under a second (measured: see §5.7), so
    // this hand-off is nominal rather than observed. The elapsed counter
    // above is the real measurement.
    const toGenerating = setTimeout(() => setStage("generating"), 600);

    try {
      const res = await fetch(`${API}/ask`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: trimmed, use_rag: useRag }),
      });
      if (!res.ok) throw await apiError(res);
      const data = await res.json();
      setResult({ ...data, transcript: transcript ?? data.transcript ?? null });
    } catch (err) {
      setError(
        err.message === "Failed to fetch"
          ? { title: "Lost contact with the assistant.", hint: "Check the backend is still running." }
          : { title: "That question could not be answered.", hint: err.message }
      );
      setBackendUp(err.message === "Failed to fetch" ? false : true);
    } finally {
      clearTimeout(toGenerating);
      setStage(null);
    }
  }

  return (
    <div className="app">
      <header className="header">
        <div>
          <h1>Local AI Assistant</h1>
          <p className="sub">
            Answers grounded in your own documents. Everything runs on this
            machine — nothing is sent to the internet.
          </p>
        </div>
        <span className={`pill ${backendUp === false ? "pill-bad" : "pill-good"}`}>
          {backendUp === null ? "connecting…" : backendUp ? "running locally" : "offline"}
        </span>
      </header>

      {backendUp === false && (
        <Banner
          kind="error"
          title="Can't reach the assistant."
          body="The backend isn't responding. Start it with `uvicorn main:app --reload`, then reload this page."
          action={{ label: "Try again", onClick: refreshDocs }}
        />
      )}

      {backendUp && chunkCount === 0 && (
        <Banner
          kind="warn"
          title="No documents loaded yet."
          body="Add a document below, then ask a question about it."
        />
      )}

      <DocumentPanel
        docs={docs}
        chunkCount={chunkCount}
        disabled={busy || backendUp === false}
        onStage={setStage}
        onError={setError}
        onUploaded={refreshDocs}
      />

      <section className="card">
        <h2>Ask a question</h2>

        <QuestionInput
          value={question}
          onChange={setQuestion}
          disabled={busy || backendUp === false}
          onSubmit={() => askText(question)}
        />

        <div className="row">
          <RecordButton
            disabled={busy || backendUp === false}
            onStage={setStage}
            onError={setError}
            onTranscript={(t) => {
              setQuestion(t);
              askText(t, t);
            }}
          />
          <RagToggle value={useRag} onChange={setUseRag} disabled={busy} />
        </div>

        {busy && (
          <div className="loading" role="status" aria-live="polite">
            <span className="spinner" />
            <span>{STAGE_LABEL[stage]}</span>
            <span className="elapsed">{elapsed.toFixed(1)}s</span>
          </div>
        )}

        {error && <Banner kind="error" title={error.title} body={error.hint} />}
      </section>

      {result && <AnswerPanel result={result} />}
      {result?.retrieved_chunks?.length > 0 && <SourcePanel result={result} />}
    </div>
  );
}

/* ====================================================================== */
/* FR3 — document upload                                                  */
/* ====================================================================== */
function DocumentPanel({ docs, chunkCount, disabled, onStage, onError, onUploaded }) {
  const [dragging, setDragging] = useState(false);
  const [note, setNote] = useState(null);
  const inputRef = useRef(null);

  async function send(file) {
    if (!file) return;
    onError(null);
    setNote(null);
    onStage("uploading");

    const form = new FormData();
    form.append("document", file);

    try {
      const res = await fetch(`${API}/upload`, { method: "POST", body: form });
      if (!res.ok) throw await apiError(res);
      const data = await res.json();
      setNote(`Added “${data.filename}” — ${data.chunks_added} passages indexed.`);
      onUploaded();
    } catch (err) {
      onError({
        title: "That document could not be added.",
        hint:
          err.message === "Failed to fetch"
            ? "The backend isn't responding."
            : err.message,
      });
    } finally {
      onStage(null);
    }
  }

  return (
    <section className="card">
      <h2>Your documents</h2>

      <div
        className={`drop ${dragging ? "drop-active" : ""} ${disabled ? "drop-off" : ""}`}
        onDragOver={(e) => {
          e.preventDefault();
          if (!disabled) setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDragging(false);
          if (!disabled) send(e.dataTransfer.files?.[0]);
        }}
        onClick={() => !disabled && inputRef.current?.click()}
      >
        <strong>Drop a document here, or click to choose one</strong>
        <span className="hint">.txt, .md or .pdf</span>
        <input
          ref={inputRef}
          type="file"
          accept=".txt,.md,.markdown,.pdf"
          hidden
          onChange={(e) => {
            send(e.target.files?.[0]);
            e.target.value = "";
          }}
        />
      </div>

      {note && <p className="note-ok">{note}</p>}

      {docs.length > 0 && (
        <>
          <ul className="doclist">
            {docs.map((d) => (
              <li key={d}>{d}</li>
            ))}
          </ul>
          <p className="hint">
            {docs.length} document{docs.length === 1 ? "" : "s"} · {chunkCount} passages
            indexed
          </p>
        </>
      )}
    </section>
  );
}

/* ====================================================================== */
/* FR1 — text input                                                        */
/* ====================================================================== */
function QuestionInput({ value, onChange, disabled, onSubmit }) {
  return (
    <div className="ask">
      <textarea
        rows={3}
        value={value}
        disabled={disabled}
        placeholder="e.g. What does the module description say about the final report?"
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            onSubmit();
          }
        }}
      />
      <button className="primary" disabled={disabled || !value.trim()} onClick={onSubmit}>
        Ask
      </button>
    </div>
  );
}

/* ====================================================================== */
/* FR2 — voice input                                                       */
/*                                                                         */
/* Transcription and answering are two separate requests on purpose. The    */
/* client genuinely knows when transcription has finished, so it can show   */
/* the user what was heard before the answer arrives rather than guessing   */
/* at progress. That makes the transcript a usable check on the speech      */
/* model instead of an after-the-fact explanation.                          */
/* ====================================================================== */
function pickMimeType() {
  const candidates = ["audio/webm", "audio/webm;codecs=opus", "audio/mp4", "audio/ogg"];
  return candidates.find((t) => window.MediaRecorder?.isTypeSupported(t)) || "";
}

function RecordButton({ disabled, onStage, onError, onTranscript }) {
  const [recording, setRecording] = useState(false);
  const recorderRef = useRef(null);
  const chunksRef = useRef([]);

  async function start() {
    onError(null);

    if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) {
      onError({
        title: "This browser can't record audio.",
        hint: "Use Chrome or Edge, or type the question instead.",
      });
      return;
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch {
      onError({
        title: "Microphone access was blocked.",
        hint: "Allow microphone access in the address bar, or type the question instead.",
      });
      return;
    }

    const mimeType = pickMimeType();
    const rec = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
    chunksRef.current = [];
    rec.ondataavailable = (e) => e.data.size > 0 && chunksRef.current.push(e.data);
    rec.onstop = async () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(chunksRef.current, { type: mimeType || "audio/webm" });
      if (blob.size < 1000) {
        onError({
          title: "That recording was too short.",
          hint: "Hold the button while you speak, then press stop.",
        });
        return;
      }
      await transcribe(blob, mimeType);
    };

    recorderRef.current = rec;
    rec.start();
    setRecording(true);
  }

  function stop() {
    recorderRef.current?.stop();
    setRecording(false);
  }

  async function transcribe(blob, mimeType) {
    onStage("transcribing");
    const ext = mimeType.includes("mp4") ? "mp4" : mimeType.includes("ogg") ? "ogg" : "webm";
    const form = new FormData();
    form.append("audio", blob, `question.${ext}`);

    try {
      const res = await fetch(`${API}/transcribe`, { method: "POST", body: form });
      if (!res.ok) throw await apiError(res);
      const data = await res.json();
      if (!data.transcript) {
        onError({
          title: "Nothing was heard in that recording.",
          hint: "Check the microphone is working, then try again.",
        });
        onStage(null);
        return;
      }
      onStage(null);
      onTranscript(data.transcript);
    } catch (err) {
      onError({
        title: "That recording could not be transcribed.",
        hint:
          err.message === "Failed to fetch" ? "The backend isn't responding." : err.message,
      });
      onStage(null);
    }
  }

  return (
    <button
      className={recording ? "record recording" : "record"}
      disabled={disabled}
      onClick={recording ? stop : start}
    >
      <span className="dot" />
      {recording ? "Stop and send" : "Ask by voice"}
    </button>
  );
}

/* ====================================================================== */
/* RAG toggle — drives the evaluation and the demonstration                */
/* ====================================================================== */
function RagToggle({ value, onChange, disabled }) {
  return (
    <label className={`toggle ${disabled ? "toggle-off" : ""}`}>
      <input
        type="checkbox"
        checked={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="track">
        <span className="knob" />
      </span>
      <span className="toggle-text">
        <strong>Use my documents</strong>
        <span className="hint">
          {value
            ? "Answers come only from your documents"
            : "Answering from the model's own memory"}
        </span>
      </span>
    </label>
  );
}

/* ====================================================================== */
/* FR5 — the answer                                                        */
/* ====================================================================== */
function AnswerPanel({ result }) {
  return (
    <section className="card">
      <h2>Answer</h2>

      {result.transcript && (
        <p className="heard">
          <span className="heard-label">Heard</span>“{result.transcript}”
        </p>
      )}

      {result.answer ? (
        <>
          <p className="answer">{result.answer}</p>
          <p className="hint">
            {result.use_rag ? "Grounded in your documents" : "From the model's own memory"}
            {typeof result.generate_s === "number" && ` · ${result.generate_s}s`}
          </p>
        </>
      ) : (
        <p className="hint">Waiting for the answer…</p>
      )}
    </section>
  );
}

/* ====================================================================== */
/* FR6 — source passages                                                   */
/* ====================================================================== */
function SourcePanel({ result }) {
  const { retrieved_chunks: chunks = [], sources = [], distances = [] } = result;

  return (
    <section className="card">
      <h2>Where this came from</h2>
      <p className="hint">
        These are the passages the assistant retrieved and was allowed to use.
        If the answer is not supported by them, it is not grounded.
      </p>

      <ol className="sources">
        {chunks.map((chunk, i) => (
          <li key={i}>
            <div className="source-head">
              <span className="source-name">{sources[i] || "unknown"}</span>
              {typeof distances[i] === "number" && (
                <span className="source-score" title="Lower is a closer match">
                  distance {distances[i].toFixed(3)}
                </span>
              )}
            </div>
            <p className="chunk">{chunk}</p>
          </li>
        ))}
      </ol>
    </section>
  );
}

/* ====================================================================== */
function Banner({ kind, title, body, action }) {
  return (
    <div className={`banner banner-${kind}`} role="alert">
      <div>
        <strong>{title}</strong>
        {body && <p>{body}</p>}
      </div>
      {action && (
        <button className="ghost" onClick={action.onClick}>
          {action.label}
        </button>
      )}
    </div>
  );
}
