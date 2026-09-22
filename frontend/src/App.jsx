/*
 * App.jsx — Locally-Deployed AI Orchestration System
 * Single-page React client for the FastAPI orchestration backend.
 *
 * Components map directly onto the functional requirements:
 *   DocumentPanel  -> FR3  upload documents into the corpus
 *   QuestionInput  -> FR1  text input
 *   RecordButton   -> FR2  voice input via MediaRecorder
 *   RagToggle      ->      drives the evaluation and the demonstration
 *   AnswerPanel    -> FR5  grounded answer
 *   SourcePanel    -> FR6  retrieved passages, the hallucination-mitigation
 *                          story made visible
 *   SystemPanel    ->      the configuration every answer is produced under
 *
 * Layout (revised after the usability study, §5.10): the question composer
 * sits at the top of the main column so it is never below the fold; the
 * document library and system settings live in a side column.
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
  const [health, setHealth] = useState(null);

  const [question, setQuestion] = useState("");
  const [useRag, setUseRag] = useState(true);

  const [stage, setStage] = useState(null);
  const [elapsed, setElapsed] = useState(0);
  const [result, setResult] = useState(null);
  const [askId, setAskId] = useState(0); // new id per answer resets the source panel
  const [error, setError] = useState(null);

  // Shared with DocumentPanel so the composer's "+" opens the same picker.
  const fileInputRef = useRef(null);

  const busy = stage !== null;
  const offline = backendUp === false;

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
    try {
      const res = await fetch(`${API}/health`);
      if (res.ok) setHealth(await res.json());
    } catch {
      /* the documents call above already reports the outage */
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

    // Retrieval completes in about 10 ms (measured: see §5.9), so this
    // hand-off is nominal rather than observed. The elapsed counter above is
    // the real measurement.
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
      setAskId((n) => n + 1);
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
            machine; nothing is sent to the internet.
          </p>
        </div>
        <span className={`pill ${offline ? "pill-bad" : "pill-good"}`}>
          <span className="pill-dot" />
          {backendUp === null ? "connecting…" : backendUp ? "running locally" : "offline"}
        </span>
      </header>

      {offline && (
        <Banner
          kind="error"
          title="Can't reach the assistant."
          body="The backend isn't responding. Start it with `uvicorn main:app --reload`, then reload this page."
          action={{ label: "Try again", onClick: refreshDocs }}
        />
      )}

      <div className="layout">
        <main className="main">
          <section className="composer" aria-label="Ask a question">
            <QuestionInput
              value={question}
              onChange={setQuestion}
              disabled={busy || offline}
              onSubmit={() => askText(question)}
            />

            <div className="toolbar">
              <button
                className="icon-btn"
                title="Add a document"
                aria-label="Add a document"
                disabled={busy || offline}
                onClick={() => fileInputRef.current?.click()}
              >
                <PlusIcon />
              </button>

              <div className="toolbar-right">
                <RagToggle value={useRag} onChange={setUseRag} disabled={busy} />
                <RecordButton
                  disabled={busy || offline}
                  onStage={setStage}
                  onError={setError}
                  onTranscript={(t) => {
                    setQuestion(t);
                    askText(t, t);
                  }}
                />
                <button
                  className="send-btn"
                  title="Ask (Enter)"
                  aria-label="Ask"
                  disabled={busy || offline || !question.trim()}
                  onClick={() => askText(question)}
                >
                  <ArrowIcon />
                </button>
              </div>
            </div>
          </section>

          {busy && (
            <div className="loading" role="status" aria-live="polite">
              <span className="spinner" />
              <span>{STAGE_LABEL[stage]}</span>
              <span className="elapsed">{elapsed.toFixed(1)}s</span>
            </div>
          )}

          {error && <Banner kind="error" title={error.title} body={error.hint} />}

          {backendUp && chunkCount === 0 && (
            <Banner
              kind="warn"
              title="No documents loaded yet."
              body="Add a document with the + button or the Documents panel, then ask a question about it."
            />
          )}

          {result ? (
            <>
              <AnswerPanel result={result} />
              {result.retrieved_chunks?.length > 0 && !isGateRefusal(result) && (
                <SourcePanel key={askId} result={result} />
              )}
            </>
          ) : (
            !busy && (
              <p className="empty">
                Ask a question about your documents. Every answer lists the
                passages it was drawn from, so you can check it.
              </p>
            )
          )}
        </main>

        <aside className="side">
          <DocumentPanel
            docs={docs}
            chunkCount={chunkCount}
            disabled={busy || offline}
            inputRef={fileInputRef}
            onStage={setStage}
            onError={setError}
            onUploaded={refreshDocs}
          />
          <SystemPanel health={health} />
        </aside>
      </div>
    </div>
  );
}

/* ====================================================================== */
/* FR3 — document upload                                                  */
/* ====================================================================== */
function DocumentPanel({ docs, chunkCount, disabled, inputRef, onStage, onError, onUploaded }) {
  const [dragging, setDragging] = useState(false);
  const [note, setNote] = useState(null);

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
      setNote(`Added “${data.filename}”: ${data.chunks_added} passages indexed.`);
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
    <section className="panel">
      <div className="panel-head">
        <h2>Documents</h2>
        <button
          className="icon-btn icon-btn-sm"
          title="Add a document"
          aria-label="Add a document"
          disabled={disabled}
          onClick={() => inputRef.current?.click()}
        >
          <PlusIcon />
        </button>
      </div>

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
        <strong>Drop a file here, or click to choose</strong>
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
          <p className="count">
            {docs.length} document{docs.length === 1 ? "" : "s"} · {chunkCount} passages indexed
          </p>
          <ul className="doclist">
            {docs.map((d) => (
              <li key={d} title={d}>
                <FileIcon />
                <span>{d}</span>
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

/* ====================================================================== */
/* The configuration every answer is produced under (read from /health)   */
/* ====================================================================== */
function SystemPanel({ health }) {
  if (!health) return null;
  const rows = [
    ["Language model", health.llm],
    ["Speech model", health.whisper && `Whisper ${health.whisper}`],
    ["Passages per answer", health.top_k],
    [
      "Refusal threshold",
      health.threshold_enabled ? `distance > ${health.distance_threshold}` : "off",
    ],
    ["Decoding", `temperature ${health.temperature}, seed ${health.seed}`],
  ].filter(([, v]) => v !== undefined && v !== null && v !== "");

  return (
    <section className="panel">
      <div className="panel-head">
        <h2>How answers are made</h2>
      </div>
      <dl className="specs">
        {rows.map(([k, v]) => (
          <div key={k}>
            <dt>{k}</dt>
            <dd>{String(v)}</dd>
          </div>
        ))}
      </dl>
    </section>
  );
}

/* ====================================================================== */
/* FR1 — text input                                                        */
/* ====================================================================== */
function QuestionInput({ value, onChange, disabled, onSubmit }) {
  return (
    <textarea
      className="question"
      rows={3}
      value={value}
      disabled={disabled}
      aria-label="Your question"
      placeholder="Ask about your documents, e.g. What does the Viterbi algorithm compute in a hidden Markov model?"
      onChange={(e) => onChange(e.target.value)}
      onKeyDown={(e) => {
        if (e.key === "Enter" && !e.shiftKey) {
          e.preventDefault();
          onSubmit();
        }
      }}
    />
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
          hint: "Press the microphone, speak, then press stop.",
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
      className={recording ? "record recording" : "icon-btn record"}
      disabled={disabled}
      title={recording ? "Stop and send" : "Ask by voice"}
      aria-label={recording ? "Stop and send" : "Ask by voice"}
      onClick={recording ? stop : start}
    >
      {recording ? (
        <>
          <span className="dot" />
          Stop and send
        </>
      ) : (
        <MicIcon />
      )}
    </button>
  );
}

/* ====================================================================== */
/* RAG toggle — drives the evaluation and the demonstration                */
/* ====================================================================== */
function RagToggle({ value, onChange, disabled }) {
  return (
    <label
      className={`toggle ${disabled ? "toggle-off" : ""}`}
      title={
        value
          ? "Answers come only from your documents"
          : "Answering from the model's own memory"
      }
    >
      <input
        type="checkbox"
        checked={value}
        disabled={disabled}
        onChange={(e) => onChange(e.target.checked)}
      />
      <span className="track">
        <span className="knob" />
      </span>
      <span className="toggle-text">{value ? "Use my documents" : "Model memory only"}</span>
    </label>
  );
}

/* A gate refusal still returns the nearest (distant) passages for logging,
   but the answer was not drawn from them, so they are not shown as sources. */
function isGateRefusal(result) {
  return (
    result?.refused === true &&
    result?.refusal_reason === "retrieval_distance" &&
    Array.isArray(result?.distances) &&
    result.distances.length > 0
  );
}

/* ====================================================================== */
/* FR5 — the answer                                                        */
/* ====================================================================== */
function AnswerPanel({ result }) {
  return (
    <section className="result">
      <h2>Answer</h2>

      {result.transcript && (
        <p className="heard">
          <span className="heard-label">Heard</span>“{result.transcript}”
        </p>
      )}

      {result.answer ? (
        <>
          <p className="answer">{result.answer}</p>
          <p className="meta">
            {result.refused ? (
              <span className="tag tag-refused">Declined: not in your documents</span>
            ) : (
              <span className={`tag ${result.use_rag ? "tag-grounded" : "tag-memory"}`}>
                {result.use_rag ? "Grounded in your documents" : "From the model's own memory"}
              </span>
            )}
            {typeof result.generate_s === "number" && result.generate_s > 0 && (
              <span>{result.generate_s}s</span>
            )}
          </p>
          {isGateRefusal(result) && (
            <p className="gate-note">
              Refused before the language model was asked: the closest passage in
              your documents is at distance{" "}
              <strong>{result.distances[0].toFixed(3)}</strong>, beyond the refusal
              threshold of <strong>{result.threshold ?? 0.53}</strong>. Nothing
              retrieved was close enough to answer from.
            </p>
          )}
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

  // Group passages by file, but keep each passage's retrieval rank (1 = closest
  // match) as its number. Files are ordered by their best-ranked passage, so the
  // closest evidence is still read first.
  const groups = [];
  const byName = new Map();
  chunks.forEach((chunk, i) => {
    const name = sources[i] || "unknown";
    if (!byName.has(name)) {
      const g = { name, passages: [] };
      byName.set(name, g);
      groups.push(g);
    }
    byName.get(name).passages.push({ rank: i + 1, chunk, distance: distances[i] });
  });

  return (
    <section className="result">
      <h2>Where this came from</h2>
      <p className="hint">
        These are the passages the assistant retrieved and was allowed to use,
        numbered by how closely they matched (1 = closest). Click a file name to
        collapse or expand it. If the answer is not supported by them, it is not grounded.
      </p>

      <div className="source-groups">
        {groups.map((g) => (
          // Open when a new answer arrives; the user can collapse any file.
          <details key={g.name} className="source-group" open>
            <summary className="source-name">
              <ChevronIcon />
              <FileIcon />
              <span className="source-file">{g.name}</span>
              <span className="source-ranks" aria-label="Retrieval ranks">
                {g.passages.map((p) => (
                  <span key={p.rank} className="source-rank">{p.rank}</span>
                ))}
              </span>
              <span className="source-count">
                {g.passages.length} passage{g.passages.length === 1 ? "" : "s"}
              </span>
            </summary>
            <ol className="sources">
              {g.passages.map((p) => (
                <li key={p.rank}>
                  <div className="source-head">
                    <span className="source-rank">{p.rank}</span>
                    {typeof p.distance === "number" && (
                      <span className="source-score" title="Lower is a closer match">
                        distance {p.distance.toFixed(3)}
                      </span>
                    )}
                  </div>
                  <p className="chunk">{p.chunk}</p>
                </li>
              ))}
            </ol>
          </details>
        ))}
      </div>
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

/* ---------------------------------------------------------------------- */
/* Icons: inline SVG, no icon library (nothing loaded from the network).  */
/* ---------------------------------------------------------------------- */
const svg = {
  width: 18,
  height: 18,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 2,
  strokeLinecap: "round",
  strokeLinejoin: "round",
  "aria-hidden": true,
};

function PlusIcon() {
  return (
    <svg {...svg}>
      <path d="M12 5v14M5 12h14" />
    </svg>
  );
}

function MicIcon() {
  return (
    <svg {...svg}>
      <rect x="9" y="3" width="6" height="11" rx="3" />
      <path d="M5 11a7 7 0 0 0 14 0M12 18v3" />
    </svg>
  );
}

function ArrowIcon() {
  return (
    <svg {...svg}>
      <path d="M12 19V5M5 12l7-7 7 7" />
    </svg>
  );
}

function ChevronIcon() {
  return (
    <svg {...svg} width={16} height={16} className="chevron">
      <path d="M9 6l6 6-6 6" />
    </svg>
  );
}

function FileIcon() {
  return (
    <svg {...svg} width={15} height={15}>
      <path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z" />
      <path d="M14 3v5h5" />
    </svg>
  );
}
