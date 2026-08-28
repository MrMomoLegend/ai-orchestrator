# A Locally-Deployed AI Orchestration System for Multimodal Task Processing and Knowledge Retrieval

BSc Computer Science final project (CM3070), University of London.
**Author:** Mokith Maheshkumar · 230655577 · **Supervisor:** Dr Andrew Yoong

A private, offline AI assistant that orchestrates three pre-trained models across three
data modalities and answers questions from your own documents — or declines to answer
when they do not contain the answer. Every component runs on the local machine. No
request leaves it at any point.

| Modality | Model | Role |
|---|---|---|
| Audio | faster-whisper (`base.en`, int8) | Speech to text |
| Text | Llama 3.1 (8B) via Ollama | Language understanding and generation |
| Vector | all-MiniLM-L6-v2 + ChromaDB | Semantic retrieval over the document corpus |

## The result

Over a 30-question set on a 3,215-chunk corpus:

| Configuration | Unsupported answers to unanswerable questions | Correct answers |
|---|---|---|
| Retrieval **off** | 15 / 15 (100%) | 15 / 15 |
| Retrieval **on** (k=10) | **0 / 15 (0%)** | 14 / 15 |

Grounding did more than suppress answers. Asked who first implemented the parser behind
the CKY algorithm, the ungrounded model invented three researchers and three dates; the
grounded system returned *"John Cocke, 1960"* with the source passage attached.

Three ablations qualify that headline, and two of them returned negative results that are
reported as measured rather than quietly dropped — see Chapter 5 of the report.

## Architecture

```
                    ┌─────────────────────────────────┐
  voice / text  ──► │  React frontend (Vite)          │
  document upload   └──────────────┬──────────────────┘
                                   ▼
                    ┌─────────────────────────────────┐
                    │  FastAPI orchestrator           │
                    │  answer_question()              │
                    └──┬────────────┬──────────────┬──┘
                       ▼            ▼              ▼
                  ┌─────────┐  ┌──────────┐  ┌───────────┐
                  │ Whisper │  │ ChromaDB │  │ Llama 3.1 │
                  │  speech │  │ retrieval│  │  via Ollama│
                  └─────────┘  └──────────┘  └───────────┘
                                   │
                    if nearest cosine distance > 0.53
                    → refuse before the model is called
```

Orchestration is written directly in FastAPI rather than delegated to LangChain, so every
stage of the pipeline is visible in one function. That is a deliberate trade — no retry
logic, no streaming, no caching — made so the model integration is inspectable.

## Endpoints

| Endpoint | Purpose |
|---|---|
| `POST /ask` | Text question. Optional `top_k`, `use_threshold`, `threshold`, `collection` |
| `POST /transcribe` | Audio → transcript |
| `POST /ask/audio` | Spoken question in a single request |
| `POST /upload` | Add a document to the corpus |
| `GET /documents` | Corpus state |
| `GET /health` | Liveness and active configuration |

## Setup

Requires Python 3.11+, Node 18+, and [Ollama](https://ollama.com).

```bash
git clone https://github.com/MrMomoLegend/ai-orchestrator.git
cd ai-orchestrator

python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt

ollama pull llama3.1:8b

# Add your own documents (.txt, .md, .pdf) to docs/ — see docs/README.md
python ingest.py               # builds both chunking collections

uvicorn main:app --reload      # backend on :8000
cd frontend && npm install && npm run dev   # frontend on :5173
```

The corpus this project was evaluated against is **not distributed with the repository**;
`docs/README.md` explains what to put there and why.

## Configuration

All defaults are set in `main.py` and overridable by environment variable.

| Variable | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `llama3.1:8b` | |
| `TOP_K` | `10` | Set by the Section 5.5 sweep, not inherited |
| `DISTANCE_THRESHOLD` | `0.53` | Cosine. Midpoint of the separating margin |
| `USE_THRESHOLD` | `1` | |
| `COLLECTION` | `sentence` | `sentence` or `fixed` — the chunking ablation |
| `WHISPER_SIZE` | `base.en` | |

## Reproducing the evaluation

Every number in Chapter 5 is produced by a script and none by hand.

```bash
python eval/exp_rag.py --experiment rag        # 5.2  RAG on/off
python eval/exp_rag.py --experiment threshold  # 5.3  threshold ablation
python eval/exp_rag.py --experiment chunking   # 5.4  chunking ablation
python eval/exp_rag.py --experiment topk       # 5.5  top-k sweep
python asr_eval/evaluate_asr.py                # 5.6  Whisper vs Vosk
python eval/make_figures_v2.py                 # all Chapter 5 figures
```

Results land in `results/` as CSV. The `rescored_*.csv` files are the ones the report
uses; `results/raw_prescore/` holds the pre-correction versions, kept because Section 4.8
documents a ground-truth relabelling and the audit trail matters.

Sections 5.7–5.10 (`exp_retrieval.py`, `exp_latency.py`, `exp_squad.py`, and the usability
study) are written and scheduled for the week 19–22 improvement phase.

## Layout

```
main.py              orchestration backend, all six endpoints, threshold logic
ingest.py            extraction, chunking, embedding — builds both collections
frontend/            React single-page interface
eval/                experiment scripts and the 30-question set
asr_eval/            self-contained speech-recognition evaluation
results/             CSVs and figures behind every Chapter 5 claim
report/              report source (docx build) and the built deliverable
archive/             superseded prototype scripts, kept for provenance
```

## Licence and third-party material

Source code in this repository is the author's own work.

The evaluation corpus is **not** included. It consisted of reading-list extracts from
published textbooks — course textbook, course textbook, course textbook, course textbook — supplied through the module and used locally under fair dealing for private study
and research. Redistributing them is not permitted, so they are excluded from this
repository and from its history. Model weights (Llama 3.1, Whisper, Vosk,
all-MiniLM-L6-v2) are downloaded at setup from their own sources under their own licences.
