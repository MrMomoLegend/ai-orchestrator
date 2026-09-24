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

The system is evaluated on a 30-question set against the public corpus in `docs/`
(15 answerable, 12 out-of-corpus, 3 related-but-unanswerable), with retrieval on and off,
and on SQuAD 2.0 (n=500). The retrieval threshold (0.53) and k (10) were fixed on an
earlier development corpus and not re-tuned, so both evaluations are out-of-sample.
Results and ablations are in `results/` and Chapter 5 of the report.

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

Requires Python 3.12+ (the pinned `numpy==2.5.0` has no 3.11 wheel), Node 18+, and
[Ollama](https://ollama.com). The first `python ingest.py` or `pytest` run downloads the
all-MiniLM-L6-v2 embedding model from Hugging Face, so it needs internet once.

```bash
git clone https://github.com/MrMomoLegend/ai-orchestrator.git
cd ai-orchestrator

python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on macOS/Linux
pip install -r requirements.txt

ollama pull llama3.1:8b

# docs/ already holds the evaluation corpus; add your own .txt/.md/.pdf alongside it
python ingest.py               # builds both chunking collections

uvicorn main:app --reload      # backend on :8000
cd frontend && npm install && npm run dev   # frontend on :5173
```

`docs/` ships with the 22 openly licensed PDFs the evaluation uses, so `python ingest.py`
works on a fresh clone. `docs/README.md` lists their sources and licence.

## Configuration

All defaults are set in `main.py` and overridable by environment variable.

| Variable | Default | Notes |
|---|---|---|
| `LLM_MODEL` | `llama3.1:8b` | |
| `TOP_K` | `10` | Set on the development corpus; re-tested frozen in Section 5.6 |
| `DISTANCE_THRESHOLD` | `0.53` | Cosine. Midpoint of the development-corpus separating margin (Section 4.5) |
| `USE_THRESHOLD` | `1` | |
| `COLLECTION` | `sentence` | `sentence` or `fixed` — the chunking ablation |
| `WHISPER_SIZE` | `base.en` | |

## Reproducing the evaluation

Every number in Chapter 5 is produced by a script and none by hand. The backend must be
running (`uvicorn main:app`) with the corpus ingested (`python ingest.py --reset`).

```bash
python eval/exp_rag.py --sweep                 # Appendix D.4  threshold sweep (no LLM calls)
python eval/exp_rag.py --experiment rag        # 5.3  RAG on/off
python eval/exp_rag.py --experiment threshold  # 5.4  threshold ablation
python eval/exp_rag.py --experiment chunking   # 5.5  chunking ablation
python eval/exp_rag.py --experiment topk       # 5.6  top-k sweep
python asr_eval/evaluate_asr.py                # 5.7  Whisper vs Vosk
python eval/exp_retrieval.py                   # 5.8  precision@k, scored from the committed labels
python eval/exp_latency.py --k 1,3,5,10        # 5.9  latency: run alone, nothing else on the GPU
python eval/exp_squad.py --n 500               # 5.2  SQuAD 2.0 (seed 42)
python eval/make_figures_v3.py                 # Figures 5.1-5.5
pytest                                         # 66 unit tests, ~3 s, no Ollama needed
pytest --cov=main --cov=ingest --cov-branch    # the same, with branch coverage (Section 4.7)
```

On a fresh clone `pytest` reports 64 passed and 2 skipped: the live-collection
distance-metric check needs `chroma_db/`, so run `python ingest.py` first to get all 66.

Results land in `results/` as CSV. **Every script overwrites its own CSVs**, so copy
`results/` before a rerun, and never rerun `exp_retrieval.py --label` (it rebuilds the
labelling worksheet). `results/raw_prescore/` holds the pre-labelling copies;
`results/old_corpus/`, `k3_archive/` and `k10_stochastic/` are development-corpus runs; the
last two are the evidence for the decoding-variance finding in Section 5.11. `make_figures.py` and
`make_figures_v2.py` are superseded (development corpus).

## Layout

```
main.py              orchestration backend, all six endpoints, threshold logic
ingest.py            extraction, chunking, embedding — builds both collections
frontend/            React single-page interface
eval/                experiment scripts and the 30-question set
asr_eval/            self-contained speech-recognition evaluation
results/             CSVs and figures behind every Chapter 5 claim
archive/             superseded prototype scripts, kept for provenance
```

## Licence and third-party material

Source code in this repository is the author's own work.

The PDFs in `docs/` are Wikipedia articles and sections of *Dive into Deep Learning*
(d2l.ai), redistributed unmodified under CC BY-SA 4.0 — see `docs/README.md`. Parameters
were developed against an earlier course-reading corpus that is not redistributable and is
not included in this repository or its history. Model weights (Llama 3.1, Whisper, Vosk,
all-MiniLM-L6-v2) are downloaded at setup from their own sources under their own licences.
