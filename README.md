# BT Support Assistant

Source-grounded IT support assistant prototype with hybrid retrieval and
window-based anomaly detection.

## Project Scope

The project combines two main capabilities:

- Hybrid retrieval over IT support tickets and knowledge-base documents using
  BM25, sentence embeddings, and FAISS.
- Window-based anomaly detection using ticket volume, category distribution,
  and semantic drift signals.

The repository includes the application code, data preparation and indexing
scripts, evaluation scripts, tests, FastAPI backend, and web interface.

This project was initiated in contact with Ozdilek Holding under the TUBITAK
2209-B program. However, no Ozdilek operational data is included in the
repository. Development and evaluation use synthetic, converted, and
open-source-supported test data. The reported results do not represent
Ozdilek's production environment.

## Installation

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Create the local environment file:

```powershell
Copy-Item .env.example .env
```

The `.env` file is ignored by Git and must not be committed.

## Data And Index Preparation

Generated datasets and indexes are intentionally excluded from the repository:

- `data/`
- `indexes/`
- `.env`
- cache, log, and test-output folders

Prepare the processed data:

```powershell
python scripts\prepare_data_for_indexing.py
```

Build the complete BM25 and FAISS indexes:

```powershell
python data_pipeline\build_indexes.py
```

For a faster local demo, an optional document limit can be used:

```powershell
python data_pipeline\build_indexes.py --limit 10000
```

The embedding model must either be available in the local Hugging Face cache
or be downloaded before running the offline demo configuration.

## Industry Ticket Export Starter

The project includes a system-agnostic starter package for future industrial
ticket data collection:

```text
integrations/ticket_collector/
```

This package reads exported CSV files from an external ticket system,
normalizes them into a common schema, masks common PII patterns, and produces
standard CSV/JSONL outputs plus a quality report. It does not connect to or
modify a live ticket system. A platform-specific read-only API adapter can be
added later after the organization's ticket system and export/API capabilities
are known.

## Running The Application

Start the FastAPI backend and web interface:

```powershell
python scripts\run_server.py
```

With the default `.env.example` configuration, open:

- Web interface: `http://127.0.0.1:8010/`
- API documentation: `http://127.0.0.1:8010/docs`
- Health check: `http://127.0.0.1:8010/health`

The application can operate with its retrieval-based fallback response
generator when `USE_REAL_LLM=false`. To enable OpenAI-based response
generation, set `USE_REAL_LLM=true` and provide `OPENAI_API_KEY` in `.env`.
API keys must never be committed.

## Evaluation

Run retrieval evaluation:

```powershell
python scripts\evaluate_retrieval.py
```

Run anomaly evaluation:

```powershell
python scripts\evaluate_anomaly.py
```

Run the automated test suite:

```powershell
python -m pytest
```

Core test suite without live-server chat scenarios:

```text
134 passed, 3 warnings
```

The live-server chat scenarios in `tests/test_chat_scenarios.py` call a
running `localhost:8000` application and can vary with the active local
server, indexes, and `.env` settings. The warnings are Pydantic deprecation
warnings and do not indicate a failing application test.

## Evaluation Summary And Limitations

The final retrieval comparison used 180 queries over the same
34,975-document index.

Results for the 80 general category/subcategory queries:

- Recall@5: 0.6875
- Recall@10: 0.8000
- nDCG@10: 0.5519
- Category Hit@5: 0.9125
- Subcategory Hit@5: 0.6875

The 100 ticket-specific queries achieved Exact Recall@5 of 0.9800. These
queries were generated from the target ticket text, and the target ticket was
not removed from the candidate index. Therefore, the exact-ticket result is a
same-record retrieval test and must not be interpreted as independent
generalization performance. The general category/subcategory results provide
the more conservative estimate for ordinary user queries.

The anomaly validation produced:

- Event precision: 0.8605
- Event recall: 0.9487
- Event F1: 0.9024
- Day-level specificity: 0.9469

The anomaly dataset is synthetic and injects volume, category, and semantic
patterns that correspond to the signals analyzed by the detector. These
metrics measure detectability under controlled validation conditions, not
performance on real organizational traffic.

Generated answers are source-grounded and follow a "no source, no answer"
policy. However, answer faithfulness, hallucination rate, and expert-rated
correctness were not independently measured. The displayed context
sufficiency score is a retrieval-oriented heuristic, not a calibrated
probability that an answer is correct.

## Important Paths

- Backend routes: `app/routers/`
- Retrieval logic: `core/retrieval/`
- RAG pipeline: `core/rag/`
- Anomaly engine: `core/anomaly/`
- Data preparation: `scripts/prepare_data_for_indexing.py`
- Index building: `data_pipeline/build_indexes.py`
- Retrieval evaluation: `scripts/evaluate_retrieval.py`
- Anomaly evaluation: `scripts/evaluate_anomaly.py`
- Web interface: `frontend/`
- Final metrics and report visuals: `reports/`
