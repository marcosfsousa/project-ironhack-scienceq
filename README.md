# ScienceQ

[![CI](https://github.com/marcosfsousa/project-ironhack-scienceq/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/marcosfsousa/project-ironhack-scienceq/actions/workflows/ci.yml)

A RAG-based chatbot that answers questions grounded in YouTube science video transcripts. Ask anything about the pre-built corpus of science videos (English, Spanish, German, French, and Portuguese), or paste any YouTube URL to ingest and query it on the fly.

Built as the final project for the [Ironhack](https://www.ironhack.com) AI Engineering course.

**Live demo:** [scienceq.app](https://scienceq.app)

---

## Demo

<p align="center">
  <img src="docs/scienceq_app_demo_v2.gif" alt="ScienceQ demo" width="780" />
</p>

---

## What it does

- Answers factual questions from a corpus of 50 science explainer videos in 5 languages (English, Spanish, German, French, Portuguese) — Veritasium, Kurzgesagt, 3Blue1Brown, PBS Space Time, Big Think, CuriosaMente, Terra X Lesch, Science Étonnante, Ciência Todo Dia, and more
- Pastes a YouTube URL → ingests it in real time → answers questions about it
- Streams answers token by token with clickable source timestamp pills
- Maintains 5-turn conversation memory with query rewriting for follow-up questions
- Stays grounded: if no relevant chunks are found above the confidence threshold, it says so rather than hallucinating
- Discloses that it is an AI on three surfaces, as EU AI Act Art. 50(1) requires: a landing-page disclosure before first interaction, an `AI` badge on every generated answer (screen-reader labelled), and a deterministic `identity` intent that answers "are you a bot?" honestly without invoking the LLM — see [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md)

---

## Architecture

```
User query
    │
    ▼
LangGraph Agent  ── keyword routing ──►  RAG chain  ──►  Pinecone top-10 (corpus + live)
                                    │                          │
                                    └──►  Metadata tool        ▼
                                                      Cohere Rerank v3.5
                                                               │
                                                               ▼ top-3
                                                      OpenAI gpt-oss-120b (via Groq)
                                                               │
                                                               ▼
                                                     Streaming answer + sources
```

Full architecture details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)  
Corpus and pipeline details: [`docs/DATASET.md`](docs/DATASET.md)

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | OpenAI `gpt-oss-120b` (answers, via Groq), `gpt-oss-20b` (rewriting, via Groq) |
| Embeddings | Cohere `embed-multilingual-v3.0` (1024d, asymmetric) |
| Reranker | Cohere Rerank v3.5 |
| Vector DB | Pinecone Serverless (cosine, AWS us-east-1) |
| Orchestration | LangChain LCEL + LangGraph |
| Tracing | LangSmith |
| UI | React 18 + Vite + TypeScript + Tailwind CSS |
| Web server | nginx (reverse proxy + SPA fallback) |
| Deployment | Google Cloud Run (API + React SPA as separate services) |

---

## Evaluation

Best checkpoint (gpt-oss-120b): **mean 4.67/5** across 33 cases — correctness 4.64, grounding 4.58, conciseness 4.79, tone 4.67.

Full results and per-checkpoint breakdown: [`docs/EVALUATION.md`](docs/EVALUATION.md)

---

## Quickstart (run locally)

**Prerequisites:** Python 3.11, a Pinecone account, a Groq API key, a Cohere API key, a LangSmith account.

```bash
git clone https://github.com/marcosfsousa/project-ironhack-scienceq
cd project-ironhack-scienceq

pip install -r requirements.txt

cp .env.example .env
# Fill in your API keys in .env

uvicorn api.main:app --reload --port 8080
```

That serves the API on `http://localhost:8080`. To run the React SPA against it, see
[`docs/FRONTEND.md`](docs/FRONTEND.md).

### Required environment variables

```
GROQ_API_KEY=
COHERE_API_KEY=
PINECONE_API_KEY=
PINECONE_INDEX_NAME=scienceq
PINECONE_NAMESPACE_CORPUS=corpus
PINECONE_NAMESPACE_LIVE=live
LANGSMITH_API_KEY=
LANGSMITH_PROJECT=scienceq
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGCHAIN_TRACING_V2=true
RERANKER_ENABLED=true
RETRIEVER_FETCH_K=10
RETRIEVER_TOP_N=3
SCORE_THRESHOLD=0.25
```

---

## Project structure

```
├── agent/              # LangGraph agent, RAG chain, retriever, tools, memory, prompts
├── api/                # FastAPI service (Cloud Run) — main, routes, service, schemas, catalog
├── data/               # metadata.json, per-video transcript/chunk/embedding files
├── docs/               # ARCHITECTURE.md, BACKEND.md, FRONTEND.md, EVALUATION.md, DATASET.md, COMPLIANCE.md
├── eval/               # Eval set, sweep scripts, LangSmith runner, results
├── frontend/           # React SPA — src/, nginx.conf.template, Vite + Tailwind config
├── pipeline/           # Corpus pipeline (extract → clean → chunk → embed → index)
├── tests/              # Unit tests
├── Dockerfile                   # FastAPI API image (Cloud Run serving, non-root appuser)
├── Dockerfile.pipeline          # Pipeline image (Cloud Run Job)
├── Dockerfile.web               # React SPA — nginx + Vite build (Cloud Run serving, non-root nginx)
├── cloudbuild-pipeline.yaml     # Cloud Build config for pipeline image
├── cloudbuild-web.yaml          # Cloud Build config for scienceq-web (React SPA)
├── cloudrun-pipeline-job.yaml   # Cloud Run Job definition
├── docker-compose.yml
├── .env.example
├── requirements.txt             # Runtime deps (API + live ingest)
└── requirements-dev.txt         # Full dev + pipeline + eval dependencies
```

---

## Docs

| Document | Contents |
|---|---|
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | Agent, RAG chain, retrieval pipeline, memory, SSE protocol |
| [`docs/BACKEND.md`](docs/BACKEND.md) | API endpoints, Cloud Run deployment, corpus pipeline, Docker |
| [`docs/FRONTEND.md`](docs/FRONTEND.md) | React SPA features, local dev, nginx, Docker |
| [`docs/EVALUATION.md`](docs/EVALUATION.md) | Eval methodology, rubric, checkpoint results |
| [`docs/DATASET.md`](docs/DATASET.md) | Corpus videos, pipeline steps, metadata schema |
| [`docs/COMPLIANCE.md`](docs/COMPLIANCE.md) | EU AI Act Article 50 obligations, status, feature tripwires |

---

## Known limitations

- Retrieval quality depends on transcript verbosity — visually-heavy videos without verbal explanation retrieve poorly
- Multi-turn pronoun resolution occasionally drifts on short follow-ups
- Live URL ingestion requires a video with available captions (auto-generated accepted). Transcript requests are made directly, using only access methods YouTube permits; the residential proxy that previously worked around datacenter IP restrictions was removed in issue #16.

## Next steps

- Whisper integration for videos without captions
- Expand non-English corpus beyond the current ES/DE/FR/PT pilot (embed-multilingual-v3.0 supports 100+ languages)
