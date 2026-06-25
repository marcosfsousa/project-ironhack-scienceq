# ScienceQ

A RAG-based chatbot that answers questions grounded in YouTube science video transcripts. Ask anything about the pre-built corpus of science videos (English, Spanish, German, French, and Portuguese), or paste any YouTube URL to ingest and query it on the fly.

Built as the final project for the [Ironhack](https://www.ironhack.com) AI Engineering course.

**Live demo:** [scienceq-web-886463515307.europe-west1.run.app](https://scienceq-web-886463515307.europe-west1.run.app)

---

## Demo

![ScienceQ Demo](docs/demo.gif)

---

## What it does

- Answers factual questions from a corpus of 50 science explainer videos in 5 languages (English, Spanish, German, French, Portuguese) — Veritasium, Kurzgesagt, 3Blue1Brown, PBS Space Time, Big Think, CuriosaMente, Terra X Lesch, Science Étonnante, Ciência Todo Dia, and more
- Pastes a YouTube URL → ingests it in real time → answers questions about it
- Streams answers token by token with clickable source timestamp pills
- Maintains 5-turn conversation memory with query rewriting for follow-up questions
- Stays grounded: if no relevant chunks are found above the confidence threshold, it says so rather than hallucinating

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

## Evaluation

Evaluated on 33 automated cases (20 English factual + 8 cross-lingual + 5 multi-turn) with GPT-4.1 as judge across four rubric dimensions (1–5). Five adversarial cases are excluded from automated scoring and reviewed manually.

| Checkpoint | Cases | Correctness | Tone | Grounding | Conciseness | Mean |
|---|---|---|---|---|---|---|
| Bootcamp — MiniLM, prompt v1 | 25 | 4.56 | 4.76 | 3.92 | 3.72 | 4.24 |
| Bootcamp — MiniLM, prompt v2 | 25 | 4.28 | **4.88** | 4.04 | 4.36 | 4.39 |
| Phase 3 — Cohere embeddings | 25 | — | — | — | — | — |
| Phase 4 — Cohere Rerank v3.5 | 25 | 4.40 | 4.84 | 3.64 | 4.12 | 4.25 |
| Phase 5 — tuned retrieval | 25 | 4.36 | 4.84 | 3.88 | 3.80 | 4.22 |
| Phase 6 — multilingual corpus | 33 | 4.38 | 4.62 | 3.62 | 4.25 | 4.22 |
| prompt-v3 — grounding tightened | 33 | 4.48 | 4.79 | 3.94 | 3.88 | 4.27 |
| **gpt-oss-120b — model swap** | **33** | **4.64** | 4.67 | **4.58** | **4.79** | **4.67** |

Phase 3 re-indexed the full corpus into a new embedding space (MiniLM → Cohere); scores are not comparable across that boundary. Phase 4 added the reranker. Phase 5 calibrated `RETRIEVER_FETCH_K`, `RETRIEVER_TOP_N`, and `SCORE_THRESHOLD` via a two-stage parameter sweep — details in [`docs/retrieval_sweep_results.md`](docs/retrieval_sweep_results.md). Phase 6 added 8 non-English videos (ES/DE/FR/PT); English queries surface non-English source pills in the UI alongside English results. prompt-v3 replaced the prohibition-framed grounding rule with a verification frame and an explicit inference ban — grounding +0.32 on multilingual cases, correctness +0.12 overall, no regressions. gpt-oss-120b swapped the LLM from `llama-3.3-70b-versatile` to `openai/gpt-oss-120b` (via Groq) with no prompt or retrieval changes — the largest single-checkpoint gain in the table: grounding +0.64, conciseness +0.91, correctness +0.16, overall mean +0.40. Tone dipped -0.12. The two previously stuck multilingual cases (ml_007 microplastics, ml_008 neurodivergence) both recovered: grounding 2 → 5 and 2 → 4 respectively.

---

## Cloud Run API (Phase 1)

The FastAPI service is deployed on Google Cloud Run:

**Base URL:** `https://scienceq-api-886463515307.europe-west1.run.app`

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe → `{"status":"ok"}` |
| `/api/chat` | POST | RAG chat — blocking, returns answer, sources, intent |
| `/api/chat/stream` | POST | RAG chat — SSE streaming, tokens + `[SOURCES]` + `[DONE]` |
| `/api/catalog` | GET | Full video catalog (corpus + live-ingested) |
| `/api/ingest` | POST | Submit a YouTube URL for live ingestion → `job_id` |
| `/api/ingest/{job_id}` | GET | Poll ingest job status |
| `/docs` | GET | Swagger UI |

**Example request:**
```bash
curl -X POST https://scienceq-api-886463515307.europe-west1.run.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"How does natural selection work?","history":[]}'
```

The original Streamlit UI remains live at [scienceq.streamlit.app](https://scienceq.streamlit.app). The React SPA (Phase 4) is the primary interface going forward.

---

## Cloud Run Pipeline Job (Phase 2)

The corpus pipeline runs as a Cloud Run Job (`scienceq-pipeline`) backed by a GCS bucket (`gs://scienceq-data`) mounted via GCS FUSE at `/app/data`. The container filesystem is ephemeral, but the FUSE mount makes writes land directly in GCS — files produced by one execution persist for the next.

**To add new videos to the corpus:**

1. Append YouTube URLs to `gs://scienceq-data/video_urls.txt`
2. Trigger the job:

```bash
gcloud run jobs execute scienceq-pipeline \
  --project=scienceq-prod \
  --region=europe-west1 \
  --wait
```

The job runs all pipeline steps in order (`extract → clean → chunk → embed → bootstrap → enrich → index`). Steps are idempotent — already-processed videos are skipped automatically.

**To rebuild the pipeline image** after code changes:

```bash
gcloud builds submit \
  --project=scienceq-prod \
  --region=europe-west1 \
  --config=cloudbuild-pipeline.yaml \
  .
```

**To update the job definition** (memory, CPU, env vars, secrets):

```bash
gcloud run jobs replace cloudrun-pipeline-job.yaml \
  --project=scienceq-prod \
  --region=europe-west1
```

---

## Live URL Ingestion (Phase 3)

Paste any YouTube URL and ScienceQ ingests it on the fly — transcript extracted, chunked, embedded, and indexed into Pinecone's `live` namespace alongside the corpus.

Because YouTube blocks transcript requests from datacenter IPs, the ingestion pipeline routes through a residential proxy (`IPROYAL_PROXY_URL`). To avoid HTTP timeouts on what is a 60–90 second operation, the API uses an async fire-and-poll pattern:

| Endpoint | Method | Description |
|---|---|---|
| `/api/ingest` | POST | Submit a URL — returns `job_id` immediately (202) |
| `/api/ingest/{job_id}` | GET | Poll for status: `pending` → `complete` or `failed` |

**Example:**

```bash
# Submit
curl -X POST https://scienceq-api-886463515307.europe-west1.run.app/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.youtube.com/watch?v=VIDEO_ID"}'
# → {"job_id":"a1b2c3d4","status":"pending"}

# Poll until complete
curl https://scienceq-api-886463515307.europe-west1.run.app/api/ingest/a1b2c3d4
# → {"status":"complete","title":"...","channel":"...","topic":"...","chunk_count":12,...}
```

Video metadata (title, channel) is fetched via the YouTube oEmbed API — no auth, no proxy, ~80ms. The residential proxy is used only for transcript extraction via `youtube-transcript-api`.

---

## React SPA (Phase 4)

The production frontend is a React 18 + Vite + TypeScript SPA deployed as a separate Cloud Run service (`scienceq-web`) behind an nginx reverse proxy.

**Live URL:** `https://scienceq-web-886463515307.europe-west1.run.app`

### Features

- Token-by-token SSE streaming with a blinking cursor during generation
- Corpus browser sidebar — videos grouped by topic, live-ingested videos shown with a LIVE badge
- Ingest panel — click `+` or paste a YouTube URL in the composer; a slide-over panel shows live step progress and confirms chunk count on completion
- Source pills and inline video embed — clickable timestamp links open the YouTube video at the right second; the top source embeds directly below the answer
- Metadata queries return a grouped, topic-filtered list that matches the sidebar count (corpus + live namespace)
- Keyboard and screen-reader accessible — `aria-expanded`, `aria-controls`, roving `tabIndex` on accent selector, `aria-label` on icon-only buttons

### Local development

```bash
cd frontend
npm install
VITE_API_TARGET=http://localhost:8080 npm run dev
# → http://localhost:5173 (proxies /api/* to the local FastAPI server)
```

### Deploy

```bash
gcloud builds submit --config cloudbuild-web.yaml --project=scienceq-prod
```

The build is two-stage: `node:20-alpine` runs `vite build`, then `nginx:alpine` serves the static bundle. `${API_URL}` is substituted at container startup via `envsubst`. Both stages run as non-root users (`nginx` user in the serving stage).

---

## Running with Docker

Three Dockerfiles cover the three deployable services. All run as non-root users.

**Prerequisites:** Docker Desktop (Linux containers mode), a `.env` file with your API keys (copy from `.env.example`).

```bash
# FastAPI API (runs as appuser)
docker build -t scienceq-api .
docker run --env-file .env -p 8080:8080 scienceq-api
```

```bash
# Streamlit app (runs as appuser)
docker build -f Dockerfile.streamlit -t scienceq-streamlit .
docker run --env-file .env -p 8501:8501 scienceq-streamlit
```

```bash
# React SPA — nginx serves the Vite bundle (runs as nginx user)
docker build -f Dockerfile.web -t scienceq-web .
docker run -e API_URL=http://host.docker.internal:8080 -p 8081:8080 scienceq-web
# → http://localhost:8081 (requires the FastAPI API running on port 8080)
```

To verify non-root users in built images:
```bash
docker inspect scienceq-api --format '{{.Config.User}}'      # → appuser
docker inspect scienceq-streamlit --format '{{.Config.User}}' # → appuser
docker inspect scienceq-web --format '{{.Config.User}}'       # → nginx
```

---

## Running with Docker Compose

Docker Compose separates the serving layer (Streamlit app) from the batch pipeline into two containers that share a `./data` volume.

**Start the app:**
```bash
docker compose up app
```

**Run the full pipeline** (extract → clean → chunk → embed → bootstrap → enrich → index):
```bash
docker compose run pipeline --full
```

**Run individual pipeline steps:**
```bash
docker compose run pipeline --steps extract,clean,chunk
docker compose run pipeline --steps enrich
docker compose run pipeline --steps embed,index --force
```

The pipeline container uses the `pipeline` profile and does **not** start automatically with `docker compose up`.

---

## Quickstart (run locally)

**Prerequisites:** Python 3.11, a Pinecone account, a Groq API key, a Cohere API key, a LangSmith account.

```bash
git clone https://github.com/marcosfsousa/project-ironhack-scienceq
cd project-ironhack-scienceq

pip install -r requirements.txt

cp .env.example .env
# Fill in your API keys in .env

streamlit run app/streamlit_app.py
```

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

## Building the corpus from scratch

If you want to index your own set of videos, run the pipeline steps in order:

```bash
pip install -r requirements-dev.txt

# Run the full pipeline (edit data/video_urls.txt with your URLs first)
python -m pipeline.run --full

# Or run individual steps
python -m pipeline.run --steps extract,clean,chunk
python -m pipeline.run --steps embed,index
python -m pipeline.run --steps embed,index --force   # re-process already-done videos
```

**Manual step between `chunk` and `embed`:** after `bootstrap` runs, open `data/metadata.json` and verify/fill in `title`, `channel`, and `topic` for each video. The `enrich` step auto-fills these via the YouTube Data API + Groq, but review the output before indexing.

Each individual script also supports `--video-id` to run on a single video. See the docstring at the top of each file for full CLI options.

The chunks will be indexed in the namespace set under `PINECONE_NAMESPACE_CORPUS` in your `.env` file.

### Running tests

```bash
python tests/run_all_tests.py
```

76 unit tests, no live API calls required, full run in ~4s.

---

## Project structure

```
├── agent/              # LangGraph agent, RAG chain, retriever, tools, memory, prompts
├── api/                # FastAPI service (Cloud Run) — main, routes, service, schemas, catalog
├── app/                # Streamlit UI (original frontend)
├── data/               # metadata.json, per-video transcript/chunk/embedding files
├── docs/               # ARCHITECTURE.md, DATASET.md, retrieval_sweep_results.md, PHASE4_PLAN.md
├── eval/               # Eval set, sweep scripts, LangSmith runner, results
├── frontend/           # React SPA — src/, nginx.conf.template, Vite + Tailwind config
├── pipeline/           # Corpus pipeline (extract → clean → chunk → embed → index)
├── tests/              # Unit tests
├── Dockerfile                   # FastAPI API image (Cloud Run serving, non-root appuser)
├── Dockerfile.streamlit         # Streamlit app image (Streamlit Cloud, non-root appuser)
├── Dockerfile.pipeline          # Pipeline image (Cloud Run Job)
├── Dockerfile.web               # React SPA — nginx + Vite build (Cloud Run serving, non-root nginx)
├── cloudbuild-pipeline.yaml     # Cloud Build config for pipeline image
├── cloudbuild-web.yaml          # Cloud Build config for scienceq-web (React SPA)
├── cloudrun-pipeline-job.yaml   # Cloud Run Job definition (Phase 2)
├── docker-compose.yml
├── .env.example
├── requirements.txt             # Runtime deps (shared by API + Streamlit Cloud)
└── requirements-dev.txt         # Full dev + pipeline + eval dependencies
```

---

## Known limitations

- Retrieval quality depends on transcript verbosity — visually-heavy videos without verbal explanation retrieve poorly
- Multi-turn pronoun resolution occasionally drifts on short follow-ups
- Live URL ingestion requires a video with available captions (auto-generated accepted). Both the Streamlit and Cloud Run deployments route transcript requests through a residential proxy to bypass YouTube's datacenter IP blocks.

## Next steps

- Citation pill rendering in streamed text — pills currently appear correctly below the answer but render as raw `[Title, mm:ss]` markers within the streamed text bubble
- Conversation-aware retrieval — follow-up questions currently embed independently; incorporating prior turns into the retrieval query would improve pronoun resolution and topic continuity
- Whisper integration for videos without captions
- Expand non-English corpus beyond the current ES/DE/FR/PT pilot (embed-multilingual-v3.0 supports 100+ languages)
