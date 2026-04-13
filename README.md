# ScienceQ

A RAG-based chatbot that answers questions grounded in YouTube science video transcripts. Ask anything about the pre-built corpus of 50 curated videos (English, Spanish, German, French, and Portuguese), or paste any YouTube URL to ingest and query it on the fly.

Built as the final project for the [Ironhack](https://www.ironhack.com) AI Engineering course.

**Live demo:** [scienceq.streamlit.app](https://scienceq.streamlit.app)

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
                                                      Groq LLM (llama-3.3-70b)
                                                               │
                                                               ▼
                                                     Streaming answer + sources
```

Full architecture details: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)  
Corpus and pipeline details: [`docs/DATASET.md`](docs/DATASET.md)

## Tech stack

| Layer | Technology |
|---|---|
| LLM | Groq — `llama-3.3-70b-versatile` |
| Embeddings | Cohere `embed-multilingual-v3.0` (1024d, asymmetric) |
| Reranker | Cohere Rerank v3.5 |
| Vector DB | Pinecone Serverless (cosine, AWS us-east-1) |
| Orchestration | LangChain LCEL + LangGraph |
| Tracing | LangSmith |
| UI | Streamlit |
| Deployment | Streamlit Community Cloud |

## Evaluation

Evaluated on 33 automated cases (20 English factual + 8 cross-lingual + 5 multi-turn) with GPT-4.1 as judge across four rubric dimensions (1–5). Five adversarial cases are excluded from automated scoring and reviewed manually.

| Checkpoint | Cases | Correctness | Tone | Grounding | Conciseness | Mean |
|---|---|---|---|---|---|---|
| Bootcamp — MiniLM, prompt v1 | 25 | 4.56 | 4.76 | 3.92 | 3.72 | 4.24 |
| Bootcamp — MiniLM, prompt v2 | 25 | 4.28 | 4.88 | **4.04** | **4.36** | 4.39 |
| Phase 3 — Cohere embeddings | 25 | — | — | — | — | — |
| Phase 4 — Cohere Rerank v3.5 | 25 | **4.40** | 4.84 | 3.64 | 4.12 | 4.25 |
| Phase 5 — tuned retrieval | 25 | 4.36 | **4.84** | 3.88 | 3.80 | 4.22 |
| Phase 6 — multilingual corpus | 33 | 4.38 | 4.62 | 3.62 | 4.25 | 4.22 |
| **prompt-v3 — grounding tightened** | **33** | **4.48** | 4.79 | 3.94 | 3.88 | **4.27** |

Phase 3 re-indexed the full corpus into a new embedding space (MiniLM → Cohere); scores are not comparable across that boundary. Phase 4 added the reranker. Phase 5 calibrated `RETRIEVER_FETCH_K`, `RETRIEVER_TOP_N`, and `SCORE_THRESHOLD` via a two-stage parameter sweep — details in [`docs/retrieval_sweep_results.md`](docs/retrieval_sweep_results.md). Phase 6 added 8 non-English videos (ES/DE/FR/PT); English queries surface non-English source pills in the UI alongside English results. prompt-v3 replaced the prohibition-framed grounding rule with a verification frame and an explicit inference ban — grounding +0.32 on multilingual cases, correctness +0.12 overall, no regressions.

---

## Running with Docker

**Prerequisites:** Docker installed, a `.env` file with your API keys (copy from `.env.example`).

```bash
docker build -t scienceq .
docker run --env-file .env -p 8501:8501 scienceq
```

Open [http://localhost:8501](http://localhost:8501).

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
├── app/                # Streamlit UI
├── data/               # metadata.json, per-video transcript/chunk/embedding files
├── docs/               # ARCHITECTURE.md, DATASET.md, retrieval_sweep_results.md
├── eval/               # Eval set, sweep scripts, LangSmith runner, results
├── pipeline/           # Corpus pipeline (extract → clean → chunk → embed → index)
├── tests/              # Unit tests
├── Dockerfile              # App image (Streamlit)
├── Dockerfile.pipeline     # Pipeline image (batch jobs)
├── docker-compose.yml
├── .env.example
├── requirements.txt        # Runtime (Streamlit Cloud)
└── requirements-dev.txt    # Full dev + pipeline + eval dependencies
```

---

## Known limitations

- Retrieval quality depends on transcript verbosity — visually-heavy videos without verbal explanation retrieve poorly
- Multi-turn pronoun resolution occasionally drifts on short follow-ups
- Live URL ingestion requires a video with available captions (auto-generated accepted). On Streamlit Community Cloud, a residential proxy is used to route transcript requests around YouTube's AWS IP blocks.

## Next steps

- Whisper integration for videos without captions
- Expand non-English corpus beyond the current ES/DE/FR/PT pilot (embed-multilingual-v3.0 supports 100+ languages)
- GPT-4.1 evaluation run for Phase 6 cross-lingual cases
