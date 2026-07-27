# Architecture — ScienceQ

## Overview

ScienceQ is a retrieval-augmented generation (RAG) system that answers questions grounded in YouTube video transcripts. It combines a curated offline corpus with on-the-fly live URL ingestion. The primary interface is a React SPA served by a Cloud Run nginx container, backed by a FastAPI service on a separate Cloud Run instance. It is the only frontend — the original Streamlit UI was sunset in issue #13.

---

## System Diagram

```
User (Browser)
     │
     ▼
React SPA  (frontend/src/)
nginx reverse proxy  (Dockerfile.web / scienceq-web Cloud Run)
     │
     ├── Paste YouTube URL ──► POST /api/ingest ──► FastAPI  (scienceq-api Cloud Run)
     │                                                   │
     │                                            Live Ingest Pipeline
     │                                            youtube-transcript-api
     │                                            cleaner → chunker → embedder
     │                                                   │
     │                                                   ▼
     │                                            Pinecone [live namespace]
     │
     ├── GET /api/catalog ──────────────────────► FastAPI
     │                                                   │
     │                                        GCS metadata.json (corpus)
     │                                      + Pinecone list/fetch (live)
     │
     └── Ask a question ──► POST /api/chat/stream ──► FastAPI
                                                          │
                                                    LangGraph Agent  (agent/agent.py)
                                                          │
                                          ┌───────────────┼───────────────┐
                                          ▼               ▼               ▼
                                    RAG intent    Metadata intent   Ingest intent
                                          │               │
                                          ▼               ▼
                                 RAGRetrieverTool  VideoMetadataTool
                                 (rag_chain.py)    (metadata.json + Pinecone live)
                                          │
                                          ▼
                                 Pinecone similarity search
                                 [corpus + live namespaces]
                                          │
                                          ▼
                                 OpenAI gpt-oss-120b  (via Groq)
                                          │
                                          ▼
                                 SSE token stream → React SPA
                                 [SOURCES] frame + [DONE] frame
                                          │
                                          ▼
                                 LangSmith  (tracing + evaluation)
```

---

## Components

### React SPA (`frontend/src/`)

The primary frontend — a React 18 + Vite + TypeScript SPA served by nginx on Cloud Run (`scienceq-web`). Key behaviours:

- **SSE streaming** — `lib/sse.ts` opens a `fetch` stream to `POST /api/chat/stream`, accumulates `data:` lines within each SSE event per spec, and dispatches one `onToken` call per event so multi-line responses (e.g. metadata lists) arrive with newlines intact
- **Corpus browser** — `useCatalog` fetches `GET /api/catalog` on mount; videos are deduplicated by `video_id`, grouped by topic, and rendered in a collapsible sidebar; live-ingested videos appear with a LIVE badge
- **Ingest panel** — `useIngest` opens an idle panel (URL input), calls `POST /api/ingest`, then polls `GET /api/ingest/:job_id` with a presentational stepper until complete or failed
- **Source pills + video embed** — `lib/citations.tsx` parses `[Title, mm:ss]` markers in the streamed answer; the top source embeds as a YouTube iframe directly below the answer
- **Accessibility** — aria roles, `aria-expanded`, `aria-controls`, roving `tabIndex` on accent selector, `aria-label` on icon-only buttons

nginx uses `proxy_set_header Host $proxy_host` (not `$host`) so Cloud Run routes the proxied `/api/*` requests by the upstream hostname rather than the frontend's hostname.

### FastAPI Service (`api/`)

The Cloud Run API service (`scienceq-api`) is the stateless seam between the React SPA and the LangGraph agent.

- **`POST /api/chat/stream`** — builds a fresh agent per request, replays client-supplied history into `ConversationMemory`, then streams tokens via `StreamingResponse(media_type="text/event-stream")`. Multi-line tokens are encoded as multiple consecutive `data:` lines within one SSE event; `[SOURCES]` and `[DONE]` frames close the stream
- **`GET /api/catalog`** — merges corpus metadata from GCS (`gs://scienceq-data/metadata.json`) with live videos fetched from Pinecone's `live` namespace via `list()` + `fetch()` on `*_000` chunk IDs, deduplicates by `video_id`, and returns the combined list
- **`POST /api/ingest` / `GET /api/ingest/:job_id`** — fire-and-poll pattern for live ingestion (see Phase 3)
- **CORS** locked to the exact `scienceq-web` Cloud Run URL + `http://localhost:5173` for local dev

### LangGraph Agent (`agent/agent.py`)

A compiled LangGraph state machine with four nodes:

```
START → classify_intent → [rag | metadata | ingest] → respond → END
```

Intent routing is zero-cost keyword matching — no LLM call required. The agent maintains a 5-turn sliding window conversation memory via a custom `ConversationMemory` class (no LangChain community dependency).

### RAG Chain (`agent/rag_chain.py`)

Built with LangChain LCEL. Two execution paths:

- **Blocking** (`answer()`) — used by the agent graph and eval runner
- **Streaming** (`stream_answer()`) — yields tokens to the agent's `stream_chat()`, which `api/service.py` forwards as SSE

Both paths share the same prompt template from `prompts.py`. The chain applies a `score_threshold=0.40` gate: queries scoring below this on all retrieved chunks receive a no-context fallback response rather than a hallucinated answer. Groq 429 rate limit errors are handled with exponential backoff (3 attempts, 2s → 4s → 8s).

### Retriever (`agent/retriever.py`)

Wraps Pinecone with namespace-aware querying. Embeddings are generated using Cohere `embed-multilingual-v3.0` (1024 dimensions, cosine similarity) with separate `input_type` values for indexing (`search_document`) and querying (`search_query`) — true asymmetric retrieval. The retriever supports:

- Single namespace queries (`corpus` or `live`)
- Multi-namespace merge (`retrieve_multi_namespace`) — embeds the query once and fans out to both namespaces, then merges and returns top-k globally by score
- Optional metadata filters by topic or channel
- Optional Cohere Rerank layer (see below)
- **Cross-lingual retrieval** — the shared multilingual embedding space means English queries retrieve semantically relevant chunks in Spanish, German, French, Portuguese, and any other Cohere-supported language without additional configuration. Each `RetrievedChunk` carries a `language` field populated from Pinecone metadata, surfaced in source citations.

**Retrieval flow with reranker enabled:**

```
query
  → Cohere embed (search_query, 1024d)
  → Pinecone top_k=10 (over-retrieve)
  → Cohere Rerank v3.5 (cross-encoder re-scoring)
  → top_n=5 passed to LLM
```

Toggled via `RERANKER_ENABLED=true/false` in `.env`. When on, `retrieve()` always fetches `RERANKER_FETCH_K=10` candidates from Pinecone regardless of the caller's `top_k`, then lets Cohere Rerank filter down to the requested count. For multi-namespace queries, 10 candidates are fetched from each namespace (20 total) before reranking.

Each `RetrievedChunk` carries both `score` (Pinecone cosine similarity) and `rerank_score` (Cohere relevance score, `None` when reranker is off), making pre/post ordering visible in LangSmith traces and UI source citations.

### Tools (`agent/tools.py`)

Two tools registered with the agent:

**RAGRetrieverTool** — answers factual questions by calling the RAG chain. Always tried first.

**VideoMetadataTool** — answers catalog queries ("what videos do you have on physics?"). Uses a three-pass matching strategy: exact match on topic/title/channel first, then a loose word match restricted to the topic field only to prevent cross-topic contamination. Merges two sources before filtering: `metadata.json` (corpus, loaded from disk) and the Pinecone `live` namespace (fetched via `list()` + `fetch()` on `*_000` chunk IDs), so the metadata list matches the sidebar count including recently ingested videos. Returns a `METADATA_LIST:<json>` signal — `service.py` intercepts this before the SSE stream and converts it to grouped plain text via `_format_metadata_list()`, which also filters by topic keyword extracted from the original question. The React client never sees the `METADATA_LIST:` prefix.

### Prompts (`agent/prompts.py`)

Three prompt components:

- **`SYSTEM_PROMPT`** — instructs the model to answer directly without hedging openers, stay grounded in retrieved context, enforce a 4-paragraph maximum, and protect prompt confidentiality
- **`NO_CONTEXT_RESPONSE`** — static fallback when no chunks meet the score threshold
- **`REWRITE_SYSTEM`** — used by the query rewriter (small model) to resolve pronouns and produce self-contained search queries for multi-turn conversations

### Live Ingest Pipeline (`pipeline/live_ingest.py`)

End-to-end pipeline triggered when the user pastes a YouTube URL:

1. Fetch video metadata (title, channel) via `yt-dlp`
2. Infer topic via `openai/gpt-oss-20b` on the first 500 words of the transcript
3. Extract transcript via `youtube-transcript-api`
4. Detect transcript language; normalise locale variants (`es-419` → `es`)
5. Clean and normalize text with language-aware filler removal
6. Chunk into ~60s windows
7. Embed with Cohere `embed-multilingual-v3.0` (`search_document` input_type)
8. Upsert to Pinecone `live` namespace with `language` stored in vector metadata

Note: `yt-dlp` is used only for metadata resolution in the live path. The corpus pipeline does not use `yt-dlp` — titles and channels are manually curated in `metadata.json`.

Includes cross-namespace duplicate detection — checks both `corpus` and `live` before indexing to avoid re-indexing videos already in the corpus.

**Deployment note:** transcript requests go out directly from whatever host runs the pipeline. The live path previously routed through an IPRoyal residential proxy to work around YouTube's datacenter IP restrictions; that proxy was removed in issue #16, along with its credential, because the project uses only access methods the platform permits. `youtube-transcript-api` and `yt-dlp` are both invoked without proxy configuration, and no proxy environment variable is read anywhere in the pipeline.

---

## Data Flow

```
YouTube URL
    │
    ▼
transcript_extractor.py  →  raw transcript JSON  (per-video)
    │
    ▼
cleaner.py               →  transcript_clean.json
    │
    ▼
chunker.py               →  chunks.json           (~60s windows)
    │
    ▼
embedder.py              →  embeddings.json        (1024-dim vectors, Cohere)
    │
    ▼
indexer.py               →  Pinecone upsert        [corpus namespace]
    │
    ▼
bootstrap_metadata.py    →  metadata.json          (video catalog)
```

At query time, the path is reversed: query → embedding → Pinecone → top-k chunks → LLM → answer.

---

## Key Design Decisions

**Hardcoded RAG-first routing** over autonomous LLM routing — eliminates routing errors during live demos and removes one LLM call per turn.

**Cohere asymmetric embeddings** — `embed-multilingual-v3.0` with separate `input_type` values (`search_document` at index time, `search_query` at retrieval time) is natively designed for asymmetric retrieval. This removes the need for title-prepended chunk text and improves precision, especially for short queries. Cosine scores with this model run higher (~0.49–0.78 for on-topic factual questions) than the old MiniLM model (~0.27–0.35). The 1024-dimensional vectors also support multilingual corpus expansion without re-indexing.

**Score threshold gate (0.40)** — calibrated via `eval/calibrate_threshold.py` against the 30-question eval set after Cohere re-indexing. `rag_factual` questions had a minimum top score of 0.49; adversarial out-of-corpus questions had a minimum of 0.39. The 0.40 gap preserves 100% hit rate on factual questions while providing a first-pass filter on the weakest off-topic matches. Adversarial intent filtering (prompt injection, out-of-scope requests) is handled by the LLM system prompt, not the score gate.

**Cohere Rerank v3.5 (optional)** — a cross-encoder reranker sits between Pinecone retrieval and the LLM. Bi-encoder cosine similarity (used at Pinecone query time) is fast but imprecise — it encodes query and document independently. A cross-encoder like Cohere Rerank jointly attends to both, producing more accurate relevance scores at the cost of one additional API call per query. The design over-retrieves 10 candidates from Pinecone, reranks them, and passes the top 5 to the LLM. Toggled via `RERANKER_ENABLED` env var to enable A/B comparison without code changes. Impact is quantified by `eval/sweep_reranker.py`, which runs the full eval set with and without the reranker and outputs a per-dimension side-by-side table.

**SSE multi-line encoding** — tokens from the agent occasionally span multiple lines (the metadata list is always multi-line). The SSE spec allows multiple `data:` lines within one event; consumers must join them with `\n` before dispatching. `service.py` encodes multi-line tokens as consecutive `data:` lines in one event (`"\n".join(f"data: {l}" for l in token.split("\n"))`). `lib/sse.ts` accumulates lines within an event and dispatches a single `onToken` call on the blank-line boundary, so `whitespace-pre-wrap` in the chat bubble renders newlines correctly.

**METADATA_LIST signal pattern** — `VideoMetadataTool` returns a `METADATA_LIST:<json>` prefix rather than a formatted string, bypassing LLM reformatting which produced inconsistent output. `service.py::_format_metadata_list()` intercepts the signal before the SSE stream and converts it to grouped plain text — topic headers with counts, then bullet lines — filtered by any topic keyword in the user's question. The React client never sees the prefix.

**Custom `ConversationMemory`** — avoids `langchain-community` dependency, which had unstable versioning during development.

**Two OpenAI models via Groq** — `openai/gpt-oss-120b` for RAG answers, `openai/gpt-oss-20b` for query rewriting and metadata resolution. The smaller model uses a separate Groq rate limit bucket.

---

## Testing

A unit test suite covers all pure pipeline and agent logic with no live API calls:

| Module | Covers |
|---|---|
| `tests/test_transcript_extractor.py` | URL parsing — watch/short/embed forms, timestamps, playlist params, malformed input |
| `tests/test_cleaner.py` | Transcript cleaning edge cases |
| `tests/test_chunker.py` | Chunk boundary conditions |
| `tests/test_live_ingest.py` | Duplicate detection, the live-ingest `parse_video_id`, ingest results |
| `tests/test_ingest_node.py` | The agent's ingest node, including error masking |
| `tests/test_generation_provenance.py` | Generation-provenance metadata on both response paths (issue #18) |
| `tests/test_source_excerpts.py` | Quotation-scale source excerpts on both response paths (issue #16) |

Run `pytest tests/` from the repo root, or `python tests/run_all_tests.py` for the same run with `-v --tb=short` preset. Both discover test files rather than reading a list, so a new file is picked up without being registered anywhere — and a local run and a CI run cannot disagree. Per-file test counts are deliberately not recorded here: nothing enforces them, and they drifted unnoticed once already ([#28](https://github.com/marcosfsousa/project-ironhack-scienceq/issues/28)).

CI (`.github/workflows/ci.yml`) gates every pull request on three checks — `Backend tests (pytest)`, `Frontend build (tsc + vite)`, and `UI tests (Playwright)`.

---

## Evaluation

Evaluated using a 38-case eval set (`eval/eval_set.json`): 20 English factual RAG cases, 8 cross-lingual RAG cases, 5 multi-turn cases, and 5 adversarial cases for manual review. GPT-4.1 is used as the judge across 4 dimensions.

| Experiment | Cases | Correctness | Tone | Grounding | Conciseness | Mean |
|---|---|---|---|---|---|---|
| prompt-v1 | 25 | 4.56 | 4.76 | 3.92 | 3.72 | 4.24 |
| prompt-v2 | 25 | 4.28 | **4.88** | 4.04 | 4.36 | 4.39 |
| Phase 3 — Cohere embeddings (reranker off) | 25 | 4.12 | 4.76 | 3.60 | 3.60 | 4.02 |
| Phase 4 — Cohere Rerank v3.5 (reranker on) | 25 | 4.40 | 4.84 | 3.64 | 4.12 | 4.25 |
| Phase 6 — Multilingual corpus | 33 | 4.38 | 4.62 | 3.62 | 4.25 | 4.22 |
| prompt-v3 — grounding tightened | 33 | 4.48 | 4.79 | 3.94 | 3.88 | 4.27 |
| **gpt-oss-120b — model swap** | **33** | **4.64** | 4.67 | **4.58** | **4.79** | **4.67** |

Phase 3 and Phase 4 scores are not comparable to prompt-v1/v2 — different embedding space (Cohere 1024d vs MiniLM 384d) and different eval methodology. Phase 6 added 8 non-English videos (ES/DE/FR/PT); cross-lingual retrieval validated via `eval/validate_multilingual.py` — 4/4 validation queries PASS with all non-English target chunks scoring above the 0.40 threshold (range: 0.52–0.70). Frontend confirmed: English queries surface non-English source pills alongside English results (verified at the time in the since-retired Streamlit UI). prompt-v3 replaced the prohibition-framed grounding rule with a verification frame ("which excerpt supports this?") and an explicit inference ban — grounding +0.32 on multilingual cases, correctness +0.12 overall, no regressions on tone or conciseness. Two cases (ml_007 microplastics, ml_008 neurodivergence) remained at grounding=2 — a corpus data gap, not fixable by prompt alone. gpt-oss-120b swapped the LLM from `llama-3.3-70b-versatile` to `openai/gpt-oss-120b` (via Groq) with no prompt or retrieval changes — the largest single-checkpoint gain in the table: grounding +0.64, conciseness +0.91, correctness +0.16, overall mean +0.40 vs. prompt-v3. Tone dipped -0.12. The two previously stuck multilingual cases (ml_007, ml_008) recovered: grounding 2 → 5 and 2 → 4 respectively.

Results are tracked in LangSmith under the `scienceq` project.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | OpenAI `gpt-oss-120b` (answers, via Groq), `gpt-oss-20b` (rewriting, via Groq) |
| Embeddings | Cohere `embed-multilingual-v3.0` — 1024 dimensions, asymmetric |
| Vector DB | Pinecone Serverless — cosine similarity, AWS us-east-1 |
| Orchestration | LangChain LCEL + LangGraph |
| Tracing | LangSmith |
| Transcripts | `youtube-transcript-api` v1.2.4 |
| Frontend | React 18 + Vite + TypeScript + Tailwind CSS |
| Web server | nginx (reverse proxy + SPA fallback) |
| Deployment | Google Cloud Run — `scienceq-api` (FastAPI) + `scienceq-web` (nginx/React) |
| Python | 3.11.9 |
