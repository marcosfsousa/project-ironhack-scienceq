# Architecture — ScienceQ

## Overview

ScienceQ is a retrieval-augmented generation (RAG) system that answers questions grounded in YouTube video transcripts. It combines a curated offline corpus with on-the-fly live URL ingestion, served through a streaming Streamlit chat interface.

---

## System Diagram

```
User (Browser)
     │
     ▼
Streamlit App  (app/streamlit_app.py)
     │
     ├── Paste YouTube URL ──► Live Ingest Pipeline
     │                              │
     │                         youtube-transcript-api
     │                         cleaner → chunker → embedder
     │                              │
     │                              ▼
     │                         Pinecone [live namespace]
     │
     └── Ask a question ──► LangGraph Agent  (agent/agent.py)
                                  │
                    ┌─────────────┼─────────────┐
                    ▼             ▼             ▼
              RAG intent    Metadata intent  Ingest intent
                    │             │
                    ▼             ▼
           RAGRetrieverTool  VideoMetadataTool
           (rag_chain.py)    (tools.py / metadata.json)
                    │
                    ▼
           Pinecone similarity search
           [corpus + live namespaces]
                    │
                    ▼
           Groq LLM  (llama-3.3-70b-versatile)
                    │
                    ▼
           Streaming answer + source citations
                    │
                    ▼
           LangSmith  (tracing + evaluation)
```

---

## Components

### Streamlit App (`app/streamlit_app.py`)

The web interface manages session state, renders streaming responses, and handles both chat input and starter button interactions. Key behaviours:

- **Intent detection** runs before the agent via `_classify_intent_fast()` to decide whether to use the streaming (RAG) or blocking (metadata/ingest) path
- **Streaming** renders tokens incrementally using a placeholder with a cursor character (`▌`)
- **Source pills** render as clickable HTML links with deep-linked YouTube timestamps
- **Video embed** persists in `st.session_state` to survive `st.rerun()` between responses, rendered in a 2-column layout (chat 60%, embed 40%) when a RAG source exists
- **Sidebar** displays the corpus catalog grouped by topic in collapsible `st.expander` sections, each showing title, channel, and duration

### LangGraph Agent (`agent/agent.py`)

A compiled LangGraph state machine with four nodes:

```
START → classify_intent → [rag | metadata | ingest] → respond → END
```

Intent routing is zero-cost keyword matching — no LLM call required. The agent maintains a 5-turn sliding window conversation memory via a custom `ConversationMemory` class (no LangChain community dependency).

### RAG Chain (`agent/rag_chain.py`)

Built with LangChain LCEL. Two execution paths:

- **Blocking** (`answer()`) — used by the agent graph and eval runner
- **Streaming** (`stream_answer()`) — yields tokens directly to the Streamlit UI

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

**VideoMetadataTool** — answers catalog queries ("what videos do you have on physics?") by searching `metadata.json`. Uses a three-pass matching strategy: exact match on topic/title/channel first, then a loose word match restricted to the topic field only to prevent cross-topic contamination. Returns a `METADATA_LIST:<json>` signal rather than a formatted string — the Streamlit app detects this prefix and renders the list directly, bypassing LLM reformatting entirely. All metadata queries are first resolved to a clean search keyword via `llama-3.1-8b-instant` before hitting the tool, normalising full sentences ("what videos do you have on mathematics?") to single terms ("mathematics").

### Prompts (`agent/prompts.py`)

Three prompt components:

- **`SYSTEM_PROMPT`** — instructs the model to answer directly without hedging openers, stay grounded in retrieved context, enforce a 4-paragraph maximum, and protect prompt confidentiality
- **`NO_CONTEXT_RESPONSE`** — static fallback when no chunks meet the score threshold
- **`REWRITE_SYSTEM`** — used by the query rewriter (small model) to resolve pronouns and produce self-contained search queries for multi-turn conversations

### Live Ingest Pipeline (`pipeline/live_ingest.py`)

End-to-end pipeline triggered when the user pastes a YouTube URL:

1. Fetch video metadata (title, channel) via `yt-dlp`
2. Infer topic via `llama-3.1-8b-instant` on the first 500 words of the transcript
3. Extract transcript via `youtube-transcript-api`
4. Detect transcript language; normalise locale variants (`es-419` → `es`)
5. Clean and normalize text with language-aware filler removal
6. Chunk into ~60s windows
7. Embed with Cohere `embed-multilingual-v3.0` (`search_document` input_type)
8. Upsert to Pinecone `live` namespace with `language` stored in vector metadata

Note: `yt-dlp` is used only for metadata resolution in the live path. The corpus pipeline does not use `yt-dlp` — titles and channels are manually curated in `metadata.json`.

Includes cross-namespace duplicate detection — checks both `corpus` and `live` before indexing to avoid re-indexing videos already in the corpus.

**Deployment note:** on Streamlit Community Cloud, YouTube blocks transcript requests from AWS IP ranges. This is resolved via an IPRoyal residential proxy: `youtube-transcript-api` is initialised with `GenericProxyConfig(http_url, https_url)` and `yt-dlp` receives a `--proxy` flag. The `_get_proxy_config()` helper reads `IPROYAL_PROXY_URL` from the environment and returns `None` when absent, so local development works unchanged.

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

**METADATA_LIST signal pattern** — `VideoMetadataTool` returns a structured `METADATA_LIST:<json>` prefix rather than a formatted string. The Streamlit app detects this prefix and renders the list directly, bypassing LLM reformatting which was producing inconsistent output. Plain-text fallbacks (no results found) are still passed through `st.markdown` as-is.

**Custom `ConversationMemory`** — avoids `langchain-community` dependency, which had unstable versioning during development.

**Two Groq models** — `llama-3.3-70b-versatile` for RAG answers, `llama-3.1-8b-instant` for query rewriting and metadata resolution. The smaller model uses a separate Groq rate limit bucket.

---

## Testing

A unit test suite covers all pure pipeline and agent logic with no live API calls:

| Module | Tests |
|---|---|
| `tests/test_transcript_extractor.py` | 12 |
| `tests/test_cleaner.py` | 22 |
| `tests/test_chunker.py` | 16 |
| `tests/test_live_ingest.py` | 14 |
| `tests/test_ingest_node.py` | 12 |
| **Total** | **76** |

All 76 tests pass. Run via `python tests/run_all_tests.py`. Tests cover duplicate detection logic, URL timestamp parsing, cleaning edge cases, chunking boundary conditions, and error masking behaviour.

---

## Evaluation

Evaluated using a 38-case eval set (`eval/eval_set.json`): 20 English factual RAG cases, 8 cross-lingual RAG cases, 5 multi-turn cases, and 5 adversarial cases for manual review. GPT-4.1 is used as the judge across 4 dimensions.

| Experiment | Correctness | Tone | Grounding | Conciseness | Mean |
|---|---|---|---|---|---|
| prompt-v1 | 4.56 | 4.76 | 3.92 | 3.72 | 4.24 |
| prompt-v2 | 4.28 | 4.88 | 4.04 | 4.36 | **4.39** |
| Phase 3 — Cohere embeddings (reranker off) | 4.12 | 4.76 | 3.60 | 3.60 | 4.02 |
| Phase 4 — Cohere Rerank v3.5 (reranker on) | 4.40 | 4.84 | 3.64 | 4.12 | **4.25** |
| Phase 6 — Multilingual corpus | — | — | — | — | — |

Phase 3 and Phase 4 scores are not comparable to prompt-v1/v2 — different embedding space (Cohere 1024d vs MiniLM 384d) and different eval methodology. Phase 6 added 8 non-English videos (ES/DE/FR/PT); cross-lingual retrieval validated via `eval/validate_multilingual.py` — 4/4 validation queries PASS with all non-English target chunks scoring above the 0.40 threshold (range: 0.52–0.70). GPT-4.1 scoring for Phase 6 is pending.

Results are tracked in LangSmith under the `scienceq` project.

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | Groq — `llama-3.3-70b-versatile` (answers), `llama-3.1-8b-instant` (rewriting) |
| Embeddings | Cohere `embed-multilingual-v3.0` — 1024 dimensions, asymmetric |
| Vector DB | Pinecone Serverless — cosine similarity, AWS us-east-1 |
| Orchestration | LangChain LCEL + LangGraph |
| Tracing | LangSmith |
| Transcripts | `youtube-transcript-api` v1.2.4 |
| Web App | Streamlit 1.55 |
| Deployment | Streamlit Community Cloud |
| Python | 3.11.9 |
