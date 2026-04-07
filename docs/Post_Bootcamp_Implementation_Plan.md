# ScienceQ — Post-Bootcamp Implementation Plan

This plan covers the improvements to ScienceQ after the Ironhack bootcamp presentation. Each phase is designed to be a self-contained unit with its own commit(s), testable outcome, and clear dependency on previous phases.

**Key architectural decision:** Embedding and reranking move from local models to Cohere's API (embed-v3/v4 + Rerank v3). This collapses three improvements — asymmetric embeddings, smaller Docker image, multilingual support — into one dependency change, and removes PyTorch (~2GB) from the runtime stack.

---

## Phase 0 — Baseline Tag + Dev/Prod Environment

**Goal:** Mark a clean separation between the bootcamp deliverable and post-bootcamp work, and establish a dev/prod environment split that protects the live Streamlit deployment during active development.

### 0a — Baseline Tag

1. Ensure `main` is clean — no uncommitted changes, all tests pass, Streamlit Cloud deployment is stable
2. Tag the current HEAD: `git tag -a v1.0-bootcamp -m "Ironhack bootcamp final deliverable"`
3. Push the tag: `git push origin v1.0-bootcamp`
4. Create a GitHub Release from the tag with a short description linking to the presentation and the live demo

**Outcome:** Any recruiter or reviewer can check out `v1.0-bootcamp` to see the original project exactly as presented. All subsequent work lives after this tag.

---

### 0b — Dev/Prod Environment Split

**Goal:** Protect the live Streamlit Cloud deployment while doing active post-bootcamp development locally. All phase work happens on `dev`; `main` is only touched on phase completion.

**Branch strategy:**

| Branch | Purpose |
|--------|---------|
| `main` | Production — Streamlit Cloud deploys from here. Never commit mid-phase work here. |
| `dev` | Development — all post-bootcamp phases are built and tested here. |

Merge rule: `dev` → `main` only when a phase is fully tested locally and ready to redeploy.

**Pinecone isolation:**

| Environment | Pinecone Index | Usage |
|-------------|----------------|-------|
| Local dev | `scienceq-dev` | All post-bootcamp experimentation |
| Production | `scienceq` | Existing live index — never touched during dev |

**Steps:**

1. Create and push the `dev` branch off current `main`:
   ```bash
   git checkout -b dev
   git push -u origin dev
   ```
2. Create a new Pinecone index named `scienceq-dev` with the same configuration as the prod index (dimensions, metric, cloud/region)
3. Add `PINECONE_INDEX_NAME` to `.env.example`, documenting both values:
   ```
   # Dev: scienceq-dev | Prod: scienceq
   PINECONE_INDEX_NAME=scienceq-dev
   ```
4. Update local `.env` to set `PINECONE_INDEX_NAME=scienceq-dev`
5. Confirm all code reads `PINECONE_INDEX_NAME` from env — hardcoded index names must be replaced
6. Verify Streamlit Cloud secrets remain unchanged (still pointing to `scienceq`)
7. Smoke-test locally on `dev`: app starts, queries hit `scienceq-dev`, prod index is untouched

**Outcome:** Commits to `dev` never trigger a Streamlit Cloud redeploy. The prod app on `main` remains stable throughout all post-bootcamp phases. Local dev uses an isolated Pinecone index with the same API keys.

**Note:** Streamlit Community Cloud free tier supports up to 3 apps. A second app deployed from `dev` can be added later for a shareable dev preview URL — not needed for local-first development.

---

## Phase 1 — Docker (Streamlit App)

**Goal:** Containerize the Streamlit app with zero code changes. Portfolio signal: "this project runs anywhere."

**Steps:**

1. Create `.dockerignore` — exclude `data/`, `eval/`, `.env`, `__pycache__/`, `.git/`, `*.pyc`, `notebooks/`, `docs/`
2. Create `Dockerfile`:
   - Base image: `python:3.11-slim`
   - Install system deps (`build-essential` for sentence-transformers C extensions)
   - Copy `requirements.txt` and install
   - Copy application code
   - Pre-download the `all-MiniLM-L6-v2` model at build time (avoids runtime download on first request)
   - Expose port 8501
   - Entrypoint: `streamlit run app/streamlit_app.py --server.port=8501 --server.address=0.0.0.0`
3. Add `HEALTHCHECK` instruction (curl against Streamlit's `/_stcore/health`)
4. Test locally: `docker build -t scienceq . && docker run --env-file .env -p 8501:8501 scienceq`
5. Verify: app loads, corpus queries work, live ingest works (local network, no AWS IP block)
6. Document in README: "Running with Docker" section
7. (Optional) Deploy to Railway or Fly.io to prove cloud portability

**Outcome:** `docker run --env-file .env scienceq` yields a working app identical to Streamlit Cloud.

**Note:** The `sentence-transformers` + PyTorch dependency makes this image ~2.5GB. This shrinks dramatically in Phase 3 when local models are replaced with Cohere API calls.

---

## Phase 2 — Docker Compose (App + Pipeline)

**Goal:** Multi-container orchestration separating the serving layer from the batch pipeline. Portfolio signal: "this person thinks about service architecture."

**Steps:**

1. Create `Dockerfile.pipeline` for the batch ingestion pipeline:
   - Same base image and Python deps
   - Entrypoint: configurable (extract, clean, chunk, embed, index, or full pipeline)
   - Mounts `data/` as a volume for intermediate artifacts
2. Create `docker-compose.yml` with two services:
   - `app` — the Streamlit container from Phase 1
   - `pipeline` — the batch processing container
   - Shared volume: `./data:/app/data`
   - Shared `.env` file for API keys
3. Add common `make` targets or a `justfile` for convenience:
   - `make up` → `docker compose up app`
   - `make pipeline-full` → `docker compose run pipeline python -m pipeline.run --full`
   - `make pipeline-index` → `docker compose run pipeline python -m pipeline.indexer`
4. Test: run the full pipeline in the pipeline container, then start the app container and verify queries return results from the freshly indexed corpus
5. Document in README: "Running with Docker Compose" section

**Outcome:** `docker compose up app` serves the app; `docker compose run pipeline` runs any pipeline stage. Clean separation of concerns, shared data volume.

---

## Phase 3 — Embedding Model Swap (Cohere embed-v3/v4)

**Goal:** Replace `all-MiniLM-L6-v2` with Cohere's asymmetric embedding model. Fixes score depression, removes PyTorch from runtime, enables multilingual in Phase 6.

**Architecture change:** Cohere embed-v3 uses separate input types — `search_document` for indexing and `search_query` for queries. This is the native fix for the asymmetric encoding that caused score depression with MiniLM's symmetric model + title-prepended index text.

**Steps:**

1. Add `cohere` to `requirements.txt`; remove `sentence-transformers` and `torch`
2. Add `COHERE_API_KEY` to `.env.example` and Streamlit secrets
3. Refactor `pipeline/embedder.py`:
   - Replace `SentenceTransformer.encode()` with `cohere.embed(input_type="search_document")`
   - Remove the `"{title} | {chunk_text}"` prepend — Cohere's asymmetric model handles this natively via `input_type`
   - Update embedding dimensions (Cohere embed-v3 = 1024, vs MiniLM's 384)
4. Refactor `agent/retriever.py`:
   - Replace `SentenceTransformer.encode()` with `cohere.embed(input_type="search_query")`
5. Re-create the Pinecone index with the new dimensionality (1024), or create a new index
6. Re-embed and re-index the full 42-video corpus into the `corpus` namespace
7. Recalibrate the score threshold:
   - Run the existing eval set against the new embeddings
   - Plot score distributions for on-topic vs off-topic queries
   - Set the new threshold empirically (expect higher scores than 0.28 — asymmetric models produce better separation)
8. Update `live_ingest.py` to use the Cohere embedding path
9. Rebuild Docker images — the image will shrink significantly without PyTorch
10. Run full eval suite and record new baselines (these are **not comparable** to v1 scores — different embedding space)
11. Update `ARCHITECTURE.md` to reflect the new embedding model and rationale
12. Tag: `git tag -a v1.1-cohere-embeddings -m "Swap to Cohere embed-v3 asymmetric embeddings"`

**Outcome:** Higher retrieval precision, better score separation between on-topic and off-topic queries, ~2GB smaller Docker image, multilingual-ready embedding space.

**Breaking change:** All existing vectors must be re-indexed. The `corpus` and `live` namespaces are wiped and rebuilt.

---

## Phase 4 — Reranker Layer (Cohere Rerank v3)

**Goal:** Add a reranking step after retrieval to improve precision before the LLM sees the results.

**Architecture change:** The retrieval flow becomes: query → Cohere embed → Pinecone top_k → **Cohere Rerank** → top_n to LLM. The reranker re-scores the retrieved chunks using a cross-encoder model, which is more accurate than bi-encoder cosine similarity but too expensive to run against the full index.

**Steps:**

1. Add reranking call in `agent/retriever.py`:
   - After Pinecone returns top_k results, pass query + chunk texts to `cohere.rerank()`
   - Retrieve top_k=10 from Pinecone, rerank, pass top_n=5 to the LLM (over-retrieve, then filter)
   - Preserve the original cosine score alongside the rerank relevance score for observability
2. Add a `RERANKER_ENABLED` flag in config to allow toggling for A/B comparison
3. Update LangSmith tracing to log both the pre-rerank and post-rerank orderings
4. Run the full eval suite with reranker enabled and compare to Phase 3 baselines
5. Analyze: does reranking change which chunks reach the LLM? Does it improve grounding scores?
6. Update `ARCHITECTURE.md` with the reranker flow diagram
7. Tag: `git tag -a v1.2-reranker -m "Add Cohere Rerank v3 after retrieval"`

**Outcome:** Measurable retrieval precision improvement, observable in LangSmith traces and eval scores.

---

## Phase 5 — Retrieval Parameter Sweeps

**Goal:** Systematic experimentation across retrieval parameters as a dedicated eval axis, separate from prompt versioning.

**This phase produces data and analysis, not code changes.**

**Experiment matrix:**

| Parameter | Values to test |
|---|---|
| top_k (Pinecone retrieval) | 5, 7, 10, 15 |
| top_n (post-rerank, to LLM) | 3, 5, 7 |
| Reranker | on / off |
| Score threshold | 0.20, 0.25, 0.30, 0.35 (recalibrated for Cohere) |

**Steps:**

1. Build a parameter sweep script (`eval/sweep_retrieval.py`) that runs the eval set across all combinations
2. Log each combination as a separate LangSmith experiment with clear naming (e.g. `cohere-k10-n5-rerank-t030`)
3. Collect results into a summary table: mean scores per rubric dimension (correctness, tone, grounding, conciseness) × parameter combination
4. Identify the Pareto-optimal configuration — the setting where no single dimension can improve without another degrading
5. Commit the winning configuration as the new default
6. Write up findings in a `docs/retrieval_sweep_results.md` with charts
7. Tag: `git tag -a v1.3-tuned-retrieval -m "Retrieval parameters tuned via systematic sweep"`

**Outcome:** Data-driven retrieval configuration. The sweep results document is itself a portfolio artifact — it shows methodical engineering, not guesswork.

---

## Phase 6 — Multilingual Corpus

**Goal:** Expand the corpus to include non-English science channels, leveraging Cohere embed-v3's native multilingual support.

**Prerequisite:** Phase 3 (Cohere embeddings) must be complete. Cohere embed-v3 supports 100+ languages in the same embedding space — no model swap needed.

**Steps:**

1. Curate 5–10 non-English science channels (Spanish, German, French, Portuguese as starting points):
   - Select channels with auto-generated captions available
   - Apply the same curation criteria as the original corpus (educational, well-structured, popular)
2. Extend `data/metadata.json` schema to include a `language` field
3. Run the full pipeline (extract → clean → chunk → embed → index) for the new videos
4. Verify cross-lingual retrieval: a query in English should retrieve relevant chunks from non-English videos (and vice versa), since they share the same embedding space
5. Add a language filter to the Streamlit sidebar (optional — the retrieval works cross-lingually, but users may want to filter)
6. Run eval: create a small multilingual eval set (5–10 questions) and verify grounding quality
7. Update README, DATASET.md, and ARCHITECTURE.md
8. Tag: `git tag -a v1.4-multilingual -m "Multilingual corpus: cross-lingual retrieval"`

**Outcome:** ScienceQ answers science questions using sources in multiple languages. This is a strong differentiator — most RAG demos are English-only.

---

## Dependency Summary

```
Phase 0 (tag) ─── no dependencies
Phase 1 (Docker) ─── no dependencies
Phase 2 (Compose) ─── depends on Phase 1
Phase 3 (Cohere embeddings) ─── no code dependency on Docker, but rebuild images after
Phase 4 (Reranker) ─── depends on Phase 3
Phase 5 (Parameter sweeps) ─── depends on Phase 4
Phase 6 (Multilingual) ─── depends on Phase 3
```

Phases 1–2 (Docker) and Phase 3 (Cohere) are independent tracks that can be worked in parallel. Phase 6 can start as soon as Phase 3 is done, without waiting for Phases 4–5.

---

## Git Tag Convention

| Tag | Phase | Description |
|---|---|---|
| `v1.0-bootcamp` | 0 | Ironhack final deliverable as presented |
| `v1.1-cohere-embeddings` | 3 | Asymmetric embeddings via Cohere |
| `v1.2-reranker` | 4 | Cohere Rerank v3 layer |
| `v1.3-tuned-retrieval` | 5 | Data-driven parameter optimization |
| `v1.4-multilingual` | 6 | Cross-lingual corpus and retrieval |

Docker phases (1–2) are committed normally without version tags — they're infrastructure, not functional milestones.

---

*ScienceQ · Post-Bootcamp Improvement Plan · Marcos Sousa · April 2026*
