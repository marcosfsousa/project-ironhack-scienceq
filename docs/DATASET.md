# Dataset — ScienceQ Corpus

## Overview

The corpus consists of 50 curated science and education YouTube videos (42 English + 8 non-English), totalling approximately 800 vectors in Pinecone. Videos were selected to maximize topic diversity, explanation density, and retrieval reliability.

Non-English languages covered: Spanish (ES), German (DE), French (FR), Portuguese (PT). Cross-lingual retrieval works natively — English queries surface relevant chunks in all four languages via Cohere `embed-multilingual-v3.0`'s shared embedding space.

---

## Corpus Statistics

| Attribute | Value |
|---|---|
| Total videos | 50 (42 EN + 8 non-EN) |
| Languages | English, Spanish, German, French, Portuguese |
| Total chunks (vectors) | ~800 |
| Chunk window | ~60 seconds |
| Embedding dimensions | 1024 |
| Pinecone namespace | `corpus` |

---

## Channel Selection

Videos were sourced from channels known for scripted, explanation-dense content with reliable closed captions:

**English channels:**

| Channel | Videos | Notes |
|---|---|---|
| Veritasium | 6 | Physics, mathematics, cognitive science |
| Kurzgesagt – In a Nutshell | 4 | Biology, cosmology, philosophy |
| 3Blue1Brown | 3 | Mathematics, technology |
| PBS Space Time | 3 | Physics, cosmology |
| Big Think | 7 | Cosmology, psychology, biology, philosophy, neuroscience |
| TED / TEDx | 4 | Psychology, education, neuroscience, biology |
| Vsauce | 2 | Philosophy, mathematics |
| Quanta Magazine | 2 | Mathematics, physics |
| Other (CGP Grey, IBM, HHMI, Science Time, Vox) | 11 | Philosophy, technology, biology, cosmology, cognitive science |

**Non-English channels (Phase 6 pilot):**

| Channel | Language | Videos | Notes |
|---|---|---|---|
| CuriosaMente | ES | 2 | Mathematics (Gödel), Psychology (neurodivergence) |
| Terra X Lesch & Co | DE | 2 | Physics (time), Philosophy (math discovered/invented) |
| Science Étonnante | FR | 1 | Physics (quantum mechanics) |
| e-penser | FR | 1 | Neuroscience (handedness) |
| Ciência Todo Dia | PT | 2 | Psychology (10k hour rule), Biology (microplastics) |

**Selection criteria:**
- One main concept per video (avoids retrieval ambiguity)
- Clear verbal explanations with causal reasoning
- Scripted or semi-scripted delivery (higher transcript quality)
- Human-generated captions preferred over auto-generated
- No pure opinion pieces or list-style ("Top 10...") videos

---

## Topic Distribution

| Topic | Videos |
|---|---|
| Physics | 7 |
| Psychology | 7 |
| Biology | 6 |
| Cosmology | 6 |
| Philosophy | 5 |
| Technology | 4 |
| Mathematics | 5 |
| Neuroscience | 4 |
| Education | 3 |
| Cognitive Science | 3 |
| **Total** | **50** |

---

## Pipeline

The corpus was built using a 6-stage offline pipeline. Each stage reads from and writes to a per-video directory under `data/videos/{video_id}/`.

### Stage 1 — Transcript Extraction (`pipeline/transcript_extractor.py`)

Fetches closed captions via `youtube-transcript-api` v1.2.4 (instance-based API). Stores raw transcript segments with start/end timestamps.

```json
{
  "video_id": "pTn6Ewhb27k",
  "title": "Why No One Has Measured The Speed Of Light",
  "channel": "Veritasium",
  "transcript": [
    { "start": 0.0, "end": 4.2, "text": "..." }
  ]
}
```

Human-generated captions are preferred. Auto-generated captions are accepted as fallback. Videos with no available captions are excluded from the corpus.

**Note on live ingestion:** for on-the-fly URL ingestion, `yt-dlp` is used separately to fetch video title and channel before transcript extraction. The corpus pipeline does not use `yt-dlp` — titles and channels are manually curated in `metadata.json`.

### Stage 2 — Cleaning (`pipeline/cleaner.py`)

Normalizes raw transcripts without altering meaning:

- Merges broken subtitle lines into complete sentences
- Normalizes punctuation and casing
- Removes sponsor segment markers
- Handles `\u00a0` non-breaking spaces (common in auto-generated captions)
- Flags empty segments rather than deleting — downstream stages skip them
- **Language-aware filler removal** — per-language regex dicts (`en`, `es`, `de`, `fr`, `pt`); locale variants (`es-419`, `de-DE`) are normalised to the base code before lookup. Falls back to English rules for unknown codes.

Timestamps are preserved exactly. The original transcript is never modified.

### Stage 3 — Chunking (`pipeline/chunker.py`)

Splits cleaned transcripts into time-window chunks:

- **Target window:** 60 seconds (configurable via `--window`)
- **Boundary logic:** closes a chunk at the nearest segment boundary once the window target is reached — never mid-sentence
- **No overlap** between chunks — clean boundaries for precise timestamp deep links
- Each chunk carries `chunk_id`, `video_id`, `start`, `end`, `duration`, `text`, and `segment_count`

```json
{
  "chunk_id": "pTn6Ewhb27k_003",
  "video_id": "pTn6Ewhb27k",
  "start": 180.4,
  "end": 241.1,
  "duration": 60.7,
  "text": "...",
  "segment_count": 14
}
```

### Stage 4 — Embedding (`pipeline/embedder.py`)

Embeds each chunk using Cohere `embed-multilingual-v3.0`:

- **Dimensions:** 1024
- **Similarity metric:** cosine
- **Input types:** `search_document` at index time, `search_query` at retrieval time (asymmetric retrieval)
- **Multilingual:** a single shared embedding space covers 100+ languages — English queries retrieve semantically relevant chunks in Spanish, German, French, Portuguese, and any other supported language without a model swap or language detection step
- Vectors are persisted to `embeddings.json` per video to avoid re-embedding during re-indexing

### Stage 5 — Indexing (`pipeline/indexer.py`)

Upserts vectors to Pinecone Serverless (AWS us-east-1, cosine similarity):

- **Namespace:** `corpus` for the pre-built corpus, `live` for on-the-fly ingestion
- **Metadata stored per vector:** `chunk_id`, `video_id`, `title`, `channel`, `topic`, `language`, `start`, `end`, `chunk_text`
- Note: the embedded text includes the title prefix, but only the plain `chunk_text` is stored in metadata — this keeps LLM context clean
- Resume logic: skips videos already indexed unless `--force` flag is passed

### Stage 6 — Metadata Bootstrap (`pipeline/bootstrap_metadata.py`)

Builds `data/metadata.json` — a flat catalog of all indexed videos used by the `VideoMetadataTool` for catalog queries. Separate from Pinecone metadata; used for browsing and filtering without requiring a vector search.

---

## Retrieval Configuration

| Parameter | Value | Rationale |
|---|---|---|
| `RETRIEVER_FETCH_K` | 10 | Over-retrieve from Pinecone before reranking |
| `RETRIEVER_TOP_N` | 3 | Chunks passed to the LLM after reranking |
| `SCORE_THRESHOLD` | 0.40 | Calibrated post-Cohere re-indexing: factual queries score 0.49–0.78; adversarial out-of-corpus queries fall below 0.40. Multilingual chunks (EN query → non-EN chunk) consistently score 0.52–0.70, comfortably above the gate. |
| Multi-namespace | True | Corpus + live queried together at runtime |

---

## Quality Control

Each video was verified through:

1. **Manual question testing** — 3–5 questions per video run against the retriever to confirm relevant chunks are returned above the score threshold
2. **Timestamp alignment** — source pills verified to deep-link to the correct moment in the video
3. **Hallucination risk check** — videos heavily dependent on visuals (without verbal explanation) were excluded
4. **Eval set coverage** — 20 of the 42 videos are covered by at least one case in `eval/eval_set.json`

---

## Eval Set

A separate evaluation dataset of 38 cases is stored in `eval/eval_set.json`:

| Type | Count | Description |
|---|---|---|
| Factual RAG | 20 | English questions answered from English corpus chunks |
| Cross-lingual RAG | 8 | English questions answered from non-English corpus chunks (ES/DE/FR/PT) |
| Multi-turn | 5 | Pronoun resolution across conversation turns |
| Adversarial | 5 | Prompt injection, out-of-scope, hallucination bait |

Adversarial cases are excluded from automated scoring and reviewed manually. The 8 cross-lingual cases validate that English queries retrieve semantically relevant chunks in all four non-English languages above the 0.40 score threshold.
