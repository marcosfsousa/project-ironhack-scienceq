# Evaluation

Evaluated on 33 automated cases (20 English factual + 8 cross-lingual + 5 multi-turn) with GPT-4.1 as judge across four rubric dimensions (1–5). Ten adversarial cases (out-of-scope, prompt-exposure, injection, hallucination-bait, and AI-identity) are excluded from automated scoring and reviewed manually.

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

**Phase 3** re-indexed the full corpus into a new embedding space (MiniLM → Cohere); scores are not comparable across that boundary.

**Phase 4** added Cohere Rerank v3.5.

**Phase 5** calibrated `RETRIEVER_FETCH_K`, `RETRIEVER_TOP_N`, and `SCORE_THRESHOLD` via a two-stage parameter sweep — details in [`retrieval_sweep_results.md`](retrieval_sweep_results.md).

**Phase 6** added 8 non-English videos (ES/DE/FR/PT); English queries surface non-English source pills in the UI alongside English results.

**prompt-v3** replaced the prohibition-framed grounding rule with a verification frame and an explicit inference ban — grounding +0.32 on multilingual cases, correctness +0.12 overall, no regressions.

**gpt-oss-120b** swapped the LLM from `llama-3.3-70b-versatile` to `openai/gpt-oss-120b` (via Groq) with no prompt or retrieval changes — the largest single-checkpoint gain in the table: grounding +0.64, conciseness +0.91, correctness +0.16, overall mean +0.40. Tone dipped -0.12. The two previously stuck multilingual cases (ml_007 microplastics, ml_008 neurodivergence) both recovered: grounding 2 → 5 and 2 → 4 respectively.
