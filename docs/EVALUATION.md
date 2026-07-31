# Evaluation

Evaluated with GPT-4.1 as judge across four rubric dimensions (1–5). The eval set now holds 46 cases, of which **36 are automated-eligible** (20 English factual + 8 cross-lingual + 5 multi-turn + 3 vague-reference multi-turn); the **10 adversarial cases are excluded from automated scoring** and reviewed manually via `eval/manual_review.json`. See [`DATASET.md`](DATASET.md#eval-set) for the full breakdown.

The set grew mid-project: the 3 vague-reference cases arrived in Phase 5 and 5 more adversarial cases in [#15](https://github.com/marcosfsousa/project-ironhack-scienceq/issues/15) (identity-disclosure and prompt-exposure guards), taking adversarial from 5 to 10. That is why the `Cases` column below changes over time rather than being constant — see the table caption.

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
| prompt-v4 — AI disclosure | 36 | 4.44 | 4.53 | 4.17 | 4.61 | 4.44 |

**The `Cases` column tracks the eval set as it grew — the rows are not all measuring the same thing.** 25-case rows predate the 8 cross-lingual cases; 33-case rows predate the 3 vague-reference cases, which were added in Phase 5 but first appear in a full run only at prompt-v4. Adversarial cases are excluded from every row (5 of them through prompt-v3, 10 from #15 onward), so no row's `Cases` count equals the eval set total of 46. The 3 vague-reference cases were exercised separately at the time by the rewriter runs (`rewriter-v1-baseline`, `rewriter-v2-tighter`) rather than being left untested — though, like the four checkpoints named below, those results files are not in this repo.

**What backs each row.** Four rows can be traced to a committed artifact: prompt v1 and prompt v2 to `eval/results/run_20260308_161303.json` and `run_20260309_084647.json`, Phase 4 to the `reranker_on` arm of `reranker_sweep_20260409_133416.json`, and Phase 5 to `run_20260410_123625.json`. Phase 3 was never scored. The remaining four — Phase 6, prompt-v3, gpt-oss-120b, prompt-v4 — have no artifact here: their results files were written but left untracked on the machine that produced them, hidden by the old `eval/results/` ignore rule, and the accompanying LangSmith push carries `case_id`, `case_type` and scores but nothing about how the run was configured.

**None of those artifacts records the retrieval config, so no row above is reproducible from what is written down.** A committed results file is history, not a recipe — which is why committing the four missing ones would not close this gap. `run_20260730_191842.json` (`rebaseline-2026-07-30-corpus-only`, [#117](https://github.com/marcosfsousa/project-ironhack-scienceq/pull/117)) is the first run whose results file records the config that produced it — commit, namespaces, reranker, `fetch_k`/`top_n`, threshold, models — and every run from that point on does the same. Treat the rows above as directional, and re-measure rather than compare against them.

**Phase 3** re-indexed the full corpus into a new embedding space (MiniLM → Cohere); scores are not comparable across that boundary.

**Phase 4** added Cohere Rerank v3.5.

**Phase 5** calibrated `RETRIEVER_FETCH_K`, `RETRIEVER_TOP_N`, and `SCORE_THRESHOLD` via a two-stage parameter sweep — details in [`retrieval_sweep_results.md`](retrieval_sweep_results.md).

**Phase 6** added 8 non-English videos (ES/DE/FR/PT); English queries surface non-English source pills in the UI alongside English results.

**prompt-v3** replaced the prohibition-framed grounding rule with a verification frame and an explicit inference ban — grounding +0.32 on multilingual cases, correctness +0.12 overall, no regressions.

**gpt-oss-120b** swapped the LLM from `llama-3.3-70b-versatile` to `openai/gpt-oss-120b` (via Groq) with no prompt or retrieval changes — the largest single-checkpoint gain in the table: grounding +0.64, conciseness +0.91, correctness +0.16, overall mean +0.40. Tone dipped -0.12. The two previously stuck multilingual cases (ml_007 microplastics, ml_008 neurodivergence) both recovered: grounding 2 → 5 and 2 → 4 respectively.

**prompt-v4** added the AI-disclosure prompt work behind the Art. 50 identity intent (see [`COMPLIANCE.md`](COMPLIANCE.md)) and is the first checkpoint to run all 36 automated-eligible cases. Its headline mean (4.44) is **not** comparable to gpt-oss-120b's 4.67, because the 3 vague-reference cases entering the pool are the hardest in the set. Restricted to the same 33 case types as the row above, prompt-v4 scores correctness 4.64, tone 4.73, grounding 4.09, conciseness 4.76, mean 4.55 — so the like-for-like change is a grounding regression (4.58 → 4.09) partly offset by tone, not the -0.23 the raw column suggests. The 3 vague-reference cases score correctness 2.33 / tone 2.33 / grounding 5.00 / conciseness 3.00: the agent stays grounded and declines to invent a referent, but resolves the pronoun poorly and answers verbosely. Vague-reference handling and the grounding dip are the two open threads from this checkpoint.
