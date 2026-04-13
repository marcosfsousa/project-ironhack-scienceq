# Multilingual Retrieval — Implementation Plan

**Prerequisite:** Phase 3 (Cohere embed-v3 swap) must be complete. Cohere embed-v3 supports 100+ languages in a shared embedding space — no additional model needed for cross-lingual retrieval.

---

## Scope

**Languages:** Spanish, German, French, Brazilian Portuguese
**Volume:** 2 videos per language (8 total) as a pilot
**Answer language:** Always English (isolates retrieval validation from generation quality)
**Caption policy:** Manual (human-uploaded) subtitles only — no auto-generated captions. Verify subs are in the native language, not English translations.

---

## Finalized Video List

### Spanish

| Channel | Video | Topic | Corpus Overlap |
|---|---|---|---|
| CuriosaMente | [¿Estrés, depresión o ansiedad? ¡Podrías ser neurodivergente!](https://www.youtube.com/watch?v=QmToxMOEls0) | Psychology | TED/Big Think psych videos |
| CuriosaMente | [Mathematics is not perfect! Gödel's incompleteness theorems](https://www.youtube.com/watch?v=_WM4XMwhRBE) | Mathematics | 3Blue1Brown, Veritasium |

Subtitles: manual + auto-generated (Latin American Spanish — same `es` language code, no special-casing needed).

### German

| Channel | Video | Topic | Corpus Overlap |
|---|---|---|---|
| Terra X Lesch & Co | [Physiker hassen die Zeit — Harald Lesch](https://www.youtube.com/watch?v=fy8jTiECCWI) | Physics | PBS Space Time |
| Terra X Lesch & Co | [Mathematik: Entdeckt oder erfunden? — Harald Lesch](https://www.youtube.com/watch?v=zUyItDq2JAk) | Philosophy / Mathematics | Vsauce, 3Blue1Brown |

Subtitles: manual + auto-generated (German).

### French

| Channel | Video | Topic | Corpus Overlap |
|---|---|---|---|
| ScienceÉtonnante | [La mécanique quantique en 7 idées — sciences inattendues #16](https://www.youtube.com/watch?v=Rj3jTw2DxXQ) | Physics | PBS Space Time, Veritasium |
| e-penser 2.0 | [Gaucher ou Droitier — quickie 07](https://www.youtube.com/watch?v=Adm-8rNBrCU) | Neuroscience / Biology | TED neuroscience videos |

Subtitles: manual + auto-generated (French — verified native-language subs, not English translations).

### Portuguese (Brazilian)

| Channel | Video | Topic | Corpus Overlap |
|---|---|---|---|
| Ciência Todo Dia | [O MITO das 10.000 HORAS](https://www.youtube.com/watch?v=QSTLljIWJ1U) | Psychology | TED psychology talks |
| Ciência Todo Dia | [Microplásticos já estão DENTRO de VOCÊ. E agora?](https://www.youtube.com/watch?v=4Yqe2j2Iw-Q) | Biology | Kurzgesagt biology videos |

Subtitles: manual + auto-generated (Brazilian Portuguese).

---

## Video Selection Criteria

All existing curation rules apply (one main concept, explanation-dense, scripted delivery), plus:

- Manual native-language subtitles required — no auto-generated-only channels
- Prefer videos whose topic overlaps with an existing English corpus video (enables cross-lingual retrieval validation)
- Log subtitle type (manual vs auto) in metadata for traceability

---

## Pipeline Changes

### 1. Extend `metadata.json` schema

Add a `language` field per video entry. Default `"en"` for all existing entries.

```json
{
  "video_id": "abc123",
  "title": "...",
  "channel": "CuriosaMente",
  "topic": "Psychology",
  "language": "es"
}
```

### 2. Language-aware cleaning (`cleaner.py`)

Keep a single `cleaner.py` with shared core logic (line merging, whitespace normalization, empty segment handling). Add a `LANG_RULES` config dict keyed by language code, loaded based on `metadata.json`'s `language` field:

```python
LANG_RULES = {
    "en": {"fillers": [...], "sponsor_patterns": [...], "punctuation_fixes": {}},
    "es": {"fillers": [...], "sponsor_patterns": [...], "punctuation_fixes": {}},
    "de": {"fillers": [...], "sponsor_patterns": [...], "punctuation_fixes": {}},
    "fr": {"fillers": [...], "sponsor_patterns": [...], "punctuation_fixes": {"colon_space": True}},
    "pt": {"fillers": [...], "sponsor_patterns": [...], "punctuation_fixes": {}},
}
```

For the pilot, minimal rulesets per language are fine — enrich as the corpus scales.

### 3. No changes to chunker, embedder, or indexer

Cohere embed-v3 handles all languages in one embedding space. Chunks are indexed into the existing `corpus` namespace alongside English vectors. No namespace-per-language split.

---

## Retrieval & Answering

- Cross-lingual retrieval works natively: an English query retrieves relevant Spanish/German/French/Portuguese chunks (and vice versa) via shared Cohere embedding space
- System prompt remains English; LLM answers in English regardless of chunk language
- No language filter in the sidebar for the pilot — add later if needed

---

## Eval

### Cross-lingual retrieval validation (manual, 2–3 queries)

For each language, pick a topic that overlaps with an English corpus video. Run the English query and verify:

1. Non-English chunks surface above the score threshold
2. Chunk ranking is reasonable (on-topic chunks outrank off-topic)
3. Timestamp deep links resolve correctly

**Suggested validation queries:**

| Query (English) | Expected hits |
|---|---|
| "What is quantum mechanics?" | ScienceÉtonnante (FR) + PBS Space Time / Veritasium (EN) |
| "Is mathematics discovered or invented?" | Terra X Lesch & Co (DE) + Vsauce / 3Blue1Brown (EN) |
| "Is the 10,000 hour rule true?" | Ciência Todo Dia (PT) + TED psychology (EN) |
| "What is Gödel's incompleteness theorem?" | CuriosaMente (ES) + 3Blue1Brown / Veritasium (EN) |

### Grounding eval (5–10 cases)

Create a small multilingual eval set added to `eval/eval_set.json`. Flag for the GPT-4.1 judge: it must verify that English answers faithfully represent non-English source content. This is a known eval limitation — document it, don't block on solving it.

---

## Deliverables

1. 8 new videos indexed (2 per language)
2. `metadata.json` extended with `language` field (all 50 entries)
3. `cleaner.py` updated with `LANG_RULES` routing
4. Cross-lingual retrieval validated manually
5. Multilingual eval set added to `eval/eval_set.json`
6. `README.md`, `DATASET.md`, `ARCHITECTURE.md` updated
7. Tag: `git tag -a v1.4-multilingual -m "Multilingual corpus: cross-lingual retrieval"`

---

## Out of Scope (Future)

- Query-language-matched answers ("respond in the language the user asked in")
- Sidebar language filter toggle
- Scaling beyond 2 videos per language
- Multilingual generation quality eval
- AssemblyAI as fallback transcription for channels without manual subs
