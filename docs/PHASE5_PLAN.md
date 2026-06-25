# Phase 5 — Quality Pass

**Goal:** Fix the two open issues carried from Phase 4 testing. No new features, no scope creep.

---

## Status

| Item | Status |
|---|---|
| Citation pills in streamed text | 🔲 Todo |
| Conversation-aware retrieval | 🔲 Todo |

---

## Issue 1 — Citation pills in streamed text

### Symptom
The LLM outputs inline citation markers in the format `[Title, mm:ss]`. Below the answer, source pills render correctly as styled badges (sources arrive in the `[SOURCES]` SSE frame). Inside the streamed text bubble, the raw `[Title, mm:ss]` text is displayed instead of styled `[n]` pills — because `message.sources` is empty while the stream is in progress.

### Root cause
`ChatMessage.tsx` calls `renderCitations(message.text, message.sources, ...)` for both the streaming and done states. `lib/citations.tsx` matches citation markers against `message.sources` to build styled pills. During streaming `message.sources = []` so no matches are found and the markers pass through as plain text.

### Fix options
**A (simpler):** During streaming, render plain text without citation processing. After `status === "done"`, apply `renderCitations` with the populated sources. This means inline pills only appear once the answer is complete — no visual change during streaming, then citations snap in.

**B (richer):** Parse `[Title, mm:ss]` markers from the text during streaming (regex only, no source lookup), render them as greyed-out `[?]` placeholders, then swap to numbered `[n]` pills when sources arrive on `[DONE]`.

Option A is the right call — simpler, no intermediate state to manage, and the answer completes fast enough that users won't notice.

### Files
- `frontend/src/components/ChatMessage.tsx` — gate `renderCitations` on `status === "done"`

---

## Issue 2 — Conversation-aware retrieval

### Symptom
Follow-up questions ("tell me more about that", "what did he say about X?") retrieve independently — the embedding query is the rewritten current question only. The rewriter resolves pronouns into a self-contained query, but doesn't incorporate prior retrieved content or conversation context into the retrieval vector, so retrieval quality on follow-ups is inconsistent.

### Root cause
In `agent/rag_chain.py`, the retrieval query passed to Pinecone is the rewritten question alone. The conversation history is injected into the LLM prompt (giving the model context to answer), but the embedding query doesn't benefit from it.

### Fix approach
Augment the retrieval query with the most recent assistant turn before embedding. Concretely: prepend the last 1–2 assistant sentences (or topic keywords extracted from them) to the rewritten question before calling `_embed_query()`. This keeps the embedding close to what the user actually wants while pulling it toward the prior topic.

This is a targeted change to `agent/rag_chain.py` — no schema changes, no new dependencies.

### Files
- `agent/rag_chain.py` — augment retrieval query with prior assistant context
- `agent/agent.py` — pass relevant history slice to the RAG chain if not already threaded

---

## Out of Scope

- Whisper integration
- Non-English corpus expansion
- Any new UI features
- Any new API endpoints
