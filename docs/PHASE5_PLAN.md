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

### Root cause (historical — fix already shipped)
`ChatMessage.tsx` called `renderCitations(message.text, message.sources, ...)` for both the streaming and done states. During streaming `message.sources = []` so no matches were found and markers passed through as plain text.

### Fix applied (Option A)
`ChatMessage.tsx` now gates `renderCitations` on `status === "done"`. During streaming, `message.text` is rendered as plain text; citation pills snap in once the answer is complete. See the `done` conditional in the `whitespace-pre-wrap` block.

### Files
- `frontend/src/components/ChatMessage.tsx` — `renderCitations` is called only when `done === true` ✓

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
