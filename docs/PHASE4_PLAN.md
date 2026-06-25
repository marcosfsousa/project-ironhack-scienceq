# Phase 4 — React SPA

**Goal:** Build a React + Vite + TS frontend that replicates Streamlit's UX against the FastAPI backend, deployed as a separate Cloud Run service for fast independent iteration.

---

## Status

| Area | Status |
|---|---|
| Frontend scaffold + config | ✅ Done (Claude Design) |
| API client (`lib/sse.ts`, `lib/ingest.ts`) | ✅ Done (Claude Design) |
| Hooks (`useChat`, `useIngest`, `useCatalog`) | ✅ Done (Claude Design) |
| All components (10) | ✅ Done (Claude Design) |
| Design tokens, Tailwind config, CSS | ✅ Done (Claude Design) |
| `POST /api/chat/stream` SSE endpoint | ❌ Not built |
| `GET /api/catalog` endpoint + `api/catalog.py` | ❌ Not built |
| CORS middleware | ❌ Not built |
| `google-cloud-storage` in `requirements.txt` | ❌ Missing |
| Move `docs/design/` → `frontend/` | ❌ Not done |
| `Dockerfile.web` + `cloudbuild-web.yaml` | ❌ Not built |
| Deploy `scienceq-web` Cloud Run service | ❌ Not done |

---

## Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Streaming | SSE from the start | 5–10s spinner is unacceptable UX; agent already has `stream_chat()` |
| Hosting | Separate Cloud Run service (`scienceq-web`, nginx + Vite build) | Independent deploy pipeline — frontend changes don't rebuild the API image |
| Corpus browser | `/api/catalog` endpoint, corpus + live videos | Full parity with Streamlit sidebar |
| Ingest UX | Client-side YouTube URL detection → `/api/ingest` + poll | Avoids extra round-trip through `/api/chat` |
| Video embed | Inline inside chat bubble, below source pills | Simpler than two-column layout; more natural on web/mobile |
| Auth | None — stays public | — |
| UI library | Tailwind CSS + inline SVGs | Claude Design used inline SVGs rather than lucide-react |

---

## Implementation Steps

### 1. API — SSE streaming endpoint ❌
**Files:** `api/routes.py`, `api/service.py`, `api/schemas.py`

- Add `stream_run_chat()` to `service.py` yielding SSE-formatted lines from `agent.stream_chat()` + `agent.last_sources`
- Add `ChatStreamRequest` schema with `question: str` + `history: list[Turn]` (distinct from `ChatRequest.message` — frontend sends `question`)
- Add `POST /api/chat/stream` returning `StreamingResponse(media_type="text/event-stream")`
- SSE event shapes (already implemented in `docs/design/src/lib/sse.ts`):
  - `data: <token>` — answer token
  - `data: [SOURCES]<json>` — source list after token stream ends
  - `data: [DONE]` — stream closed
- Keep blocking `POST /api/chat` as-is

> **Field name note:** `lib/sse.ts` sends `{ question, history }`. The new stream endpoint must use `question` (not `message` like `ChatRequest`) or `sse.ts` must be updated.

### 2. API — Catalog endpoint ❌
**Files:** `api/routes.py`, new `api/catalog.py`, `requirements.txt`

- Add `google-cloud-storage` to `requirements.txt`
- Read corpus from `gs://scienceq-data/metadata.json` via `google-cloud-storage`
- Read live videos from Pinecone `live` namespace via `list()` + `fetch()` on `*_000` chunk IDs
- Return merged list: `[{video_id, title, channel, topic, duration, url, source: "corpus"|"live"}]`
- Grant `Storage Object Viewer` to compute SA `886463515307-compute@developer.gserviceaccount.com` on `scienceq-data` bucket

### 3. API — CORS middleware ❌
**Files:** `api/main.py`

- Allow `https://scienceq-web-*.europe-west1.run.app` + `http://localhost:5173` for dev
- Lock to exact `scienceq-web` URL once known (deploy web first → get URL → update → redeploy API)

### 4. Frontend — Move scaffold into place ❌
**Action:** Copy `docs/design/` contents → `frontend/`

The Claude Design bundle at `docs/design/` is a runnable scaffold. Move it:
```
docs/design/src/          → frontend/src/
docs/design/index.html    → frontend/index.html
docs/design/package.json  → frontend/package.json
docs/design/vite.config.ts → frontend/vite.config.ts
docs/design/tsconfig.json → frontend/tsconfig.json
docs/design/tailwind.config.ts → frontend/tailwind.config.ts
docs/design/postcss.config.js → frontend/postcss.config.js
```
`docs/design/reference/` stays in place (visual source of truth, not deployed).

Local dev after move:
```bash
cd frontend && npm install
VITE_API_TARGET=http://localhost:8080 npm run dev
```

### 5. Infrastructure ❌
**Files:** `Dockerfile.web`, `cloudbuild-web.yaml`, `frontend/nginx.conf`

- `Dockerfile.web` — two-stage: `node:20-alpine` build (`vite build`) → `nginx:alpine` serve
- `frontend/nginx.conf` — `add_header X-Accel-Buffering "off"` required for SSE through nginx; serve `index.html` for all non-API routes (SPA fallback)
- `cloudbuild-web.yaml` — `VITE_API_URL` substitution baked at `vite build` time
- Add to `.dockerignore`: `frontend/node_modules`, `frontend/.env.local`

### 6. Deploy ❌
- Deploy `scienceq-web` first → get auto-generated Cloud Run URL
- Update CORS in `api/main.py` with exact URL
- Redeploy API (`cloudbuild.yaml`)

---

## Known Gaps / Things to Verify

- **`question` vs `message` field:** `lib/sse.ts` POSTs `{ question, history }`. The new stream endpoint schema must match this (use `question`), or update `sse.ts` to send `message`. Decide before building Step 1.
- **`METADATA_LIST:` signal:** The original plan called for parsing `METADATA_LIST:<json>` from the answer and rendering a structured list. The Claude Design scaffold does not implement this — `ChatMessage.tsx` handles plain text + citations only. Confirm whether the metadata intent path still needs this or whether the corpus browser makes it redundant.
- **Inline citation format:** `lib/citations.tsx` expects backend answers to contain `[Title, mm:ss]` inline markers. Verify the Groq prompt emits this format; if not, adjust the regex.
- **`rerank_score` nullability:** `types.ts` declares `rerank_score: number` (non-optional) but `api/schemas.py` has `rerank_score: Optional[float]`. Update `types.ts` to `rerank_score: number | null`.

---

## Data Shapes

```typescript
// POST /api/chat/stream — request body
{ question: string; history: { role: string; content: string }[] }

// POST /api/chat/stream — SSE events
// data: <token>
// data: [SOURCES][{"title","timestamp","link","score","rerank_score","text"}, ...]
// data: [DONE]

// POST /api/ingest → 202
{ job_id: string; status: "pending" }

// GET /api/ingest/:job_id
{
  status: "pending" | "complete" | "failed";
  title?: string;
  channel?: string;
  topic?: string;
  chunk_count?: number;
  already_indexed?: boolean;
  error?: string;
}

// GET /api/catalog
[{
  video_id: string;
  title: string;
  channel: string;
  topic: string;
  duration: string;
  url: string;
  source: "corpus" | "live";
}]
```

---

## Open Risks

- **CORS chicken-and-egg** — `scienceq-web` URL is auto-generated on first deploy; CORS in the API can't reference it until after the web service exists. Fix: deploy web → get URL → update API CORS → redeploy API.
- **GCS catalog read** — API compute SA needs `Storage Object Viewer` on `scienceq-data`. Easy to forget.
- **Pinecone `list()` for live catalog** — may need pagination if live namespace grows; start without, add later.
- **SSE through nginx** — nginx buffers by default and breaks SSE. Must set `X-Accel-Buffering: off`.
- **`VITE_API_URL` baked at build time** — must be known before `vite build` runs in Cloud Build. Local dev needs a `frontend/.env.local` with `VITE_API_TARGET`.

---

## Out of Scope

- Auth
- Retiring Streamlit (runs in parallel until SPA is validated)
- Whisper integration
- Expanding non-English corpus
- Dark/light mode toggle

---

## Design Handoff

✅ Complete. Claude Design delivered a full runnable scaffold at `docs/design/`.

- `docs/design/src/` — all React/TS source (components, hooks, lib, types, CSS tokens)
- `docs/design/reference/ScienceQ.prototype.dc.html` — original HTML prototype, **visual source of truth**; when Tailwind translation and prototype disagree, prototype wins on visuals
- `docs/design/reference/scienceq-data.js` — mock catalog + canned answers used by the prototype
