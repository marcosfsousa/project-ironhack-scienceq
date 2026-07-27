# Backend

---

## Cloud Run API

The FastAPI service is deployed on Google Cloud Run:

**Base URL:** `https://scienceq-api-886463515307.europe-west1.run.app`

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness probe → `{"status":"ok"}` |
| `/api/chat` | POST | RAG chat — blocking, returns answer, sources, intent |
| `/api/chat/stream` | POST | RAG chat — SSE streaming, tokens + `[SOURCES]` + `[DONE]` |
| `/api/catalog` | GET | Full video catalog (corpus + live-ingested) |
| `/api/ingest` | POST | Submit a YouTube URL for live ingestion → `job_id` |
| `/api/ingest/{job_id}` | GET | Poll ingest job status |
| `/docs` | GET | Swagger UI |

**Example request:**
```bash
curl -X POST https://scienceq-api-886463515307.europe-west1.run.app/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message":"How does natural selection work?","history":[]}'
```

The original Streamlit UI remains live at [scienceq.streamlit.app](https://scienceq.streamlit.app). The React SPA is the primary interface going forward.

---

## Cloud Run Pipeline Job

The corpus pipeline runs as a Cloud Run Job (`scienceq-pipeline`) backed by a GCS bucket (`gs://scienceq-data`) mounted via GCS FUSE at `/app/data`. The container filesystem is ephemeral, but the FUSE mount makes writes land directly in GCS — files produced by one execution persist for the next.

**To add new videos to the corpus:**

1. Append YouTube URLs to `gs://scienceq-data/video_urls.txt`. GCS objects can't be appended in place, so pull the file down, add to it, and put it back. The local `video_urls.txt` here is a throwaway working copy — not the repo-tracked `data/video_urls.txt` used for local builds:

```bash
gcloud storage cp gs://scienceq-data/video_urls.txt video_urls.txt

echo "https://www.youtube.com/watch?v=VIDEO_ID" >> video_urls.txt

gcloud storage cp video_urls.txt gs://scienceq-data/video_urls.txt
```

2. Trigger the job:

```bash
gcloud run jobs execute scienceq-pipeline \
  --project=scienceq-prod \
  --region=europe-west1 \
  --wait
```

The job runs all pipeline steps in order (`extract → clean → chunk → embed → bootstrap → enrich → index`). Steps are idempotent — already-processed videos are skipped automatically.

**To rebuild the pipeline image** after code changes:

```bash
gcloud builds submit \
  --project=scienceq-prod \
  --region=europe-west1 \
  --config=cloudbuild-pipeline.yaml \
  .
```

**To update the job definition** (memory, CPU, env vars, secrets):

```bash
gcloud run jobs replace cloudrun-pipeline-job.yaml \
  --project=scienceq-prod \
  --region=europe-west1
```

---

## Live URL Ingestion

Paste any YouTube URL and ScienceQ ingests it on the fly — transcript extracted, chunked, embedded, and indexed into Pinecone's `live` namespace alongside the corpus.

Transcript requests are made directly, using only access methods YouTube permits — the residential proxy that previously worked around datacenter IP restrictions was removed in issue #16. Verified from a proxy-free Cloud Run revision on 25 July 2026: a previously unseen video ingested end-to-end (20 chunks, `already_indexed: false`), so YouTube now serves transcript requests from GCP datacenter IPs and the proxy came out with no feature impact. To avoid HTTP timeouts on what is a 60–90 second operation, the API uses an async fire-and-poll pattern:

| Endpoint | Method | Description |
|---|---|---|
| `/api/ingest` | POST | Submit a URL — returns `job_id` immediately (202) |
| `/api/ingest/{job_id}` | GET | Poll for status: `pending` → `complete` or `failed` |

**Example:**

```bash
# Submit
curl -X POST https://scienceq-api-886463515307.europe-west1.run.app/api/ingest \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.youtube.com/watch?v=VIDEO_ID"}'
# → {"job_id":"a1b2c3d4","status":"pending"}

# Poll until complete
curl https://scienceq-api-886463515307.europe-west1.run.app/api/ingest/a1b2c3d4
# → {"status":"complete","title":"...","channel":"...","topic":"...","chunk_count":12,...}
```

Video metadata (title, channel) is fetched via the YouTube oEmbed API — no auth, ~80ms.

---

## Running with Docker

Three Dockerfiles cover the three deployable services. All run as non-root users.

**Prerequisites:** Docker Desktop (Linux containers mode), a `.env` file with your API keys (copy from `.env.example`).

```bash
# FastAPI API (runs as appuser)
docker build -t scienceq-api .
docker run --env-file .env -p 8080:8080 scienceq-api
```

```bash
# Streamlit app (runs as appuser)
docker build -f Dockerfile.streamlit -t scienceq-streamlit .
docker run --env-file .env -p 8501:8501 scienceq-streamlit
```

See [`docs/FRONTEND.md`](FRONTEND.md) for the React SPA (`Dockerfile.web`) Docker instructions.

To verify non-root users in built images:
```bash
docker inspect scienceq-api --format '{{.Config.User}}'       # → appuser
docker inspect scienceq-streamlit --format '{{.Config.User}}' # → appuser
docker inspect scienceq-web --format '{{.Config.User}}'       # → nginx
```

---

## Running with Docker Compose

Docker Compose separates the serving layer (Streamlit app) from the batch pipeline into two containers that share a `./data` volume.

**Start the app:**
```bash
docker compose up app
```

**Run the full pipeline** (extract → clean → chunk → embed → bootstrap → enrich → index):
```bash
docker compose run pipeline --full
```

**Run individual pipeline steps:**
```bash
docker compose run pipeline --steps extract,clean,chunk
docker compose run pipeline --steps enrich
docker compose run pipeline --steps embed,index --force
```

The pipeline container uses the `pipeline` profile and does **not** start automatically with `docker compose up`.

---

## Building the corpus from scratch

If you want to index your own set of videos, run the pipeline steps in order:

```bash
pip install -r requirements-dev.txt

# Run the full pipeline (edit data/video_urls.txt with your URLs first)
python -m pipeline.run --full

# Or run individual steps
python -m pipeline.run --steps extract,clean,chunk
python -m pipeline.run --steps embed,index
python -m pipeline.run --steps embed,index --force   # re-process already-done videos
```

**Manual step between `chunk` and `embed`:** after `bootstrap` runs, open `data/metadata.json` and verify/fill in `title`, `channel`, and `topic` for each video. The `enrich` step auto-fills these via the YouTube Data API + Groq, but review the output before indexing.

Each individual script also supports `--video-id` to run on a single video. See the docstring at the top of each file for full CLI options.

The chunks will be indexed in the namespace set under `PINECONE_NAMESPACE_CORPUS` in your `.env` file.

---

## Running tests

```bash
python tests/run_all_tests.py
```

150 unit tests, no live API calls required, full run in well under a second.
