# Frontend — React SPA

The production frontend is a React 18 + Vite + TypeScript SPA deployed as a separate Cloud Run service (`scienceq-web`) behind an nginx reverse proxy.

**Live URL:** [https://scienceq.app](https://scienceq.app)

---

## Features

- Token-by-token SSE streaming with a blinking cursor during generation
- Corpus browser sidebar — videos grouped by topic, live-ingested videos shown with a LIVE badge
- Ingest panel — click `+` or paste a YouTube URL in the composer; a slide-over panel shows live step progress and confirms chunk count on completion
- Source pills and inline video embed — clickable timestamp links open the YouTube video at the right second; the top source embeds directly below the answer
- Metadata queries return a grouped, topic-filtered list that matches the sidebar count (corpus + live namespace)
- Keyboard and screen-reader accessible — `aria-expanded`, `aria-controls`, roving `tabIndex` on accent selector, `aria-label` on icon-only buttons

---

## Local development

```bash
cd frontend
npm install
VITE_API_TARGET=http://localhost:8080 npm run dev
# → http://localhost:5173 (proxies /api/* to the local FastAPI server)
```

---

## Deploy

```bash
gcloud builds submit --config cloudbuild-web.yaml --project=scienceq-prod
```

The build is two-stage: `node:20-alpine` runs `vite build`, then `nginx:alpine` serves the static bundle. `${API_URL}` is substituted at container startup via `envsubst`. Both stages run as non-root users (`nginx` user in the serving stage).

---

## Docker

```bash
# React SPA — nginx serves the Vite bundle (runs as nginx user)
docker build -f Dockerfile.web -t scienceq-web .
docker run -e API_URL=http://host.docker.internal:8080 -p 8081:8080 scienceq-web
# → http://localhost:8081 (requires the FastAPI API running on port 8080)
```
