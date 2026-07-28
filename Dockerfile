# ScienceQ FastAPI service — deployed to Cloud Run.
FROM python:3.11-slim

WORKDIR /app

# Install dependencies before copying code (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code. api/ and agent/ ship wholesale; data/metadata.json is
# required because the agent's metadata tool reads it.
#
# pipeline/ ships four of its ten modules (#52) — the closure of what the API
# path imports, traced rather than guessed:
#
#   api/routes.py:19, agent/agent.py:71  ->  live_ingest
#   pipeline/live_ingest.py:70-72        ->  cleaner, chunker, embedder
#
# The six left out are batch-pipeline only (bootstrap_metadata, enrich_metadata,
# indexer, run, sponsorblock, transcript_extractor) and still ship in
# Dockerfile.pipeline, which COPYs pipeline/ wholesale. Bare module imports
# resolve through the sys.path bridge in api/__init__.py:17-20, so a partial
# copy works as long as the named modules are present.
#
# This is an enumerated module list, so it drifts: add an import to live_ingest
# and this image breaks at *import* time, which no amount of building catches.
# Two things enforce it, and neither is optional maintenance —
#   .github/workflows/ci.yml   builds this file and runs `import api.main`
#   tests/test_declared_imports.py
#                              TestShippedSourceMatchesDockerfile keeps the list
#                              below equal to SHIPPED_MODULES, and
#                              TestShippedSourceIsSelfContained fails if shipped
#                              code imports a module left out here.
#
# One deliberate loose end. pipeline/chunker.py:180 imports sponsorblock lazily,
# inside the skip_sponsors branch of chunk_transcript, whose only caller is
# run() — the batch entrypoint, which does not exist in this image. That import
# is unreachable here, so leaving sponsorblock out trades a silent problem for a
# loud one: if the API path ever grows a caller, it raises ImportError instead
# of quietly running against an undeclared dependency version, which is the
# drift #42 had to fix. Recorded as a documented seam in DOCUMENTED_SEAMS.
COPY api/ api/
COPY agent/ agent/
COPY pipeline/chunker.py \
     pipeline/cleaner.py \
     pipeline/embedder.py \
     pipeline/live_ingest.py \
     pipeline/
COPY data/metadata.json data/metadata.json

RUN addgroup --system --gid 1001 appuser && \
    adduser --system --uid 1001 --gid 1001 appuser && \
    chown -R appuser:appuser /app
USER appuser

# Cloud Run injects $PORT; default to 8080 for local `docker run`.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
