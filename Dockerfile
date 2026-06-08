# ScienceQ FastAPI service — deployed to Cloud Run.
# (The Streamlit image lives in Dockerfile.streamlit.)
FROM python:3.11-slim

WORKDIR /app

# Install dependencies before copying code (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code. pipeline/ + data/metadata.json are required because
# importing the agent pulls in live_ingest and the metadata tool.
COPY api/ api/
COPY agent/ agent/
COPY pipeline/ pipeline/
COPY data/metadata.json data/metadata.json

# Cloud Run injects $PORT; default to 8080 for local `docker run`.
ENV PORT=8080
EXPOSE 8080

CMD ["sh", "-c", "exec uvicorn api.main:app --host 0.0.0.0 --port ${PORT}"]
