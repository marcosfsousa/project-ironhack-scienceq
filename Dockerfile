FROM python:3.11-slim

# curl: required for HEALTHCHECK
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies before copying code (better layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/
COPY agent/ agent/
COPY pipeline/ pipeline/

# corpus browser sidebar reads this file; returns [] gracefully if missing
COPY data/metadata.json data/metadata.json

EXPOSE 8501

# start-period accounts for Streamlit's ~20s cold start
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app/streamlit_app.py", \
    "--server.port=8501", \
    "--server.address=0.0.0.0"]
