"""
routes.py
---------
HTTP routes for the ScienceQ API.

Phase 1 exposes a single blocking endpoint, ``POST /api/chat``. Streaming
(SSE) and the catalog endpoint are deferred to follow-up commits.
"""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, BackgroundTasks, HTTPException

from live_ingest import ingest_url

from .schemas import (
    ChatRequest,
    ChatResponse,
    IngestJobCreated,
    IngestRequest,
    IngestStatusResponse,
)
from .service import run_chat

log = logging.getLogger(__name__)

router = APIRouter()

# In-memory job store — persists for the lifetime of the Cloud Run instance.
# Acceptable for a single-instance demo; jobs are lost if the instance recycles.
_jobs: dict[str, dict] = {}


def _run_ingest(job_id: str, url: str) -> None:
    try:
        result = ingest_url(url)
        if result.success or result.already_indexed:
            _jobs[job_id] = {
                "status": "complete",
                "title": result.title,
                "channel": result.channel,
                "topic": result.topic,
                "chunk_count": result.chunk_count,
                "already_indexed": result.already_indexed,
            }
        else:
            _jobs[job_id] = {"status": "failed", "error": result.error}
    except Exception as exc:
        log.exception("Unhandled error during background ingest for job %s", job_id)
        _jobs[job_id] = {"status": "failed", "error": str(exc)}


@router.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Answer a question against the indexed corpus (blocking).

    Defined as a sync handler so FastAPI runs it in a worker thread — the
    underlying agent call is blocking I/O, so this keeps the event loop free.
    """
    try:
        return run_chat(request.message, request.history)
    except Exception as exc:  # noqa: BLE001 — translate any failure to HTTP
        err = str(exc).lower()
        if "429" in err or "rate_limit" in err or "too many requests" in err:
            # Mirror the Streamlit UI's graceful rate-limit handling.
            raise HTTPException(
                status_code=503,
                detail="The service is currently at capacity. Please try again in a few minutes.",
            )
        log.exception("Unhandled error answering chat request")
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred while answering your question.",
        )


@router.post("/api/ingest", status_code=202, response_model=IngestJobCreated)
def ingest(request: IngestRequest, background_tasks: BackgroundTasks) -> IngestJobCreated:
    """
    Kick off async ingestion of a YouTube URL.

    Returns immediately with a job_id. Poll GET /api/ingest/{job_id} for status.
    """
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {"status": "pending"}
    background_tasks.add_task(_run_ingest, job_id, request.url)
    return IngestJobCreated(job_id=job_id, status="pending")


@router.get("/api/ingest/{job_id}", response_model=IngestStatusResponse)
def ingest_status(job_id: str) -> IngestStatusResponse:
    """Poll the status of a previously submitted ingest job."""
    job = _jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    return IngestStatusResponse(**job)
