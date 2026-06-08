"""
routes.py
---------
HTTP routes for the ScienceQ API.

Phase 1 exposes a single blocking endpoint, ``POST /api/chat``. Streaming
(SSE) and the catalog endpoint are deferred to follow-up commits.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from .schemas import ChatRequest, ChatResponse
from .service import run_chat

log = logging.getLogger(__name__)

router = APIRouter()


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
