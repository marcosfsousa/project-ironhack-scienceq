"""
schemas.py
----------
Pydantic request/response models for the ScienceQ API.

The API is stateless: the client sends the recent conversation `history`
with every request, and the server rebuilds memory from it (no server-side
session store). See api/service.py.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


class Turn(BaseModel):
    """One prior conversation turn supplied by the client."""

    role: str  # "user" | "assistant"
    content: str


class Source(BaseModel):
    """
    One source citation.

    Mirrors the dict shape returned by the blocking path
    (RAGResponse.source_chunks_for_display in agent/rag_chain.py).
    `rerank_score` is None when the Cohere reranker is disabled.
    """

    title: str
    timestamp: str
    link: str
    score: float
    rerank_score: Optional[float] = None
    text: str


class ChatRequest(BaseModel):
    message: str
    history: list[Turn] = Field(default_factory=list)


class GenerationMetadata(BaseModel):
    """
    Machine-readable generation provenance for one answer.

    Additive substrate for EU AI Act Art. 50(2) output marking: every answer
    states whether its text is AI-generated and, if so, by which model. Purely
    additive — existing clients that ignore the field keep working unchanged.

    `ai_generated` is True only for LLM-generated prose; the no-context
    fallback, catalog listings, and ingest status messages are code-assembled
    and reported `ai_generated=False` with `model=None`. `model` is read from
    the runtime generation config at response time, not hard-coded. `mode`
    discriminates the producing path: "generated" | "no_context" | "metadata"
    | "ingest", plus "static" — the conservative fallback used when provenance
    is missing or the intent is unrecognised (see `service._provenance_dict`
    and `agent._derive_provenance`). "static" always carries
    `ai_generated=False`, so an unmapped path can never claim an AI origin.

    Note: this model validates the blocking `POST /api/chat` response only. The
    streaming `[META]` frame is serialised straight from `_provenance_dict`,
    so keep the two in step by hand.
    """

    ai_generated: bool
    model: Optional[str] = None
    mode: Literal["generated", "no_context", "metadata", "ingest", "static"]


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    intent: str  # "rag" | "metadata" | "ingest"
    generation: GenerationMetadata


class ChatStreamRequest(BaseModel):
    question: str
    history: list[Turn] = Field(default_factory=list)


class IngestRequest(BaseModel):
    url: str


class IngestJobCreated(BaseModel):
    job_id: str
    status: Literal["pending"]


class IngestStatusResponse(BaseModel):
    status: str  # "pending" | "complete" | "failed"
    title: Optional[str] = None
    channel: Optional[str] = None
    topic: Optional[str] = None
    chunk_count: Optional[int] = None
    already_indexed: Optional[bool] = None
    error: Optional[str] = None
