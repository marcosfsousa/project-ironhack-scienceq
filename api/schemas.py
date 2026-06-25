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


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    intent: str  # "rag" | "metadata" | "ingest"


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
