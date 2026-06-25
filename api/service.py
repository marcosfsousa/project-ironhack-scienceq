"""
service.py
----------
The stateless seam between the HTTP layer and the existing agent.

Each request builds a fresh ``YouTubeQAAgent`` and replays the client-supplied
history into its ConversationMemory before answering. This keeps the service
stateless so it scales cleanly across Cloud Run instances. Instantiation is
cheap — the LangGraph build is light and the Pinecone/Cohere/Groq clients are
module-level singletons reused across requests.
"""

from __future__ import annotations

import json
import logging
from typing import Generator

from agent import YouTubeQAAgent  # resolved via sys.path setup in api/main.py

from .schemas import ChatResponse, Turn

log = logging.getLogger(__name__)


def _build_agent(history: list[Turn]) -> YouTubeQAAgent:
    """Return a fresh agent seeded with prior (user, assistant) turns."""
    agent = YouTubeQAAgent()
    pending_user: str | None = None
    for turn in history:
        if turn.role == "user":
            pending_user = turn.content
        elif turn.role == "assistant" and pending_user is not None:
            agent.memory.save_turn(pending_user, turn.content)
            pending_user = None
    return agent


def run_chat(message: str, history: list[Turn]) -> ChatResponse:
    """Answer a single message (blocking) with conversation context."""
    agent = _build_agent(history)
    resp = agent.chat(message)
    return ChatResponse(answer=resp.answer, sources=resp.sources, intent=resp.intent)


def _format_metadata_list(raw: str) -> str:
    """
    Convert the METADATA_LIST:<json> signal from VideoMetadataTool into
    human-readable plain text for the React frontend.

    The signal is a Streamlit-specific internal format. The SSE path must
    normalise it before forwarding so the React client never sees the prefix.
    """
    try:
        payload = json.loads(raw[len("METADATA_LIST:"):])
    except Exception:
        return raw  # unparseable — return as-is rather than crashing
    if not payload:
        return "No videos found matching your query."
    by_topic: dict[str, list[dict]] = {}
    for v in payload:
        by_topic.setdefault(v.get("topic", "Other"), []).append(v)
    count = len(payload)
    lines = [f"Here {'is' if count == 1 else 'are'} {count} video{'s' if count != 1 else ''} I know about:\n"]
    for topic, videos in by_topic.items():
        lines.append(topic)
        for v in videos:
            dur = f" — {v['duration']}" if v.get("duration") else ""
            lines.append(f"  • {v['title']} by {v['channel']}{dur}")
        lines.append("")
    return "\n".join(lines).strip()


def stream_run_chat(question: str, history: list[Turn]) -> Generator[str, None, None]:
    """
    Yield SSE-formatted strings for POST /api/chat/stream.

    Protocol (matches frontend/src/lib/sse.ts):
      data: <token>          — one per LLM token
      data: [SOURCES]<json>  — single frame after tokens end (RAG only)
      data: [DONE]           — always the final frame, including on error

    Exceptions are caught, logged, and surfaced as a final error token before
    [DONE] so the frontend always receives a terminal frame and the chat
    bubble exits the streaming state with a user-readable message.

    METADATA_LIST:<json> tokens (emitted by the metadata intent) are converted
    to plain text before forwarding — the React frontend has no handler for
    that Streamlit-specific signal prefix.
    """
    try:
        agent = _build_agent(history)
        for token in agent.stream_chat(question):
            if not token:
                continue
            if token.startswith("METADATA_LIST:"):
                token = _format_metadata_list(token)
            yield f"data: {token}\n\n"
        chunks = agent._streamed_chunks
        if chunks:
            sources = [
                {
                    "title":        c.title,
                    "timestamp":    c.timestamp_label,
                    "link":         c.youtube_link,
                    "score":        c.score,
                    "rerank_score": c.rerank_score,
                    "text":         c.text,
                }
                for c in chunks
            ]
            yield f"data: [SOURCES]{json.dumps(sources)}\n\n"
    except Exception as exc:
        log.exception("Unhandled error in stream_run_chat")
        err = str(exc).lower()
        if "429" in err or "rate_limit" in err or "too many requests" in err:
            msg = "The service is currently at capacity. Please try again in a few minutes."
        else:
            msg = "An unexpected error occurred while answering your question."
        yield f"data: {msg}\n\n"
    yield "data: [DONE]\n\n"
