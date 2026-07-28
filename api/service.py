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

import itertools
import json
import logging
from typing import Generator

from agent import YouTubeQAAgent  # resolved via sys.path setup in api/main.py

from .excerpt import quotation_excerpt
from .schemas import ChatResponse, GenerationMetadata, Turn

log = logging.getLogger(__name__)

# Sentinel distinguishing an exhausted generator from a falsy first token.
_STREAM_EMPTY = object()


def _provenance_dict(prov) -> dict:
    """
    Normalise a GenerationProvenance (or None) into the wire/schema shape.

    Defaults conservatively — an absent provenance is reported as non-generated
    static text rather than silently claiming an AI origin.
    """
    return {
        "ai_generated": bool(getattr(prov, "ai_generated", False)),
        "model":        getattr(prov, "model", None),
        "mode":         getattr(prov, "mode", "static"),
    }


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


def _trim_source_excerpts(sources: list[dict]) -> list[dict]:
    """
    Shorten each source's `text` to a quotation while leaving every other
    field alone. The source object's shape is unchanged — only the text field's
    content gets shorter — so existing API clients keep working (issue #16).

    Both response paths funnel through here — `run_chat` and the streaming
    `[SOURCES]` frame — so the two cannot drift apart.
    """
    return [{**s, "text": quotation_excerpt(s.get("text"))} for s in sources]


def run_chat(message: str, history: list[Turn]) -> ChatResponse:
    """Answer a single message (blocking) with conversation context."""
    agent = _build_agent(history)
    resp = agent.chat(message)
    return ChatResponse(
        answer=resp.answer,
        sources=_trim_source_excerpts(resp.sources),
        intent=resp.intent,
        generation=GenerationMetadata(**_provenance_dict(resp.provenance)),
    )


def _format_metadata_list(raw: str, question: str = "") -> str:
    """
    Convert the METADATA_LIST:<json> signal from VideoMetadataTool into
    human-readable plain text for the React frontend.

    The signal is an agent-internal format, never part of the wire protocol.
    The SSE path must normalise it before forwarding so the client never sees
    the prefix.

    If the question mentions a topic that matches a known topic in the corpus,
    the list is filtered to that topic only.
    """
    try:
        payload = json.loads(raw[len("METADATA_LIST:"):])
    except Exception:
        return raw  # unparseable — return as-is rather than crashing
    if not payload:
        return "No videos found matching your query."

    # Build topic index and attempt topic-based filtering from the question.
    all_topics = {v.get("topic", "Other") for v in payload}
    q_lower = question.lower()
    matched_topic = next(
        (t for t in all_topics if t and t.lower() in q_lower), None
    )
    if matched_topic:
        payload = [v for v in payload if v.get("topic", "Other") == matched_topic]

    by_topic: dict[str, list[dict]] = {}
    for v in payload:
        by_topic.setdefault(v.get("topic", "Other"), []).append(v)

    count = len(payload)
    topic_label = f" on {matched_topic}" if matched_topic else ""
    lines = [f"Here {'is' if count == 1 else 'are'} {count} video{'s' if count != 1 else ''}{topic_label} in the corpus:\n"]
    for topic, videos in by_topic.items():
        lines.append(f"{topic}  ({len(videos)})")
        for v in videos:
            dur = f"  {v['duration']}" if v.get("duration") else ""
            lines.append(f"  • {v['title']} — {v['channel']}{dur}")
        lines.append("")
    return "\n".join(lines).strip()


def stream_run_chat(question: str, history: list[Turn]) -> Generator[str, None, None]:
    """
    Yield SSE-formatted strings for POST /api/chat/stream.

    Protocol (matches frontend/src/lib/sse.ts):
      data: [META]<json>     — generation provenance, emitted before any token
      data: <token>          — one per LLM token
      data: [SOURCES]<json>  — single frame after tokens end (RAG only)
      data: [DONE]           — always the final frame, including on error

    Exceptions are caught, logged, and surfaced as a final error token before
    [DONE] so the frontend always receives a terminal frame and the chat
    bubble exits the streaming state with a user-readable message.

    METADATA_LIST:<json> tokens (emitted by the metadata intent) are converted
    to plain text before forwarding — the signal is agent-internal and the
    frontend has no handler for that prefix.
    """
    try:
        agent = _build_agent(history)
        stream = agent.stream_chat(question)

        # Advance to the first token before emitting anything. stream_chat sets
        # _last_provenance after intent classification / retrieval but before it
        # yields, so pulling one item populates provenance without forwarding a
        # token — letting the provenance frame lead the stream (Art. 50(2)).
        first = next(stream, _STREAM_EMPTY)
        meta = _provenance_dict(agent._last_provenance)
        yield f"data: [META]{json.dumps(meta)}\n\n"

        tokens = () if first is _STREAM_EMPTY else itertools.chain((first,), stream)
        for token in tokens:
            if not token:
                continue
            if token.startswith("METADATA_LIST:"):
                token = _format_metadata_list(token, question=question)
            # Encode multi-line tokens correctly: each line needs its own "data: " prefix.
            sse_data = "\n".join(f"data: {line}" for line in token.split("\n"))
            yield f"{sse_data}\n\n"
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
            sources = _trim_source_excerpts(sources)
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
