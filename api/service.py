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

from agent import YouTubeQAAgent  # resolved via sys.path setup in api/main.py

from .schemas import ChatResponse, Turn


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
