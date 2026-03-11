"""
memory.py
---------
Conversation memory for the ScienceQ Bot agent.

Manages a sliding window of the last K conversation turns as a plain
list of LangChain message objects. No external dependencies beyond
langchain-core (already required by the rest of the project).

Exposes:
  create_memory()              → fresh ConversationMemory instance
  ConversationMemory methods:
    .save_turn(human, ai)      → add a turn, drop oldest if over window
    .to_history()              → list of messages for prompt injection
    .clear()                   → reset all history

Design decisions:
  - Plain list instead of ConversationBufferWindowMemory (no langchain-community needed)
  - Fresh instance per session (no singleton — avoids cross-session bleed)
  - 5-turn window (configurable via MEMORY_WINDOW_K)
  - In-process only: resets on restart (sufficient for demo)
  - to_history() output is compatible with MessagesPlaceholder("history")
    in rag_chain.py's ChatPromptTemplate

Usage:
  from memory import create_memory

  memory = create_memory()
  memory.save_turn("How does a neural network learn?", "A neural network learns by...")
  history = memory.to_history()   # pass to answer() or stream_answer()
  memory.clear()
"""

from __future__ import annotations

import logging

from langchain_core.messages import AIMessage, HumanMessage, BaseMessage

# ── Logging ────────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)

# ── Config ─────────────────────────────────────────────────────────────────────
MEMORY_WINDOW_K = 5   # Number of conversation turns (human + AI pairs) to retain


# ── Memory class ───────────────────────────────────────────────────────────────

class ConversationMemory:
    """
    Sliding window conversation memory backed by a plain list.

    Stores up to K turns (each turn = one HumanMessage + one AIMessage).
    Oldest turns are dropped automatically when the window is exceeded.
    """

    def __init__(self, k: int = MEMORY_WINDOW_K):
        self.k        = k
        self._turns:  list[tuple[str, str]] = []   # (human_input, ai_output) pairs

    def save_turn(self, human_input: str, ai_output: str) -> None:
        """
        Save a conversation turn and trim to window size.

        Args:
            human_input: The user's question.
            ai_output:   The agent's response.
        """
        self._turns.append((human_input, ai_output))
        if len(self._turns) > self.k:
            self._turns.pop(0)   # Drop oldest turn
        log.debug(f"Memory: {len(self._turns)}/{self.k} turns stored.")

    def to_history(self) -> list[BaseMessage]:
        """
        Return conversation history as a flat list of LangChain message objects.
        Compatible with MessagesPlaceholder("history") in ChatPromptTemplate.

        Returns an empty list if no turns have been saved yet.
        """
        messages: list[BaseMessage] = []
        for human_input, ai_output in self._turns:
            messages.append(HumanMessage(content=human_input))
            messages.append(AIMessage(content=ai_output))
        return messages

    def clear(self) -> None:
        """Clear all conversation history."""
        self._turns.clear()
        log.info("Conversation memory cleared.")

    @property
    def turn_count(self) -> int:
        """Number of turns currently stored."""
        return len(self._turns)

    def __repr__(self) -> str:
        return f"ConversationMemory(k={self.k}, turns={self.turn_count})"


# ── Factory ────────────────────────────────────────────────────────────────────

def create_memory(k: int = MEMORY_WINDOW_K) -> ConversationMemory:
    """
    Create a fresh ConversationMemory instance for a new session.

    Args:
        k: Number of turns to retain (default: 5).

    Returns:
        Empty ConversationMemory ready to use.
    """
    return ConversationMemory(k=k)
