"""
agent.py
--------
LangGraph agent entry point for the YouTube QA Bot.

Architecture:
  - Hardcoded RAG-first routing: always calls RAGRetrieverTool first
  - Falls back to VideoMetadataTool for catalog/listing queries
  - ConversationBufferWindowMemory (5 turns) per session
  - LangSmith tracing via environment variables (set in rag_chain.py)

Graph structure:
  [START] → classify_intent → [rag_node | metadata_node] → respond → [END]

Routing logic:
  - METADATA queries: "what videos", "list", "do you have", "what topics",
                      "which channels", "what's available"
  - Everything else  → RAG (default)

Usage:
  python agent.py          # interactive loop, corpus namespace
  python agent.py --debug  # prints full retrieved context per turn
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from memory import create_memory
from tools import get_tools, RAGRetrieverTool, VideoMetadataTool
from retriever import PINECONE_NAMESPACE_CORPUS

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

load_dotenv()

# ── Intent classification ──────────────────────────────────────────────────────

# Keywords that signal a metadata/catalog query rather than a factual question
METADATA_INTENT_KEYWORDS = [
    "what videos",
    "which videos",
    "list videos",
    "what topics",
    "which topics",
    "what channels",
    "which channels",
    "do you have",
    "what do you have",
    "what's available",
    "what is available",
    "show me videos",
    "browse",
    "catalog",
]


RESOLVE_SYSTEM = (
    "You are a query resolver. Given a conversation history and a metadata query, "
    "extract the specific topic, channel, or keyword the user is asking about. "
    "Return ONLY the resolved search term (e.g. 'Physics', 'Veritasium', 'neural networks'). "
    "No explanation, no preamble."
)


def resolve_metadata_query(question: str, history: list) -> str:
    """
    Resolve a vague metadata query like 'what videos do you have about that topic?'
    into a concrete search term using conversation history.

    Uses llama-3.1-8b-instant (separate Groq rate limit bucket).
    Falls back to original question on any error.
    """
    if not history:
        return question

    history_text = ""
    for msg in history:
        role = "Human" if msg.__class__.__name__ == "HumanMessage" else "Assistant"
        history_text += f"{role}: {msg.content}\n"

    user_prompt = (
        f"Conversation history:\n{history_text.strip()}\n\n"
        f"Metadata query: {question}\n"
        f"Resolved search term:"
    )

    try:
        from langchain_groq import ChatGroq
        from langchain_core.messages import HumanMessage as HMsg, SystemMessage as SMsg
        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY", ""),
            model="llama-3.1-8b-instant",
            temperature=0.0,
            max_tokens=32,
        )
        response = llm.invoke([SMsg(content=RESOLVE_SYSTEM), HMsg(content=user_prompt)])
        resolved = response.content.strip()
        if resolved and resolved != question:
            log.info(f"Metadata query resolved: {question!r} -> {resolved!r}")
        return resolved or question
    except Exception as e:
        log.warning(f"Metadata resolve failed ({e}) — using original query.")
        return question


def classify_intent(question: str) -> str:
    """
    Classify whether a question needs RAG retrieval or metadata lookup.

    Returns:
      "metadata"  — question is about what's in the library
      "rag"       — question is about content/concepts (default)
    """
    q = question.lower().strip()
    if any(kw in q for kw in METADATA_INTENT_KEYWORDS):
        return "metadata"
    return "rag"


# ── LangGraph state ────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    question:    str
    answer:      str
    intent:      str
    tool_output: str
    messages:    Annotated[list[BaseMessage], add_messages]


# ── Graph nodes ────────────────────────────────────────────────────────────────

def make_classify_node(debug: bool = False):
    def classify_node(state: AgentState) -> AgentState:
        intent = classify_intent(state["question"])
        log.info(f"Intent classified: {intent!r} for question: {state['question']!r}")
        return {**state, "intent": intent}
    return classify_node


def make_rag_node(rag_tool: RAGRetrieverTool, debug: bool = False):
    def rag_node(state: AgentState) -> AgentState:
        log.info("Routing to RAGRetrieverTool")
        tool_output = rag_tool._run(state["question"])
        if debug:
            print(f"\n[DEBUG] RAG tool output:\n{tool_output}\n")
        # Extract just the answer portion for the final response
        answer = _extract_answer(tool_output)
        return {**state, "tool_output": tool_output, "answer": answer}
    return rag_node


def make_metadata_node(metadata_tool: VideoMetadataTool, debug: bool = False):
    def metadata_node(state: AgentState) -> AgentState:
        log.info("Routing to VideoMetadataTool")
        # Resolve vague references using conversation history
        history = state.get("messages", [])
        resolved_query = resolve_metadata_query(state["question"], history)
        tool_output = metadata_tool._run(resolved_query)
        if debug:
            print(f"\n[DEBUG] Metadata tool output:\n{tool_output}\n")
        # Strip internal prefix — not meant for end users
        display = tool_output
        for prefix in ("METADATA RESULT: ", "METADATA RESULT:\n"):
            if display.startswith(prefix):
                display = display[len(prefix):]
                break
        return {**state, "tool_output": tool_output, "answer": display}
    return metadata_node


def make_respond_node():
    def respond_node(state: AgentState) -> AgentState:
        """Final node — answer is already set by the tool node. Pass through."""
        return state
    return respond_node


# ── Routing function ───────────────────────────────────────────────────────────

def route_intent(state: AgentState) -> str:
    """LangGraph conditional edge — routes based on classified intent."""
    return state.get("intent", "rag")


# ── Answer extraction ──────────────────────────────────────────────────────────

def _extract_answer(tool_output: str) -> str:
    """
    Extract the clean answer text from RAGRetrieverTool's formatted output.
    Strips the 'RETRIEVAL RESULT:' header and 'Sources:' footer for display.
    """
    if "RETRIEVAL RESULT: No relevant content" in tool_output:
        # Return the response line directly
        for line in tool_output.splitlines():
            if line.startswith("Response:"):
                return line.replace("Response:", "").strip()
        return tool_output

    # Extract the Answer block
    if "Answer:\n" in tool_output and "\n\nSources:" in tool_output:
        answer_start = tool_output.index("Answer:\n") + len("Answer:\n")
        answer_end   = tool_output.index("\n\nSources:")
        return tool_output[answer_start:answer_end].strip()

    # Fallback — return full output
    return tool_output


def _extract_sources(tool_output: str) -> str:
    """Extract the Sources block from RAGRetrieverTool output for display."""
    if "\n\nSources:\n" in tool_output:
        return tool_output.split("\n\nSources:\n", 1)[1].strip()
    return ""


# ── Agent class ────────────────────────────────────────────────────────────────

class YouTubeQAAgent:
    """
    LangGraph-based agent for YouTube video question answering.

    Each instance owns its own:
      - LangGraph compiled graph
      - Conversation memory (5-turn window)
      - Tool instances (RAGRetrieverTool, VideoMetadataTool)

    Create one instance per user session.
    """

    def __init__(
        self,
        namespace: str = PINECONE_NAMESPACE_CORPUS,
        debug: bool = False,
    ):
        self.namespace = namespace
        self.debug     = debug
        self.memory    = create_memory()

        # Instantiate tools
        tools = get_tools(namespace=namespace)
        self.rag_tool      : RAGRetrieverTool  = tools[0]
        self.metadata_tool : VideoMetadataTool = tools[1]

        # Build and compile the graph
        self.graph = self._build_graph()
        log.info(f"YouTubeQAAgent ready | namespace: {namespace}")

    def _build_graph(self):
        builder = StateGraph(AgentState)

        # Add nodes
        builder.add_node("classify",  make_classify_node(self.debug))
        builder.add_node("rag",       make_rag_node(self.rag_tool, self.debug))
        builder.add_node("metadata",  make_metadata_node(self.metadata_tool, self.debug))
        builder.add_node("respond",   make_respond_node())

        # Edges
        builder.add_edge(START, "classify")
        builder.add_conditional_edges(
            "classify",
            route_intent,
            {"rag": "rag", "metadata": "metadata"},
        )
        builder.add_edge("rag",      "respond")
        builder.add_edge("metadata", "respond")
        builder.add_edge("respond",  END)

        return builder.compile()

    def ask(self, question: str) -> dict:
        """
        Ask a question and return the agent's response.

        Args:
            question: The user's question.

        Returns:
            dict with keys:
              - answer:   str — the agent's response
              - sources:  str — formatted source citations (empty for metadata queries)
              - intent:   str — "rag" or "metadata"
              - grounded: bool — False if no relevant chunks were found
        """
        # Inject current memory into the RAG tool before invoking
        self.rag_tool.history = self.memory.to_history()

        # Run the graph
        initial_state: AgentState = {
            "question":    question,
            "answer":      "",
            "intent":      "",
            "tool_output": "",
            "messages":    self.memory.to_history(),
        }
        result = self.graph.invoke(initial_state)

        answer      = result.get("answer", "")
        tool_output = result.get("tool_output", "")
        intent      = result.get("intent", "rag")
        sources     = _extract_sources(tool_output) if intent == "rag" else ""
        grounded    = "RETRIEVAL RESULT: No relevant content" not in tool_output

        # Save turn to memory
        self.memory.save_turn(question, answer)

        return {
            "answer":   answer,
            "sources":  sources,
            "intent":   intent,
            "grounded": grounded,
        }

    def reset(self) -> None:
        """Clear conversation memory. Call this for a 'New conversation' action."""
        self.memory.clear()
        log.info("Agent memory reset.")


# ── CLI interactive loop ───────────────────────────────────────────────────────

def run_interactive(debug: bool = False) -> None:
    """
    Interactive terminal loop for testing the agent.
    Preserves conversation memory across turns.
    Type 'exit' or 'quit' to stop.
    Type 'reset' to clear memory and start a new conversation.
    Type 'sources' to show sources from the last answer.
    """
    print("\n" + "═"*60)
    print("  YouTube QA Bot — Interactive Mode")
    print("  Commands: 'exit' | 'reset' | 'sources'")
    print("═"*60 + "\n")

    agent = YouTubeQAAgent(debug=debug)
    last_sources = ""

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question:
            continue

        if question.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        if question.lower() == "reset":
            agent.reset()
            last_sources = ""
            print("── Memory cleared. Starting new conversation. ──\n")
            continue

        if question.lower() == "sources":
            if last_sources:
                print(f"\nSources:\n{last_sources}\n")
            else:
                print("No sources from last answer.\n")
            continue

        # Ask the agent
        result = agent.ask(question)

        print(f"\nBot: {result['answer']}\n")

        if result["sources"]:
            last_sources = result["sources"]
            print(f"── Sources ──────────────────────────────────────")
            print(result["sources"])
            print()

        if not result["grounded"]:
            print("── [No relevant content found in video library] ──\n")


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YouTube QA Bot — interactive agent.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Print full tool output for each turn",
    )
    args = parser.parse_args()
    run_interactive(debug=args.debug)
