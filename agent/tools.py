"""
tools.py
--------
LangChain tools for the YouTube QA Bot LangGraph agent.

Tools defined here:
  - RAGRetrieverTool     → answers questions using Pinecone + Groq RAG chain
  - VideoMetadataTool    → looks up video catalog info from metadata.json

Design decision:
  Tools return formatted strings (not dicts) so the LLM can reason over
  the output naturally. Structured data is embedded in the string in a
  readable format rather than raw JSON.

Usage:
  from tools import get_tools
  tools = get_tools(namespace="corpus")
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Type

from dotenv import load_dotenv
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from rag_chain import answer, RAGResponse, DEFAULT_SCORE_THRESHOLD
from retriever import PINECONE_NAMESPACE_CORPUS, PINECONE_NAMESPACE_LIVE

# ── Logging ────────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)

# ── Env ────────────────────────────────────────────────────────────────────────
load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────────
# tools.py lives in agent/, metadata.json lives in data/
DATA_DIR      = Path(__file__).parent.parent / "data"
METADATA_PATH = DATA_DIR / "metadata.json"


# ── Metadata loader ────────────────────────────────────────────────────────────

def _load_metadata() -> list[dict]:
    """
    Load metadata.json. Returns empty list if file not found.
    Called once per tool invocation — file is small (~28 entries), no caching needed.
    Handles both formats:
      - dict keyed by video_id: {"abc123": {"title": ...}, ...}
      - list of dicts:          [{"video_id": "abc123", "title": ...}, ...]
    """
    if not METADATA_PATH.exists():
        log.warning(f"metadata.json not found at {METADATA_PATH}")
        return []
    data = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    # Handle nested structure: {"videos": {"abc123": {...}, ...}, "created_at": ...}
    if isinstance(data, dict) and "videos" in data:
        videos_dict = data["videos"]
        return [
            {"video_id": vid_id, **entry} if "video_id" not in entry else entry
            for vid_id, entry in videos_dict.items()
            if isinstance(entry, dict)
        ]
    # Fallback: flat dict keyed by video_id
    if isinstance(data, dict):
        return [
            {"video_id": vid_id, **entry} if "video_id" not in entry else entry
            for vid_id, entry in data.items()
            if isinstance(entry, dict)
        ]
    return data

BROWSE_TRIGGERS = [
    "available videos", "all videos", "list videos", "what videos",
    "show videos", "what do you have", "everything", "full list",
    "what's indexed", "what is indexed", "show me everything",
    "browse", "what can i ask", "indexed videos",
]

# ══════════════════════════════════════════════════════════════════════════════
# Tool 1: RAGRetrieverTool
# ══════════════════════════════════════════════════════════════════════════════

class RAGRetrieverInput(BaseModel):
    question: str = Field(
        description=(
            "The user's question about a YouTube video or scientific topic. "
            "Should be a complete, self-contained question."
        )
    )


class RAGRetrieverTool(BaseTool):
    """
    Answers questions by searching the YouTube transcript corpus using
    semantic similarity (Pinecone) and generating a grounded response (Groq LLM).

    Always call this tool first for any factual or conceptual question.
    Returns the answer with source video titles and timestamps.
    """

    name: str = "rag_retriever"
    description: str = (
        "Use this tool to answer any question about scientific topics, concepts, "
        "or ideas covered in the YouTube video library. "
        "Input should be the user's question as a complete sentence. "
        "Returns a grounded answer with source video titles and timestamps. "
        "Always try this tool first before VideoMetadataTool."
    )
    args_schema: Type[BaseModel] = RAGRetrieverInput

    # These are set at instantiation time via get_tools()
    namespace:        str   = PINECONE_NAMESPACE_CORPUS
    top_k:            int   = 5
    score_threshold:  float = DEFAULT_SCORE_THRESHOLD
    multi_namespace:  bool  = False

    # Conversation history injected by the agent before each tool call
    history: list = Field(default_factory=list)

    def _run(self, question: str) -> str:
        """
        Run the RAG chain and return a formatted string result.
        The agent receives this string and incorporates it into its response.
        """
        log.info(f"RAGRetrieverTool called | question: {question!r}")

        result: RAGResponse = answer(
            question,
            namespace=self.namespace,
            top_k=self.top_k,
            score_threshold=self.score_threshold,
            history=self.history,
            multi_namespace=self.multi_namespace,
        )

        if not result.grounded:
            return (
                "RETRIEVAL RESULT: No relevant content found in the video library.\n"
                f"Response: {result.answer}"
            )

        # Format sources for the agent to reference
        source_lines = "\n".join(
            f"  - \"{chunk['title']}\" [{chunk['timestamp']}] "
            f"(score: {chunk['score']:.2f}) → {chunk['link']}"
            for chunk in result.source_chunks_for_display
        )

        return (
            f"RETRIEVAL RESULT: Found {len(result.chunks)} relevant chunk(s).\n\n"
            f"Answer:\n{result.answer}\n\n"
            f"Sources:\n{source_lines}"
        )

    async def _arun(self, question: str) -> str:
        raise NotImplementedError("Async not supported — use _run")


# ══════════════════════════════════════════════════════════════════════════════
# Tool 2: VideoMetadataTool
# ══════════════════════════════════════════════════════════════════════════════

class VideoMetadataInput(BaseModel):
    query: str = Field(
        description=(
            "A search query to find videos in the catalog. Can be: "
            "a topic name (e.g. 'Physics', 'Biology'), "
            "a channel name (e.g. 'Veritasium', 'Kurzgesagt'), "
            "a keyword from a video title, "
            "or 'all' to list every video in the library."
        )
    )


class VideoMetadataTool(BaseTool):
    """
    Looks up video metadata from the local catalog (metadata.json).
    Use this tool when the user asks:
      - What videos do you have about [topic]?
      - Which channels are in your library?
      - Do you have any videos about [subject]?
      - List all available videos.
    Do NOT use this tool to answer conceptual questions — use RAGRetrieverTool for those.
    """

    name: str = "video_metadata"
    description: str = (
        "Use this tool to look up what videos are available in the library. "
        "Input can be a topic (e.g. 'Physics'), a channel name (e.g. 'Veritasium'), "
        "a keyword from a video title, or 'all' to list everything. "
        "Use this when the user asks what content is available, not to answer "
        "factual questions about science topics."
    )
    args_schema: Type[BaseModel] = VideoMetadataInput

    def _run(self, query: str) -> str:
        """
        Search metadata.json and return matching video entries as a
        formatted string the agent can present directly to the user.
        """
        log.info(f"VideoMetadataTool called | query: {query!r}")

        videos = _load_metadata()
        if not videos:
            return "METADATA RESULT: Video catalog not available."

        query_lower = query.lower().strip()

        # 'all' → return full catalog
        if query_lower == "all" or any(trigger in query_lower for trigger in BROWSE_TRIGGERS):
            matches = videos
                    
        else:
            matches = [
                v for v in videos
                if (
                    query_lower in v.get("title",   "").lower() or
                    query_lower in v.get("topic",   "").lower() or
                    query_lower in v.get("channel", "").lower()
                )
            ]

        if not matches:
            # Try a looser match — any word in the query
            words = [w for w in query_lower.split() if len(w) > 3]
            matches = [
                v for v in videos
                if any(
                    w in v.get("title",   "").lower() or
                    w in v.get("topic",   "").lower() or
                    w in v.get("channel", "").lower()
                    for w in words
                )
            ]

        if not matches:
            # Summarise available topics and channels as a fallback
            topics   = sorted({v.get("topic",   "Unknown") for v in videos})
            channels = sorted({v.get("channel", "Unknown") for v in videos})
            return (
                f"METADATA RESULT: No videos found matching '{query}'.\n\n"
                f"Available topics:   {', '.join(topics)}\n"
                f"Available channels: {', '.join(channels)}"
            )

        # Format results
        label = "in the catalog" if any(t in query_lower for t in BROWSE_TRIGGERS) else f"matching '{query}'"
        lines = [f"METADATA RESULT: Found {len(matches)} video(s) {label}:\n"]
        for v in matches:
            video_id = v.get("video_id", "")
            title    = v.get("title",    "Unknown")
            channel  = v.get("channel",  "Unknown")
            topic    = v.get("topic",    "Unknown")
            duration = v.get("duration", "Unknown")
            url      = f"https://www.youtube.com/watch?v={video_id}" if video_id else "N/A"
            lines.append(
                f"  • {title}\n"
                f"    Channel: {channel} | Topic: {topic} | Duration: {duration}\n"
                f"    URL: {url}"
            )

        return "\n".join(lines)

    async def _arun(self, query: str) -> str:
        raise NotImplementedError("Async not supported — use _run")


# ══════════════════════════════════════════════════════════════════════════════
# Tool factory
# ══════════════════════════════════════════════════════════════════════════════

def get_tools(
    namespace: str = PINECONE_NAMESPACE_CORPUS,
    top_k: int = 5,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    multi_namespace: bool = False,
    history: Optional[list] = None,
) -> list[BaseTool]:
    """
    Instantiate and return all agent tools.

    Args:
        namespace:       Pinecone namespace for RAG retrieval.
        top_k:           Number of chunks to retrieve.
        score_threshold: Minimum similarity score for retrieval.
        multi_namespace: Query both corpus + live namespaces if True.
        history:         Conversation history for multi-turn context.

    Returns:
        [RAGRetrieverTool, VideoMetadataTool]
    """
    rag_tool = RAGRetrieverTool(
        namespace=namespace,
        top_k=top_k,
        score_threshold=score_threshold,
        multi_namespace=multi_namespace,
        history=history or [],
    )
    metadata_tool = VideoMetadataTool()

    return [rag_tool, metadata_tool]


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke-test agent tools.")
    parser.add_argument("--rag",      type=str, default=None,
                        help="Test RAGRetrieverTool with this question")
    parser.add_argument("--metadata", type=str, default=None,
                        help="Test VideoMetadataTool with this query (try 'all')")
    args = parser.parse_args()

    tools = get_tools()
    rag_tool, meta_tool = tools[0], tools[1]

    if args.rag:
        print(f"\n{'═'*60}")
        print(f"RAGRetrieverTool | question: {args.rag}")
        print('═'*60)
        print(rag_tool._run(args.rag))

    if args.metadata:
        print(f"\n{'═'*60}")
        print(f"VideoMetadataTool | query: {args.metadata}")
        print('═'*60)
        print(meta_tool._run(args.metadata))

    if not args.rag and not args.metadata:
        print("Usage:")
        print('  python tools.py --rag "How does a neural network learn?"')
        print('  python tools.py --metadata "Physics"')
        print('  python tools.py --metadata all')