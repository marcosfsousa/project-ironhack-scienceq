"""
retriever.py
------------
Pinecone similarity search wrapper for the YouTube QA Bot.

Wraps the Pinecone index with:
  - Namespace-aware querying (corpus | live)
  - Enriched query embedding (title-prepend not needed at query time —
    plain query text is intentionally used; the index handles asymmetry)
  - Top-k retrieval with configurable k
  - Optional topic/channel metadata filtering
  - Returns typed RetrievedChunk objects (not raw Pinecone dicts)
  - Graceful fallback: returns [] instead of raising on empty results

Usage (standalone smoke test):
  python retriever.py --query "How does entropy work?" --k 5
  python retriever.py --query "What is dark matter?" --namespace live
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Env ────────────────────────────────────────────────────────────────────────
load_dotenv()

PINECONE_API_KEY          = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME       = os.getenv("PINECONE_INDEX_NAME", "youtube-qa-bot")
PINECONE_NAMESPACE_CORPUS = os.getenv("PINECONE_NAMESPACE_CORPUS", "corpus")
PINECONE_NAMESPACE_LIVE   = os.getenv("PINECONE_NAMESPACE_LIVE", "live")

EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_TOP_K    = 5


# ── Data types ─────────────────────────────────────────────────────────────────

@dataclass
class RetrievedChunk:
    """
    A single result from a Pinecone similarity search.
    All fields are populated from the vector metadata stored at index time.
    """
    chunk_id:   str
    video_id:   str
    title:      str
    channel:    str
    topic:      str
    start:      float
    end:        float
    text:       str
    score:      float                        # cosine similarity (0–1)
    namespace:  str                          # which namespace this came from

    @property
    def youtube_link(self) -> str:
        """Deep link to the exact timestamp in the video."""
        t = int(self.start)
        return f"https://www.youtube.com/watch?v={self.video_id}&t={t}s"

    @property
    def timestamp_label(self) -> str:
        """Human-readable timestamp range, e.g. '1:02 – 2:03'."""
        def fmt(s: float) -> str:
            m, sec = divmod(int(s), 60)
            return f"{m}:{sec:02d}"
        return f"{fmt(self.start)} – {fmt(self.end)}"

    def to_context_string(self) -> str:
        """
        Formatted string for injection into the LLM prompt.
        Includes source attribution so the model can cite timestamps.
        """
        return (
            f"[Source: \"{self.title}\" | {self.timestamp_label} | "
            f"{self.youtube_link}]\n{self.text}"
        )


# ── Singletons ─────────────────────────────────────────────────────────────────
# Both model and Pinecone index are loaded once per process.

_embed_model: Optional[SentenceTransformer] = None
_pinecone_index = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        log.info(f"Loading embedding model: {EMBED_MODEL_NAME}")
        _embed_model = SentenceTransformer(EMBED_MODEL_NAME)
    return _embed_model


def _get_index():
    global _pinecone_index
    if _pinecone_index is None:
        if not PINECONE_API_KEY:
            raise EnvironmentError(
                "PINECONE_API_KEY not set. Check your .env file."
            )
        pc = Pinecone(api_key=PINECONE_API_KEY)
        _pinecone_index = pc.Index(PINECONE_INDEX_NAME)
        log.info(f"Connected to Pinecone index: {PINECONE_INDEX_NAME}")
    return _pinecone_index


# ── Core retrieval ─────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    *,
    namespace: str = "corpus",
    top_k: int = DEFAULT_TOP_K,
    filter_topic: Optional[str] = None,
    filter_channel: Optional[str] = None,
    score_threshold: float = 0.0,
) -> list[RetrievedChunk]:
    """
    Embed a query and return the top-k most similar chunks from Pinecone.

    Args:
        query:            Natural language question or search text.
        namespace:        Pinecone namespace — "corpus" (pre-built) or "live" (on-the-fly).
                          Accepts the namespace string directly or the convenience
                          constants PINECONE_NAMESPACE_CORPUS / PINECONE_NAMESPACE_LIVE.
        top_k:            Number of results to return (default: 5).
        filter_topic:     Optional metadata filter — only return chunks from this topic
                          (e.g. "Physics", "Biology"). Case-sensitive, matches metadata.json.
        filter_channel:   Optional metadata filter — only return chunks from this channel
                          (e.g. "Veritasium"). Case-sensitive.
        score_threshold:  Minimum cosine similarity score to include a result (default: 0.0).
                          Raise to 0.3–0.4 to filter low-confidence matches.

    Returns:
        List of RetrievedChunk objects, sorted by score descending.
        Returns an empty list if no results meet the threshold.
    """
    if not query.strip():
        log.warning("retrieve() called with empty query — returning []")
        return []

    model = _get_embed_model()
    index = _get_index()

    # Embed the query
    query_vector = model.encode(query, normalize_embeddings=True).tolist()

    # Build optional metadata filter
    pinecone_filter: dict = {}
    if filter_topic:
        pinecone_filter["topic"] = {"$eq": filter_topic}
    if filter_channel:
        pinecone_filter["channel"] = {"$eq": filter_channel}

    # Query Pinecone
    query_kwargs: dict = {
        "vector":          query_vector,
        "top_k":           top_k,
        "namespace":       namespace,
        "include_metadata": True,
    }
    if pinecone_filter:
        query_kwargs["filter"] = pinecone_filter

    try:
        response = index.query(**query_kwargs)
    except Exception as e:
        log.error(f"Pinecone query failed: {e}")
        return []

    # Parse results
    chunks: list[RetrievedChunk] = []
    for match in response.get("matches", []):
        score = match.get("score", 0.0)
        if score < score_threshold:
            continue

        meta = match.get("metadata", {})
        chunk = RetrievedChunk(
            chunk_id  = meta.get("chunk_id",   match["id"]),
            video_id  = meta.get("video_id",   ""),
            title     = meta.get("title",      "Unknown"),
            channel   = meta.get("channel",    "Unknown"),
            topic     = meta.get("topic",      "Unknown"),
            start     = float(meta.get("start", 0.0)),
            end       = float(meta.get("end",   0.0)),
            text      = meta.get("chunk_text", ""),
            score     = round(score, 4),
            namespace = namespace,
        )
        chunks.append(chunk)

    log.info(
        f"Retrieved {len(chunks)} chunk(s) for query "
        f"(namespace={namespace!r}, top_k={top_k}, threshold={score_threshold})"
    )
    return chunks


def retrieve_multi_namespace(
    query: str,
    *,
    top_k: int = DEFAULT_TOP_K,
    score_threshold: float = 0.0,
) -> list[RetrievedChunk]:
    """
    Query both 'corpus' and 'live' namespaces and merge results.
    Useful when the user has ingested a live URL alongside the pre-built corpus.

    Returns top_k results globally (not top_k per namespace), sorted by score.
    """
    corpus_results = retrieve(
        query,
        namespace=PINECONE_NAMESPACE_CORPUS,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    live_results = retrieve(
        query,
        namespace=PINECONE_NAMESPACE_LIVE,
        top_k=top_k,
        score_threshold=score_threshold,
    )
    combined = corpus_results + live_results
    combined.sort(key=lambda c: c.score, reverse=True)
    return combined[:top_k]


def format_context_for_llm(chunks: list[RetrievedChunk]) -> str:
    """
    Format a list of retrieved chunks into a single context block
    ready for injection into the LLM system or user prompt.

    Returns an empty string if chunks is empty.
    """
    if not chunks:
        return ""
    parts = [f"--- Chunk {i+1} ---\n{chunk.to_context_string()}"
             for i, chunk in enumerate(chunks)]
    return "\n\n".join(parts)


# ── CLI smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Smoke-test the Pinecone retriever."
    )
    parser.add_argument("--query",     type=str, required=True,
                        help="Search query")
    parser.add_argument("--k",         type=int, default=DEFAULT_TOP_K,
                        help=f"Number of results (default: {DEFAULT_TOP_K})")
    parser.add_argument("--namespace", type=str, default="corpus",
                        choices=["corpus", "live"],
                        help="Pinecone namespace (default: corpus)")
    parser.add_argument("--topic",     type=str, default=None,
                        help="Filter by topic (e.g. Physics)")
    parser.add_argument("--channel",   type=str, default=None,
                        help="Filter by channel (e.g. Veritasium)")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Minimum similarity score (default: 0.0)")
    args = parser.parse_args()

    results = retrieve(
        args.query,
        namespace=args.namespace,
        top_k=args.k,
        filter_topic=args.topic,
        filter_channel=args.channel,
        score_threshold=args.threshold,
    )

    if not results:
        print("\n⚠  No results returned.")
    else:
        print(f"\n✓ {len(results)} result(s) for query: \"{args.query}\"\n")
        for i, chunk in enumerate(results, 1):
            print(f"{'─'*60}")
            print(f"[{i}] score={chunk.score:.4f}  |  {chunk.title}")
            print(f"    channel : {chunk.channel}  |  topic: {chunk.topic}")
            print(f"    time    : {chunk.timestamp_label}")
            print(f"    link    : {chunk.youtube_link}")
            print(f"    text    : {chunk.text[:200]}{'...' if len(chunk.text) > 200 else ''}")
        print(f"\n{'─'*60}")
        print("\n── Context string for LLM ──────────────────────────────")
        print(format_context_for_llm(results[:2]))
