"""
retriever.py
------------
Pinecone similarity search wrapper for the ScienceQ.

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

import cohere
import dataclasses
from dotenv import load_dotenv
from pinecone import Pinecone

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

COHERE_MODEL  = "embed-multilingual-v3.0"

# ── Retrieval tuning config ────────────────────────────────────────────────────
# Env-driven so the parameter sweep script can patch them at runtime without
# code changes (same pattern as RERANKER_ENABLED).
RERANKER_ENABLED  = os.getenv("RERANKER_ENABLED",  "false").lower() == "true"
RERANKER_MODEL    = "rerank-v3.5"
RETRIEVER_FETCH_K = int(os.getenv("RETRIEVER_FETCH_K", "10"))
RETRIEVER_TOP_N   = int(os.getenv("RETRIEVER_TOP_N",   "5"))
SCORE_THRESHOLD   = float(os.getenv("SCORE_THRESHOLD", "0.40"))


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
    language:   str
    start:      float
    end:        float
    text:       str
    score:      float                        # cosine similarity (0–1)
    namespace:  str                          # which namespace this came from
    rerank_score: Optional[float] = None     # Cohere Rerank relevance score (None when reranker off)

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
# Both Cohere client and Pinecone index are loaded once per process.

_cohere_client: Optional[cohere.Client] = None
_pinecone_index = None


def _get_cohere_client() -> cohere.Client:
    global _cohere_client
    if _cohere_client is None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise EnvironmentError("COHERE_API_KEY not set. Check your .env file.")
        _cohere_client = cohere.Client(api_key=api_key)
        log.info(f"Cohere client initialised. Model: {COHERE_MODEL}")
    return _cohere_client


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

def _rerank(
    query: str,
    chunks: list["RetrievedChunk"],
    top_n: int,
) -> list["RetrievedChunk"]:
    """
    Re-score retrieved chunks with Cohere Rerank and return the top_n.

    Each returned chunk has rerank_score set (the Cohere relevance score, 0–1).
    The original cosine score is preserved on the object for observability.
    Chunks are returned sorted by rerank_score descending.
    """
    if not chunks:
        return chunks
    client = _get_cohere_client()
    resp = client.rerank(
        model     = RERANKER_MODEL,
        query     = query,
        documents = [c.text for c in chunks],
        top_n     = min(top_n, len(chunks)),
    )
    reranked: list[RetrievedChunk] = [
        dataclasses.replace(chunks[r.index], rerank_score=round(r.relevance_score, 4))
        for r in resp.results
    ]

    log.info(
        f"Reranked {len(chunks)} → {len(reranked)} chunks "
        f"(model={RERANKER_MODEL}, top_n={top_n})"
    )
    return reranked


def _embed_query(query: str) -> list[float]:
    """Embed a single query string with search_query input_type."""
    client = _get_cohere_client()
    resp = client.embed(
        texts=[query],
        model=COHERE_MODEL,
        input_type="search_query",
        embedding_types=["float"],
    )
    raw = resp.embeddings
    # Cohere SDK v4 returns a list directly; v5 wraps in EmbeddingList with .float_
    return (raw.float_ if hasattr(raw, "float_") else raw)[0]


def _query_pinecone(
    query_vector: list[float],
    *,
    namespace: str,
    top_k: int,
    filter_topic: Optional[str],
    filter_channel: Optional[str],
    score_threshold: float,
) -> list[RetrievedChunk]:
    """Run a pre-computed vector query against one Pinecone namespace."""
    index = _get_index()

    pinecone_filter: dict = {}
    if filter_topic:
        pinecone_filter["topic"] = {"$eq": filter_topic}
    if filter_channel:
        pinecone_filter["channel"] = {"$eq": filter_channel}

    query_kwargs: dict = {
        "vector":           query_vector,
        "top_k":            top_k,
        "namespace":        namespace,
        "include_metadata": True,
    }
    if pinecone_filter:
        query_kwargs["filter"] = pinecone_filter

    try:
        response = index.query(**query_kwargs)
    except Exception as e:
        log.error(f"Pinecone query failed: {e}")
        return []

    chunks: list[RetrievedChunk] = []
    for match in response.get("matches", []):
        score = match.get("score", 0.0)
        if score < score_threshold:
            continue
        meta = match.get("metadata", {})
        chunks.append(RetrievedChunk(
            chunk_id  = meta.get("chunk_id",   match["id"]),
            video_id  = meta.get("video_id",   ""),
            title     = meta.get("title",      "Unknown"),
            channel   = meta.get("channel",    "Unknown"),
            topic     = meta.get("topic",      "Unknown"),
            language  = meta.get("language",   "en"),
            start     = float(meta.get("start", 0.0)),
            end       = float(meta.get("end",   0.0)),
            text      = meta.get("chunk_text", ""),
            score     = round(score, 4),
            namespace = namespace,
        ))

    log.info(
        f"Retrieved {len(chunks)} chunk(s) "
        f"(namespace={namespace!r}, top_k={top_k}, threshold={score_threshold})"
    )
    return chunks


def retrieve(
    query: str,
    *,
    namespace: str = "corpus",
    top_k: Optional[int] = None,
    filter_topic: Optional[str] = None,
    filter_channel: Optional[str] = None,
    score_threshold: Optional[float] = None,
) -> list[RetrievedChunk]:
    """
    Embed a query and return the top-k most similar chunks from Pinecone.

    Args:
        query:            Natural language question or search text.
        namespace:        Pinecone namespace — "corpus" (pre-built) or "live" (on-the-fly).
        top_k:            Chunks to return after reranking (or directly from Pinecone when
                          reranker is off). Defaults to RETRIEVER_TOP_N env var (default 5).
        filter_topic:     Optional metadata filter — only return chunks from this topic
                          (e.g. "Physics", "Biology"). Case-sensitive, matches metadata.json.
        filter_channel:   Optional metadata filter — only return chunks from this channel
                          (e.g. "Veritasium"). Case-sensitive.
        score_threshold:  Minimum cosine similarity score to include a result.
                          Defaults to SCORE_THRESHOLD env var (default 0.40).

    Returns:
        List of RetrievedChunk objects, sorted by score descending.
        Returns an empty list if no results meet the threshold.
    """
    if not query.strip():
        log.warning("retrieve() called with empty query — returning []")
        return []

    effective_top_k     = top_k if top_k is not None else RETRIEVER_TOP_N
    effective_threshold = score_threshold if score_threshold is not None else SCORE_THRESHOLD

    # When reranking, over-retrieve from Pinecone then let Cohere Rerank filter down.
    fetch_k = RETRIEVER_FETCH_K if RERANKER_ENABLED else effective_top_k
    chunks  = _query_pinecone(
        _embed_query(query),
        namespace=namespace,
        top_k=fetch_k,
        filter_topic=filter_topic,
        filter_channel=filter_channel,
        score_threshold=effective_threshold,
    )

    if RERANKER_ENABLED and chunks:
        chunks = _rerank(query, chunks, top_n=effective_top_k)

    return chunks


def retrieve_multi_namespace(
    query: str,
    *,
    top_k: Optional[int] = None,
    score_threshold: Optional[float] = None,
) -> list[RetrievedChunk]:
    """
    Query both 'corpus' and 'live' namespaces and merge results.
    Embeds the query once and reuses the vector for both namespace queries.

    Returns top_k results globally (not top_k per namespace), sorted by score.
    Defaults for top_k and score_threshold fall through to module-level env vars.
    """
    if not query.strip():
        log.warning("retrieve_multi_namespace() called with empty query — returning []")
        return []

    effective_top_k     = top_k if top_k is not None else RETRIEVER_TOP_N
    effective_threshold = score_threshold if score_threshold is not None else SCORE_THRESHOLD

    query_vector = _embed_query(query)
    # When reranking, over-retrieve from each namespace so the reranker has
    # enough candidates across both corpus and live to choose from.
    fetch_k = RETRIEVER_FETCH_K if RERANKER_ENABLED else effective_top_k
    corpus_results = _query_pinecone(
        query_vector,
        namespace=PINECONE_NAMESPACE_CORPUS,
        top_k=fetch_k,
        filter_topic=None,
        filter_channel=None,
        score_threshold=effective_threshold,
    )
    live_results = _query_pinecone(
        query_vector,
        namespace=PINECONE_NAMESPACE_LIVE,
        top_k=fetch_k,
        filter_topic=None,
        filter_channel=None,
        score_threshold=effective_threshold,
    )
    combined = corpus_results + live_results

    if RERANKER_ENABLED and combined:
        return _rerank(query, combined, top_n=effective_top_k)

    combined.sort(key=lambda c: c.score, reverse=True)
    return combined[:effective_top_k]


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
    parser.add_argument("--k",         type=int, default=RETRIEVER_TOP_N,
                        help=f"Number of results (default: {RETRIEVER_TOP_N})")
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
