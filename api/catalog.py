"""
catalog.py
----------
Builds the GET /api/catalog payload by merging two sources:

  corpus — metadata.json from GCS bucket (gs://scienceq-data/metadata.json)
  live   — first-chunk vectors (*_000) from Pinecone live namespace

Live entries take precedence: if the same video_id appears in both,
the live entry wins (it has the freshest metadata).

Falls back gracefully: if either source fails the other is still returned.
"""

from __future__ import annotations

import json
import logging
import os

log = logging.getLogger(__name__)

_BUCKET     = "scienceq-data"
_BLOB       = "metadata.json"
_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "scienceq-prod")
_NS_LIVE    = os.getenv("PINECONE_NAMESPACE_LIVE", "live")


def _parse_metadata(raw: dict | list) -> list[dict]:
    """Mirror the parsing logic from agent/tools.py _load_metadata()."""
    if isinstance(raw, dict) and "videos" in raw:
        videos_dict = raw["videos"]
        return [
            {"video_id": vid_id, **entry} if "video_id" not in entry else entry
            for vid_id, entry in videos_dict.items()
            if isinstance(entry, dict)
        ]
    if isinstance(raw, dict):
        return [
            {"video_id": vid_id, **entry} if "video_id" not in entry else entry
            for vid_id, entry in raw.items()
            if isinstance(entry, dict)
        ]
    return list(raw)


def _corpus_videos() -> list[dict]:
    from google.cloud import storage  # imported here to avoid import-time cost

    client = storage.Client()
    blob   = client.bucket(_BUCKET).blob(_BLOB)
    raw    = json.loads(blob.download_as_text())
    entries = _parse_metadata(raw)
    out = []
    for v in entries:
        video_id = v.get("video_id", "")
        out.append({
            "video_id": video_id,
            "title":    v.get("title",    ""),
            "channel":  v.get("channel",  ""),
            "topic":    v.get("topic",    ""),
            "duration": v.get("duration", ""),
            "url":      f"https://www.youtube.com/watch?v={video_id}",
            "source":   "corpus",
        })
    return out


def _live_videos() -> list[dict]:
    from pinecone import Pinecone  # imported here; Pinecone client is a module singleton in retriever.py

    pc    = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
    index = pc.Index(_INDEX_NAME)

    # Collect all IDs in the live namespace, then filter for first-chunk entries.
    # Chunk IDs follow the pattern <video_id>_<seq> where seq is zero-padded (e.g. _000).
    all_ids: list[str] = []
    for page in index.list(namespace=_NS_LIVE):
        all_ids.extend(page)

    first_chunk_ids = [id_ for id_ in all_ids if id_.endswith("_000")]
    if not first_chunk_ids:
        return []

    # Pinecone fetch accepts at most 1,000 IDs per call.
    _FETCH_BATCH = 1_000
    all_vectors: dict = {}
    for i in range(0, len(first_chunk_ids), _FETCH_BATCH):
        batch_resp = index.fetch(ids=first_chunk_ids[i : i + _FETCH_BATCH], namespace=_NS_LIVE)
        all_vectors.update(batch_resp.vectors)

    out  = []
    for vec in all_vectors.values():
        m        = vec.metadata or {}
        video_id = m.get("video_id", "")
        out.append({
            "video_id": video_id,
            "title":    m.get("title",    ""),
            "channel":  m.get("channel",  ""),
            "topic":    m.get("topic",    ""),
            "duration": m.get("duration", ""),
            "url":      f"https://www.youtube.com/watch?v={video_id}",
            "source":   "live",
        })
    return out


def get_catalog() -> list[dict]:
    """Return merged corpus + live catalog. Live entries deduplicate corpus."""
    live: list[dict] = []
    try:
        live = _live_videos()
    except Exception:
        log.exception("Failed to load live catalog from Pinecone")

    corpus: list[dict] = []
    try:
        corpus = _corpus_videos()
    except Exception:
        log.exception("Failed to load corpus catalog from GCS")

    seen = {v["video_id"] for v in live}
    return live + [v for v in corpus if v["video_id"] not in seen]
