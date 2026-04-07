"""
embedder.py
-----------
Embeds transcript chunks produced by chunker.py using Cohere's API.
Reads  → data/videos/{video_id}/chunks.json
Writes → data/videos/{video_id}/embeddings.json

Why write embeddings to disk?
  Cohere charges per token. Persisting vectors lets you re-index into Pinecone
  (e.g. after tweaking namespaces or metadata) without paying to re-embed.

Embedding model:
  cohere embed-multilingual-v3.0
  - 1024-dimensional vectors
  - Supports 100+ languages (ready for Phase 6 multilingual corpus)
  - input_type="search_document" for asymmetric retrieval

Output format (embeddings.json):
  {
    "video_id":      "...",
    "model":         "embed-multilingual-v3.0",
    "dimension":     1024,
    "embedded_at":   "...",
    "chunk_count":   N,
    "embeddings": [
      {
        "chunk_id": "aircAruvnKk_000",
        "vector":   [0.123, ...]   // 1024 floats
      },
      ...
    ]
  }

Guarantees:
  - chunks.json is NEVER modified
  - Resume support: skips videos where embeddings.json already exists
  - Progress log written to data/logs/embedding_log.json

Usage:
  python embedder.py                        # embed all videos
  python embedder.py --video-id aircAruvnKk # embed one video
  python embedder.py --dry-run              # print stats, write nothing
  python embedder.py --force                # re-embed even if already done
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import cohere
from dotenv import load_dotenv

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR   = Path("data")
VIDEOS_DIR = DATA_DIR / "videos"
LOGS_DIR   = DATA_DIR / "logs"
EMBED_LOG  = LOGS_DIR / "embedding_log.json"

# ── Model config ───────────────────────────────────────────────────────────────
COHERE_MODEL     = "embed-multilingual-v3.0"
DIMENSION        = 1024
COHERE_BATCH_SIZE = 96   # Cohere API limit per request


# ── Cohere client (singleton) ──────────────────────────────────────────────────

_co: cohere.Client | None = None

def _get_client() -> cohere.Client:
    global _co
    if _co is None:
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            raise EnvironmentError("COHERE_API_KEY not set. Add it to your .env file.")
        _co = cohere.Client(api_key=api_key)
        log.info(f"Cohere client initialised. Model: {COHERE_MODEL} ({DIMENSION}d)")
    return _co


def _embed(texts: list[str]) -> list[list[float]]:
    """
    Embed texts with input_type='search_document'. Batches in groups of 96.
    Returns a list of 1024-float vectors in the same order as the input.
    """
    client = _get_client()
    all_vectors: list[list[float]] = []
    for i in range(0, len(texts), COHERE_BATCH_SIZE):
        resp = client.embed(
            texts=texts[i : i + COHERE_BATCH_SIZE],
            model=COHERE_MODEL,
            input_type="search_document",
            embedding_types=["float"],
        )
        raw = resp.embeddings
        # SDK v5: resp.embeddings.float_ | SDK v4: resp.embeddings (already a list)
        batch = raw.float_ if hasattr(raw, "float_") else raw
        all_vectors.extend(batch)
    return all_vectors


# ── Per-video embedding ────────────────────────────────────────────────────────

def embed_video(
    video_id: str,
    force:    bool,
    dry_run:  bool,
) -> dict | None:
    """
    Load chunks.json, embed all chunk texts via Cohere, write embeddings.json.
    Returns a stats dict on success, None if chunks.json not found.
    """
    chunks_path = VIDEOS_DIR / video_id / "chunks.json"
    embed_path  = VIDEOS_DIR / video_id / "embeddings.json"

    if not chunks_path.exists():
        log.warning(f"  chunks.json not found, skipping: {chunks_path}")
        return None

    # Resume support — skip if already done with same model (unless --force)
    if embed_path.exists() and not force:
        existing = json.loads(embed_path.read_text(encoding="utf-8"))
        if existing.get("model") == COHERE_MODEL:
            log.info(f"  Already embedded ({COHERE_MODEL}), skipping: {video_id}")
            return {"video_id": video_id, "skipped": True, "reason": "already_embedded"}
        log.info(f"  Model mismatch in embeddings.json — re-embedding: {video_id}")

    data   = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = data["chunks"]

    if not chunks:
        log.warning(f"  No chunks found in {chunks_path}, skipping.")
        return None

    chunk_ids = [c["chunk_id"] for c in chunks]
    texts     = [c["text"]     for c in chunks]

    log.info(f"  Embedding {len(texts)} chunks via Cohere...")
    vectors = _embed(texts)

    embedded_at = datetime.utcnow().isoformat() + "Z"
    stats = {
        "video_id":    video_id,
        "chunk_count": len(chunks),
        "dimension":   DIMENSION,
        "model":       COHERE_MODEL,
        "embedded_at": embedded_at,
    }

    if dry_run:
        log.info("  DRY RUN — embeddings computed but not written.")
        return stats

    output = {
        "video_id":    video_id,
        "model":       COHERE_MODEL,
        "dimension":   DIMENSION,
        "embedded_at": embedded_at,
        "chunk_count": len(chunks),
        "embeddings": [
            {"chunk_id": cid, "vector": vec}
            for cid, vec in zip(chunk_ids, vectors)
        ],
    }

    embed_path.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return stats


# ── Log helpers ────────────────────────────────────────────────────────────────

def load_embed_log() -> dict:
    if EMBED_LOG.exists():
        return json.loads(EMBED_LOG.read_text())
    return {"embedded": [], "skipped": [], "failed": []}


def save_embed_log(log_data: dict) -> None:
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    EMBED_LOG.write_text(json.dumps(log_data, indent=2))


# ── Main ───────────────────────────────────────────────────────────────────────

def run(
    video_id: str | None,
    force:    bool,
    dry_run:  bool,
) -> None:

    if dry_run:
        log.info("DRY RUN — no files will be written.")

    targets = (
        [video_id] if video_id
        else sorted(p.name for p in VIDEOS_DIR.iterdir() if p.is_dir())
    )
    if not targets:
        log.warning(f"No video folders found in {VIDEOS_DIR}")
        return

    log.info(f"Found {len(targets)} video(s) to embed.")
    embed_log  = load_embed_log()
    total_ok   = 0
    total_skip = 0
    total_fail = 0

    for vid_id in targets:
        log.info(f"Embedding: {vid_id}")
        try:
            stats = embed_video(vid_id, force=force, dry_run=dry_run)

            if stats is None:
                embed_log["skipped"].append({
                    "video_id":   vid_id,
                    "reason":     "chunks_file_missing",
                    "skipped_at": datetime.utcnow().isoformat() + "Z",
                })
                total_skip += 1
                continue

            if stats.get("skipped"):
                total_skip += 1
                continue

            log.info(
                f"  ✓ {stats['chunk_count']} chunks embedded | "
                f"dim: {stats['dimension']} | model: {COHERE_MODEL}"
            )
            embed_log["embedded"].append(stats)
            total_ok += 1

        except Exception as e:
            log.error(f"  ✗ Failed: {vid_id} — {e}")
            embed_log["failed"].append({
                "video_id":  vid_id,
                "error":     str(e),
                "failed_at": datetime.utcnow().isoformat() + "Z",
            })
            total_fail += 1

    if not dry_run:
        save_embed_log(embed_log)

    log.info(
        f"\n── Embedding complete ───────────────────────\n"
        f"  ✓ Embedded : {total_ok}\n"
        f"  ↷ Skipped  : {total_skip}  (already done or missing chunks)\n"
        f"  ✗ Failed   : {total_fail}\n"
        + ("  (dry run — nothing written)" if dry_run else
           f"  Log saved → {EMBED_LOG}")
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Embed YouTube transcript chunks using Cohere API."
    )
    parser.add_argument("--video-id", type=str, default=None,
                        help="Embed a single video ID (default: embed all)")
    parser.add_argument("--force", action="store_true",
                        help="Re-embed videos that already have embeddings.json")
    parser.add_argument("--dry-run", action="store_true",
                        help="Compute embeddings but write nothing to disk")
    args = parser.parse_args()

    run(video_id=args.video_id, force=args.force, dry_run=args.dry_run)
