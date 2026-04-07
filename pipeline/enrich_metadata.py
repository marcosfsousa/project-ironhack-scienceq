"""
enrich_metadata.py
------------------
Auto-enriches data/metadata.json using:
  - YouTube Data API v3  →  title, channel name  (requires YOUTUBE_DATA_API_KEY)
  - Groq LLM             →  topic classification (requires GROQ_API_KEY)

Only processes entries with missing title, channel, or topic.
Safe to re-run — already-filled fields are never overwritten.
Checkpoints after each video so progress survives interruption.

Valid topics: Biology, Chemistry, Cosmology, Mathematics, Other,
              Physics, Psychology, Technology

Usage:
  python pipeline/enrich_metadata.py
  python pipeline/enrich_metadata.py --dry-run
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv
from groq import Groq

# bootstrap_metadata.py is a sibling module — import its VALID_TOPICS
sys.path.insert(0, str(Path(__file__).parent))
from bootstrap_metadata import VALID_TOPICS

load_dotenv()

# ── Logging ────────────────────────────────────────────────────────────────────
log = logging.getLogger(__name__)
if not log.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    ))
    log.addHandler(_handler)
    log.setLevel(logging.INFO)
    log.propagate = False

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_DIR      = Path("data")
VIDEOS_DIR    = DATA_DIR / "videos"
METADATA_PATH = DATA_DIR / "metadata.json"

# ── Constants ──────────────────────────────────────────────────────────────────
YOUTUBE_API_URL = "https://www.googleapis.com/youtube/v3/videos"
GROQ_MODEL      = "llama-3.3-70b-versatile"
SAMPLE_CHARS    = 600   # transcript chars sent to Groq for topic classification
YT_BATCH_SIZE   = 50   # YouTube Data API max IDs per request


# ── YouTube Data API ───────────────────────────────────────────────────────────

def fetch_videos_info(video_ids: list[str], api_key: str) -> dict[str, dict]:
    """
    Fetch title and channel for multiple video IDs in batched API calls.
    YouTube Data API v3 accepts up to 50 IDs per request.
    Returns {video_id: {"title": ..., "channel": ...}} for found videos.
    """
    results = {}
    for i in range(0, len(video_ids), YT_BATCH_SIZE):
        batch = video_ids[i : i + YT_BATCH_SIZE]
        try:
            resp = requests.get(
                YOUTUBE_API_URL,
                params={"part": "snippet", "id": ",".join(batch), "key": api_key},
                timeout=10,
            )
            resp.raise_for_status()
            for item in resp.json().get("items", []):
                vid_id  = item["id"]
                snippet = item["snippet"]
                results[vid_id] = {
                    "title":   snippet.get("title", ""),
                    "channel": snippet.get("channelTitle", ""),
                }
        except Exception as exc:
            log.warning(f"  YouTube API batch error ({batch}): {exc}")
    return results


# ── Topic classification ───────────────────────────────────────────────────────

def _load_transcript_sample(video_id: str) -> str:
    """Read the first SAMPLE_CHARS of transcript text from clean or raw file."""
    for fname in ("transcript_clean.json", "transcript_raw.json"):
        path = VIDEOS_DIR / video_id / fname
        if path.exists():
            try:
                data     = json.loads(path.read_text(encoding="utf-8"))
                segments = data.get("transcript", [])
                text     = " ".join(s.get("text", "") for s in segments[:30])
                return text[:SAMPLE_CHARS]
            except Exception:
                pass
    return ""


def classify_topic(title: str, sample: str, client: Groq) -> str:
    """
    Use Groq to classify the video topic from title and a transcript sample.
    Falls back to 'Other' if the model returns an unexpected value.
    """
    valid_list = ", ".join(sorted(VALID_TOPICS))
    prompt = (
        f"Classify this YouTube science video into exactly one topic.\n\n"
        f"Title: {title}\n"
        f"Transcript sample: {sample}\n\n"
        f"Valid topics: {valid_list}\n\n"
        f"Reply with ONLY the topic name, nothing else."
    )
    try:
        completion = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
        topic = completion.choices[0].message.content.strip()
        if topic in VALID_TOPICS:
            return topic
        for valid in VALID_TOPICS:
            if valid.lower() == topic.lower():
                return valid
        log.warning(f"  Groq returned unexpected topic '{topic}', falling back to Other")
        return "Other"
    except Exception as exc:
        log.warning(f"  Groq topic classification failed: {exc}")
        return "Other"


# ── Main enrichment loop ───────────────────────────────────────────────────────

def run(dry_run: bool = False) -> None:
    if dry_run:
        log.info("DRY RUN — metadata.json will not be written.")

    if not METADATA_PATH.exists():
        log.warning("metadata.json not found. Run bootstrap_metadata.py first.")
        return

    metadata   = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    videos     = metadata.get("videos", {})
    yt_api_key = os.getenv("YOUTUBE_DATA_API_KEY", "")
    groq_key   = os.getenv("GROQ_API_KEY", "")

    if not yt_api_key:
        log.warning("YOUTUBE_DATA_API_KEY not set — title/channel enrichment skipped.")
    if not groq_key:
        log.warning("GROQ_API_KEY not set — topic classification skipped.")

    groq_client = Groq(api_key=groq_key) if groq_key else None

    needs_enrichment = [
        vid_id for vid_id, entry in videos.items()
        if not entry.get("title") or not entry.get("channel") or not entry.get("topic")
    ]

    if not needs_enrichment:
        log.info("All entries already enriched. Nothing to do.")
        return

    log.info(f"Enriching {len(needs_enrichment)} entry/entries...")

    # ── Batch-fetch title + channel for all videos needing it ─────────────────
    needs_yt = [v for v in needs_enrichment
                if not videos[v].get("title") or not videos[v].get("channel")]
    yt_info  = fetch_videos_info(needs_yt, yt_api_key) if yt_api_key and needs_yt else {}

    changed = 0

    for vid_id in needs_enrichment:
        entry = videos[vid_id]
        log.info(f"  Processing: {vid_id}")

        # ── Title + channel from pre-fetched batch ────────────────────────────
        if yt_info.get(vid_id):
            info = yt_info[vid_id]
            if info.get("title") and not entry.get("title"):
                entry["title"] = info["title"]
                log.info(f"    title   → {info['title']}")
            if info.get("channel") and not entry.get("channel"):
                entry["channel"] = info["channel"]
                log.info(f"    channel → {info['channel']}")

        # ── Topic from Groq ───────────────────────────────────────────────────
        if groq_client and not entry.get("topic"):
            title  = entry.get("title", vid_id)
            sample = _load_transcript_sample(vid_id)
            topic  = classify_topic(title, sample, groq_client)
            entry["topic"] = topic
            log.info(f"    topic   → {topic}")

        videos[vid_id] = entry
        changed += 1

        # Checkpoint after each video so progress survives interruption
        if not dry_run:
            METADATA_PATH.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    if dry_run:
        log.info(f"\n(dry run) Would have enriched {changed} entry/entries.")
    else:
        log.info(f"\n✓ metadata.json updated — {changed} entry/entries enriched.")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Auto-enrich metadata.json via YouTube API + Groq."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be written without writing anything",
    )
    args = parser.parse_args()
    run(dry_run=args.dry_run)
