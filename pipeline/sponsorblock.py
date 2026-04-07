"""
sponsorblock.py
---------------
Thin wrapper around the SponsorBlock public API.
https://sponsor.ajay.app/api/docs

Used by chunker.py to filter sponsor/intro/outro segments before chunking.
Returns empty list (and logs a warning) on any API failure so chunking
can always proceed without crashing.

No auth required. Rate limits are generous for batch pipeline use.
"""

import json
import logging

import requests

log = logging.getLogger(__name__)

SPONSORBLOCK_API = "https://sponsor.ajay.app/api/skipSegments"

# Categories to filter out. "sponsor" is the main one; others remove
# non-educational padding (intros, outros, self-promo, interaction bait).
SKIP_CATEGORIES = ["sponsor", "selfpromo", "interaction", "intro", "outro"]


def get_sponsor_segments(video_id: str, timeout: int = 5) -> list[dict]:
    """
    Fetch sponsor/intro/outro segments for a video from SponsorBlock.

    Returns list of {"start": float, "end": float} dicts, or [] if the video
    is not in the database or the API call fails.
    """
    try:
        resp = requests.get(
            SPONSORBLOCK_API,
            params={
                "videoID":    video_id,
                "categories": json.dumps(SKIP_CATEGORIES),
            },
            timeout=timeout,
        )
        if resp.status_code == 404:
            return []  # video not in SponsorBlock database
        resp.raise_for_status()
        segments = resp.json()
        parsed = [
            {"start": s["segment"][0], "end": s["segment"][1]}
            for s in segments
        ]
        log.info(
            f"  SponsorBlock: {len(parsed)} segment(s) to skip for {video_id} "
            f"({[s['category'] for s in segments]})"
        )
        return parsed
    except Exception as exc:
        log.warning(f"  SponsorBlock error for {video_id}: {exc}")
        return []


def is_sponsored(start: float, end: float, sponsor_segments: list[dict]) -> bool:
    """Return True if the segment overlaps any sponsor segment."""
    return any(start < seg["end"] and end > seg["start"] for seg in sponsor_segments)
