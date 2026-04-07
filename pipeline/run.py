"""
run.py
------
Full pipeline orchestrator for ScienceQ.

Steps (in order):
  extract   – download transcripts from data/video_urls.txt
  clean     – clean raw transcripts
  chunk     – chunk into 60s windows (SponsorBlock filtering on by default)
  embed     – generate sentence-transformer embeddings
  bootstrap – generate starter metadata.json entries
  enrich    – auto-fill title/channel via YouTube API, topic via Groq
  index     – upsert vectors to Pinecone (corpus namespace)

Usage:
  python -m pipeline.run --full
  python -m pipeline.run --steps extract,clean,chunk
  python -m pipeline.run --steps enrich
  python -m pipeline.run --full --dry-run

Run from the project root. Reads .env automatically.
"""

import argparse
import functools
import logging
import sys
from pathlib import Path

# Ensure sibling modules in pipeline/ are importable regardless of invocation
sys.path.insert(0, str(Path(__file__).parent))

import bootstrap_metadata
import chunker
import cleaner
import embedder
import enrich_metadata
import indexer
import transcript_extractor

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Step registry ──────────────────────────────────────────────────────────────
ALL_STEPS = ["extract", "clean", "chunk", "embed", "bootstrap", "enrich", "index"]


def _run_extract() -> None:
    log.info("━━ Step: extract ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    transcript_extractor.run(
        input_path  = Path("data/video_urls.txt"),
        delay_min   = 3.0,
        delay_max   = 8.0,
        max_retries = 3,
    )


def _run_clean() -> None:
    log.info("━━ Step: clean ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    cleaner.run(video_id=None, dry_run=False, force=False)


def _run_chunk(skip_sponsors: bool = True) -> None:
    log.info("━━ Step: chunk ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    chunker.run(
        video_id      = None,
        window        = 60.0,
        dry_run       = False,
        force         = False,
        skip_sponsors = skip_sponsors,
    )


def _run_embed() -> None:
    log.info("━━ Step: embed ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    embedder.run(video_id=None, force=False, dry_run=False)


def _run_bootstrap() -> None:
    log.info("━━ Step: bootstrap ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    bootstrap_metadata.run(dry_run=False)


def _run_enrich() -> None:
    log.info("━━ Step: enrich ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    enrich_metadata.run(dry_run=False)


def _run_index() -> None:
    log.info("━━ Step: index ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    indexer.run(video_id=None, namespace="corpus", force=False, dry_run=False)


STEP_RUNNERS = {
    "extract":   _run_extract,
    "clean":     _run_clean,
    "chunk":     _run_chunk,
    "embed":     _run_embed,
    "bootstrap": _run_bootstrap,
    "enrich":    _run_enrich,
    "index":     _run_index,
}


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ScienceQ pipeline orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m pipeline.run --full\n"
            "  python -m pipeline.run --steps extract,clean,chunk\n"
            "  python -m pipeline.run --steps enrich\n"
        ),
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--full",
        action="store_true",
        help=f"Run all steps in order: {' → '.join(ALL_STEPS)}",
    )
    group.add_argument(
        "--steps",
        type=str,
        help=f"Comma-separated subset of steps to run. Available: {', '.join(ALL_STEPS)}",
    )

    parser.add_argument(
        "--no-skip-sponsors",
        action="store_true",
        help="Disable SponsorBlock filtering during the chunk step",
    )

    args = parser.parse_args()

    if args.full:
        steps = ALL_STEPS
    else:
        steps = [s.strip() for s in args.steps.split(",")]
        unknown = [s for s in steps if s not in ALL_STEPS]
        if unknown:
            parser.error(f"Unknown step(s): {', '.join(unknown)}. Available: {', '.join(ALL_STEPS)}")

    skip_sponsors = not args.no_skip_sponsors

    # Bake skip_sponsors into the chunk runner so all runners share the same () interface
    runners = {
        **STEP_RUNNERS,
        "chunk": functools.partial(_run_chunk, skip_sponsors=skip_sponsors),
    }

    log.info(f"ScienceQ pipeline — running steps: {' → '.join(steps)}")

    for step in steps:
        runners[step]()

    log.info("━━ Pipeline complete ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")


if __name__ == "__main__":
    main()
