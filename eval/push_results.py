"""
push_results.py
---------------
One-off script to push existing local eval result JSON files to LangSmith.
No agent calls, no judge calls — reads from eval/results/run_*.json only.

Usage:
  # Push all result files (auto-detects experiment name from JSON):
  python eval/push_results.py

  # Push a specific file:
  python eval/push_results.py --file eval/results/run_20260309_084647.json

  # Override experiment name for a specific file:
  python eval/push_results.py --file eval/results/run_20260308_XXXXXX.json --experiment-name prompt-v1

Requirements:
  LANGSMITH_API_KEY in .env
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

_EVAL_DIR    = Path(__file__).resolve().parent
_RESULTS_DIR = _EVAL_DIR / "results"
DATASET_NAME = "youtube-qa-bot-eval-set"


# ══════════════════════════════════════════════════════════════════════════════
# LangSmith helpers (copied + fixed from run_evals.py)
# ══════════════════════════════════════════════════════════════════════════════

def _get_or_create_dataset(ls_client) -> str:
    """Return the LangSmith dataset ID, creating the dataset if it doesn't exist."""
    existing = list(ls_client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        dataset_id = str(existing[0].id)
        log.info(f"Found existing LangSmith dataset: {DATASET_NAME} (id={dataset_id})")
        return dataset_id

    log.info(f"Dataset '{DATASET_NAME}' not found — creating empty dataset.")
    dataset = ls_client.create_dataset(
        dataset_name=DATASET_NAME,
        description="YouTube QA Bot eval set — 25 automated cases (20 factual + 5 multi-turn)",
    )
    return str(dataset.id)


def _push_results(ls_client, dataset_id: str, experiment_name: str, results: list[dict]) -> None:
    """Push a list of result dicts to LangSmith as a named experiment."""
    pushed = 0
    for result in results:
        if result.get("skipped") or result.get("error"):
            continue

        run_id = str(uuid.uuid4())
        ls_client.create_run(
            id       = run_id,
            name     = experiment_name,
            run_type = "chain",
            inputs   = {"question": result["question"]},
            outputs  = {"answer":   result["answer"]},
            extra    = {
                "case_id":   result["case_id"],
                "case_type": result["case_type"],
                "scores":    result.get("scores", {}),
            },
        )
        ls_client.update_run(run_id, end_time=datetime.now(timezone.utc))

        scores = result.get("scores", {})
        for dim, val in scores.items():
            if dim == "mean":
                ls_client.create_feedback(
                    run_id=run_id,
                    key="mean_score",
                    score=round(val / 5, 4),  # normalize 1-5 → 0.0-1.0
                )
            elif isinstance(val, dict) and "score" in val:
                ls_client.create_feedback(
                    run_id=run_id,
                    key=dim,
                    score=round(val["score"] / 5, 4),  # normalize 1-5 → 0.0-1.0
                    comment=val.get("reason", ""),
                )
        pushed += 1

    log.info(f"Pushed {pushed} results to LangSmith experiment: '{experiment_name}'")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def push_file(filepath: Path, experiment_name_override: str | None, ls_client, dataset_id: str) -> None:
    data = json.loads(filepath.read_text(encoding="utf-8"))

    exp_name = experiment_name_override or data.get("experiment")
    if not exp_name:
        log.warning(f"No experiment name found in {filepath.name} and none provided — skipping.")
        return

    results = data.get("results", [])
    scored  = [r for r in results if r.get("scores") and not r.get("error")]

    if len(scored) < 10:
        log.warning(f"Skipping {filepath.name} — only {len(scored)} scored case(s). "
                    "Looks like a test run. Use --file to push it explicitly if intended.")
        return

    log.info(f"\nFile:       {filepath.name}")
    log.info(f"Experiment: {exp_name}")
    log.info(f"Cases:      {len(scored)} scored / {len(results)} total")

    _push_results(ls_client, dataset_id, exp_name, results)


def main():
    parser = argparse.ArgumentParser(description="Push local eval results to LangSmith.")
    parser.add_argument("--file", metavar="PATH", help="Push a specific result JSON file.")
    parser.add_argument("--experiment-name", metavar="NAME", help="Override experiment name.")
    args = parser.parse_args()

    # ── LangSmith client ───────────────────────────────────────────────────────
    langsmith_key = os.getenv("LANGSMITH_API_KEY")
    if not langsmith_key:
        log.error("LANGSMITH_API_KEY not set in .env — cannot push to LangSmith.")
        sys.exit(1)

    try:
        from langsmith import Client as LSClient
        ls_client = LSClient()
        log.info("LangSmith client initialised.")
    except Exception as e:
        log.error(f"Failed to initialise LangSmith client: {e}")
        sys.exit(1)

    dataset_id = _get_or_create_dataset(ls_client)

    # ── Resolve files to push ──────────────────────────────────────────────────
    if args.file:
        files = [Path(args.file)]
        if not files[0].exists():
            log.error(f"File not found: {files[0]}")
            sys.exit(1)
    else:
        files = sorted(_RESULTS_DIR.glob("run_*.json"))
        if not files:
            log.error(f"No run_*.json files found in {_RESULTS_DIR}")
            sys.exit(1)
        log.info(f"Found {len(files)} result file(s) in {_RESULTS_DIR}:")
        for f in files:
            log.info(f"  {f.name}")

    # ── Push each file ─────────────────────────────────────────────────────────
    for filepath in files:
        # Only allow experiment-name override when pushing a single file
        override = args.experiment_name if len(files) == 1 else None
        push_file(filepath, override, ls_client, dataset_id)

    log.info("\nDone. Check your LangSmith dashboard for the experiments.")


if __name__ == "__main__":
    main()
