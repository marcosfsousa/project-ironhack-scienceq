"""
sweep_reranker.py
-----------------
Side-by-side evaluation of the ScienceQ RAG pipeline with and without
the Cohere Rerank v3.5 layer.

Runs the full automated eval set (rag_factual + rag_multi_turn) twice:
  - Run A: RERANKER_ENABLED=false  (baseline — cosine similarity only)
  - Run B: RERANKER_ENABLED=true   (Pinecone top 10 → Cohere Rerank → top 5)

Scores both runs with GPT-4.1 on the standard four dimensions:
  correctness, tone, grounding, conciseness

Outputs:
  eval/results/reranker_sweep_<timestamp>.json
    Full results for both runs (answers + scores per case).
  Console table showing per-dimension and per-case delta (B − A).

Usage:
  cd project-ironhack-scienceq
  python eval/sweep_reranker.py

  # Dry-run — list cases, skip agent calls:
  python eval/sweep_reranker.py --dry-run

  # Single case for quick sanity check:
  python eval/sweep_reranker.py --case rag_001

Requirements:
  OPENAI_API_KEY   — GPT-4.1 judge
  COHERE_API_KEY   — reranker (run B)
  PINECONE_*       — retrieval
  GROQ_API_KEY     — LLM answer generation
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import openai
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
_EVAL_DIR    = Path(__file__).resolve().parent
_ROOT        = _EVAL_DIR.parent
_AGENT_DIR   = _ROOT / "agent"
_PIPELINE_DIR = _ROOT / "pipeline"
_RESULTS_DIR = _EVAL_DIR / "results"
_RESULTS_DIR.mkdir(exist_ok=True)

for p in [str(_AGENT_DIR), str(_PIPELINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

EVAL_SET_PATH    = _EVAL_DIR / "eval_set.json"
JUDGE_MODEL      = "gpt-4.1"
RATE_LIMIT_SLEEP = 1.0    # seconds between judge calls
INTER_CASE_SLEEP = 3.0    # seconds between cases (Groq rate limit)
DIMS             = ["correctness", "tone", "grounding", "conciseness"]

# ── Reuse judge from run_evals ─────────────────────────────────────────────────
from run_evals import score_answer, run_rag_case, run_multi_turn_case  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Single-run executor
# ══════════════════════════════════════════════════════════════════════════════

def _run_cases(
    cases: list[dict],
    reranker_on: bool,
    judge_client: openai.OpenAI,
) -> list[dict]:
    """
    Run all cases through the agent with RERANKER_ENABLED set accordingly.
    Returns a list of result dicts (one per case) with scores.
    """
    # Patch the env var and reload the retriever singleton so the flag takes effect.
    os.environ["RERANKER_ENABLED"] = "true" if reranker_on else "false"

    # Import after env is set — retriever reads RERANKER_ENABLED at module level,
    # so we force a reimport to pick up the new value.
    import importlib
    import retriever as _retriever_mod
    _retriever_mod.RERANKER_ENABLED = reranker_on
    # Also reset the Cohere/Pinecone singletons so they re-read config cleanly.
    _retriever_mod._cohere_client = None
    _retriever_mod._pinecone_index = None

    # Import agent *after* patching retriever so it picks up the updated module.
    from agent import YouTubeQAAgent

    label  = "RERANKER=ON " if reranker_on else "RERANKER=OFF"
    total  = len(cases)
    results: list[dict] = []

    log.info(f"\n{'═'*60}")
    log.info(f"  Starting run: {label}  ({total} cases)")
    log.info(f"{'═'*60}")

    for i, case in enumerate(cases, start=1):
        case_id   = case["id"]
        case_type = case["type"]
        question  = (
            case["turns"][2]["content"]
            if case_type == "rag_multi_turn"
            else case["question"]
        )
        reference = case["reference_answer"]

        log.info(f"\n[{label}] [{i}/{total}] {case_id} ({case_type})")
        log.info(f"  Q: {question[:100]}")

        agent = YouTubeQAAgent()

        try:
            if case_type == "rag_multi_turn":
                bot_answer = run_multi_turn_case(case, agent)
            else:
                bot_answer = run_rag_case(case, agent)
        except Exception as e:
            log.error(f"  Agent call failed: {e}")
            results.append({
                "case_id":    case_id,
                "case_type":  case_type,
                "question":   question,
                "reference":  reference,
                "answer":     "",
                "scores":     {},
                "error":      str(e),
            })
            continue

        log.info(f"  A: {bot_answer[:120]}...")

        log.info(f"  Scoring with {JUDGE_MODEL}...")
        scores = score_answer(question, bot_answer, reference, judge_client)

        log.info(
            f"  Scores → correctness={scores['correctness']['score']} "
            f"tone={scores['tone']['score']} "
            f"grounding={scores['grounding']['score']} "
            f"conciseness={scores['conciseness']['score']} "
            f"mean={scores['mean']}"
        )

        results.append({
            "case_id":   case_id,
            "case_type": case_type,
            "topic":     case.get("topic", ""),
            "video":     case.get("video_title", ""),
            "question":  question,
            "reference": reference,
            "answer":    bot_answer,
            "scores":    scores,
            "error":     None,
        })

        if i < total:
            time.sleep(INTER_CASE_SLEEP)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# Comparison table
# ══════════════════════════════════════════════════════════════════════════════

def _print_comparison(
    results_off: list[dict],
    results_on: list[dict],
) -> None:
    """Print a side-by-side comparison table to stdout."""

    def avg(results: list[dict], dim: str) -> float:
        scored = [r for r in results if r.get("scores") and not r.get("error")]
        if not scored:
            return 0.0
        return sum(r["scores"][dim]["score"] for r in scored) / len(scored)

    def mean_avg(results: list[dict]) -> float:
        scored = [r for r in results if r.get("scores") and not r.get("error")]
        if not scored:
            return 0.0
        return sum(r["scores"]["mean"] for r in scored) / len(scored)

    print(f"\n{'═'*72}")
    print(f"  RERANKER SWEEP — Side-by-side Comparison")
    print(f"{'═'*72}")
    print(f"  {'Dimension':<16}  {'OFF':>6}  {'ON':>6}  {'Δ':>7}")
    print(f"  {'─'*16}  {'─'*6}  {'─'*6}  {'─'*7}")

    for dim in DIMS:
        off_score = avg(results_off, dim)
        on_score  = avg(results_on,  dim)
        delta     = on_score - off_score
        sign      = "+" if delta >= 0 else ""
        print(f"  {dim:<16}  {off_score:>6.2f}  {on_score:>6.2f}  {sign}{delta:>6.2f}")

    off_mean = mean_avg(results_off)
    on_mean  = mean_avg(results_on)
    delta    = on_mean - off_mean
    sign     = "+" if delta >= 0 else ""
    print(f"  {'─'*16}  {'─'*6}  {'─'*6}  {'─'*7}")
    print(f"  {'MEAN':<16}  {off_mean:>6.2f}  {on_mean:>6.2f}  {sign}{delta:>6.2f}")
    print(f"{'═'*72}")

    # Per-case delta table
    by_id_off = {r["case_id"]: r for r in results_off}
    by_id_on  = {r["case_id"]: r for r in results_on}

    print(f"\n  Per-case mean scores (OFF → ON):")
    print(f"  {'Case ID':<12}  {'Type':<16}  {'OFF':>5}  {'ON':>5}  {'Δ':>6}")
    print(f"  {'─'*12}  {'─'*16}  {'─'*5}  {'─'*5}  {'─'*6}")

    all_ids = sorted(set(by_id_off) | set(by_id_on))
    for cid in all_ids:
        r_off = by_id_off.get(cid)
        r_on  = by_id_on.get(cid)
        if not r_off or not r_on:
            continue
        if r_off.get("error") or r_on.get("error"):
            print(f"  {cid:<12}  {'ERROR':<16}")
            continue
        s_off = r_off["scores"]["mean"]
        s_on  = r_on["scores"]["mean"]
        delta = s_on - s_off
        sign  = "+" if delta >= 0 else ""
        ctype = r_off.get("case_type", "")[:16]
        print(f"  {cid:<12}  {ctype:<16}  {s_off:>5.2f}  {s_on:>5.2f}  {sign}{delta:>5.2f}")

    print(f"{'═'*72}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run(
    dry_run:     bool          = False,
    case_filter: Optional[str] = None,
) -> None:

    # ── Load eval set ──────────────────────────────────────────────────────────
    if not EVAL_SET_PATH.exists():
        log.error(f"eval_set.json not found at {EVAL_SET_PATH}")
        sys.exit(1)

    eval_set  = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    all_cases = eval_set["cases"]

    # Only automated cases (skip adversarial — no reference answer)
    automated = [c for c in all_cases if c["type"] != "adversarial"]

    if case_filter:
        automated = [c for c in automated if c["id"] == case_filter]
        if not automated:
            log.error(f"No automated case found with id={case_filter!r}")
            sys.exit(1)

    log.info(f"Loaded {len(automated)} automated cases.")

    if dry_run:
        log.info("DRY RUN — cases to be evaluated:")
        for c in automated:
            q = c["turns"][2]["content"] if c["type"] == "rag_multi_turn" else c["question"]
            log.info(f"  [{c['id']}] ({c['type']}) {q[:80]}")
        return

    # ── Initialise judge ───────────────────────────────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        log.error("OPENAI_API_KEY not set — required for GPT-4.1 judge.")
        sys.exit(1)
    judge_client = openai.OpenAI(api_key=openai_key)
    log.info(f"Judge: {JUDGE_MODEL}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    # ── Run A: reranker OFF ────────────────────────────────────────────────────
    results_off = _run_cases(automated, reranker_on=False, judge_client=judge_client)

    # ── Run B: reranker ON ────────────────────────────────────────────────────
    results_on = _run_cases(automated, reranker_on=True, judge_client=judge_client)

    # ── Print comparison ───────────────────────────────────────────────────────
    _print_comparison(results_off, results_on)

    # ── Save results ───────────────────────────────────────────────────────────
    output = {
        "sweep":       "reranker",
        "timestamp":   timestamp,
        "judge_model": JUDGE_MODEL,
        "total_cases": len(automated),
        "runs": {
            "reranker_off": results_off,
            "reranker_on":  results_on,
        },
    }
    out_path = _RESULTS_DIR / f"reranker_sweep_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"Results saved to {out_path}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Side-by-side eval: reranker ON vs OFF."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List cases without running the agent or judge.",
    )
    parser.add_argument(
        "--case",
        metavar="CASE_ID",
        help="Run a single case (e.g. rag_001) for quick sanity check.",
    )
    args = parser.parse_args()

    run(dry_run=args.dry_run, case_filter=args.case)
