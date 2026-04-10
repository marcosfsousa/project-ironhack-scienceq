"""
sweep_retrieval.py
------------------
Two-stage retrieval parameter sweep for ScienceQ.

Runs the full automated eval set (rag_factual + rag_multi_turn) across a matrix
of retrieval parameters, scoring each combination with GPT-4.1 on four rubric
dimensions: correctness, tone, grounding, conciseness.

Stage 1 — Retrieval shape
  Sweeps RETRIEVER_FETCH_K × RETRIEVER_TOP_N at a fixed threshold of 0.0 (no
  filter) so retrieval-shape differences are not muddied by cosine filtering.
  Reranker is on for all combinations.

  Matrix (11 combos):
    fetch_k ∈ {5, 7, 10, 15}
    top_n   ∈ {3, 5, 7}  where top_n ≤ fetch_k

Stage 2 — Threshold calibration
  Locks in the Stage 1 winner (passed via --fetch-k / --top-n) and sweeps
  SCORE_THRESHOLD across {0.20, 0.25, 0.30, 0.35, 0.40} to bracket the current
  production default (0.40) and find the empirically optimal value.

  Matrix (5 combos):
    threshold ∈ {0.20, 0.25, 0.30, 0.35, 0.40}

After Stage 2 completes, the top 5 configurations by mean score (across both
stages) are pushed to LangSmith as named experiments.

Outputs (per stage):
  eval/results/sweep_retrieval_stage{1,2}_<timestamp>.json
    Full per-case data: question, reference, answer, all 4 scores + reasons.
  eval/results/sweep_retrieval_stage{1,2}_<timestamp>.csv
    One row per combo — aggregate scores only. Drop-in for pandas/matplotlib.

Usage:
  cd project-ironhack-scienceq

  # Stage 1 — retrieval shape:
  python eval/sweep_retrieval.py --stage 1

  # Stage 2 — threshold calibration (pass Stage 1 winner):
  python eval/sweep_retrieval.py --stage 2 --fetch-k 10 --top-n 5

  # Dry-run — list combos without running anything:
  python eval/sweep_retrieval.py --stage 1 --dry-run
  python eval/sweep_retrieval.py --stage 2 --fetch-k 10 --top-n 5 --dry-run

  # Single-case smoke test:
  python eval/sweep_retrieval.py --stage 1 --case rag_001

Requirements:
  OPENAI_API_KEY   — GPT-4.1 judge
  COHERE_API_KEY   — reranker
  PINECONE_*       — retrieval
  GROQ_API_KEY     — LLM answer generation
  LANGSMITH_API_KEY — top-5 push (optional; results still saved locally if absent)
"""

from __future__ import annotations

import argparse
import csv
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
_EVAL_DIR     = Path(__file__).resolve().parent
_ROOT         = _EVAL_DIR.parent
_AGENT_DIR    = _ROOT / "agent"
_PIPELINE_DIR = _ROOT / "pipeline"
_RESULTS_DIR  = _EVAL_DIR / "results"
_RESULTS_DIR.mkdir(exist_ok=True)

for p in [str(_AGENT_DIR), str(_PIPELINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

EVAL_SET_PATH    = _EVAL_DIR / "eval_set.json"
JUDGE_MODEL      = "gpt-4.1"
RATE_LIMIT_SLEEP = 1.0   # seconds between judge API calls
INTER_CASE_SLEEP = 3.0   # seconds between cases (Groq rate limit)
DIMS             = ["correctness", "tone", "grounding", "conciseness"]

# ── Stage matrices ─────────────────────────────────────────────────────────────

# Stage 1: all (fetch_k, top_n) pairs where top_n ≤ fetch_k.
STAGE1_THRESHOLD = 0.0   # no cosine filter — isolate retrieval-shape signal
STAGE1_COMBOS = [
    (fetch_k, top_n)
    for fetch_k in [5, 7, 10, 15]
    for top_n in [3, 5, 7]
    if top_n <= fetch_k
]  # 11 combinations

# Stage 2: threshold sweep at locked fetch_k/top_n (set via CLI after Stage 1).
STAGE2_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40]

TOP_N_FOR_LANGSMITH = 5   # push only the top N configs to avoid dashboard clutter

# ── Reuse judge + case runners from run_evals ──────────────────────────────────
from run_evals import score_answer, run_rag_case, run_multi_turn_case  # noqa: E402


# ══════════════════════════════════════════════════════════════════════════════
# Module-level patching helpers
# ══════════════════════════════════════════════════════════════════════════════

def _patch_retriever(fetch_k: int, top_n: int, threshold: float) -> None:
    """
    Patch retriever module attributes and reset Cohere/Pinecone singletons so
    the next agent instantiation picks up the new parameter values.

    Uses the same pattern as sweep_reranker.py — both os.environ and direct
    module attribute assignment are needed because the module was already
    imported and its module-level constants are already bound.
    """
    os.environ["RERANKER_ENABLED"]  = "true"
    os.environ["RETRIEVER_FETCH_K"] = str(fetch_k)
    os.environ["RETRIEVER_TOP_N"]   = str(top_n)
    os.environ["SCORE_THRESHOLD"]   = str(threshold)

    import retriever as _r
    _r.RERANKER_ENABLED   = True
    _r.RETRIEVER_FETCH_K  = fetch_k
    _r.RETRIEVER_TOP_N    = top_n
    _r.SCORE_THRESHOLD    = threshold
    _r._cohere_client     = None
    _r._pinecone_index    = None


# ══════════════════════════════════════════════════════════════════════════════
# Single-combo runner
# ══════════════════════════════════════════════════════════════════════════════

def _run_combo(
    cases:        list[dict],
    combo_label:  str,
    fetch_k:      int,
    top_n:        int,
    threshold:    float,
    judge_client: openai.OpenAI,
) -> list[dict]:
    """
    Patch retriever params, run all cases, score each with the LLM judge.
    Returns a list of result dicts (one per case).
    """
    _patch_retriever(fetch_k, top_n, threshold)

    # Import agent after patching so it picks up the updated retriever state.
    from agent import YouTubeQAAgent

    total   = len(cases)
    results: list[dict] = []

    log.info(f"\n{'═'*60}")
    log.info(f"  Combo: {combo_label}  ({total} cases)")
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

        log.info(f"\n[{combo_label}] [{i}/{total}] {case_id} ({case_type})")
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
                "topic":      case.get("topic", ""),
                "video":      case.get("video_title", ""),
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
# Aggregation helpers
# ══════════════════════════════════════════════════════════════════════════════

def _aggregate(results: list[dict]) -> dict:
    """Compute per-dimension averages and overall mean over a list of case results."""
    scored = [r for r in results if r.get("scores") and not r.get("error")]
    n      = len(scored)
    if n == 0:
        return {d: 0.0 for d in DIMS} | {"mean": 0.0, "n_cases": 0, "n_errors": len(results)}

    dim_avgs = {
        d: round(sum(r["scores"][d]["score"] for r in scored) / n, 4)
        for d in DIMS
    }
    mean = round(sum(r["scores"]["mean"] for r in scored) / n, 4)
    return dim_avgs | {"mean": mean, "n_cases": n, "n_errors": len(results) - n}


def _print_summary_table(combos: list[dict]) -> None:
    """Print a ranked summary table to stdout."""
    ranked = sorted(combos, key=lambda c: c["agg"]["mean"], reverse=True)
    print(f"\n{'═'*80}")
    print(f"  {'Label':<22}  {'correct':>7}  {'tone':>7}  {'ground':>7}  {'concise':>7}  {'mean':>7}")
    print(f"  {'─'*22}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}  {'─'*7}")
    for c in ranked:
        a = c["agg"]
        print(
            f"  {c['label']:<22}  {a['correctness']:>7.3f}  {a['tone']:>7.3f}  "
            f"{a['grounding']:>7.3f}  {a['conciseness']:>7.3f}  {a['mean']:>7.3f}"
        )
    print(f"{'═'*80}\n")


# ══════════════════════════════════════════════════════════════════════════════
# Output writers
# ══════════════════════════════════════════════════════════════════════════════

def _save_json(combos: list[dict], stage: int, timestamp: str) -> Path:
    output = {
        "stage":       stage,
        "timestamp":   timestamp,
        "judge_model": JUDGE_MODEL,
        "combos":      {c["label"]: c["results"] for c in combos},
        "summary":     {c["label"]: c["agg"]     for c in combos},
    }
    path = _RESULTS_DIR / f"sweep_retrieval_stage{stage}_{timestamp}.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"JSON saved → {path}")
    return path


def _save_csv(combos: list[dict], stage: int, timestamp: str) -> Path:
    path = _RESULTS_DIR / f"sweep_retrieval_stage{stage}_{timestamp}.csv"
    fieldnames = ["label", "fetch_k", "top_n", "threshold"] + DIMS + ["mean", "n_cases", "n_errors"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for c in combos:
            row = {
                "label":     c["label"],
                "fetch_k":   c["fetch_k"],
                "top_n":     c["top_n"],
                "threshold": c["threshold"],
            } | c["agg"]
            writer.writerow(row)
    log.info(f"CSV saved  → {path}")
    return path


# ══════════════════════════════════════════════════════════════════════════════
# LangSmith — push top N configs
# ══════════════════════════════════════════════════════════════════════════════

def _push_top_configs(all_combos: list[dict]) -> None:
    """
    Rank all combos by mean score and push the top N to LangSmith as named
    experiments. Reuses _push_experiment_results from run_evals.py.
    """
    try:
        from langsmith import Client as LSClient
        from run_evals import _push_experiment_results
        ls_client = LSClient()
    except Exception as e:
        log.warning(f"LangSmith unavailable — skipping push: {e}")
        return

    ranked = sorted(all_combos, key=lambda c: c["agg"]["mean"], reverse=True)
    top    = ranked[:TOP_N_FOR_LANGSMITH]

    log.info(f"\nPushing top {len(top)} configs to LangSmith...")
    for c in top:
        # Derive experiment name from label (e.g. k10_n3_t0.25 → cohere-k10-n3-t025)
        exp_name = "cohere-" + c["label"].replace("_", "-").replace(".", "")
        log.info(f"  → {exp_name}  (mean={c['agg']['mean']:.3f})")
        try:
            _push_experiment_results(ls_client, None, exp_name, c["results"])
        except Exception as e:
            log.warning(f"  Push failed for {exp_name}: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Stage runners
# ══════════════════════════════════════════════════════════════════════════════

def run_stage1(
    cases:       list[dict],
    judge_client: openai.OpenAI,
    dry_run:     bool = False,
    case_filter: Optional[str] = None,
) -> list[dict]:
    """
    Stage 1: sweep fetch_k × top_n at threshold=0.0 (no cosine filter).
    Returns list of combo dicts — each with label, params, results, and agg.
    """
    combos_to_run = STAGE1_COMBOS

    if dry_run:
        log.info(f"\nStage 1 — DRY RUN ({len(combos_to_run)} combos, threshold=0.0):")
        for fetch_k, top_n in combos_to_run:
            label = f"k{fetch_k}_n{top_n}_t0.0"
            log.info(f"  {label}  ({len(cases)} cases)")
        return []

    all_combos: list[dict] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for fetch_k, top_n in combos_to_run:
        label   = f"k{fetch_k}_n{top_n}_t{STAGE1_THRESHOLD}"
        results = _run_combo(
            cases,
            combo_label=label,
            fetch_k=fetch_k,
            top_n=top_n,
            threshold=STAGE1_THRESHOLD,
            judge_client=judge_client,
        )
        agg = _aggregate(results)
        all_combos.append({
            "label":     label,
            "fetch_k":   fetch_k,
            "top_n":     top_n,
            "threshold": STAGE1_THRESHOLD,
            "results":   results,
            "agg":       agg,
        })
        log.info(f"  → {label} complete: mean={agg['mean']:.3f}")

    _print_summary_table(all_combos)
    _save_json(all_combos, stage=1, timestamp=timestamp)
    _save_csv(all_combos,  stage=1, timestamp=timestamp)

    return all_combos


def run_stage2(
    cases:        list[dict],
    judge_client:  openai.OpenAI,
    fetch_k:      int,
    top_n:        int,
    dry_run:      bool = False,
    case_filter:  Optional[str] = None,
    stage1_combos: Optional[list[dict]] = None,
) -> list[dict]:
    """
    Stage 2: sweep score_threshold at the Stage 1 winning fetch_k/top_n.
    After completion, pushes the top 5 configs (across both stages) to LangSmith.
    Returns list of combo dicts.
    """
    if dry_run:
        log.info(f"\nStage 2 — DRY RUN (fetch_k={fetch_k}, top_n={top_n}):")
        for t in STAGE2_THRESHOLDS:
            label = f"k{fetch_k}_n{top_n}_t{t}"
            log.info(f"  {label}  ({len(cases)} cases)")
        return []

    all_combos: list[dict] = []
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    for threshold in STAGE2_THRESHOLDS:
        label   = f"k{fetch_k}_n{top_n}_t{threshold}"
        results = _run_combo(
            cases,
            combo_label=label,
            fetch_k=fetch_k,
            top_n=top_n,
            threshold=threshold,
            judge_client=judge_client,
        )
        agg = _aggregate(results)
        all_combos.append({
            "label":     label,
            "fetch_k":   fetch_k,
            "top_n":     top_n,
            "threshold": threshold,
            "results":   results,
            "agg":       agg,
        })
        log.info(f"  → {label} complete: mean={agg['mean']:.3f}")

    _print_summary_table(all_combos)
    _save_json(all_combos, stage=2, timestamp=timestamp)
    _save_csv(all_combos,  stage=2, timestamp=timestamp)

    # Push top 5 across both stages to LangSmith.
    # If stage1_combos not passed in-process, load from the most recent stage1 file.
    if stage1_combos is None:
        stage1_files = sorted(_RESULTS_DIR.glob("sweep_retrieval_stage1_*.json"))
        if stage1_files:
            raw = json.loads(stage1_files[-1].read_text(encoding="utf-8"))
            stage1_combos = [
                {"label": label, "results": results, "agg": raw["summary"][label]}
                for label, results in raw["combos"].items()
            ]
            log.info(f"Loaded {len(stage1_combos)} Stage 1 combos from {stage1_files[-1].name}")
        else:
            log.warning("No Stage 1 results found — LangSmith push will rank Stage 2 only.")
            stage1_combos = []

    _push_top_configs(stage1_combos + all_combos)

    return all_combos


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Two-stage retrieval parameter sweep for ScienceQ.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--stage", type=int, required=True, choices=[1, 2],
        help="Which stage to run: 1 (retrieval shape) or 2 (threshold calibration).",
    )
    parser.add_argument(
        "--fetch-k", type=int, default=None,
        help="Stage 2 only: fetch_k from Stage 1 winner.",
    )
    parser.add_argument(
        "--top-n", type=int, default=None,
        help="Stage 2 only: top_n from Stage 1 winner.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="List combos and cases without running agent or judge.",
    )
    parser.add_argument(
        "--case", metavar="CASE_ID",
        help="Run a single case by id (e.g. rag_001) for a quick smoke test.",
    )
    args = parser.parse_args()

    # ── Validate stage 2 args ──────────────────────────────────────────────────
    if args.stage == 2 and (args.fetch_k is None or args.top_n is None):
        parser.error("--stage 2 requires --fetch-k and --top-n (Stage 1 winner).")

    # ── Load eval set ──────────────────────────────────────────────────────────
    if not EVAL_SET_PATH.exists():
        log.error(f"eval_set.json not found at {EVAL_SET_PATH}")
        sys.exit(1)

    eval_set  = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    automated = [c for c in eval_set["cases"] if c["type"] != "adversarial"]

    if args.case:
        automated = [c for c in automated if c["id"] == args.case]
        if not automated:
            log.error(f"No automated case found with id={args.case!r}")
            sys.exit(1)

    log.info(f"Eval cases loaded: {len(automated)} automated")

    if args.dry_run:
        if args.stage == 1:
            run_stage1(automated, judge_client=None, dry_run=True)
        else:
            run_stage2(automated, judge_client=None, fetch_k=args.fetch_k,
                       top_n=args.top_n, dry_run=True)
        return

    # ── Initialise judge ───────────────────────────────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        log.error("OPENAI_API_KEY not set — required for GPT-4.1 judge.")
        sys.exit(1)
    judge_client = openai.OpenAI(api_key=openai_key)
    log.info(f"Judge: {JUDGE_MODEL}")

    # ── Run the requested stage ────────────────────────────────────────────────
    if args.stage == 1:
        combos = run_stage1(automated, judge_client)
        if combos:
            best = max(combos, key=lambda c: c["agg"]["mean"])
            log.info(
                f"\nStage 1 winner: {best['label']}  (mean={best['agg']['mean']:.3f})\n"
                f"  → Run Stage 2 with: "
                f"--stage 2 --fetch-k {best['fetch_k']} --top-n {best['top_n']}"
            )
    else:
        run_stage2(
            automated,
            judge_client,
            fetch_k=args.fetch_k,
            top_n=args.top_n,
        )


if __name__ == "__main__":
    main()
