"""
calibrate_threshold.py
----------------------
Sweep score thresholds using eval_set.json to find the optimal
DEFAULT_SCORE_THRESHOLD for the Cohere embed-multilingual-v3.0 index.

Strategy:
  - Run all 30 eval questions through the retriever at threshold=0.0
  - Record the top-1 score per question
  - Print a score distribution by question type
  - Suggest a threshold that maximises hits on rag_factual/multi_turn
    while keeping adversarial questions at 0 results

Usage:
  python eval/calibrate_threshold.py
  python eval/calibrate_threshold.py --top-k 10
"""

import argparse
import json
import sys
from pathlib import Path

# Make agent/ importable from the project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent.retriever import retrieve_multi_namespace

EVAL_PATH    = Path(__file__).parent / "eval_set.json"
RAG_CHAIN_PATH = Path(__file__).parent.parent / "agent" / "rag_chain.py"

CANDIDATE_THRESHOLDS = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]


def get_question(case: dict) -> str:
    """Extract the question string from any case type."""
    if "question" in case:
        return case["question"]
    # multi-turn: use first user turn
    for turn in case.get("turns", []):
        if turn["role"] == "user":
            return turn["content"]
    return ""


def run_calibration(top_k: int) -> None:
    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))["cases"]

    print(f"\n{'─'*72}")
    print(f"  Threshold calibration — {len(cases)} eval questions, top_k={top_k}")
    print(f"{'─'*72}\n")

    results = []
    for case in cases:
        question = get_question(case)
        if not question:
            continue

        chunks = retrieve_multi_namespace(query=question, top_k=top_k, score_threshold=0.0)
        top_score = chunks[0].score if chunks else 0.0

        results.append({
            "id":        case["id"],
            "type":      case["type"],
            "question":  question[:60] + ("…" if len(question) > 60 else ""),
            "top_score": top_score,
            "n_chunks":  len(chunks),
        })

        tag = f"[{case['type'][:3].upper()}]"
        print(f"  {tag} {case['id']:8s}  score={top_score:.4f}  chunks={len(chunks):2d}  {question[:50]}")

    # ── Score distribution by type ────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  Score distribution by type\n")

    by_type: dict[str, list[float]] = {}
    for r in results:
        by_type.setdefault(r["type"], []).append(r["top_score"])

    for qtype, scores in sorted(by_type.items()):
        avg   = sum(scores) / len(scores)
        mn    = min(scores)
        mx    = max(scores)
        zeros = sum(1 for s in scores if s == 0.0)
        print(f"  {qtype:20s}  n={len(scores):2d}  avg={avg:.4f}  min={mn:.4f}  max={mx:.4f}  zero_hits={zeros}")

    # ── Threshold sweep ───────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    print("  Threshold sweep (hit = at least 1 chunk returned)\n")
    print(f"  {'threshold':>10}  {'rag_factual':>12}  {'rag_multi_turn':>15}  {'adversarial':>12}")
    print(f"  {'─'*10}  {'─'*12}  {'─'*15}  {'─'*12}")

    rag_scores  = [r["top_score"] for r in results if r["type"] == "rag_factual"]
    mt_scores   = [r["top_score"] for r in results if r["type"] == "rag_multi_turn"]
    adv_scores  = [r["top_score"] for r in results if r["type"] == "adversarial"]

    best_threshold = None
    best_rag_hits  = -1

    for t in CANDIDATE_THRESHOLDS:
        rag_hits = sum(1 for s in rag_scores  if s >= t)
        mt_hits  = sum(1 for s in mt_scores   if s >= t)
        adv_hits = sum(1 for s in adv_scores  if s >= t)

        rag_pct = rag_hits / len(rag_scores) * 100 if rag_scores else 0
        mt_pct  = mt_hits  / len(mt_scores)  * 100 if mt_scores  else 0
        adv_pct = adv_hits / len(adv_scores) * 100 if adv_scores else 0

        flag = ""
        # Best = max rag_factual hits with 0 adversarial hits
        if adv_hits == 0 and rag_hits > best_rag_hits:
            best_rag_hits  = rag_hits
            best_threshold = t
            flag = "  ◀ suggested"

        print(
            f"  {t:>10.2f}  "
            f"{rag_hits:2d}/{len(rag_scores)} ({rag_pct:5.1f}%)  "
            f"{mt_hits:2d}/{len(mt_scores)} ({mt_pct:5.1f}%)    "
            f"{adv_hits:2d}/{len(adv_scores)} ({adv_pct:5.1f}%)"
            f"{flag}"
        )

    # ── Recommendation ────────────────────────────────────────────────────────
    print(f"\n{'─'*72}")
    if best_threshold is not None:
        print(f"  Suggested threshold: {best_threshold}")
        print(f"  Update DEFAULT_SCORE_THRESHOLD in agent/rag_chain.py")
    else:
        print("  No threshold found that filters all adversarial — check score distribution above.")
        print("  Consider whether adversarial questions happen to match corpus content.")
    print(f"{'─'*72}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calibrate retrieval score threshold using eval_set.json."
    )
    parser.add_argument(
        "--top-k", type=int, default=5,
        help="top_k passed to retrieve_multi_namespace (default: 5)",
    )
    args = parser.parse_args()
    run_calibration(top_k=args.top_k)
