"""
validate_multilingual.py
------------------------
Cross-lingual retrieval validation for Phase 6.

Runs the 4 planned validation queries against the corpus namespace and checks
that non-English chunks surface above the score threshold alongside English ones.

Usage:
    python eval/validate_multilingual.py
    python eval/validate_multilingual.py --threshold 0.35 --top-k 12
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "agent"))
from retriever import retrieve, SCORE_THRESHOLD

# ── Validation cases ───────────────────────────────────────────────────────────

QUERIES = [
    {
        "query":    "What is quantum mechanics?",
        "expected": ["Rj3jTw2DxXQ"],   # Science Étonnante (FR)
        "note":     "FR — Science Étonnante should surface alongside PBS Space Time / Veritasium",
    },
    {
        "query":    "Is mathematics discovered or invented?",
        "expected": ["zUyItDq2JAk"],   # Terra X Lesch (DE)
        "note":     "DE — Terra X should surface alongside Vsauce / 3Blue1Brown",
    },
    {
        "query":    "Is the 10000 hour rule true?",
        "expected": ["QSTLljIWJ1U"],   # Ciência Todo Dia (PT)
        "note":     "PT — Ciência Todo Dia should surface alongside TED psychology",
    },
    {
        "query":    "What is Gödel's incompleteness theorem?",
        "expected": ["_WM4XMwhRBE"],   # CuriosaMente (ES)
        "note":     "ES — CuriosaMente should surface alongside Veritasium / 3Blue1Brown",
    },
]

def run(threshold: float, top_k: int) -> None:
    print(f"\nCross-lingual retrieval validation")
    print(f"threshold={threshold}  top_k={top_k}  namespace=corpus")
    print("=" * 70)

    all_passed = True

    for case in QUERIES:
        query    = case["query"]
        expected = set(case["expected"])
        note     = case["note"]

        print(f"\nQuery: \"{query}\"")
        print(f"Expect: {note}")
        print("-" * 70)

        chunks = retrieve(
            query,
            namespace="corpus",
            top_k=top_k,
            score_threshold=0.0,   # fetch everything, display threshold separately
        )

        if not chunks:
            print("  [NO RESULTS]")
            all_passed = False
            continue

        expected_found   = set()
        expected_cleared = set()

        for i, c in enumerate(chunks, 1):
            cleared = c.score >= threshold
            label   = c.language.upper()
            marker  = "✓" if cleared else "·"
            flag    = " ◄ TARGET" if c.video_id in expected else ""

            print(
                f"  {marker} [{label}] {c.score:.4f}  {c.title[:45]:<45}"
                f"  {c.channel[:22]}{flag}"
            )

            if c.video_id in expected:
                expected_found.add(c.video_id)
                if cleared:
                    expected_cleared.add(c.video_id)

        # Per-query verdict
        missing  = expected - expected_found
        below_th = expected_found - expected_cleared

        if not missing and not below_th:
            print(f"\n  PASS — all expected non-English chunks present and above threshold")
        elif missing:
            print(f"\n  FAIL — expected video(s) not in top-{top_k}: {missing}")
            all_passed = False
        else:
            print(f"\n  FAIL — expected video(s) found but below threshold ({threshold}): {below_th}")
            all_passed = False

    print("\n" + "=" * 70)
    print("OVERALL:", "PASS" if all_passed else "FAIL — see details above")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cross-lingual retrieval validation.")
    parser.add_argument("--threshold", type=float, default=SCORE_THRESHOLD,
                        help=f"Score threshold (default: {SCORE_THRESHOLD})")
    parser.add_argument("--top-k",    type=int,   default=10,
                        help="Results to fetch per query (default: 10)")
    args = parser.parse_args()

    run(threshold=args.threshold, top_k=args.top_k)
