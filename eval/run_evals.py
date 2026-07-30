"""
run_evals.py
------------
LangSmith evaluation runner for the ScienceQ.

Loads eval_set.json, runs each case through the live agent, scores every
answer with an LLM-as-judge on four rubric dimensions, and pushes results
to LangSmith as a named dataset + experiment for cross-run comparison.

Scoring dimensions (each 1–5):
  correctness  — key facts match the reference answer
  tone         — direct answer without librarian hedging openers
  grounding    — stays within transcript context, no external invention
  conciseness  — 2–4 paragraphs, no padding

Adversarial cases (type == "adversarial") are skipped in automated eval
and written to a separate manual_review.json file for human inspection.

Multi-turn cases (type == "rag_multi_turn") use real two-turn agent calls
with live memory so the query rewriter is exercised.

Usage:
  cd project-ironhack-youtube-chatbot
  python eval/run_evals.py

  # Dry-run — skip agent calls, print case list only:
  python eval/run_evals.py --dry-run

  # Run a single case by id:
  python eval/run_evals.py --case rag_001

  # Override experiment name:
  python eval/run_evals.py --experiment-name "prompt-v2"

Requirements:
  OPENAI_API_KEY in .env      — used by GPT-4.1 judge
  LANGSMITH_API_KEY in .env   — used to push results
  LANGCHAIN_TRACING_V2=true   — enables LangSmith tracing

Output:
  eval/results/run_<timestamp>.json   — full results locally
  eval/manual_review.json             — adversarial cases for human review
  LangSmith dashboard                 — dataset + experiment with scores
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import uuid
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

# ── Path setup ─────────────────────────────────────────────────────────────────
_EVAL_DIR     = Path(__file__).resolve().parent
_ROOT         = _EVAL_DIR.parent
_AGENT_DIR    = _ROOT / "agent"
_PIPELINE_DIR = _ROOT / "pipeline"
_RESULTS_DIR  = _EVAL_DIR / "results"
_RESULTS_DIR.mkdir(exist_ok=True)

for p in [str(_AGENT_DIR), str(_PIPELINE_DIR)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from agent import YouTubeQAAgent  # noqa: E402
import agent as _agent_mod  # noqa: E402
import rag_chain as _rag_chain  # noqa: E402
import retriever as _retriever  # noqa: E402

# ── Config ─────────────────────────────────────────────────────────────────────
JUDGE_MODEL    = "gpt-4.1"              # stronger than the evaluated model (gpt-oss-120b)
EVAL_SET_PATH  = _EVAL_DIR / "eval_set.json"
MANUAL_REVIEW_PATH = _EVAL_DIR / "manual_review.json"
RATE_LIMIT_SLEEP   = 1.0               # seconds between judge calls (OpenAI safety)
INTER_CASE_SLEEP   = 3.0               # seconds between cases (Groq rate limit safety)


# ── Run configuration capture ──────────────────────────────────────────────────
#
# Every checkpoint before #114 recorded only the judge model and case counts, so
# the retrieval config a given score was produced under is unrecoverable — the
# run resolved RETRIEVER_TOP_N / SCORE_THRESHOLD / RERANKER_ENABLED through
# retriever.py's module-level os.getenv, which silently falls back to a code
# literal when the key is absent from .env. That is how Phase 6 validated
# against a threshold the Phase 5 sweep had already rejected (see the comment
# above the constants in agent/retriever.py). Recording the config here is what
# stops the same question being unanswerable about this run.

# Env vars whose absence means the run used a code fallback rather than an
# explicit setting. Keyed by var name → the module attribute holding the
# effective value.
_TRACKED_ENV = {
    "RERANKER_ENABLED":   (_retriever, "RERANKER_ENABLED"),
    "RETRIEVER_FETCH_K":  (_retriever, "RETRIEVER_FETCH_K"),
    "RETRIEVER_TOP_N":    (_retriever, "RETRIEVER_TOP_N"),
    "SCORE_THRESHOLD":    (_retriever, "SCORE_THRESHOLD"),
    "PINECONE_INDEX_NAME": (_retriever, "PINECONE_INDEX_NAME"),
}


def _git_commit() -> Optional[str]:
    """
    Short SHA of the code under test, or None if it cannot be determined.

    EVAL_GIT_COMMIT takes precedence because the container route has no way to
    work it out for itself: the API image ships no git binary, and under a git
    worktree .git is a file pointing outside the bind mount. Both make the
    subprocess fall back to None, which would leave the one field that pins
    *which code produced this score* empty. Pass it in from the host:

        docker run -e EVAL_GIT_COMMIT=$(git rev-parse --short HEAD) ...
    """
    pinned = os.getenv("EVAL_GIT_COMMIT", "").strip()
    if pinned:
        return pinned
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=_ROOT, capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _capture_run_config() -> dict:
    """
    Snapshot the retrieval and model configuration this run actually used.

    Reads the *effective* module-level values rather than os.getenv, so an
    override applied by monkeypatching the module (the pattern
    sweep_retrieval.py and sweep_reranker.py both use) is recorded as what
    really ran. `env_source` separately records whether each value came from an
    explicit env var or from the code fallback — the distinction that made the
    pre-#114 checkpoints unverifiable.

    Also pins the git commit, since prompt revisions and eval-set growth move
    scores independently of config and are not otherwise recoverable from a
    results file.
    """
    return {
        "git_commit": _git_commit(),
        "retrieval": {
            "multi_namespace":   _agent_mod.MULTI_NAMESPACE,
            "namespaces_queried": (
                [_retriever.PINECONE_NAMESPACE_CORPUS, _retriever.PINECONE_NAMESPACE_LIVE]
                if _agent_mod.MULTI_NAMESPACE
                else [_retriever.PINECONE_NAMESPACE_CORPUS]
            ),
            "reranker_enabled":  _retriever.RERANKER_ENABLED,
            "reranker_model":    _retriever.RERANKER_MODEL,
            "retriever_fetch_k": _retriever.RETRIEVER_FETCH_K,
            "retriever_top_n":   _retriever.RETRIEVER_TOP_N,
            "score_threshold":   _retriever.SCORE_THRESHOLD,
            "embedding_model":   _retriever.COHERE_MODEL,
            "pinecone_index":    _retriever.PINECONE_INDEX_NAME,
        },
        "models": {
            "answer_model":       _rag_chain.GROQ_MODEL,
            "answer_temperature": _rag_chain.GROQ_TEMPERATURE,
            "rewrite_model":      _rag_chain.REWRITE_MODEL,
            "judge_model":        JUDGE_MODEL,
        },
        # "env" = explicit value in the environment/.env; "code-default" = the
        # os.getenv fallback in agent/retriever.py was used.
        "env_source": {
            name: ("env" if os.getenv(name) is not None else "code-default")
            for name in _TRACKED_ENV
        },
    }


def _log_run_config(cfg: dict) -> None:
    """Print the captured config up front so it is visible in the run log too."""
    r, m = cfg["retrieval"], cfg["models"]
    log.info("─── Run config ───────────────────────────────────────────")
    log.info(f"  commit         {cfg['git_commit'] or 'UNKNOWN — pass EVAL_GIT_COMMIT to pin it'}")
    log.info(f"  answer model   {m['answer_model']} (temp={m['answer_temperature']})")
    log.info(f"  judge model    {m['judge_model']}")
    log.info(f"  namespaces     {', '.join(r['namespaces_queried'])}"
             f"{'' if r['multi_namespace'] else '  (corpus-only — not the production path)'}")
    log.info(f"  reranker       {r['reranker_enabled']} ({r['reranker_model']})")
    log.info(f"  fetch_k/top_n  {r['retriever_fetch_k']}/{r['retriever_top_n']}")
    log.info(f"  threshold      {r['score_threshold']}")
    log.info(f"  index          {r['pinecone_index']}")
    fallbacks = [n for n, src in cfg["env_source"].items() if src == "code-default"]
    if fallbacks:
        log.warning(
            "  Using code defaults (not set in env): " + ", ".join(sorted(fallbacks))
        )

# ── Judge prompts ──────────────────────────────────────────────────────────────

_JUDGE_SYSTEM = """\
You are a strict, consistent evaluator scoring answers produced by a \
RAG-based YouTube QA chatbot. Score the given answer on the requested \
dimension using the provided rubric. Respond ONLY with a JSON object \
containing two keys: "score" (integer 1–5) and "reason" (one sentence \
explaining the score). No preamble, no markdown, no extra keys.\
"""

_RUBRIC = {
    "correctness": (
        "Does the answer contain the key facts from the reference answer?\n"
        "5 = all key facts present and accurate\n"
        "4 = most key facts present, minor omissions\n"
        "3 = some key facts present, notable gaps\n"
        "2 = few key facts present, mostly incomplete\n"
        "1 = key facts absent or contradicted"
    ),
    "tone": (
        "Does the answer lead directly without hedging openers such as "
        "'According to the video', 'The provided context', 'Based on the transcript', "
        "or 'I don't have information beyond what was discussed'?\n"
        "5 = leads directly, no hedging openers anywhere\n"
        "4 = mostly direct, one minor hedge\n"
        "3 = some hedging but answer is still useful\n"
        "2 = noticeable hedging that weakens the answer\n"
        "1 = dominated by hedging or librarian framing"
    ),
    "grounding": (
        "Does the answer stay within what the transcript context would contain, "
        "without adding external facts, statistics, or claims not derivable from "
        "a science YouTube video on this topic?\n"
        "5 = fully grounded, nothing added beyond reasonable inference\n"
        "4 = mostly grounded, one borderline external detail\n"
        "3 = some external additions but core answer is grounded\n"
        "2 = notable external additions that could be hallucinated\n"
        "1 = substantial hallucination or fabricated claims"
    ),
    "conciseness": (
        "Is the answer appropriately concise — 2 to 4 paragraphs, no padding, "
        "no unnecessary repetition, no meta-commentary about the question?\n"
        "5 = ideal length, dense and useful throughout\n"
        "4 = slightly long but no wasted content\n"
        "3 = some padding or repetition\n"
        "2 = noticeably too long or too short\n"
        "1 = severely padded or a single unhelpful sentence"
    ),
}


# ══════════════════════════════════════════════════════════════════════════════
# Judge
# ══════════════════════════════════════════════════════════════════════════════

def _score_dimension(
    dimension: str,
    question: str,
    answer: str,
    reference: str,
    client: openai.OpenAI,
) -> dict:
    """
    Call GPT-4.1 to score a single dimension.
    Returns {"score": int, "reason": str} or {"score": 0, "reason": str} on error.
    Uses JSON mode to guarantee parseable output.
    """
    rubric_text = _RUBRIC[dimension]

    user_prompt = f"""\
Dimension: {dimension.upper()}

Rubric:
{rubric_text}

Question:
{question}

Reference answer (ground truth):
{reference}

Bot answer to evaluate:
{answer}

Respond with JSON only: {{"score": <1-5>, "reason": "<one sentence>"}}
"""
    try:
        response = client.chat.completions.create(
            model           = JUDGE_MODEL,
            max_tokens      = 256,
            response_format = {"type": "json_object"},
            messages        = [
                {"role": "system", "content": _JUDGE_SYSTEM},
                {"role": "user",   "content": user_prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()
        result = json.loads(raw)
        assert "score" in result and "reason" in result
        return result
    except Exception as e:
        log.warning(f"Judge call failed for dimension={dimension}: {e}")
        return {"score": 0, "reason": f"Judge error: {e}"}


def score_answer(
    question: str,
    answer: str,
    reference: str,
    client: openai.OpenAI,
) -> dict:
    """
    Score an answer on all 4 dimensions.
    Returns a dict of {dimension: {score, reason}} plus an aggregate mean.
    """
    scores = {}
    for dim in _RUBRIC:
        scores[dim] = _score_dimension(dim, question, answer, reference, client)
        time.sleep(RATE_LIMIT_SLEEP)

    valid = [v["score"] for v in scores.values() if v["score"] > 0]
    scores["mean"] = round(sum(valid) / len(valid), 2) if valid else 0.0
    return scores


# ══════════════════════════════════════════════════════════════════════════════
# Agent runner
# ══════════════════════════════════════════════════════════════════════════════

def run_rag_case(case: dict, agent: YouTubeQAAgent) -> str:
    """Run a single factual RAG case. Returns the bot's answer."""
    resp = agent.chat(case["question"])
    return resp.answer


def run_multi_turn_case(case: dict, agent: YouTubeQAAgent) -> str:
    """
    Run a multi-turn case with real two-turn agent calls (live memory).
    Turn 1 seeds the memory; Turn 2 is the question being evaluated.
    Returns the answer to Turn 2 only.
    """
    turns = case["turns"]

    # Turn 1 — seed memory (not evaluated)
    turn1_question = turns[0]["content"]
    log.info(f"  Multi-turn turn 1: {turn1_question!r}")
    agent.chat(turn1_question)

    # Turn 2 — evaluated question
    turn2_question = turns[2]["content"]
    log.info(f"  Multi-turn turn 2: {turn2_question!r}")
    resp = agent.chat(turn2_question)
    return resp.answer


# ══════════════════════════════════════════════════════════════════════════════
# LangSmith dataset + experiment
# ══════════════════════════════════════════════════════════════════════════════

def _get_or_create_langsmith_dataset(
    ls_client,
    dataset_name: str,
    cases: list[dict],
) -> str:
    """
    Get existing LangSmith dataset by name or create it from eval_set cases.
    Returns the dataset ID.
    """
    def _build_examples(subset: list[dict]) -> list[dict]:
        out = []
        for c in subset:
            if c["type"] == "adversarial":
                continue
            question = c["turns"][2]["content"] if c["type"] in ("rag_multi_turn", "rag_multi_turn_vague") else c["question"]
            out.append({
                "inputs":  {"question": question, "case_id": c["id"]},
                "outputs": {"reference_answer": c["reference_answer"]},
            })
        return out

    # Check if dataset already exists
    existing = list(ls_client.list_datasets(dataset_name=dataset_name))
    if existing:
        dataset_id = str(existing[0].id)
        log.info(f"Using existing LangSmith dataset: {dataset_name} (id={dataset_id})")

        # Append any cases that are missing from the dataset
        existing_ids = {
            ex.inputs.get("case_id")
            for ex in ls_client.list_examples(dataset_id=dataset_id)
        }
        missing_cases = [c for c in cases if c["id"] not in existing_ids]
        if missing_cases:
            examples = _build_examples(missing_cases)
            if examples:
                ls_client.create_examples(
                    inputs     = [e["inputs"]  for e in examples],
                    outputs    = [e["outputs"] for e in examples],
                    dataset_id = dataset_id,
                )
                log.info(f"Appended {len(examples)} missing examples to LangSmith dataset.")
        return dataset_id

    log.info(f"Creating LangSmith dataset: {dataset_name}")
    dataset = ls_client.create_dataset(
        dataset_name = dataset_name,
        description  = "ScienceQ eval set — automated cases (factual + multi-turn)",
    )

    examples = _build_examples(cases)
    ls_client.create_examples(
        inputs     = [e["inputs"]  for e in examples],
        outputs    = [e["outputs"] for e in examples],
        dataset_id = dataset.id,
    )
    log.info(f"Uploaded {len(examples)} examples to LangSmith dataset.")
    return str(dataset.id)


def _push_experiment_results(
    ls_client,
    dataset_id: str,
    experiment_name: str,
    results: list[dict],
    run_config: Optional[dict] = None,
) -> None:
    """
    Push scored results to LangSmith as a named experiment run.
    Each result becomes a feedback entry on the corresponding example.

    `run_config` rides along on every run's `extra` so a LangSmith experiment is
    self-describing even when no local results file survives — which is the
    state every pre-#114 checkpoint is in.
    """
    try:
        for result in results:
            if result.get("skipped"):
                continue
            # Generate run_id explicitly — create_run() returns None in
            # langsmith SDK >= 0.0.69, so we can't rely on its return value.
            run_id = str(uuid.uuid4())
            ls_client.create_run(
                id          = run_id,
                name        = experiment_name,
                run_type    = "chain",
                inputs      = {"question": result["question"]},
                outputs     = {"answer":   result["answer"]},
                extra       = {
                    "case_id":   result["case_id"],
                    "case_type": result["case_type"],
                    "scores":    result["scores"],
                    "config":    run_config or {},
                },
            )
            ls_client.update_run(run_id, end_time=datetime.now(timezone.utc))

            # Attach scores as feedback
            scores = result.get("scores", {})
            for dim, val in scores.items():
                if dim == "mean":
                    ls_client.create_feedback(
                        run_id = run_id,
                        key    = "mean_score",
                        score  = round(val / 5, 4),  # normalize 1-5 → 0.0-1.0
                    )
                elif isinstance(val, dict) and "score" in val:
                    ls_client.create_feedback(
                        run_id  = run_id,
                        key     = dim,
                        score   = round(val["score"] / 5, 4),  # normalize 1-5 → 0.0-1.0
                        comment = val.get("reason", ""),
                    )
        log.info(f"Pushed {len(results)} results to LangSmith experiment: {experiment_name}")
    except Exception as e:
        log.warning(f"LangSmith push failed (results still saved locally): {e}")


# ══════════════════════════════════════════════════════════════════════════════
# Main runner
# ══════════════════════════════════════════════════════════════════════════════

def run(
    dry_run:         bool          = False,
    case_filter:     Optional[str] = None,
    type_filter:     Optional[str] = None,
    experiment_name: Optional[str] = None,
    multi_namespace: bool          = False,
) -> None:

    # ── Pin retrieval scope ────────────────────────────────────────────────────
    # Corpus-only by default. The 'live' namespace accumulates every URL
    # ingested through production, so scoring against it means the retrieval
    # pool moves between runs and a checkpoint cannot be compared to the next
    # one. --multi-namespace opts back into the production path when the point
    # of the run is to measure what users actually get.
    _agent_mod.MULTI_NAMESPACE = multi_namespace

    # ── Load eval set ──────────────────────────────────────────────────────────
    if not EVAL_SET_PATH.exists():
        log.error(f"eval_set.json not found at {EVAL_SET_PATH}")
        sys.exit(1)

    eval_set = json.loads(EVAL_SET_PATH.read_text(encoding="utf-8"))
    all_cases = eval_set["cases"]
    log.info(f"Loaded {len(all_cases)} cases from eval_set.json")

    # Apply case filter
    if case_filter:
        all_cases = [c for c in all_cases if c["id"] == case_filter]
        if not all_cases:
            log.error(f"No case found with id={case_filter!r}")
            sys.exit(1)

    # Apply type filter
    if type_filter:
        all_cases = [c for c in all_cases if c["type"] == type_filter]
        if not all_cases:
            log.error(f"No cases found with type={type_filter!r}")
            sys.exit(1)
        log.info(f"Type filter applied: {len(all_cases)} case(s) with type={type_filter!r}")

    # Split adversarial out for manual review
    automated = [c for c in all_cases if c["type"] != "adversarial"]
    adversarial = [c for c in all_cases if c["type"] == "adversarial"]

    log.info(
        f"Cases: {len(automated)} automated, "
        f"{len(adversarial)} adversarial (flagged for manual review)"
    )

    # Write adversarial cases to manual_review.json
    if adversarial:
        MANUAL_REVIEW_PATH.write_text(
            json.dumps({"cases": adversarial}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        log.info(f"Adversarial cases written to {MANUAL_REVIEW_PATH}")

    # ── Capture the config this run will use ──────────────────────────────────
    # Above the dry-run exit deliberately: --dry-run is then a free preflight
    # that shows the exact config a real run would use — including which values
    # fall back to code defaults — without spending an API call. Also means a
    # run aborted midway still has its config on record.
    run_config = _capture_run_config()
    _log_run_config(run_config)

    if dry_run:
        log.info("DRY RUN — listing automated cases only:")
        for c in automated:
            q = c["turns"][2]["content"] if c["type"] in ("rag_multi_turn", "rag_multi_turn_vague") else c["question"]
            log.info(f"  [{c['id']}] ({c['type']}) {q[:80]}")
        return

    # ── Initialise clients ─────────────────────────────────────────────────────
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        log.error("OPENAI_API_KEY not set in .env — required for GPT-4.1 judge.")
        sys.exit(1)

    judge_client = openai.OpenAI(api_key=openai_key)
    log.info(f"Judge model: {JUDGE_MODEL}")

    # LangSmith (optional — results always saved locally even if LS fails)
    ls_client    = None
    dataset_id   = None
    ls_available = False
    try:
        from langsmith import Client as LSClient
        ls_client    = LSClient()
        ls_available = True
        log.info("LangSmith client initialised.")
    except Exception as e:
        log.warning(f"LangSmith unavailable — results will be local only: {e}")

    # ── Set experiment name ────────────────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    exp_name  = experiment_name or f"youtube-qa-bot-eval-{timestamp}"

    # ── Create/fetch LangSmith dataset ────────────────────────────────────────
    if ls_available:
        try:
            dataset_id = _get_or_create_langsmith_dataset(
                ls_client,
                dataset_name = "youtube-qa-bot-eval-set",
                cases        = all_cases,
            )
        except Exception as e:
            log.warning(f"LangSmith dataset setup failed: {e}")

    # ── Run cases ──────────────────────────────────────────────────────────────
    results = []
    total   = len(automated)

    for i, case in enumerate(automated, start=1):
        case_id   = case["id"]
        case_type = case["type"]
        question  = (
            case["turns"][2]["content"]
            if case_type in ("rag_multi_turn", "rag_multi_turn_vague")
            else case["question"]
        )
        reference = case["reference_answer"]

        log.info(f"\n[{i}/{total}] {case_id} ({case_type})")
        log.info(f"  Q: {question[:100]}")

        # Fresh agent per case — no memory bleed between cases
        agent = YouTubeQAAgent()

        # ── Get bot answer ─────────────────────────────────────────────────────
        try:
            if case_type in ("rag_multi_turn", "rag_multi_turn_vague"):
                answer = run_multi_turn_case(case, agent)
            else:
                answer = run_rag_case(case, agent)
        except Exception as e:
            log.error(f"  Agent call failed: {e}")
            results.append({
                "case_id":   case_id,
                "case_type": case_type,
                "question":  question,
                "reference": reference,
                "answer":    "",
                "scores":    {},
                "error":     str(e),
                "skipped":   False,
            })
            continue

        log.info(f"  A: {answer[:120]}...")

        # ── Score with Claude judge ────────────────────────────────────────────
        log.info(f"  Scoring with {JUDGE_MODEL}...")
        scores = score_answer(question, answer, reference, judge_client)

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
            "answer":    answer,
            "scores":    scores,
            "error":     None,
            "skipped":   False,
        })

        # Pause between cases to avoid Groq rate limits
        if i < total:
            time.sleep(INTER_CASE_SLEEP)

    # ── Summary ────────────────────────────────────────────────────────────────
    scored = [r for r in results if r.get("scores") and not r.get("error")]
    if scored:
        dims = ["correctness", "tone", "grounding", "conciseness"]
        log.info("\n─── Eval Summary ─────────────────────────────────────────")
        for dim in dims:
            avg = sum(r["scores"][dim]["score"] for r in scored) / len(scored)
            log.info(f"  {dim:<14} avg={avg:.2f}/5")
        overall = sum(r["scores"]["mean"] for r in scored) / len(scored)
        log.info(f"  {'overall mean':<14} avg={overall:.2f}/5")
        log.info(f"  Cases scored: {len(scored)}/{total}")

        # Flag low-scoring cases
        low = [r for r in scored if r["scores"]["mean"] < 3.0]
        if low:
            log.info(f"\n  ⚠ Low-scoring cases (mean < 3.0):")
            for r in low:
                log.info(f"    {r['case_id']} — mean={r['scores']['mean']} — {r['question'][:60]}")

    # ── Save results locally ───────────────────────────────────────────────────
    output = {
        "experiment":  exp_name,
        "timestamp":   timestamp,
        "judge_model": JUDGE_MODEL,
        "config":      run_config,
        "total_cases": total,
        "scored":      len(scored),
        "adversarial_flagged": len(adversarial),
        "results":     results,
    }
    out_path = _RESULTS_DIR / f"run_{timestamp}.json"
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info(f"\nResults saved to {out_path}")

    # ── Push to LangSmith ──────────────────────────────────────────────────────
    if ls_available and ls_client and dataset_id:
        _push_experiment_results(ls_client, dataset_id, exp_name, results, run_config)
        log.info(f"LangSmith experiment: {exp_name}")
    elif ls_available and not dataset_id:
        log.warning("LangSmith dataset_id is None — skipping experiment push. "
                    "Run without --case filter to upload the full dataset first.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run ScienceQ evaluations and push results to LangSmith."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List cases without running the agent or judge.",
    )
    parser.add_argument(
        "--case",
        metavar="CASE_ID",
        help="Run a single case by id (e.g. rag_001, mt_003).",
    )
    parser.add_argument(
        "--experiment-name",
        metavar="NAME",
        help="Override the LangSmith experiment name (default: auto-timestamped).",
    )
    parser.add_argument(
        "--type",
        metavar="TYPE",
        help="Run only cases of this type (e.g. rag_multilingual, rag_factual, rag_multi_turn).",
    )
    parser.add_argument(
        "--multi-namespace",
        action="store_true",
        help="Query corpus + live, as production does. Default is corpus-only, "
             "so the retrieval pool is stable between checkpoints.",
    )
    args = parser.parse_args()

    run(
        dry_run         = args.dry_run,
        case_filter     = args.case,
        type_filter     = args.type,
        experiment_name = args.experiment_name,
        multi_namespace = args.multi_namespace,
    )
