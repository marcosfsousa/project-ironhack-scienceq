"""
excerpt.py
----------
Quotation-scale trimming for source excerpts (issue #16).

Every answer cites the transcript chunks that grounded it. Shipping each
chunk's full text redistributes more third-party transcript content than the
product needs to make its grounding convincing — a short representative
quotation, plus the title, timestamp, and link that let a reader click through
to the original, evidences the source just as well.

This module owns that rule as a single pure function so the API response
boundary can apply it in one place: `api/service.py` calls it on both the
blocking `POST /api/chat` response and the streaming `[SOURCES]` frame, so
every API client inherits the same behaviour and the two paths cannot diverge.

Scope note: this used to be narrower than it looks, because the legacy Streamlit
UI imported the agent directly and bypassed the API. That surface was sunset in
issue #13, so the API is now the only *user-facing* path by which chunk text is
distributed. What still bypasses this module is developer tooling that prints to
a terminal, not to a product surface: the `agent.py` REPL (via `last_sources`,
which carries `chunk_text`) and the `retriever.py` CLI. `agent/tools.py` holds
chunks too but reads only title, timestamp, link, and score. Adding another
user-facing surface that imports the agent directly would reopen the gap.

The cut is sentence-aware rather than a character chop. Auto-generated YouTube
captions frequently carry no sentence punctuation at all, so there is a
word-boundary fallback that never severs a word mid-way. The same fallback
catches chunks whose only in-budget "sentence end" is an abbreviation or a
personal initial — "E. coli", "Dr. Smith" — which a purely sentence-aware cut
would reduce to two characters.
"""

from __future__ import annotations

import re

# Roughly one to two sentences of transcript prose. Long enough that the
# quotation still evidences why the source supports the answer, short enough
# that it reads as a citation rather than a copy of the chunk.
MAX_EXCERPT_CHARS = 240

# At most two sentence boundaries, even when more would fit inside the
# character budget. Counted as regex matches rather than true sentences: a
# leading abbreviation consumes one of the two, so "Dr. X did Y. Then Z."
# yields a single real sentence. Erring short is the safe direction for a
# quotation, so this is left as is.
MAX_EXCERPT_SENTENCES = 2

# A sentence-bounded cut shorter than this is a stub, not a quotation. The
# terminator regex cannot tell a sentence end from an abbreviation or a
# personal initial, so a chunk opening on "E.", "Dr." or "J." has a boundary
# two or three characters in. Where that is the only boundary inside the
# budget, taking it would ship "E." as the excerpt — below this floor the
# sentence cut is rejected and the word-boundary fallback runs instead.
MIN_EXCERPT_CHARS = 40

# A sentence terminator followed by whitespace or end-of-text. Requiring the
# trailing boundary keeps decimals ("0.95") and abbreviations mid-token from
# being read as sentence ends.
_SENTENCE_END = re.compile(r"[.!?…]+(?=\s|$)")

# Trailing characters left dangling by a mid-sentence cut, stripped so the
# ellipsis reads cleanly rather than following a stray comma or dash.
_TRIM_TRAILING_CHARS = " ,;:—–-"


def quotation_excerpt(text: str | None) -> str:
    """
    Reduce a transcript chunk to a short representative quotation.

    Text already at or under `MAX_EXCERPT_CHARS` is returned unchanged. Longer
    text is cut at the last sentence boundary that fits (at most
    `MAX_EXCERPT_SENTENCES`), or — when the text has no usable terminator, or
    the only one inside the budget is too early to be a quotation — at the last
    whole word inside the budget, followed by an ellipsis.

    The result is always a prefix of the input, so the quotation is verbatim
    and never reorders or paraphrases what the speaker said. Idempotent:
    trimming an already-trimmed excerpt returns it unchanged.
    """
    if not text:
        return ""

    stripped = text.strip()
    if len(stripped) <= MAX_EXCERPT_CHARS:
        return stripped

    # Prefer a real sentence boundary inside the budget. Filter by the budget
    # first and cap the count second: slicing to MAX_EXCERPT_SENTENCES up front
    # discards every later boundary whenever the leading ones are abbreviations,
    # which at 60-second chunk windows leaves the abbreviation as the only
    # candidate and collapses the excerpt to a stub.
    ends = [m.end() for m in _SENTENCE_END.finditer(stripped)]
    fitting = [e for e in ends if e <= MAX_EXCERPT_CHARS][:MAX_EXCERPT_SENTENCES]
    if fitting and fitting[-1] >= MIN_EXCERPT_CHARS:
        return stripped[: fitting[-1]].strip()

    # No usable terminator in range — fall back to the last whole word. The ellipsis
    # is part of the budget, so a single unbroken run of characters longer than
    # the budget (no space to fall back to) still comes in at the limit rather
    # than one over it.
    window = stripped[: MAX_EXCERPT_CHARS - 1]
    last_space = window.rfind(" ")
    if last_space > 0:
        window = window[:last_space]
    return window.rstrip(_TRIM_TRAILING_CHARS) + "…"
