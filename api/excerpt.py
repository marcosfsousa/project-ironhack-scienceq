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

Scope note: the legacy Streamlit UI (`app/streamlit_app.py`) imports the agent
directly rather than calling this API, so it does not pass through here. That
costs nothing today — it renders only title, timestamp, and link, never the
chunk text — but it does mean "the API trims" is not the same claim as
"everything that can reach a chunk trims".

The cut is sentence-aware rather than a character chop. Auto-generated YouTube
captions frequently carry no sentence punctuation at all, so there is a
word-boundary fallback that never severs a word mid-way.
"""

from __future__ import annotations

import re

# Roughly one to two sentences of transcript prose. Long enough that the
# quotation still evidences why the source supports the answer, short enough
# that it reads as a citation rather than a copy of the chunk.
MAX_EXCERPT_CHARS = 240

# At most two sentences, even when more would fit inside the character budget.
MAX_EXCERPT_SENTENCES = 2

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
    `MAX_EXCERPT_SENTENCES`), or — when the text has no usable terminator — at
    the last whole word inside the budget, followed by an ellipsis.

    The result is always a prefix of the input, so the quotation is verbatim
    and never reorders or paraphrases what the speaker said. Idempotent:
    trimming an already-trimmed excerpt returns it unchanged.
    """
    if not text:
        return ""

    stripped = text.strip()
    if len(stripped) <= MAX_EXCERPT_CHARS:
        return stripped

    # Prefer a real sentence boundary inside the budget.
    ends = [m.end() for m in _SENTENCE_END.finditer(stripped)]
    fitting = [e for e in ends[:MAX_EXCERPT_SENTENCES] if e <= MAX_EXCERPT_CHARS]
    if fitting:
        return stripped[: fitting[-1]].strip()

    # No terminator in range — fall back to the last whole word. The ellipsis
    # is part of the budget, so a single unbroken run of characters longer than
    # the budget (no space to fall back to) still comes in at the limit rather
    # than one over it.
    window = stripped[: MAX_EXCERPT_CHARS - 1]
    last_space = window.rfind(" ")
    if last_space > 0:
        window = window[:last_space]
    return window.rstrip(_TRIM_TRAILING_CHARS) + "…"
