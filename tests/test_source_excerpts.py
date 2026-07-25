# tests/test_source_excerpts.py

"""
Contract tests for quotation-scale source excerpts (issue #16).

Two seams are covered:
  - api.excerpt.quotation_excerpt — the pure text-trimming rule.
  - api.service — that both response paths (blocking `POST /api/chat` and the
    streaming `[SOURCES]` frame) apply the rule, with the agent monkeypatched
    so the tests never touch the network.

Tests assert externally observable behaviour — what the API returns — not how
the trimming is implemented.
"""

import json
from types import SimpleNamespace

from api import service
from api.excerpt import MAX_EXCERPT_CHARS, MIN_EXCERPT_CHARS, quotation_excerpt


# A realistic full transcript chunk: several sentences, well over quotation scale.
FULL_CHUNK = (
    "CRISPR stands for Clustered Regularly Interspaced Short Palindromic Repeats. "
    "It was originally discovered as part of the bacterial immune system, which uses "
    "it to remember and destroy viruses that have attacked it before. "
    "Scientists realised this mechanism could be repurposed as a general-purpose "
    "gene-editing tool. Today it is used in laboratories all over the world to make "
    "precise edits to the genomes of plants, animals, and human cells."
)

# Auto-generated captions frequently carry no sentence punctuation at all.
UNPUNCTUATED_CHUNK = (
    "so what we're going to do here is take a look at the way these cells divide and "
    "you can see that each one splits into two identical copies and that process "
    "repeats over and over again until you end up with an entire organism made of "
    "trillions of cells all descended from that single original cell"
)

# A chunk opening on an abbreviation the terminator regex reads as a sentence
# end, with the next real terminator falling outside the budget. Chunks are
# 60-second windows, so at normal speech rates almost every one exceeds the
# budget and reaches the sentence logic — an opening "E.", "Dr." or personal
# initial is the common shape in this corpus, not an edge case.
ABBREVIATION_OPENING_CHUNK = (
    "E. coli is the workhorse of molecular biology because it grows fast, it is "
    "cheap to culture, and its genome has been mapped in exhaustive detail by "
    "generations of researchers working across dozens of laboratories on every "
    "continent over the course of the last century. That makes it the default "
    "chassis for almost any first experiment."
)


# ── The trimming rule ───────────────────────────────────────────────────────────

class TestQuotationExcerpt:

    def test_short_text_passes_through_unchanged(self):
        short = "CRISPR is a gene-editing tool."
        assert quotation_excerpt(short) == short

    def test_short_unpunctuated_text_passes_through_unchanged(self):
        short = "this is a caption fragment with no terminator"
        assert quotation_excerpt(short) == short

    def test_full_chunk_is_trimmed_to_quotation_scale(self):
        out = quotation_excerpt(FULL_CHUNK)
        assert len(out) < len(FULL_CHUNK)
        assert len(out) <= MAX_EXCERPT_CHARS

    def test_full_chunk_ends_on_a_sentence_boundary(self):
        out = quotation_excerpt(FULL_CHUNK)
        assert out.endswith((".", "!", "?"))
        # Sentence-aware, not a character chop: the cut lands on a real
        # terminator from the source, so no word is severed mid-way.
        assert FULL_CHUNK.startswith(out)

    def test_trims_to_at_most_two_sentences(self):
        out = quotation_excerpt(FULL_CHUNK)
        terminators = sum(out.count(c) for c in ".!?")
        assert 1 <= terminators <= 2

    def test_excerpt_remains_meaningful(self):
        # The quotation must still evidence why the source supports the answer,
        # so it keeps the opening sentence intact rather than a fragment.
        out = quotation_excerpt(FULL_CHUNK)
        assert out.startswith("CRISPR stands for Clustered Regularly")

    def test_unpunctuated_chunk_cuts_on_a_word_boundary(self):
        out = quotation_excerpt(UNPUNCTUATED_CHUNK)
        assert len(out) <= MAX_EXCERPT_CHARS
        assert out.endswith("…")
        # No severed word: everything before the ellipsis is a whole-word prefix.
        body = out[:-1]
        assert UNPUNCTUATED_CHUNK.startswith(body)
        assert UNPUNCTUATED_CHUNK[len(body)] == " "

    def test_is_idempotent(self):
        once = quotation_excerpt(FULL_CHUNK)
        assert quotation_excerpt(once) == once
        once_unpunctuated = quotation_excerpt(UNPUNCTUATED_CHUNK)
        assert quotation_excerpt(once_unpunctuated) == once_unpunctuated

    def test_never_exceeds_the_budget(self):
        """The ellipsis counts against the budget, not on top of it."""
        cases = (
            FULL_CHUNK,
            UNPUNCTUATED_CHUNK,
            "x" * 300,                      # one unbroken run — no space to fall back to
            "word " * 200,
            "no terminator anywhere in this caption " * 10,
        )
        for text in cases:
            assert len(quotation_excerpt(text)) <= MAX_EXCERPT_CHARS

    def test_unbroken_run_still_yields_an_excerpt(self):
        out = quotation_excerpt("x" * 300)
        assert out.endswith("…")
        assert set(out[:-1]) == {"x"}

    def test_abbreviation_does_not_truncate_at_the_first_boundary(self):
        """
        A terminal-looking abbreviation ("Dr.", "U.S.") is a sentence end by
        the regex, but it must not reduce the quotation to a stub — the cut
        takes the last boundary that fits, not the first.
        """
        text = (
            "Dr. Jane Goodall spent decades studying chimpanzees in Gombe, and what "
            "she found overturned the assumption that tool use was uniquely human. "
            "Her observations changed primatology permanently, and they reshaped how "
            "field researchers have approached every long-term primate study since."
        )
        # Guard the guard: under the budget the function returns early and the
        # sentence logic this test exists to cover is never reached.
        assert len(text) > MAX_EXCERPT_CHARS
        out = quotation_excerpt(text)
        assert out.startswith("Dr. Jane Goodall spent decades")
        assert out.endswith("uniquely human.")

    def test_abbreviation_opening_falls_back_rather_than_collapsing(self):
        """
        Regression (#25 review): when the abbreviation is the *only* boundary
        inside the budget, taking it shipped a two-character excerpt ("E.").
        Such a cut is rejected and the word-boundary fallback runs instead.
        """
        assert len(ABBREVIATION_OPENING_CHUNK) > MAX_EXCERPT_CHARS
        out = quotation_excerpt(ABBREVIATION_OPENING_CHUNK)
        assert len(out) >= MIN_EXCERPT_CHARS
        assert len(out) <= MAX_EXCERPT_CHARS
        assert out.endswith("…")
        assert out.startswith("E. coli is the workhorse")

    def test_no_opening_abbreviation_yields_a_stub(self):
        """The same trap across the abbreviations this corpus actually carries."""
        for opening in ("E.", "J.", "Dr.", "Mr.", "St.", "No.", "vs."):
            text = opening + " " + "some padding words to run past the budget " * 8
            out = quotation_excerpt(text)
            assert len(out) >= MIN_EXCERPT_CHARS, (opening, out)

    def test_never_returns_more_than_it_was_given(self):
        for text in (FULL_CHUNK, UNPUNCTUATED_CHUNK, "short.", ""):
            assert len(quotation_excerpt(text)) <= max(len(text), 1)

    def test_empty_and_whitespace_are_safe(self):
        assert quotation_excerpt("") == ""
        assert quotation_excerpt("   ") == ""
        assert quotation_excerpt(None) == ""


# ── Helpers / fakes ─────────────────────────────────────────────────────────────

def _chunk(text):
    """A retrieved chunk as the streaming path sees it."""
    return SimpleNamespace(
        title="Genetic Engineering Will Change Everything Forever – CRISPR",
        timestamp_label="3:11",
        youtube_link="https://www.youtube.com/watch?v=jAhjPd4uNFY&t=191",
        score=0.95,
        rerank_score=0.93,
        text=text,
    )


def _source_dict(text):
    """A source as the blocking path sees it (source_chunks_for_display shape)."""
    return {
        "title":        "Genetic Engineering Will Change Everything Forever – CRISPR",
        "timestamp":    "3:11",
        "link":         "https://www.youtube.com/watch?v=jAhjPd4uNFY&t=191",
        "score":        0.95,
        "rerank_score": 0.93,
        "text":         text,
    }


class _FakeBlockingAgent:
    def __init__(self, response):
        self._response = response

    def chat(self, message):
        return self._response


class _FakeStreamAgent:
    def __init__(self, tokens, chunks=()):
        self._tokens = list(tokens)
        self._streamed_chunks = list(chunks)
        self._last_provenance = SimpleNamespace(
            ai_generated=True, model="openai/gpt-oss-120b", mode="generated"
        )

    def stream_chat(self, question):
        for tok in self._tokens:
            yield tok


def _sources_frame(chunks):
    """Extract and parse the [SOURCES] payload from raw SSE output."""
    for raw_event in "".join(chunks).split("\n\n"):
        for line in raw_event.split("\n"):
            if line.startswith("data: [SOURCES]"):
                return json.loads(line[len("data: [SOURCES]"):])
    return None


# ── Blocking path ───────────────────────────────────────────────────────────────

class TestBlockingSourceExcerpts:

    def test_source_text_is_trimmed_to_quotation_scale(self, monkeypatch):
        resp = SimpleNamespace(
            answer="CRISPR edits DNA.",
            sources=[_source_dict(FULL_CHUNK)],
            intent="rag",
            provenance=SimpleNamespace(
                ai_generated=True, model="openai/gpt-oss-120b", mode="generated"
            ),
        )
        monkeypatch.setattr(service, "_build_agent", lambda history: _FakeBlockingAgent(resp))

        out = service.run_chat("What is CRISPR?", [])
        assert len(out.sources) == 1
        assert out.sources[0].text == quotation_excerpt(FULL_CHUNK)
        assert len(out.sources[0].text) < len(FULL_CHUNK)

    def test_source_field_names_are_unchanged(self, monkeypatch):
        """API stability: only the text field's content shrinks (user story 6)."""
        resp = SimpleNamespace(
            answer="CRISPR edits DNA.",
            sources=[_source_dict(FULL_CHUNK)],
            intent="rag",
            provenance=SimpleNamespace(
                ai_generated=True, model="openai/gpt-oss-120b", mode="generated"
            ),
        )
        monkeypatch.setattr(service, "_build_agent", lambda history: _FakeBlockingAgent(resp))

        src = service.run_chat("What is CRISPR?", []).sources[0].model_dump()
        assert set(src) == {"title", "timestamp", "link", "score", "rerank_score", "text"}
        assert src["title"].startswith("Genetic Engineering")
        assert src["timestamp"] == "3:11"
        assert src["link"].endswith("t=191")
        assert src["score"] == 0.95
        assert src["rerank_score"] == 0.93

    def test_short_source_text_survives_intact(self, monkeypatch):
        short = "CRISPR stands for Clustered Regularly Interspaced Short Palindromic Repeats."
        resp = SimpleNamespace(
            answer="CRISPR edits DNA.",
            sources=[_source_dict(short)],
            intent="rag",
            provenance=SimpleNamespace(
                ai_generated=True, model="openai/gpt-oss-120b", mode="generated"
            ),
        )
        monkeypatch.setattr(service, "_build_agent", lambda history: _FakeBlockingAgent(resp))

        assert service.run_chat("What is CRISPR?", []).sources[0].text == short


# ── Streaming path ──────────────────────────────────────────────────────────────

class TestStreamingSourceExcerpts:

    def test_sources_frame_text_is_trimmed(self, monkeypatch):
        agent = _FakeStreamAgent(tokens=["CRISPR ", "edits DNA."], chunks=[_chunk(FULL_CHUNK)])
        monkeypatch.setattr(service, "_build_agent", lambda history: agent)

        sources = _sources_frame(list(service.stream_run_chat("What is CRISPR?", [])))
        assert sources is not None
        assert sources[0]["text"] == quotation_excerpt(FULL_CHUNK)
        assert len(sources[0]["text"]) < len(FULL_CHUNK)

    def test_sources_frame_keeps_its_field_names(self, monkeypatch):
        agent = _FakeStreamAgent(tokens=["CRISPR."], chunks=[_chunk(FULL_CHUNK)])
        monkeypatch.setattr(service, "_build_agent", lambda history: agent)

        sources = _sources_frame(list(service.stream_run_chat("q", [])))
        assert set(sources[0]) == {"title", "timestamp", "link", "score", "rerank_score", "text"}

    def test_both_paths_agree_on_the_same_chunk(self, monkeypatch):
        """The trimming is applied server-side once, so clients cannot diverge."""
        agent = _FakeStreamAgent(tokens=["x"], chunks=[_chunk(FULL_CHUNK)])
        monkeypatch.setattr(service, "_build_agent", lambda history: agent)
        streamed = _sources_frame(list(service.stream_run_chat("q", [])))[0]["text"]

        resp = SimpleNamespace(
            answer="x",
            sources=[_source_dict(FULL_CHUNK)],
            intent="rag",
            provenance=SimpleNamespace(
                ai_generated=True, model="openai/gpt-oss-120b", mode="generated"
            ),
        )
        monkeypatch.setattr(service, "_build_agent", lambda history: _FakeBlockingAgent(resp))
        blocking = service.run_chat("q", []).sources[0].text

        assert streamed == blocking
