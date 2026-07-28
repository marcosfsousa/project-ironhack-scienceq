# tests/test_last_sources.py

"""
Contract tests for the single source-display shape (issue #30).

`YouTubeQAAgent.last_sources` used to return one dict shape after a streamed
answer (title, video_id, channel, start, end, chunk_text) and another after a
blocking one (title, timestamp, link, score, rerank_score, text). The two shared
only `title`. The agent REPL, which only ever calls `chat()`, read the streaming
keys and so printed every source as `[Title]  0s–0s  — ...`.

Two seams are covered:
  - agent.last_sources — that both paths return the shape
    `rag_chain.chunks_for_display` defines, so a caller need not know which ran.
  - agent._cli — that the REPL's `sources` command prints the real timestamp and
    excerpt. This is the seam the bug was visible at, so it is asserted on the
    printed output rather than on the formatter's internals.

Chunks are real `RetrievedChunk` instances: a fake with hand-written attributes
would keep passing if the chunk renamed `timestamp_label` or `text`.
"""

# `RAGResponse` and `chunks_for_display` are taken from `agent.agent` rather than
# from `agent.rag_chain`: the agent imports rag_chain by its *bare* name, so the
# package path would load a second copy of the module and the tests would compare
# against a different function object than the one `last_sources` calls. See
# conftest.py for the bare-vs-package split this comes from.
from agent.agent import YouTubeQAAgent, RAGResponse, chunks_for_display, _cli
from agent.retriever import RetrievedChunk


DISPLAY_KEYS = {"title", "timestamp", "link", "score", "rerank_score", "text"}

EXCERPT = (
    "CRISPR stands for Clustered Regularly Interspaced Short Palindromic Repeats, "
    "originally found in the bacterial immune system."
)


def _chunk(start=191.0, end=254.0, text=EXCERPT):
    return RetrievedChunk(
        chunk_id="jAhjPd4uNFY-3",
        video_id="jAhjPd4uNFY",
        title="Genetic Engineering Will Change Everything Forever – CRISPR",
        channel="Kurzgesagt",
        topic="biology",
        language="en",
        start=start,
        end=end,
        text=text,
        score=0.95,
        namespace="corpus",
        rerank_score=0.93,
    )


def _agent(streamed=(), last_response=None):
    """
    An agent carrying only the two fields `last_sources` reads.

    Built without __init__ on purpose: the real constructor builds the LangGraph
    and its tools, which reach for Pinecone/Groq credentials, and none of that
    is under test here.
    """
    agent = object.__new__(YouTubeQAAgent)
    agent._streamed_chunks = list(streamed)
    agent._last_response = last_response
    return agent


def _rag_response(chunks):
    return RAGResponse(
        answer="CRISPR is a gene-editing tool.",
        chunks=list(chunks),
        question="What is CRISPR?",
        namespace="corpus",
    )


# ── One shape, both paths ───────────────────────────────────────────────────────

class TestLastSourcesShape:

    def test_streaming_path_returns_the_display_shape(self):
        sources = _agent(streamed=[_chunk()]).last_sources
        assert [set(s) for s in sources] == [DISPLAY_KEYS]

    def test_blocking_path_returns_the_display_shape(self):
        sources = _agent(last_response=_rag_response([_chunk()])).last_sources
        assert [set(s) for s in sources] == [DISPLAY_KEYS]

    def test_both_paths_agree_on_the_same_chunk(self):
        chunk = _chunk()
        streamed = _agent(streamed=[chunk]).last_sources
        blocking = _agent(last_response=_rag_response([chunk])).last_sources
        assert streamed == blocking

    def test_display_shape_matches_the_shared_helper(self):
        """The helper is the definition; last_sources must not re-derive it."""
        chunk = _chunk()
        assert _agent(streamed=[chunk]).last_sources == chunks_for_display([chunk])

    def test_timestamp_and_text_are_populated(self):
        """The two fields the REPL prints — the ones that used to come out blank."""
        source = _agent(streamed=[_chunk()]).last_sources[0]
        assert source["timestamp"] == "3:11 – 4:14"
        assert source["text"] == EXCERPT

    def test_streamed_chunks_win_over_a_stale_blocking_response(self):
        streamed = _chunk(text="streamed chunk")
        stale    = _chunk(text="blocking chunk")
        agent = _agent(streamed=[streamed], last_response=_rag_response([stale]))
        assert [s["text"] for s in agent.last_sources] == ["streamed chunk"]

    def test_no_sources_at_all_is_empty(self):
        assert _agent().last_sources == []

    def test_rag_response_without_chunks_is_empty(self):
        assert _agent(last_response=_rag_response([])).last_sources == []


# ── The REPL 'sources' command ──────────────────────────────────────────────────

class _FakeAgent:
    """Stands in for YouTubeQAAgent inside _cli; only `sources` is exercised."""

    def __init__(self, sources):
        self._sources = sources

    @property
    def last_sources(self):
        return self._sources


def _run_cli(monkeypatch, inputs, sources):
    replies = iter(inputs)
    monkeypatch.setattr("builtins.input", lambda *_: next(replies))
    monkeypatch.setattr("agent.agent.YouTubeQAAgent", lambda: _FakeAgent(sources))
    _cli()


class TestReplSourcesCommand:

    def test_prints_timestamp_and_excerpt(self, monkeypatch, capsys):
        _run_cli(monkeypatch, ["sources", "quit"], chunks_for_display([_chunk()]))
        out = capsys.readouterr().out
        assert "3:11 – 4:14" in out
        assert EXCERPT[:80] in out

    def test_does_not_print_a_zero_timestamp_or_empty_excerpt(self, monkeypatch, capsys):
        """The exact regression from issue #30: `[Title]  0s–0s  — ...`."""
        _run_cli(monkeypatch, ["sources", "quit"], chunks_for_display([_chunk()]))
        out = capsys.readouterr().out
        assert "0s–0s" not in out      # start/end read off a shape without them
        assert "— ..." not in out       # chunk_text read off a shape without it

    def test_reports_when_there_are_no_sources(self, monkeypatch, capsys):
        _run_cli(monkeypatch, ["sources", "quit"], [])
        assert "No RAG sources" in capsys.readouterr().out
