# tests/test_generation_provenance.py

"""
Contract tests for the additive generation-provenance metadata (issue #18).

Two seams are covered:
  - agent._derive_provenance — the (intent, grounded) → provenance mapping.
  - api.service — the blocking response's `generation` object and the
    streaming path's [META] frame ordering, with the agent monkeypatched so
    the tests never touch the network.
"""

from types import SimpleNamespace

import pytest

# Import `api` first: its package __init__ puts agent/ on sys.path and its
# submodules import the agent by the bare `agent` name (→ agent/agent.py as a
# top-level module). Pulling GenerationProvenance from that same `agent` module
# afterwards avoids binding sys.modules['agent'] to the agent/ *package*, which
# would shadow `from agent import YouTubeQAAgent` inside api.service.
from api import service
from api.schemas import ChatResponse
from agent import GenerationProvenance, _derive_provenance


# ── _derive_provenance: the core mapping ────────────────────────────────────────

class TestDeriveProvenance:

    def test_grounded_rag_is_ai_generated_with_model(self):
        prov = _derive_provenance("rag", grounded=True)
        assert prov.ai_generated is True
        assert prov.mode == "generated"
        assert prov.model  # non-empty runtime model id, not hard-coded here

    def test_ungrounded_rag_is_static_no_context(self):
        prov = _derive_provenance("rag", grounded=False)
        assert prov.ai_generated is False
        assert prov.model is None
        assert prov.mode == "no_context"

    def test_metadata_listing_is_not_ai_generated(self):
        prov = _derive_provenance("metadata", grounded=False)
        assert prov.ai_generated is False
        assert prov.model is None
        assert prov.mode == "metadata"

    def test_ingest_status_is_not_ai_generated(self):
        prov = _derive_provenance("ingest", grounded=False)
        assert prov.ai_generated is False
        assert prov.model is None
        assert prov.mode == "ingest"


# ── Helpers / fakes ─────────────────────────────────────────────────────────────

def _prov(ai_generated, model, mode):
    return SimpleNamespace(ai_generated=ai_generated, model=model, mode=mode)


class _FakeBlockingAgent:
    def __init__(self, response):
        self._response = response

    def chat(self, message):
        return self._response


class _FakeStreamAgent:
    """Mirrors the real agent: sets _last_provenance before the first yield."""

    def __init__(self, tokens, provenance, chunks=()):
        self._tokens = list(tokens)
        self._prov = provenance
        self._streamed_chunks = list(chunks)
        self._last_provenance = None

    def stream_chat(self, question):
        self._last_provenance = self._prov
        for tok in self._tokens:
            yield tok


def _parse_sse(chunks):
    """Turn raw SSE output chunks into a list of event payload strings."""
    events = []
    for raw_event in "".join(chunks).split("\n\n"):
        if not raw_event.strip():
            continue
        data = "\n".join(
            line[len("data:"):].lstrip(" ")
            for line in raw_event.split("\n")
            if line.startswith("data:")
        )
        events.append(data)
    return events


# ── Blocking path: ChatResponse.generation ──────────────────────────────────────

class TestBlockingProvenance:

    def test_generated_answer_carries_model_and_flag(self, monkeypatch):
        resp = SimpleNamespace(
            answer="Neural nets learn by adjusting weights.",
            sources=[],
            intent="rag",
            provenance=_prov(True, "openai/gpt-oss-120b", "generated"),
        )
        monkeypatch.setattr(service, "_build_agent", lambda history: _FakeBlockingAgent(resp))

        out: ChatResponse = service.run_chat("How do neural nets learn?", [])
        assert out.generation.ai_generated is True
        assert out.generation.model == "openai/gpt-oss-120b"
        assert out.generation.mode == "generated"

    def test_static_fallback_flagged_not_generated(self, monkeypatch):
        resp = SimpleNamespace(
            answer="I don't have information about that in the available videos.",
            sources=[],
            intent="rag",
            provenance=_prov(False, None, "no_context"),
        )
        monkeypatch.setattr(service, "_build_agent", lambda history: _FakeBlockingAgent(resp))

        out = service.run_chat("What's the capital of France?", [])
        assert out.generation.ai_generated is False
        assert out.generation.model is None
        assert out.generation.mode == "no_context"

    def test_missing_provenance_defaults_to_static(self, monkeypatch):
        resp = SimpleNamespace(answer="x", sources=[], intent="rag", provenance=None)
        monkeypatch.setattr(service, "_build_agent", lambda history: _FakeBlockingAgent(resp))

        out = service.run_chat("hi", [])
        assert out.generation.ai_generated is False
        assert out.generation.model is None


# ── Streaming path: [META] frame ordering ───────────────────────────────────────

class TestStreamingProvenance:

    def test_meta_frame_leads_before_first_token(self, monkeypatch):
        agent = _FakeStreamAgent(
            tokens=["Hello ", "world."],
            provenance=_prov(True, "openai/gpt-oss-120b", "generated"),
        )
        monkeypatch.setattr(service, "_build_agent", lambda history: agent)

        events = _parse_sse(list(service.stream_run_chat("q", [])))

        # First event is the provenance frame, and it precedes any answer token.
        assert events[0].startswith("[META]")
        import json
        meta = json.loads(events[0][len("[META]"):])
        assert meta == {"ai_generated": True, "model": "openai/gpt-oss-120b", "mode": "generated"}

        meta_idx = 0
        first_token_idx = next(
            i for i, e in enumerate(events)
            if not e.startswith("[META]") and not e.startswith("[SOURCES]") and e != "[DONE]"
        )
        assert meta_idx < first_token_idx
        assert events[first_token_idx] == "Hello "
        assert events[-1] == "[DONE]"

    def test_meta_frame_present_for_static_stream(self, monkeypatch):
        agent = _FakeStreamAgent(
            tokens=["I don't have information about that in the available videos."],
            provenance=_prov(False, None, "no_context"),
        )
        monkeypatch.setattr(service, "_build_agent", lambda history: agent)

        events = _parse_sse(list(service.stream_run_chat("q", [])))
        import json
        assert events[0].startswith("[META]")
        meta = json.loads(events[0][len("[META]"):])
        assert meta["ai_generated"] is False
        assert meta["model"] is None
        assert meta["mode"] == "no_context"
        assert events[-1] == "[DONE]"
