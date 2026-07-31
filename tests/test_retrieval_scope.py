# tests/test_retrieval_scope.py

"""
``MULTI_NAMESPACE`` decides which Pinecone namespaces an answer is retrieved
from, and it is the only switch in the codebase that two callers need to
disagree about on purpose.

Production must query ``corpus`` **and** ``live``: a user who ingests a URL
expects to ask about it in the next message, and ``live`` is where that video
lands. An eval run must query ``corpus`` alone: ``live`` accumulates every URL
ingested through production, so a checkpoint scored against it is measured on a
pool that changed between runs and cannot be compared to the next checkpoint.
``agent/agent.py`` holds the production default; ``eval/run_evals.py`` overrides
it and records the value it used.

Nothing held either end. Deferred finding
``agent/agent.py:MULTI_NAMESPACE:no-test-coverage`` (#119, from the review of
#117): a later edit that flips the default — or any future code that rebinds it
inside the API process — would narrow production retrieval to the curated corpus
with the whole suite still green, and the symptom is not a crash. It is answers
that quietly stop knowing about anything a user ingested.


What is asserted
----------------

Three things, because the switch has three halves that can drift apart:

1. The production default is ``True``.
2. Both call sites forward the *module global*, read at call time — the blocking
   path (``rag_node`` → ``rag_chain.answer``) and the streaming path
   (``stream_chat`` → ``rag_chain.stream_answer``). The streaming one is what
   the SSE endpoint actually serves, so a test covering only ``rag_node`` would
   miss the path production runs.
3. The eval runner still opts *in* to the production scope rather than out of
   it: ``run(multi_namespace=False)`` by default, ``--multi-namespace``
   as ``store_true``, and the assignment that pushes the flag onto the agent
   module. If that flag ever defaults to ``True``, checkpoints silently start
   scoring against a moving pool again.

Point 2 is asserted by driving the global to ``True`` *and* ``False`` and
watching what reaches the retrieval call. Asserting only the default would pass
against a hardcoded literal, which is the bug shape being guarded.


Why the eval half is read rather than imported
----------------------------------------------

``eval/run_evals.py`` cannot be imported from this suite, and the reason is not
style:

- It does ``sys.path.insert(0, agent/)`` and then ``from agent import
  YouTubeQAAgent``, which rebinds ``sys.modules['agent']`` to the bare module.
  ``tests/conftest.py`` pins that name to the *package* for the whole session,
  and the two are mutually exclusive inside one interpreter — importing it here
  would break unrelated test files depending on collection order.
- It imports ``openai`` at module scope, which CI does not install: the backend
  job installs ``requirements.txt`` plus a pinned pytest, never
  ``requirements-dev.txt``.

So the eval-side defaults are read from the source with ``ast``, the same
approach and the same reason as ``tests/test_env_defaults.py`` parsing
``agent/retriever.py`` rather than importing it. The parsers are pinned against
a fixture below, because an assertion that passes when the parser finds nothing
is not checking anything.
"""

import ast
from pathlib import Path

import pytest

# Bare imports, matching how agent/agent.py imports its own siblings — see
# tests/test_last_sources.py for the bare-vs-package split this comes from.
from rag_chain import RAGResponse

import agent.agent as agent_mod
from agent.agent import rag_node, YouTubeQAAgent


_REPO_ROOT   = Path(__file__).resolve().parent.parent
_RUN_EVALS   = _REPO_ROOT / "eval" / "run_evals.py"

QUESTION = "What causes black holes to form?"


# ── Spies ──────────────────────────────────────────────────────────────────────

class _Spy:
    """Records every call and returns a fixed result."""

    def __init__(self, result):
        self.result = result
        self.calls: list[tuple[tuple, dict]] = []

    def __call__(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result

    @property
    def scope(self):
        """The ``multi_namespace`` the single recorded call was made with."""
        assert len(self.calls) == 1, f"expected one retrieval call, got {len(self.calls)}"
        args, kwargs = self.calls[0]
        assert "multi_namespace" in kwargs, (
            "retrieval scope was not passed by keyword; a positional argument "
            f"here is unreadable at the call site. Got args={args!r}"
        )
        return kwargs["multi_namespace"]


def _state(question=QUESTION):
    return {
        "messages":     [],
        "question":     question,
        "intent":       "rag",
        "answer":       "",
        "rag_response": None,
    }


def _agent():
    """
    An agent carrying only the fields ``stream_chat`` touches.

    Built without ``__init__`` on purpose: the real constructor compiles the
    LangGraph and its tools, which reach for Pinecone/Groq credentials, and the
    graph is not what is under test — the retrieval call inside one node is.
    """
    agent = object.__new__(YouTubeQAAgent)
    agent._streamed_chunks = []
    agent._last_provenance = None
    agent.memory = _NoMemory()
    return agent


class _NoMemory:
    """The two memory calls ``stream_chat`` makes, doing nothing."""

    def to_history(self):
        return []

    def save_turn(self, question, answer):
        pass


# ── The production default ─────────────────────────────────────────────────────

class TestProductionDefault:

    def test_default_is_multi_namespace(self):
        assert agent_mod.MULTI_NAMESPACE is True, (
            "agent/agent.py has stopped querying the 'live' namespace by "
            "default. Every video a user ingests goes to 'live', so this makes "
            "the product unable to answer about the URL it just accepted. If "
            "this is deliberate, the eval override in eval/run_evals.py is now "
            "the same value as production and stops meaning anything."
        )


# ── Both retrieval paths forward it, at call time ──────────────────────────────

class TestRetrievalCallsForwardTheScope:
    """
    Both parametrised over True and False: the default alone would pass against
    a call site that hardcodes ``multi_namespace=True`` and ignores the global,
    which is exactly the state an eval run would then silently misreport.
    """

    @pytest.mark.parametrize("scope", [True, False])
    def test_rag_node_forwards_it(self, monkeypatch, scope):
        spy = _Spy(RAGResponse(
            answer="Stars above roughly 20 solar masses collapse.",
            chunks=[], question=QUESTION, namespace="corpus",
        ))
        monkeypatch.setattr(agent_mod, "answer", spy)
        monkeypatch.setattr(agent_mod, "MULTI_NAMESPACE", scope)

        rag_node(_state())

        assert spy.scope is scope

    @pytest.mark.parametrize("scope", [True, False])
    def test_stream_chat_forwards_it(self, monkeypatch, scope):
        spy = _Spy((iter(["Stars ", "collapse."]), []))
        monkeypatch.setattr(agent_mod, "stream_answer", spy)
        monkeypatch.setattr(agent_mod, "MULTI_NAMESPACE", scope)

        # stream_chat is a generator; the retrieval call only happens once it is
        # driven, so the tokens have to be consumed.
        assert "".join(_agent().stream_chat(QUESTION)) == "Stars collapse."
        assert spy.scope is scope

    def test_the_global_is_read_at_call_time(self, monkeypatch):
        # The override pattern eval/run_evals.py uses is a plain rebind of the
        # module attribute, long after import. A call site that captured the
        # value at import — a default argument, a module-level alias — would
        # keep passing the two tests above and ignore the override entirely.
        spy = _Spy(RAGResponse(answer="a", chunks=[], question=QUESTION, namespace="corpus"))
        monkeypatch.setattr(agent_mod, "answer", spy)

        monkeypatch.setattr(agent_mod, "MULTI_NAMESPACE", False)
        rag_node(_state())
        first = spy.calls[0][1]["multi_namespace"]

        monkeypatch.setattr(agent_mod, "MULTI_NAMESPACE", True)
        rag_node(_state())
        second = spy.calls[1][1]["multi_namespace"]

        assert (first, second) == (False, True)


# ── Reading eval/run_evals.py ──────────────────────────────────────────────────

def _parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _function(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"no function named {name!r}")


def _defaults(fn: ast.FunctionDef) -> dict:
    """``{parameter: default}`` for every defaulted parameter, literals only."""
    out = {}
    positional = fn.args.posonlyargs + fn.args.args
    for arg, default in zip(positional[len(positional) - len(fn.args.defaults):],
                            fn.args.defaults):
        if isinstance(default, ast.Constant):
            out[arg.arg] = default.value
    for arg, default in zip(fn.args.kwonlyargs, fn.args.kw_defaults):
        if isinstance(default, ast.Constant):
            out[arg.arg] = default.value
    return out


def _add_argument(tree: ast.Module, flag: str) -> dict:
    """``{keyword: value}`` for the ``add_argument`` call declaring ``flag``."""
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "add_argument"):
            continue
        if any(isinstance(a, ast.Constant) and a.value == flag for a in node.args):
            return {
                kw.arg: kw.value.value
                for kw in node.keywords
                if isinstance(kw.value, ast.Constant)
            }
    raise AssertionError(f"no add_argument call declares {flag!r}")


def _assigns_global_from(fn: ast.FunctionDef, attr: str, param: str) -> bool:
    """Is there an ``<anything>.attr = param`` assignment inside ``fn``?"""
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assign):
            continue
        targets = [t for t in node.targets
                   if isinstance(t, ast.Attribute) and t.attr == attr]
        if targets and isinstance(node.value, ast.Name) and node.value.id == param:
            return True
    return False


class TestEvalRunsCorpusOnlyByDefault:
    """The other end of the invariant: an eval opts *in* to production scope."""

    def test_run_defaults_to_corpus_only(self):
        defaults = _defaults(_function(_parse(_RUN_EVALS), "run"))
        assert defaults["multi_namespace"] is False, (
            "eval/run_evals.py run() now queries 'live' by default. The live "
            "namespace grows with every URL production ingests, so checkpoints "
            "stop being comparable to each other and a regression cannot be "
            "distinguished from a changed retrieval pool."
        )

    def test_the_flag_is_opt_in(self):
        kwargs = _add_argument(_parse(_RUN_EVALS), "--multi-namespace")
        assert kwargs.get("action") == "store_true"

    def test_the_flag_reaches_the_agent_module(self):
        # Without this assignment both tests above would pass while the flag
        # did nothing at all.
        assert _assigns_global_from(
            _function(_parse(_RUN_EVALS), "run"), "MULTI_NAMESPACE", "multi_namespace"
        ), ("run() no longer pushes multi_namespace onto the agent module, so "
            "the flag and the default it documents have no effect on retrieval.")


# ── The parsers' own seams ─────────────────────────────────────────────────────

class TestGuardIsNotVacuous:

    def test_run_evals_is_where_this_thinks_it_is(self):
        assert _RUN_EVALS.is_file(), f"{_RUN_EVALS} is missing"

    def test_defaults_are_read_off_a_known_signature(self, tmp_path):
        source = tmp_path / "m.py"
        source.write_text(
            "def run(a, b=1, *, c=False, d=None, e=compute()):\n    pass\n",
            encoding="utf-8",
        )
        assert _defaults(_function(_parse(source), "run")) == {
            "b": 1, "c": False, "d": None,   # `a` has no default, `e` is computed
        }

    def test_add_argument_is_matched_on_the_flag(self, tmp_path):
        source = tmp_path / "m.py"
        source.write_text(
            "p.add_argument('--other', action='store_false')\n"
            "p.add_argument('-m', '--multi-namespace', action='store_true', help='h')\n",
            encoding="utf-8",
        )
        assert _add_argument(_parse(source), "--multi-namespace") == {
            "action": "store_true", "help": "h",
        }

    def test_a_missing_flag_is_an_error_not_a_pass(self, tmp_path):
        source = tmp_path / "m.py"
        source.write_text("p.add_argument('--other')\n", encoding="utf-8")
        with pytest.raises(AssertionError):
            _add_argument(_parse(source), "--multi-namespace")

    def test_the_assignment_check_distinguishes_the_two_shapes(self, tmp_path):
        wired = tmp_path / "wired.py"
        wired.write_text(
            "def run(multi_namespace=False):\n    m.MULTI_NAMESPACE = multi_namespace\n",
            encoding="utf-8",
        )
        inert = tmp_path / "inert.py"
        inert.write_text(
            "def run(multi_namespace=False):\n    m.MULTI_NAMESPACE = True\n",
            encoding="utf-8",
        )
        assert _assigns_global_from(
            _function(_parse(wired), "run"), "MULTI_NAMESPACE", "multi_namespace")
        assert not _assigns_global_from(
            _function(_parse(inert), "run"), "MULTI_NAMESPACE", "multi_namespace")
