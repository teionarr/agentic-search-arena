"""M5 Langfuse tracing (§11) — mocked, no network, no real Langfuse client.

Asserts: (a) disabled by default -> NullTracer, ZERO client calls; (b) enabled + keys ->
one trace/query with the three span types correctly nested; (c) a seeded SENTINEL secret
never appears in any span payload (redact applied); (d) enabled=true but missing keys ->
silently disabled.
"""

from arena.adapters.base import EvidenceDoc
from arena.config import ArenaConfig, Query, load_config
from arena.pipeline import run_arena
from arena.scope import INCLUDED, Scope, ScopeEntry
from arena.tracing import (
    LangfuseTracer,
    NullTracer,
    build_tracer,
)
from _fakes import FakeAdapter, FakeLLM, sync_gather


# ---- A fake Langfuse client/observation that records every call (no network) ----

class FakeObs:
    """Records name/input/output and children so tests can assert nesting + payloads."""

    def __init__(self, name, input=None, output=None):
        self.name = name
        self.input = input
        self.output = output
        self.children = []
        self.ended = False

    def start_observation(self, *, name, input=None, output=None):
        child = FakeObs(name, input=input, output=output)
        self.children.append(child)
        return child

    def update(self, *, output=None):
        self.output = output

    def end(self):
        self.ended = True


class FakeLangfuseClient:
    def __init__(self):
        self.traces = []
        self.flushed = 0
        self.auth_checks = 0

    def start_observation(self, *, name, input=None, output=None):
        t = FakeObs(name, input=input, output=output)
        self.traces.append(t)
        return t

    def auth_check(self):
        self.auth_checks += 1
        return True

    def flush(self):
        self.flushed += 1


_LANGFUSE_ENV = ("LANGFUSE_PUBLIC_KEY", "LANGFUSE_SECRET_KEY", "LANGFUSE_HOST")


def _set_langfuse_keys(monkeypatch):
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-fake")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-fake")
    monkeypatch.setenv("LANGFUSE_HOST", "https://fake.langfuse")


def _clear_langfuse_keys(monkeypatch):
    for k in _LANGFUSE_ENV:
        monkeypatch.delenv(k, raising=False)


def _scope(names):
    return Scope(entries=[ScopeEntry(n, INCLUDED) for n in names])


def _run(tracer, queries=None):
    cfg = ArenaConfig(evidence_budget_tokens=600)
    queries = queries or [Query(query="what is the capital of France?")]
    docs = [EvidenceDoc(url="http://a", title="A", content="Paris is the capital of France.")]
    adapters = [FakeAdapter("tavily", docs), FakeAdapter("brave", docs)]
    scope = _scope([a.name for a in adapters])
    return run_arena(cfg, queries, adapters, scope, FakeLLM(), FakeLLM(),
                     search_gatherer=sync_gather, tracer=tracer)


# ---- (a) disabled by default -> NullTracer, ZERO client calls ----

def test_config_langfuse_disabled_by_default():
    assert load_config(None).langfuse_enabled is False


def test_config_langfuse_enabled_from_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("langfuse:\n  enabled: true\n")
    assert load_config(str(p)).langfuse_enabled is True


def test_build_tracer_disabled_is_nulltracer(monkeypatch):
    _set_langfuse_keys(monkeypatch)  # keys present but config disabled -> still NullTracer
    tracer = build_tracer(enabled=False)
    assert isinstance(tracer, NullTracer)
    assert tracer.enabled is False


def test_disabled_run_makes_zero_client_calls():
    client = FakeLangfuseClient()
    # build_tracer(enabled=False, ...) is what the pipeline holds when disabled; the injected
    # client must be untouched, proving a disabled tracer never reaches an available client.
    tracer = build_tracer(enabled=False, client=client)
    assert isinstance(tracer, NullTracer)
    _run(tracer)
    assert client.traces == [] and client.flushed == 0


# ---- (d) enabled=true but missing keys -> silently disabled ----

def test_enabled_but_missing_keys_silently_disabled(monkeypatch):
    _clear_langfuse_keys(monkeypatch)
    tracer = build_tracer(enabled=True)  # no client injected, no keys -> NullTracer, no error
    assert isinstance(tracer, NullTracer)


def test_enabled_with_keys_builds_langfuse_tracer(monkeypatch):
    _set_langfuse_keys(monkeypatch)
    client = FakeLangfuseClient()
    tracer = build_tracer(enabled=True, client=client)
    assert isinstance(tracer, LangfuseTracer) and tracer.enabled is True


# ---- (b) enabled + keys -> one trace/query with three span types correctly nested ----

def test_one_trace_per_query_with_nested_span_types(monkeypatch):
    _set_langfuse_keys(monkeypatch)
    client = FakeLangfuseClient()
    tracer = build_tracer(enabled=True, client=client)
    queries = [Query(query="capital of France?"), Query(query="capital of Spain?")]
    _run(tracer, queries=queries)

    assert len(client.traces) == 2  # one trace per query
    assert client.flushed == 1
    for trace in client.traces:
        names = [c.name for c in trace.children]
        # provider.search per provider (2), one reader.synthesize per provider (2), one judge.compare
        assert sum(n.startswith("provider.search") for n in names) == 2
        assert names.count("reader.synthesize") == 2
        assert names.count("judge.compare") == 1
        assert all(c.ended for c in trace.children)  # every child span closed


def test_span_payloads_carry_expected_shape(monkeypatch):
    _set_langfuse_keys(monkeypatch)
    client = FakeLangfuseClient()
    tracer = build_tracer(enabled=True, client=client)
    _run(tracer)
    trace = client.traces[0]
    reader = next(c for c in trace.children if c.name == "reader.synthesize")
    judge = next(c for c in trace.children if c.name == "judge.compare")
    assert "prompt" in reader.input and "answer" in reader.output
    assert "outcome" in judge.output and "rationales" in judge.output


# ---- (c) seeded SENTINEL secret NEVER appears in any span payload (redact applied) ----

def test_secret_sentinel_never_in_span_payload(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "SENTINEL_SECRET_123")
    _set_langfuse_keys(monkeypatch)
    client = FakeLangfuseClient()
    tracer = build_tracer(enabled=True, client=client)

    cfg = ArenaConfig(evidence_budget_tokens=600)
    # Evidence content embeds the sentinel; it must be scrubbed before reaching any span.
    docs = [EvidenceDoc(url="u", title="t", content="leaked SENTINEL_SECRET_123 in evidence")]
    adapters = [FakeAdapter("tavily", docs), FakeAdapter("brave", docs)]
    scope = _scope([a.name for a in adapters])
    run_arena(cfg, [Query(query="q?")], adapters, scope, FakeLLM(), FakeLLM(),
              search_gatherer=sync_gather, tracer=tracer)

    def _walk(obs):
        blob = repr(obs.input) + repr(obs.output)
        assert "SENTINEL_SECRET_123" not in blob
        for c in obs.children:
            _walk(c)

    assert client.traces
    for t in client.traces:
        _walk(t)


def test_langfuse_key_never_in_span_payload(monkeypatch):
    # The Langfuse keys themselves must also be scrubbed from any exported payload.
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-fake")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "SENTINEL_LF_SECRET")
    monkeypatch.setenv("LANGFUSE_HOST", "https://fake.langfuse")
    client = FakeLangfuseClient()
    tracer = build_tracer(enabled=True, client=client)
    docs = [EvidenceDoc(url="u", title="t", content="mentions SENTINEL_LF_SECRET here")]
    adapters = [FakeAdapter("tavily", docs), FakeAdapter("brave", docs)]
    scope = _scope([a.name for a in adapters])
    run_arena(ArenaConfig(evidence_budget_tokens=600), [Query(query="q?")], adapters, scope,
              FakeLLM(), FakeLLM(), search_gatherer=sync_gather, tracer=tracer)

    def _walk(obs):
        assert "SENTINEL_LF_SECRET" not in (repr(obs.input) + repr(obs.output))
        for c in obs.children:
            _walk(c)

    for t in client.traces:
        _walk(t)
