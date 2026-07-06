"""Cross-family secondary judge (§5): factory routing, OpenAIClient drop-in behavior.

No live API calls — the langchain client is always mocked/injected. What matters is that an
``openai:``-prefixed judge.secondary yields a NON-Claude client with the exact LLMClient
interface, so the judge/pipeline code stays family-blind.
"""

import inspect

import pytest

from arena.judge import PairwiseVerdict
from arena.llm import LLMClient, OpenAIClient, build_llm_client


# ---- factory routing ----------------------------------------------------------------------

def test_factory_routes_openai_prefix_to_openai_client():
    client = build_llm_client("openai:gpt-5.2")
    assert isinstance(client, OpenAIClient)
    assert client.model == "gpt-5.2"  # prefix stripped: the raw OpenAI model id


def test_factory_routes_bare_id_to_anthropic_client():
    client = build_llm_client("claude-sonnet-4-6")
    assert isinstance(client, LLMClient)
    assert client.model == "claude-sonnet-4-6"


def test_factory_strips_explicit_claude_prefix():
    client = build_llm_client("claude:claude-haiku-4-5")
    assert isinstance(client, LLMClient)
    assert client.model == "claude-haiku-4-5"


# ---- OpenAIClient.structured: same pydantic schema instances the judge expects -------------

class _FakeStructuredRunnable:
    """Stands in for ChatOpenAI().with_structured_output(schema)."""

    def __init__(self, result):
        self._result = result
        self.invocations = []

    def invoke(self, messages):
        self.invocations.append(messages)
        return self._result


class _FakeChatOpenAI:
    def __init__(self, result):
        self.runnable = _FakeStructuredRunnable(result)
        self.schemas = []

    def with_structured_output(self, schema):
        self.schemas.append(schema)
        return self.runnable


def test_structured_returns_schema_instance_via_mocked_inner_client():
    fake = _FakeChatOpenAI(PairwiseVerdict(winner="a", rationale="better sourced"))
    llm = OpenAIClient(model="gpt-5.2", client=fake)
    verdict = llm.structured("sys", "user", PairwiseVerdict, max_tokens=512)
    assert isinstance(verdict, PairwiseVerdict)
    assert verdict.winner == "A"  # field_validator normalization applies as with the Claude judge
    assert fake.schemas == [PairwiseVerdict]
    # system/user land as (role, content) messages
    assert fake.runnable.invocations == [[("system", "sys"), ("user", "user")]]


def test_structured_validates_dict_result_into_schema():
    fake = _FakeChatOpenAI({"winner": "b", "rationale": ""})
    llm = OpenAIClient(model="gpt-5.2", client=fake)
    verdict = llm.structured("sys", "user", PairwiseVerdict)
    assert isinstance(verdict, PairwiseVerdict) and verdict.winner == "B"


def test_structured_returns_none_on_unvalidatable_result():
    fake = _FakeChatOpenAI({"not_a_field": 1, "winner": None})
    llm = OpenAIClient(model="gpt-5.2", client=fake)
    assert llm.structured("sys", "user", PairwiseVerdict) is None  # skip, never a bad verdict


# ---- retry policy mirrors LLMClient -------------------------------------------------------

class _Err(Exception):
    def __init__(self, status=None):
        self.status_code = status


class _RaisingRunnable:
    def __init__(self, exc):
        self.calls = 0
        self._exc = exc

    def invoke(self, messages):
        self.calls += 1
        raise self._exc

    def with_structured_output(self, schema):
        return self


def test_non_retryable_gives_up_after_one_call():
    runnable = _RaisingRunnable(_Err(400))
    llm = OpenAIClient(model="gpt-5.2", client=runnable, max_retries=4)
    assert llm.structured("sys", "user", PairwiseVerdict) is None
    assert runnable.calls == 1  # did NOT waste 4 attempts on a 400


def test_retryable_exhausts_budget_then_skips(monkeypatch):
    monkeypatch.setattr("arena.llm.time.sleep", lambda s: None)
    runnable = _RaisingRunnable(_Err(429))
    llm = OpenAIClient(model="gpt-5.2", client=runnable, max_retries=3)
    assert llm.structured("sys", "user", PairwiseVerdict) is None
    assert runnable.calls == 3


# ---- missing key: clear error, not a silent skip -------------------------------------------

def test_missing_openai_key_raises_clear_error(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    llm = OpenAIClient(model="gpt-5.2")  # no injected client -> real lazy init path
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm.structured("sys", "user", PairwiseVerdict)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        llm.complete("sys", "user")


# ---- cost: honest zero, never fabricated ----------------------------------------------------

def test_cost_usd_is_zero_not_fabricated():
    fake = _FakeChatOpenAI(PairwiseVerdict(winner="tie"))
    llm = OpenAIClient(model="gpt-5.2", client=fake)
    llm.structured("sys", "user", PairwiseVerdict)
    assert llm.cost_usd() == 0.0


# ---- entrypoints build the secondary judge through the factory -----------------------------

def test_run_arena_secondary_uses_factory():
    import run_arena
    assert run_arena.build_llm_client is build_llm_client  # entrypoint imports the factory
    src = inspect.getsource(run_arena.main)
    assert "build_llm_client(config.judge_secondary)" in src


def test_tier_b_secondary_uses_factory():
    from arena import tier_b
    src = inspect.getsource(tier_b.main)
    assert "build_llm_client(config.judge_secondary)" in src
