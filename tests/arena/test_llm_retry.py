"""Retry classification: transient errors retry, 4xx (credits/auth/bad-request) don't."""

from arena.llm import LLMClient, _is_retryable


class _Err(Exception):
    def __init__(self, status=None):
        self.status_code = status


def test_is_retryable():
    assert _is_retryable(_Err(429)) is True          # rate limit
    assert _is_retryable(_Err(500)) is True           # server error
    assert _is_retryable(_Err(400)) is False          # bad request / insufficient credits
    assert _is_retryable(_Err(401)) is False          # auth
    assert _is_retryable(Exception("timeout")) is True  # no status_code -> transient


class _FakeClient:
    def __init__(self, exc):
        self.calls = 0
        self._exc = exc
        self.messages = self

    def create(self, **kwargs):
        self.calls += 1
        raise self._exc


def test_non_retryable_gives_up_after_one_call():
    client = _FakeClient(_Err(400))  # e.g. insufficient credits
    llm = LLMClient(model="m", client=client, max_retries=4)
    assert llm.complete("sys", "user") is None
    assert client.calls == 1          # did NOT waste 4 attempts on a 400
