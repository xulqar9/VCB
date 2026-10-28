"""OpenAICompatLLM.stream retry semantics.

The key guarantee: a request is only retried while it has produced nothing.
Once a token has been streamed, a transient error must propagate instead of
restarting the request (which would re-speak the reply's start).
"""

import asyncio
from types import SimpleNamespace

import httpx
import openai
import pytest

from vcb.config import LLMConfig
from vcb.llm.openai_compat import OpenAICompatLLM

_REQ = httpx.Request("POST", "http://localhost/v1/chat/completions")


def _chunk(text):
    """A streaming event shaped like an OpenAI chat.completion chunk."""
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text))])


_EMPTY = SimpleNamespace(choices=[])   # keep-alive event with no choice


class _FakeStream:
    def __init__(self, events):
        self._events = events
        self.closed = 0

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for ev in self._events:
            if isinstance(ev, BaseException):
                raise ev
            yield ev

    async def close(self):
        self.closed += 1


def _returns_stream(events):
    return lambda: _FakeStream(events)


def _raises(exc):
    def _f():
        raise exc
    return _f


class _FakeCompletions:
    def __init__(self, script):
        self._script = script
        self.calls = 0
        self.sent = []

    async def create(self, **kwargs):
        self.sent.append(kwargs)
        behavior = self._script[min(self.calls, len(self._script) - 1)]
        self.calls += 1
        return behavior()          # raises (pre-stream) or returns a stream


def _make_llm(script, max_retries=2):
    cfg = LLMConfig()
    cfg.max_retries = max_retries
    cfg.retry_backoff_s = 0.0      # no real sleeping in tests
    llm = OpenAICompatLLM(cfg)
    completions = _FakeCompletions(script)
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return llm, completions


async def _collect(llm):
    out = []
    async for delta in llm.stream([{"role": "user", "content": "hi"}]):
        out.append(delta)
    return out


def test_midstream_error_does_not_retry_or_duplicate():
    err = openai.APIConnectionError(message="dropped", request=_REQ)
    llm, comp = _make_llm([_returns_stream([_chunk("Hello"), _chunk(" world"), err])],
                          max_retries=5)

    out = []

    async def _run():
        with pytest.raises(openai.APIConnectionError):
            async for delta in llm.stream([{"role": "user", "content": "hi"}]):
                out.append(delta)

    asyncio.run(_run())
    assert out == ["Hello", " world"]      # streamed once, not duplicated
    assert comp.calls == 1                  # no retry after tokens were emitted


def test_cancelled_stream_hangs_up():
    """A barge-in cancels the consumer mid-stream; the HTTP response must be
    closed so the server stops generating the abandoned reply (otherwise the
    next request waits behind that ghost -- Ollama runs one slot per model).
    A stream read to the end is closed too."""
    stream = _FakeStream([_chunk("Hello"), _chunk(" there"), _chunk(" friend")])
    llm, _comp = _make_llm([lambda: stream])

    async def _run():
        agen = llm.stream([{"role": "user", "content": "hi"}])
        assert await agen.__anext__() == "Hello"
        await agen.aclose()                 # what task cancellation does to the generator

    asyncio.run(_run())
    assert stream.closed == 1

    stream = _FakeStream([_chunk("Hi")])
    llm, _comp = _make_llm([lambda: stream])
    assert asyncio.run(_collect(llm)) == ["Hi"]
    assert stream.closed == 1


def test_prestream_error_retries_then_succeeds():
    err = openai.APITimeoutError(request=_REQ)
    llm, comp = _make_llm(
        [_raises(err), _returns_stream([_chunk("Hi"), _chunk(" there")])],
        max_retries=2,
    )
    out = asyncio.run(_collect(llm))
    assert out == ["Hi", " there"]
    assert comp.calls == 2                  # first failed pre-stream, second worked


def test_prestream_error_exhausts_retries():
    err = openai.APIConnectionError(message="down", request=_REQ)
    llm, comp = _make_llm([_raises(err)], max_retries=1)

    async def _run():
        with pytest.raises(openai.APIConnectionError):
            await _collect(llm)

    asyncio.run(_run())
    assert comp.calls == 2                  # initial try + 1 retry


def test_empty_and_none_deltas_are_skipped():
    llm, _ = _make_llm([_returns_stream([_chunk(None), _EMPTY, _chunk("ok")])])
    out = asyncio.run(_collect(llm))
    assert out == ["ok"]


def _status_error(cls, status, body):
    resp = httpx.Response(status, request=_REQ, json=body)
    return cls(message=str(body), response=resp, body=body)


def test_no_credit_is_a_config_error_not_retried():
    from vcb.llm.base import LLMConfigError

    err = _status_error(openai.RateLimitError, 429, {"error": {
        "type": "insufficient_quota", "code": "credit_balance_exhausted"}})
    llm, comp = _make_llm([_raises(err)], max_retries=3)

    async def _run():
        with pytest.raises(LLMConfigError) as info:
            await _collect(llm)
        return info.value

    exc = asyncio.run(_run())
    assert comp.calls == 1                  # pointless to retry an empty account
    assert "credit" in str(exc) and "credit" in exc.spoken


def test_bad_key_is_a_config_error():
    from vcb.llm.base import LLMConfigError

    err = _status_error(openai.AuthenticationError, 401, {"error": {"code": "invalid_api_key"}})
    llm, _ = _make_llm([_raises(err)])

    async def _run():
        with pytest.raises(LLMConfigError, match="API key"):
            await _collect(llm)

    asyncio.run(_run())


def test_plain_rate_limit_retries_before_first_token():
    err = _status_error(openai.RateLimitError, 429, {"error": {"code": "rate_limit_exceeded"}})
    llm, comp = _make_llm([_raises(err), _returns_stream([_chunk("ok")])], max_retries=2)
    assert asyncio.run(_collect(llm)) == ["ok"]
    assert comp.calls == 2


class _RecordingCompletions:
    def __init__(self):
        self.kwargs = None

    async def create(self, **kwargs):
        self.kwargs = kwargs
        return _FakeStream([_chunk("ok")])


def _recording_llm(**cfg_overrides):
    cfg = LLMConfig()
    for k, v in cfg_overrides.items():
        setattr(cfg, k, v)
    llm = OpenAICompatLLM(cfg)
    comp = _RecordingCompletions()
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=comp))
    asyncio.run(_collect(llm))
    return comp.kwargs


def test_unset_optional_params_are_not_sent():
    # Gemini's OpenAI endpoint rejects frequency_penalty outright.
    kwargs = _recording_llm()
    assert "frequency_penalty" not in kwargs
    assert "presence_penalty" not in kwargs
    assert "reasoning_effort" not in kwargs


def test_set_optional_params_are_sent():
    kwargs = _recording_llm(frequency_penalty=0.2, reasoning_effort="minimal")
    assert kwargs["frequency_penalty"] == 0.2
    assert kwargs["reasoning_effort"] == "minimal"


def _bad_request(body):
    resp = httpx.Response(400, request=_REQ, json=body)
    return openai.BadRequestError(message=str(body), response=resp, body=body)


def test_unsupported_thinking_level_falls_back_to_an_allowed_one():
    # Groq's gpt-oss rejects Gemini's "minimal".
    err = _bad_request({"error": {"type": "invalid_request_error", "message":
                                  "`reasoning_effort` must be one of `low`, `medium`, or `high`"}})
    llm, comp = _make_llm([_raises(err), _returns_stream([_chunk("ok")])])
    llm.cfg.reasoning_effort = "minimal"
    assert asyncio.run(_collect(llm)) == ["ok"]
    assert [k.get("reasoning_effort") for k in comp.sent] == ["minimal", "low"]
    assert llm.cfg.reasoning_effort == "low"      # sticks for the session


def test_rejected_thinking_level_without_alternatives_is_dropped():
    body = [{"error": {"code": 400, "status": "INVALID_ARGUMENT",
                       "message": "Thinking level MINIMAL is not supported for this model."}}]
    llm, comp = _make_llm([_raises(_bad_request(body)), _returns_stream([_chunk("ok")])])
    llm.cfg.reasoning_effort = "minimal"
    assert asyncio.run(_collect(llm)) == ["ok"]
    assert "reasoning_effort" not in comp.sent[1]
    assert llm.cfg.reasoning_effort is None


def test_other_bad_request_is_a_readable_config_error():
    from vcb.llm.base import LLMConfigError

    body = [{"error": {"code": 400, "status": "INVALID_ARGUMENT",
                       "message": "Unknown name \"foo\": Cannot find field."}}]
    llm, comp = _make_llm([_raises(_bad_request(body))])
    llm.cfg.reasoning_effort = "minimal"

    async def _run():
        with pytest.raises(LLMConfigError) as info:
            await _collect(llm)
        return info.value

    exc = asyncio.run(_run())
    assert comp.calls == 1
    assert "Cannot find field" in str(exc)
    assert "Thinking" in str(exc)
