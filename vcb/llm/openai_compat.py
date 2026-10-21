"""OpenAI-compatible streaming client (llama.cpp server, LM Studio, Ollama,
vLLM, ...). Cancellation closes the HTTP stream, which stops server-side
generation. Transient failures retry with backoff, but only until the first
token is streamed -- retrying after that would re-speak the reply's start."""

import asyncio
import logging
from collections.abc import AsyncIterator

import openai

from ..config import LLMConfig
from .base import LLMConfigError, LLMEngine

log = logging.getLogger("vcb.llm")


def _error_code(exc: "openai.APIStatusError") -> str:
    body = exc.body
    if isinstance(body, list) and body:      # Gemini wraps the error in a list
        body = body[0]
    body = body if isinstance(body, dict) else {}
    err = body.get("error", body)
    if isinstance(err, dict):
        return str(err.get("code") or err.get("type") or "")
    return ""


def _config_error(exc: "openai.APIStatusError", cfg: LLMConfig) -> LLMConfigError | None:
    """Map provider errors the user must fix to a clear message; None means
    the error isn't a configuration problem."""
    where = cfg.base_url
    if isinstance(exc, openai.AuthenticationError):
        return LLMConfigError(
            f"the API key was rejected by {where} -- check it in VCB settings",
            "My API key was rejected. Please check it in the VCB settings.")
    code = _error_code(exc)
    if isinstance(exc, openai.RateLimitError) and (
            "quota" in code or "credit" in code or "billing" in code):
        return LLMConfigError(
            f"your account at {where} has no credit left ({code}) -- add credit "
            "with the provider, or switch backend in VCB settings",
            "My API account is out of credit, so I can't answer right now.")
    if isinstance(exc, openai.PermissionDeniedError):
        return LLMConfigError(
            f"access denied by {where} for model {cfg.model!r} ({exc.message})",
            "My API account isn't allowed to use this model.")
    if isinstance(exc, openai.NotFoundError):
        return LLMConfigError(
            f"model {cfg.model!r} was not found at {where} -- check the model "
            "name in VCB settings",
            "I couldn't find the configured model. Please check its name in the settings.")
    if isinstance(exc, openai.BadRequestError):
        hint = (" -- try another 'Thinking' level in VCB settings"
                if cfg.reasoning_effort else "")
        return LLMConfigError(
            f"{where} rejected the request for model {cfg.model!r}: "
            f"{_error_message(exc)}{hint}",
            "My AI provider rejected the request. Please check the model settings.")
    return None


def _error_message(exc: "openai.APIStatusError") -> str:
    body = exc.body
    if isinstance(body, list) and body:      # Gemini wraps the error in a list
        body = body[0]
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict) and err.get("message"):
            return str(err["message"])
    return exc.message


class OpenAICompatLLM(LLMEngine):
    def __init__(self, cfg: LLMConfig) -> None:
        self.cfg = cfg
        self._client: openai.AsyncOpenAI | None = None

    async def load(self) -> None:
        self._client = openai.AsyncOpenAI(
            base_url=self.cfg.base_url,
            api_key=self.cfg.api_key,
            timeout=self.cfg.request_timeout_s,
            max_retries=0,               # we handle retries ourselves
        )
        log.info("LLM client -> %s (model=%s)", self.cfg.base_url, self.cfg.model)

    async def stream(self, messages: list[dict[str, str]],
                     temperature: float | None = None,
                     max_tokens: int | None = None) -> AsyncIterator[str]:
        assert self._client is not None, "call load() first"
        attempt = 0
        while True:
            yielded = False              # once true, a retry would re-speak the reply
            try:
                kwargs = {
                    "model": self.cfg.model,
                    "messages": messages,
                    "max_tokens": max_tokens if max_tokens is not None else self.cfg.max_tokens,
                    "temperature": temperature if temperature is not None else self.cfg.temperature,
                    "top_p": self.cfg.top_p,
                    "stream": True,
                }
                # Only send penalties when set: 0 is the API default anyway, and
                # some OpenAI-compatible APIs (e.g. Google Gemini) reject the fields.
                if self.cfg.frequency_penalty:
                    kwargs["frequency_penalty"] = self.cfg.frequency_penalty
                if self.cfg.presence_penalty:
                    kwargs["presence_penalty"] = self.cfg.presence_penalty
                if self.cfg.reasoning_effort:
                    kwargs["reasoning_effort"] = self.cfg.reasoning_effort
                response = await self._client.chat.completions.create(**kwargs)
                async for event in response:
                    if event.choices and event.choices[0].delta.content:
                        yielded = True
                        yield event.choices[0].delta.content
                return
            except asyncio.CancelledError:
                raise                    # barge-in: propagate, stream closes
            except openai.APIStatusError as exc:
                config_error = _config_error(exc, self.cfg)
                if config_error is not None:
                    raise config_error from None
                # A plain rate limit or server hiccup: retry like a connection
                # error, but never after tokens were already voiced.
                if yielded or not (isinstance(exc, openai.RateLimitError)
                                   or exc.status_code >= 500):
                    raise
                attempt += 1
                if attempt > self.cfg.max_retries:
                    raise
                delay = self.cfg.retry_backoff_s * attempt * 2
                log.warning("LLM busy (HTTP %d), retry %d in %.1fs",
                            exc.status_code, attempt, delay)
                await asyncio.sleep(delay)
            except (openai.APIConnectionError, openai.APITimeoutError) as exc:
                # Only retry a request that produced nothing yet. Restarting after
                # tokens were already streamed would speak the reply's start twice.
                if yielded:
                    raise
                attempt += 1
                if attempt > self.cfg.max_retries:
                    raise
                delay = self.cfg.retry_backoff_s * attempt
                log.warning("LLM request failed (%s), retry %d in %.1fs", exc, attempt, delay)
                await asyncio.sleep(delay)

    async def close(self) -> None:
        if self._client is not None:
            await self._client.close()
            self._client = None
