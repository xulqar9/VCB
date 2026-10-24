"""LLM interface: message history in, async stream of text deltas out."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field


@dataclass
class ToolCallRequest:
    """Native (OpenAI-style) tool calls the model made, yielded once at the end
    of a stream instead of text: [(call_id, tool_name, args, extra)]. `extra`
    is provider data that must be echoed back with the call (e.g. Gemini's
    thought_signature), or None."""
    calls: list[tuple[str, str, dict, dict | None]] = field(default_factory=list)


class LLMConfigError(RuntimeError):
    """The request was refused for a reason the user has to fix (bad API key,
    no credit, unknown model) -- retrying won't help. `spoken` is a short
    explanation safe to say out loud."""

    def __init__(self, message: str, spoken: str) -> None:
        super().__init__(message)
        self.spoken = spoken


class LLMEngine(ABC):
    @abstractmethod
    async def load(self) -> None: ...

    @abstractmethod
    def stream(self, messages: list[dict],
               temperature: float | None = None,
               max_tokens: int | None = None,
               tools: list[dict] | None = None) -> AsyncIterator["str | ToolCallRequest"]:
        """Yield text deltas. Must abort promptly when the consumer cancels.
        temperature/max_tokens override the config defaults for this call.
        With `tools` (OpenAI function specs), a ToolCallRequest may be yielded
        last if the model chose to call tools."""

    @abstractmethod
    async def close(self) -> None: ...
