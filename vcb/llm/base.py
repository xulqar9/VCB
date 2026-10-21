"""LLM interface: message history in, async stream of text deltas out."""

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator


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
    def stream(self, messages: list[dict[str, str]],
               temperature: float | None = None,
               max_tokens: int | None = None) -> AsyncIterator[str]:
        """Yield text deltas. Must abort promptly when the consumer cancels.
        temperature/max_tokens override the config defaults for this call."""

    @abstractmethod
    async def close(self) -> None: ...
