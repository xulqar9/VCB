"""TTS interface: text in, float32 audio out."""

from abc import ABC, abstractmethod
from typing import AsyncIterator

import numpy as np


class TTSEngine(ABC):
    sample_rate: int
    supports_tags: bool = False          # True if engine voices <laugh>/<sigh> etc.

    @abstractmethod
    async def load(self) -> None: ...

    @abstractmethod
    async def synthesize(self, text: str) -> np.ndarray:
        """Returns float32 mono audio at self.sample_rate."""

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        """Yield float32 audio in order. Default: one shot via synthesize().
        Engines that generate incrementally override this to cut first-audio latency.
        """
        audio = await self.synthesize(text)
        if len(audio):
            yield audio

    @abstractmethod
    async def close(self) -> None: ...

