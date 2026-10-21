"""STT interface. Implement transcribe() to add a new engine."""

from abc import ABC, abstractmethod

import numpy as np


class STTEngine(ABC):
    @abstractmethod
    async def load(self) -> None: ...

    @abstractmethod
    async def transcribe(self, audio: np.ndarray, sample_rate: int) -> str:
        """audio: float32 mono. Returns transcript text (may be empty)."""

    @abstractmethod
    async def close(self) -> None: ...
