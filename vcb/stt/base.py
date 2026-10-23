"""STT interface. Implement transcribe() to add a new engine."""

from abc import ABC, abstractmethod

import numpy as np


class STTEngine(ABC):
    @abstractmethod
    async def load(self) -> None: ...

    @abstractmethod
    async def transcribe(self, audio: np.ndarray, sample_rate: int,
                         context: str | None = None) -> str:
        """audio: float32 mono. context: recent conversation / vocabulary that
        may help recognition (engines may ignore it). Returns transcript text
        (may be empty)."""

    @abstractmethod
    async def close(self) -> None: ...
