"""Kokoro TTS (82M, 24 kHz). Synthesis runs in a worker thread; per-chunk
latency on GPU is well under 100 ms for short phrases."""

import asyncio
import logging

import numpy as np

from ..config import TTSConfig
from .base import TTSEngine

log = logging.getLogger("vcb.tts")


class KokoroTTS(TTSEngine):
    sample_rate = 24_000

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._pipeline = None

    async def load(self) -> None:
        from kokoro import KPipeline

        # Kokoro voice names start with their language code (af_* = American,
        # bf_* = British); keep the pipeline's phonemizer in step with the voice.
        lang = self.cfg.voice[0] if self.cfg.voice[:1] in ("a", "b") else self.cfg.lang_code

        def _load():
            return KPipeline(lang_code=lang, device=self.cfg.device)

        self._pipeline = await asyncio.to_thread(_load)
        log.info("kokoro loaded (voice=%s)", self.cfg.voice)

    async def synthesize(self, text: str) -> np.ndarray:
        assert self._pipeline is not None, "call load() first"

        def _run() -> np.ndarray:
            parts = []
            for _gs, _ps, audio in self._pipeline(
                text, voice=self.cfg.voice, speed=self.cfg.speed
            ):
                parts.append(audio.detach().cpu().numpy().astype(np.float32))
            if not parts:
                return np.zeros(0, dtype=np.float32)
            return np.concatenate(parts)

        return await asyncio.to_thread(_run)

    async def close(self) -> None:
        self._pipeline = None
