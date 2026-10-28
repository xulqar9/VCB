"""Kokoro TTS (82M, 24 kHz). Synthesis runs in a worker thread; per-chunk
latency on GPU is well under 100 ms for short phrases."""

import asyncio
import logging
import warnings

import numpy as np

from ..config import TTSConfig
from .base import TTSEngine

log = logging.getLogger("vcb.tts")

REPO_ID = "hexgrad/Kokoro-82M"


class KokoroTTS(TTSEngine):
    sample_rate = 24_000

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._pipeline = None

    async def load(self) -> None:
        # Kokoro's model code trips two torch notices (dropout on a 1-layer
        # LSTM, the old weight_norm API); neither affects output.
        warnings.filterwarnings("ignore", message="dropout option adds dropout")
        warnings.filterwarnings("ignore", message=".*weight_norm.* is deprecated")
        from kokoro import KPipeline

        # Kokoro voice names start with their language code (af_* = American,
        # bf_* = British); keep the pipeline's phonemizer in step with the voice.
        lang = self.cfg.voice[0] if self.cfg.voice[:1] in ("a", "b") else self.cfg.lang_code

        def _load():
            pipeline = KPipeline(lang_code=lang, repo_id=REPO_ID, device=self.cfg.device)
            # The first synthesis fetches the voice pack, starts the phonemizer
            # (spaCy, espeak) and picks CUDA kernels: ~5 s live. Pay it now, not
            # on the user's first reply (or the greeting, which they'd talk over).
            for _ in pipeline("Hello there.", voice=self.cfg.voice, speed=self.cfg.speed):
                pass
            return pipeline

        self._pipeline = await asyncio.to_thread(_load)
        log.info("kokoro loaded on %s (voice=%s)", self.cfg.device, self.cfg.voice)

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
