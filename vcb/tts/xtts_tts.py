"""XTTS v2 voice cloning. Speaker latents are computed once from the
reference wav at load time, so per-chunk synthesis stays fast (~200-400 ms
on a modern GPU). Note: XTTS v2 weights are CPML-licensed (non-commercial).
"""

import asyncio
import logging
import os

import numpy as np

from ..config import TTSConfig, resolve_path
from .base import TTSEngine

log = logging.getLogger("vcb.tts.xtts")


class XTTSv2TTS(TTSEngine):
    sample_rate = 24_000

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._gpt_cond_latent = None
        self._speaker_embedding = None

    async def load(self) -> None:
        os.environ.setdefault("COQUI_TOS_AGREED", "1")

        def _load():
            from TTS.api import TTS

            api = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(self.cfg.device)
            model = api.synthesizer.tts_model
            gpt_cond, spk_emb = model.get_conditioning_latents(
                audio_path=[resolve_path(self.cfg.voice_wav)]
            )
            return model, gpt_cond, spk_emb

        self._model, self._gpt_cond_latent, self._speaker_embedding = (
            await asyncio.to_thread(_load)
        )
        log.info("xtts v2 loaded, voice cloned from %s", self.cfg.voice_wav)

    async def synthesize(self, text: str) -> np.ndarray:
        assert self._model is not None, "call load() first"

        def _run() -> np.ndarray:
            out = self._model.inference(
                text,
                self.cfg.language,
                self._gpt_cond_latent,
                self._speaker_embedding,
                speed=self.cfg.speed,
            )
            return np.asarray(out["wav"], dtype=np.float32)

        return await asyncio.to_thread(_run)

    async def close(self) -> None:
        self._model = None
        self._gpt_cond_latent = None
        self._speaker_embedding = None
