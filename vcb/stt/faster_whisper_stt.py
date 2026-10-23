"""faster-whisper STT. GPU inference runs in a worker thread so the event
loop (and audio callbacks) never block."""

import asyncio
import logging
import os
import sys

import numpy as np

from ..config import STTConfig
from .base import STTEngine

log = logging.getLogger("vcb.stt")


def _register_cuda_dlls() -> None:
    """CTranslate2 on Windows needs cuBLAS/cuDNN DLLs from the nvidia-* wheels."""
    if sys.platform != "win32":
        return
    import site

    for base in site.getsitepackages():
        for pkg in ("cublas", "cudnn", "cuda_nvrtc"):
            path = os.path.join(base, "nvidia", pkg, "bin")
            if os.path.isdir(path):
                os.add_dll_directory(path)
                os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]


class FasterWhisperSTT(STTEngine):
    def __init__(self, cfg: STTConfig) -> None:
        self.cfg = cfg
        self._model = None

    async def load(self) -> None:
        _register_cuda_dlls()
        from faster_whisper import WhisperModel

        def _load():
            model = WhisperModel(
                self.cfg.model,
                device=self.cfg.device,
                device_index=self.cfg.device_index,
                compute_type=self.cfg.compute_type,
            )
            # The first transcription pays one-time GPU setup costs (seconds);
            # do it now on a second of silence, not on the user's first words.
            segments, _info = model.transcribe(
                np.zeros(16_000, dtype=np.float32), language=self.cfg.language,
                beam_size=1,
            )
            list(segments)
            return model

        self._model = await asyncio.to_thread(_load)
        where = (f"cuda:{self.cfg.device_index}" if self.cfg.device == "cuda"
                 else self.cfg.device)
        log.info("faster-whisper '%s' loaded on %s", self.cfg.model, where)

    async def transcribe(self, audio: np.ndarray, sample_rate: int,
                         context: str | None = None) -> str:
        assert self._model is not None, "call load() first"
        if sample_rate != 16_000:
            raise ValueError("faster-whisper expects 16 kHz audio")

        def _run() -> str:
            segments, _info = self._model.transcribe(
                audio,
                language=self.cfg.language,
                beam_size=self.cfg.beam_size,
                vad_filter=False,        # we already segmented with Silero
                condition_on_previous_text=False,
                # Recent conversation + known names/jargon steer spelling
                # ("Plat 3", "boost") without changing what was said.
                initial_prompt=context or None,
            )
            return " ".join(s.text.strip() for s in segments).strip()

        return await asyncio.to_thread(_run)

    async def close(self) -> None:
        self._model = None
