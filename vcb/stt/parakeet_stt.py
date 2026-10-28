"""NVIDIA Parakeet-TDT 0.6B v3 through onnx-asr: the default speech
recognition for English and the other 24 European languages it covers.

Why it's the default: on a 5 s utterance it's ~8x faster than Whisper
large-v3-turbo on the same GPU, and about 2x faster *on the CPU* than Whisper
was on the GPU, so it needs no GPU-specific build (AMD and CPU-only PCs run
the identical model). Its transducer decoder also emits nothing for silence
and noise, where Whisper invents "Thank you." -- fewer phantom turns.

onnx-asr is pure NumPy + onnxruntime (no torch, no NeMo). Trade-off: no
prompt biasing, so VCB's conversation context isn't used for spelling hints.
"""

import asyncio
import logging

import numpy as np

from ..config import STTConfig
from .base import STTEngine
from .cuda_dlls import register_cuda_dlls

log = logging.getLogger("vcb.stt")

MODEL = "nemo-parakeet-tdt-0.6b-v3"
_MAX_SECONDS = 30                        # onnx-asr's single-pass limit; VAD caps at 30 s too


class ParakeetSTT(STTEngine):
    def __init__(self, cfg: STTConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._provider = "CPUExecutionProvider"

    async def load(self) -> None:
        register_cuda_dlls()
        import onnx_asr
        import onnxruntime as ort

        # Parakeet's graph has two ops the CUDA provider runs on the CPU;
        # onnxruntime warns about the Memcpy nodes it inserts for them.
        # Microseconds per call -- only show real errors.
        ort.set_default_logger_severity(3)
        providers = ["CPUExecutionProvider"]
        if self.cfg.device == "cuda":
            if "CUDAExecutionProvider" in ort.get_available_providers():
                providers = [("CUDAExecutionProvider", {"device_id": self.cfg.device_index}),
                             "CPUExecutionProvider"]
            else:
                log.info("onnxruntime has no CUDA provider here; parakeet runs on the CPU "
                         "(fast enough: ~0.2 s per utterance)")
        self._provider = providers[0][0] if isinstance(providers[0], tuple) else providers[0]

        def _load():
            model = onnx_asr.load_model(MODEL, providers=providers,
                                        quantization=self.cfg.parakeet_quantization or None)
            # First call pays session/kernel setup; do it on silence now.
            model.recognize(np.zeros(16_000, dtype=np.float32), sample_rate=16_000)
            return model

        self._model = await asyncio.to_thread(_load)
        where = (f"cuda:{self.cfg.device_index}" if self._provider == "CUDAExecutionProvider"
                 else "cpu")
        log.info("parakeet-tdt-0.6b-v3 loaded on %s", where)

    async def transcribe(self, audio: np.ndarray, sample_rate: int,
                         context: str | None = None) -> str:
        assert self._model is not None, "call load() first"
        if sample_rate != 16_000:
            raise ValueError("parakeet expects 16 kHz audio")
        audio = np.ascontiguousarray(audio[: _MAX_SECONDS * 16_000], dtype=np.float32)

        def _run() -> str:
            return (self._model.recognize(audio, sample_rate=16_000) or "").strip()

        return await asyncio.to_thread(_run)

    async def close(self) -> None:
        self._model = None
