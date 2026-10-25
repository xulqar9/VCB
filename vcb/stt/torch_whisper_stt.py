"""Whisper STT on plain PyTorch (transformers), for GPUs that faster-whisper's
CTranslate2 backend can't use -- AMD Radeon through torch's ROCm build. Same
models, run as torch modules: anywhere torch works, this works. Inference runs
in a worker thread so the event loop (and audio callbacks) never block."""

import asyncio
import logging

import numpy as np

from ..config import STTConfig
from .base import STTEngine

log = logging.getLogger("vcb.stt")

# faster-whisper model names -> the original OpenAI checkpoints they were
# converted from (anything else is taken as a Hugging Face repo id or path).
_HF_MODELS = {
    "tiny": "openai/whisper-tiny", "tiny.en": "openai/whisper-tiny.en",
    "base": "openai/whisper-base", "base.en": "openai/whisper-base.en",
    "small": "openai/whisper-small", "small.en": "openai/whisper-small.en",
    "medium": "openai/whisper-medium", "medium.en": "openai/whisper-medium.en",
    "large-v2": "openai/whisper-large-v2", "large-v3": "openai/whisper-large-v3",
    "large-v3-turbo": "openai/whisper-large-v3-turbo", "turbo": "openai/whisper-large-v3-turbo",
}
_DTYPES = {"float16": "float16", "bfloat16": "bfloat16", "float32": "float32",
           "int8": "float32", "int8_float16": "float16"}   # no int8 kernels in torch here
_MAX_SECONDS = 30                        # Whisper's window; longer audio is clipped


class TorchWhisperSTT(STTEngine):
    def __init__(self, cfg: STTConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._processor = None
        self._device = None
        self._dtype = None

    @property
    def model_id(self) -> str:
        return _HF_MODELS.get(self.cfg.model, self.cfg.model)

    async def load(self) -> None:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        if self.cfg.device == "cuda":
            self._device = torch.device("cuda", self.cfg.device_index)
        else:
            self._device = torch.device(self.cfg.device)
        dtype = _DTYPES.get(self.cfg.compute_type, "float32")
        if self._device.type == "cpu" and dtype != "float32":
            dtype = "float32"            # half precision is slow/unsupported on CPU
        self._dtype = getattr(torch, dtype)

        def _load():
            processor = WhisperProcessor.from_pretrained(self.model_id)
            model = WhisperForConditionalGeneration.from_pretrained(
                self.model_id, dtype=self._dtype, low_cpu_mem_usage=True,
            ).to(self._device).eval()
            self._processor, self._model = processor, model
            # First call pays one-time costs (kernel selection, allocator
            # growth): do it on a second of silence, not the user's first words.
            self._transcribe(np.zeros(16_000, dtype=np.float32), None)

        await asyncio.to_thread(_load)
        log.info("whisper (torch) '%s' loaded on %s (%s)", self.model_id,
                 self._device, dtype)

    def _transcribe(self, audio: np.ndarray, context: str | None) -> str:
        import torch

        audio = audio[: _MAX_SECONDS * 16_000]
        features = self._processor(
            audio, sampling_rate=16_000, return_tensors="pt",
        ).input_features.to(self._device, self._dtype)
        kwargs = {
            "language": self.cfg.language, "task": "transcribe",
            "num_beams": self.cfg.beam_size, "do_sample": False,
            "return_timestamps": False,
        }
        if context:
            # Like faster-whisper's initial_prompt: recent conversation and
            # known names steer spelling without changing what was said.
            kwargs["prompt_ids"] = torch.from_numpy(
                self._processor.get_prompt_ids(context)).to(self._device)
        with torch.inference_mode():
            ids = self._model.generate(features, **kwargs)
        text = self._processor.batch_decode(ids, skip_special_tokens=True)[0]
        if context and text.startswith(context):
            text = text[len(context):]   # older transformers echo the prompt
        return text.strip()

    async def transcribe(self, audio: np.ndarray, sample_rate: int,
                         context: str | None = None) -> str:
        assert self._model is not None, "call load() first"
        if sample_rate != 16_000:
            raise ValueError("whisper expects 16 kHz audio")
        return await asyncio.to_thread(self._transcribe, audio, context)

    async def close(self) -> None:
        self._model = self._processor = None
