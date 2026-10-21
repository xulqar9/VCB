"""Sesame CSM-1B TTS via HuggingFace Transformers (Llama backbone + Mimi codec,
24 kHz). A base generation model with no fixed voice: it invents one per run
(seed it with reference audio for consistency). Unlike Orpheus it does not voice
<laugh>/<sigh> tags; it's here to A/B against Orpheus for naturalness.

Runs in-process on cfg.csm_device (default cuda:0, the first GPU visible under
VCB_CUDA_DEVICES). The model is
gated on HuggingFace: accept the terms at https://huggingface.co/sesame/csm-1b
and run `hf auth login` once before first use.
"""

import asyncio
import logging
import os

import numpy as np

from ..config import TTSConfig, resolve_path
from .base import TTSEngine

log = logging.getLogger("vcb.tts.csm")


class SesameCSMTTS(TTSEngine):
    sample_rate = 24_000
    supports_tags = False                # CSM does not voice <laugh>/<sigh> tags

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._processor = None
        self._ref = None                 # (text, 24kHz float32 array) or None

    async def load(self) -> None:
        import torch
        from transformers import AutoProcessor, CsmForConditionalGeneration

        dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }.get(self.cfg.csm_dtype, torch.bfloat16)
        device = self.cfg.csm_device
        model_id = self.cfg.csm_model_id

        def _load():
            try:
                processor = AutoProcessor.from_pretrained(model_id)
                model = CsmForConditionalGeneration.from_pretrained(
                    model_id, dtype=dtype
                ).to(device)
            except OSError as exc:
                raise RuntimeError(
                    f"Could not load CSM model {model_id!r}. It is gated: accept the "
                    "terms at https://huggingface.co/sesame/csm-1b and run `hf auth "
                    "login` (or set HF_TOKEN)."
                ) from exc
            model.eval()
            self._maybe_compile(model)
            return processor, model

        self._processor, self._model = await asyncio.to_thread(_load)
        self._load_reference()
        log.info(
            "csm loaded on %s (speaker=%d, voice=%s)",
            device, self.cfg.csm_speaker_id, "seeded" if self._ref else "random",
        )

    def _maybe_compile(self, model) -> None:
        """Optional torch.compile + static KV cache. Speeds up steady-state
        generation after a slow first-call warmup; needs a working triton.
        """
        if not self.cfg.csm_compile:
            return
        import torch

        try:
            model.generation_config.cache_implementation = "static"
            if hasattr(model, "depth_decoder"):
                model.depth_decoder.generation_config.cache_implementation = "static"
            # reduce-overhead (CUDA graphs) breaks CSM's backbone+depth two-stage
            # loop ("tensor overwritten by a subsequent run"); plain inductor is safe.
            model.forward = torch.compile(model.forward, mode="default")
            log.info("csm: torch.compile enabled (expect a slow first call)")
        except Exception as exc:  # never let compile take down synthesis
            log.warning("csm: torch.compile unavailable, running eager (%s)", exc)

    def _load_reference(self) -> None:
        """Preload the reference clip so every synthesis conditions on the same
        voice. Requires both a readable wav and its transcript.
        """
        ref_audio, ref_text = resolve_path(self.cfg.csm_ref_audio), self.cfg.csm_ref_text
        if not (ref_audio and ref_text):
            return
        if not os.path.isfile(ref_audio):
            log.warning("csm: reference %r not found, voice will vary per call", ref_audio)
            return
        import librosa

        wav, _ = librosa.load(ref_audio, sr=self.sample_rate, mono=True)
        self._ref = (ref_text, wav.astype(np.float32))
        log.info("csm: seeded voice from %s (%.1fs)", ref_audio, len(wav) / self.sample_rate)

    async def synthesize(self, text: str) -> np.ndarray:
        assert self._model is not None, "call load() first"
        import torch

        device = self.cfg.csm_device
        speaker = str(self.cfg.csm_speaker_id)

        def _run() -> np.ndarray:
            if self._ref is not None:
                ref_text, ref_wav = self._ref
                conversation = [
                    {"role": speaker, "content": [
                        {"type": "text", "text": ref_text},
                        {"type": "audio", "path": ref_wav},
                    ]},
                    {"role": speaker, "content": [{"type": "text", "text": text}]},
                ]
                inputs = self._processor.apply_chat_template(
                    conversation, tokenize=True, return_dict=True
                ).to(device)
                # The reference waveform stays float32; cast it to the model dtype
                # so the bf16/fp16 Mimi codec encoder doesn't reject it.
                if "input_values" in inputs:
                    inputs["input_values"] = inputs["input_values"].to(self._model.dtype)
            else:
                inputs = self._processor(
                    f"[{self.cfg.csm_speaker_id}]{text}", add_special_tokens=True
                ).to(device)
            with torch.inference_mode():
                out = self._model.generate(
                    **inputs,
                    output_audio=True,
                    max_new_tokens=self.cfg.csm_max_new_tokens,
                )
            audio = out[0] if isinstance(out, (list, tuple)) else out
            if hasattr(audio, "detach"):
                audio = audio.detach().to(torch.float32).cpu().numpy()
            audio = np.asarray(audio, dtype=np.float32).reshape(-1)
            return audio

        return await asyncio.to_thread(_run)

    async def close(self) -> None:
        self._model = None
        self._processor = None
        self._ref = None
