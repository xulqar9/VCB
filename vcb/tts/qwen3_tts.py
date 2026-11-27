"""Qwen3-TTS (Alibaba Qwen) through the official `qwen-tts` package.

Two checkpoints cover the two voice modes: CustomVoice (built-in speakers) for
the default voice and Base (reference-clip cloning) for the clone run. The
package pins transformers==4.57.3, so install it in its own virtualenv, not in
VCB's. Weights (Apache-2.0) come from the local Hugging Face cache; torch and
qwen_tts are imported inside load(), so importing this module costs nothing.

Optional adapter: it is not wired into VCB's engine factory. Qwen3-TTS has no
inline sound tags (its maintainers confirmed it), so canonical <tag>s are
dropped before synthesis instead of being spelled out loud. Its only style
control is the free-text `instruct` prompt, which is off by default so every
engine in a comparison receives identical text.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import TTSConfig, resolve_path
from .base import TTSEngine

log = logging.getLogger("vcb.tts.qwen3_tts")

# Nothing to map: every tag is dropped (with its spacing) rather than voiced.
_QWEN3_TAGS: dict[str, str] = {}
_ANGLE_TAG = re.compile(r"<\s*([a-zA-Z_]+)\s*>")
_DTYPES = ("bfloat16", "float16", "float32")


def to_qwen3_text(text: str) -> str:
    out = _ANGLE_TAG.sub(lambda m: _QWEN3_TAGS.get(m.group(1).lower()) or "\x00", text)
    out = re.sub(r" *\x00 *(?=[,.;:!?])", "", out)
    return re.sub(r" *\x00 *", " ", out).strip()


def _snapshot(repo: str) -> str:
    """Local folder of a model. An existing path is used as is; a Hub id is
    resolved from the Hugging Face cache (downloaded only when the environment
    is not offline, i.e. the benchmark was started with --allow-download)."""
    local = resolve_path(repo)
    if local and os.path.isdir(local):
        return local
    from huggingface_hub import snapshot_download

    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")
    try:
        return snapshot_download(repo, local_files_only=offline)
    except Exception as exc:
        raise FileNotFoundError(
            f"could not get {repo} from the Hugging Face cache ({type(exc).__name__}); "
            "download it first: python tools/tts_benchmark.py download --engines qwen3-tts"
        ) from exc


def _mono_float32(audio) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


class Qwen3TTS(TTSEngine):
    sample_rate = 24_000
    supports_tags = False

    @dataclass
    class Options:
        custom_voice_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"   # default-voice run
        base_model: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"                   # clone run
        speaker: str = "Ryan"              # one of the CustomVoice speakers
        language: str = "English"
        reference_text: str = ""           # exact transcript of the clip (clone run)
        dtype: str = "bfloat16"
        attn_implementation: str = "sdpa"  # flash-attn is not needed (nor on Windows)
        x_vector_only: bool = False        # clone from the speaker embedding alone; no transcript needed
        instruct: str = ""                 # optional style prompt; empty = none

    def __init__(self, cfg: TTSConfig, options: Qwen3TTS.Options | None = None) -> None:
        self.cfg = cfg
        self.opt = options or self.Options()
        if self.opt.dtype not in _DTYPES:
            raise ValueError(f"dtype must be one of {', '.join(_DTYPES)}, not {self.opt.dtype!r}")
        self._clone = bool(cfg.voice_wav)
        self._model = None
        self._prompt = None

    async def load(self) -> None:
        o = self.opt
        ref = ""
        if self._clone:
            ref = resolve_path(self.cfg.voice_wav)
            if not Path(ref).is_file():
                raise FileNotFoundError(f"reference clip not found: {ref}")
            if not o.reference_text.strip() and not o.x_vector_only:
                raise ValueError(
                    "Qwen3-TTS voice cloning needs the exact transcript of the reference clip "
                    "(set reference_text, or x_vector_only = true to clone from the voice alone)"
                )
        device = "cuda:0" if self.cfg.device == "cuda" else self.cfg.device
        repo = o.base_model if self._clone else o.custom_voice_model

        def _load():
            import torch
            from qwen_tts import Qwen3TTSModel

            model = Qwen3TTSModel.from_pretrained(
                _snapshot(repo), device_map=device, dtype=getattr(torch, o.dtype),
                attn_implementation=o.attn_implementation,
            )
            prompt = None
            if self._clone:
                # The clip is analysed once and the result reused for every sentence.
                prompt = model.create_voice_clone_prompt(
                    ref_audio=ref, ref_text=o.reference_text.strip() or None,
                    x_vector_only_mode=o.x_vector_only,
                )
            return model, prompt

        self._model, self._prompt = await asyncio.to_thread(_load)
        log.info("qwen3-tts loaded on %s (%s), voice: %s", device, repo,
                 ref if self._clone else f"speaker {o.speaker}")

    async def synthesize(self, text: str) -> np.ndarray:
        assert self._model is not None, "call load() first"
        text = to_qwen3_text(text)
        if not text:
            raise ValueError("nothing left to say once the unsupported tags are dropped")
        o = self.opt

        def _run():
            if self._clone:
                return self._model.generate_voice_clone(
                    text=text, language=o.language, voice_clone_prompt=self._prompt)
            return self._model.generate_custom_voice(
                text=text, language=o.language, speaker=o.speaker, instruct=o.instruct or None)

        wavs, sr = await asyncio.to_thread(_run)
        self.sample_rate = int(sr)
        return _mono_float32(wavs[0])

    async def close(self) -> None:
        self._model = None
        self._prompt = None
