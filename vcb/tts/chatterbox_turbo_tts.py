"""Chatterbox-Turbo (Resemble AI) voice cloning with native paralinguistic
tags. Conditionals are prepared once from the reference wav at load time and
reused for every chunk. Turbo voices [laugh]/[cough]/[chuckle] etc. natively,
so the <tag>s the LLM emits are rewritten to Turbo's square-bracket syntax.
"""

import asyncio
import logging
import re
from pathlib import Path

import numpy as np

from ..config import TTSConfig, resolve_path
from .base import TTSEngine

log = logging.getLogger("vcb.tts.chatterbox_turbo")

# Turbo only ACTS a fixed set of learned tokens (ResembleAI/chatterbox-turbo
# added_tokens.json). The LLM emits angle tags with its own vocabulary, so map
# each to Turbo's exact square-bracket token; anything unmapped is dropped, or
# Turbo would spell the literal word out ("giggle") instead of performing it.
_TURBO_TAGS = {
    "laugh": "[laugh]", "chuckle": "[chuckle]", "giggle": "[chuckle]",
    "sigh": "[sigh]", "gasp": "[gasp]", "groan": "[groan]",
    "cough": "[cough]", "clear_throat": "[clear throat]", "ahem": "[clear throat]",
    "sniff": "[sniff]", "sniffle": "[sniff]",
    "cry": "[crying]", "crying": "[crying]", "sob": "[crying]",
    "shush": "[shush]", "whisper": "[whispering]", "whispering": "[whispering]",
    "angry": "[angry]", "happy": "[happy]", "surprised": "[surprised]",
    "fear": "[fear]", "sarcastic": "[sarcastic]", "dramatic": "[dramatic]",
}
_ANGLE_TAG = re.compile(r"<\s*([a-zA-Z_]+)\s*>")


def _to_turbo_tags(text: str) -> str:
    return _ANGLE_TAG.sub(
        lambda m: _TURBO_TAGS.get(m.group(1).lower().replace(" ", "_"), ""), text
    )


class ChatterboxTurboTTS(TTSEngine):
    sample_rate = 24_000
    supports_tags = True

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._model = None

    async def load(self) -> None:
        voice = resolve_path(self.cfg.voice_wav)
        has_voice = bool(voice) and Path(voice).is_file()
        if not has_voice:
            log.warning("voice file %r not found -- using chatterbox's built-in "
                        "voice (pick your own .wav in the VCB settings)", voice)

        def _load():
            from chatterbox.tts_turbo import ChatterboxTurboTTS as _Turbo

            model = _Turbo.from_pretrained(device=self.cfg.device)
            if has_voice:
                model.prepare_conditionals(voice, exaggeration=self.cfg.exaggeration)
            # The first generation pays one-time costs (lazy watermarker load,
            # CUDA kernel selection, allocator growth) -- several seconds live.
            # Pay them now instead of on the user's first reply.
            model.generate("Hello there, nice to meet you.")
            return model

        self._model = await asyncio.to_thread(_load)
        self.sample_rate = self._model.sr
        log.info(
            "chatterbox-turbo loaded on %s (sr=%d), voice: %s", self.cfg.device,
            self.sample_rate, voice if has_voice else "built-in",
        )

    async def synthesize(self, text: str) -> np.ndarray:
        assert self._model is not None, "call load() first"
        text = _to_turbo_tags(text)

        def _run() -> np.ndarray:
            wav = self._model.generate(text)
            return wav.squeeze(0).detach().cpu().numpy().astype(np.float32)

        return await asyncio.to_thread(_run)

    async def close(self) -> None:
        self._model = None
