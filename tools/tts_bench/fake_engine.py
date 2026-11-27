"""Hardware-free stand-in engines. They exist only so the benchmark harness
(timeouts, failures, GPU pinning, WAV/CSV/JSON/HTML output) can be tested without
loading a model. They are registered as the hidden engines "fake" and "fake-native".

Failures are injected through the sentence text:
    [[FAIL]]    raises an exception
    [[HANG]]    never finishes (exercises the per-synthesis timeout)
    [[SILENT]]  returns silence
    [[EMPTY]]   returns no audio

"fake-native" stands in for one of VCB's own engines (a TTSConfig and nothing else, text
prepared by VCB's sanitize step). Its markers are plain lower-case words, because that step
would strip brackets and lower-case shouting:
    zzecho      fails with the text and the voice settings it was given
    zzlookup    looks up a host name while synthesizing, as Coqui's model manager does on every
                load (with TTS_BENCH_FAKE_LOOKUP set it also does so while loading)
"""

from __future__ import annotations

import asyncio
import os
import re
import socket
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vcb.config import TTSConfig
from vcb.tts.base import TTSEngine

_TAGS = {"laugh": "(laugh)", "sigh": "(sigh)"}
_ANGLE_TAG = re.compile(r"<\s*([a-zA-Z_]+)\s*>")


def to_fake_tags(text: str) -> str:
    """Canonical <tag> -> the fake engine's "(tag)"; tags it lacks are dropped."""
    out = _ANGLE_TAG.sub(lambda m: _TAGS.get(m.group(1).lower()) or "\x00", text)
    out = re.sub(r" *\x00 *(?=[,.;:!?])", "", out)     # no gap before punctuation
    return re.sub(r" *\x00 *", " ", out).strip()       # no double space


@dataclass
class FakeOptions:
    latency_s: float = 0.05           # delay before each audio chunk
    sample_rate: int = 22_050
    seconds_per_char: float = 0.05    # audio length grows with the text


class FakeTTS(TTSEngine):
    Options = FakeOptions
    supports_tags = True
    extra_tags = frozenset(_TAGS)

    def __init__(self, cfg: TTSConfig, options: FakeOptions | None = None) -> None:
        self.cfg = cfg
        self.options = options or FakeOptions()
        self.sample_rate = self.options.sample_rate

    async def load(self) -> None:
        voice = self.cfg.voice_wav
        if voice and not Path(voice).is_file():
            raise FileNotFoundError(f"reference clip not found: {voice}")
        await asyncio.sleep(0.05)
        # Lets tests prove which GPU the worker was pinned to.
        self.visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")

    def _audio(self, text: str, fraction: float = 1.0) -> np.ndarray:
        seconds = max(0.2, len(text) * self.options.seconds_per_char) * fraction
        t = np.arange(int(seconds * self.sample_rate)) / self.sample_rate
        return (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)

    async def _check(self, text: str) -> None:
        if "[[FAIL]]" in text:
            raise RuntimeError("injected failure")
        if "[[HANG]]" in text:
            await asyncio.sleep(3600)

    async def synthesize(self, text: str) -> np.ndarray:
        text = to_fake_tags(text)
        await self._check(text)
        await asyncio.sleep(self.options.latency_s)
        if "[[EMPTY]]" in text:
            return np.zeros(0, dtype=np.float32)
        if "[[SILENT]]" in text:
            return np.zeros(self.sample_rate, dtype=np.float32)
        return self._audio(text)

    async def synthesize_stream(self, text: str):
        """Two chunks, so time-to-first-audio is shorter than the total time."""
        text = to_fake_tags(text)
        await self._check(text)
        if "[[EMPTY]]" in text or "[[SILENT]]" in text:
            audio = await self.synthesize(text)
            if len(audio):
                yield audio
            return
        for _ in range(2):
            await asyncio.sleep(self.options.latency_s)
            yield self._audio(text, 0.5)

    async def close(self) -> None:
        return None


class FakeNativeTTS(FakeTTS):
    """A stand-in for one of VCB's own engines: no Options, no tags of its own."""

    Options = None
    supports_tags = False
    extra_tags = frozenset()
    mood_tags = frozenset({"excited"})        # like IndexTTS, it performs a mood tag

    def __init__(self, cfg: TTSConfig) -> None:
        super().__init__(cfg)

    def _phone_home(self, host: str) -> None:
        try:
            socket.getaddrinfo(host, 443)
        except OSError:
            pass                      # a library that cannot reach its server carries on

    async def load(self) -> None:
        await super().load()
        if os.environ.get("TTS_BENCH_FAKE_LOOKUP"):
            self._phone_home("bench-load.invalid")

    async def synthesize_stream(self, text: str):
        if "zzlookup" in text:
            self._phone_home("bench-synth.invalid")
        if "zzecho" in text:
            raise RuntimeError(f"heard {text!r}; voice_wav={self.cfg.voice_wav!r}; "
                               f"csm_ref_audio={self.cfg.csm_ref_audio!r}; "
                               f"csm_ref_text={self.cfg.csm_ref_text!r}")
        async for chunk in super().synthesize_stream(text):
            yield chunk
