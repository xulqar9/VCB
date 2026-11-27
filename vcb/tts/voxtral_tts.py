"""Voxtral TTS (Mistral AI, mistralai/Voxtral-4B-TTS-2603) served by vLLM-Omni.

Voxtral's supported runtime is a vLLM-Omni server, which exists for Linux only
(use WSL2 on Windows). This adapter starts that server as a child process bound
to 127.0.0.1, waits for /health and calls its OpenAI-style /v1/audio/speech
endpoint, so no text or audio leaves the machine; the server is stopped by
close(). Weights are read from the local Hugging Face cache (~8 GB).

LICENSE: CC BY-NC 4.0, so the weights are NON-COMMERCIAL. The open weights have
preset voices only (no cloning) and no inline sound tags.

VRAM: vLLM reserves GPU memory up front (the deploy config takes about 0.9 of
the card), so a benchmark's peak-VRAM figure is the server's reservation, not
what the model needs. Streaming (default) asks for raw 16-bit little-endian
24 kHz mono PCM, per the vLLM-Omni recipe, which makes time-to-first-audio
real; `stream = false` fetches one WAV instead.

Optional adapter: it is not wired into VCB's engine factory, and vllm, torch
and the rest are only imported by the server process.
"""

from __future__ import annotations

import asyncio
import http.client
import io
import json
import logging
import os
import re
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import AsyncIterator, Iterator

import numpy as np

from ..config import TTSConfig, resolve_path
from .base import TTSEngine

log = logging.getLogger("vcb.tts.voxtral")

# Voxtral has no inline sound tags: canonical <tag>s are dropped (with their
# spacing) instead of being read out loud.
_VOXTRAL_TAGS: dict[str, str] = {}
_ANGLE_TAG = re.compile(r"<\s*([a-zA-Z_]+)\s*>")

# Requests go to the local server only, never through an HTTP proxy.
_LOCAL = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def to_voxtral_text(text: str) -> str:
    out = _ANGLE_TAG.sub(lambda m: _VOXTRAL_TAGS.get(m.group(1).lower()) or "\x00", text)
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
            "download it first: python tools/tts_benchmark.py download --engines voxtral"
        ) from exc


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _sink():
    """The server's log goes to this process's stderr (the benchmark's worker.log)."""
    try:
        return sys.stderr.fileno()
    except (AttributeError, OSError, ValueError):
        return subprocess.DEVNULL


def _read_wav(data: bytes) -> tuple[np.ndarray, int]:
    import soundfile

    audio, rate = soundfile.read(io.BytesIO(data), dtype="float32", always_2d=True)
    return audio.mean(axis=1), int(rate)


def _stop(proc: subprocess.Popen, grace_s: float = 20.0) -> None:
    if proc.poll() is not None:
        return
    proc.terminate()
    try:
        proc.wait(grace_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(10)


class VoxtralTTS(TTSEngine):
    sample_rate = 24_000
    supports_tags = False

    @dataclass
    class Options:
        model: str = "mistralai/Voxtral-4B-TTS-2603"   # Hub id (cached) or a local folder
        voice: str = "casual_female"                    # one of the 20 preset voices
        language: str = "English"
        stream: bool = True                             # raw PCM chunks; False = one WAV
        port: int = 0                                   # 0 = pick a free port
        startup_timeout_s: float = 1200.0
        request_timeout_s: float = 120.0
        trust_remote_code: bool = False                 # the recipes run fine without it
        server_args: list[str] = field(default_factory=list)   # extra `vllm serve` flags

    def __init__(self, cfg: TTSConfig, options: VoxtralTTS.Options | None = None) -> None:
        self.cfg = cfg
        self.opt = options or self.Options()
        self._proc: subprocess.Popen | None = None
        self._base = ""

    async def load(self) -> None:
        if self.cfg.voice_wav:
            raise ValueError("the open Voxtral weights have no voice cloning: run it with its preset voices")
        if sys.platform == "win32":
            raise RuntimeError("vLLM-Omni has no Windows build: run the benchmark inside WSL2 for Voxtral")
        try:
            await asyncio.to_thread(self._start_server)
        except BaseException:
            await self.close()      # never leave a half-started server holding VRAM
            raise
        log.info("voxtral server ready at %s (voice %s, %s)", self._base, self.opt.voice,
                 "streaming PCM" if self.opt.stream else "WAV")

    def _start_server(self) -> None:
        o = self.opt
        folder = _snapshot(o.model)      # fail early, in plain words, if the weights are missing
        port = o.port or _free_port()
        self._base = f"http://127.0.0.1:{port}"
        local = os.path.isdir(resolve_path(o.model))
        cmd = [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", folder if local else o.model,
               "--omni", "--host", "127.0.0.1", "--port", str(port)]
        if local:
            cmd += ["--served-model-name", o.model]
        if o.trust_remote_code:
            cmd.append("--trust-remote-code")
        cmd += [str(arg) for arg in o.server_args]
        env = dict(os.environ)
        for name in ("VLLM_NO_USAGE_STATS", "DO_NOT_TRACK"):     # vLLM's usage telemetry stays off
            env.setdefault(name, "1")
        log.info("starting vLLM-Omni: %s", " ".join(cmd))
        # Same process group as this worker, so the benchmark's kill of the worker's tree takes the server with it.
        self._proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=_sink(), stderr=_sink(), env=env)
        self._wait_ready()

    def _wait_ready(self) -> None:
        deadline = time.monotonic() + self.opt.startup_timeout_s
        while True:
            code = self._proc.poll()
            if code is not None:
                raise RuntimeError(f"the vLLM-Omni server exited with code {code} before it was ready (see worker.log)")
            try:
                with _LOCAL.open(self._base + "/health", timeout=5) as resp:
                    if resp.status == 200:
                        return
            except (OSError, http.client.HTTPException):
                pass     # still starting up
            if time.monotonic() > deadline:
                raise TimeoutError(f"the vLLM-Omni server was not ready after {self.opt.startup_timeout_s:.0f} s")
            time.sleep(2)

    def _chunks(self, text: str) -> Iterator[np.ndarray]:
        o = self.opt
        body = {"model": o.model, "input": text, "voice": o.voice, "language": o.language}
        if o.stream:
            body.update(stream=True, stream_format="audio", response_format="pcm")
        else:
            body["response_format"] = "wav"
        request = urllib.request.Request(
            self._base + "/v1/audio/speech", data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            response = _LOCAL.open(request, timeout=o.request_timeout_s)
        except urllib.error.HTTPError as exc:
            detail = exc.read(300).decode("utf-8", "replace")
            raise RuntimeError(f"the Voxtral server answered HTTP {exc.code}: {detail}") from exc
        with response:
            if not o.stream:
                audio, self.sample_rate = _read_wav(response.read())
                yield audio
                return
            odd = b""        # a network chunk can end in the middle of a 16-bit sample
            while True:
                data = response.read1(1 << 16)
                if not data:
                    break
                data = odd + data
                whole = len(data) - len(data) % 2
                odd = data[whole:]
                if whole:
                    yield np.frombuffer(data[:whole], dtype="<i2").astype(np.float32) / 32768.0

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        assert self._proc is not None, "call load() first"
        text = to_voxtral_text(text)
        if not text:
            raise ValueError("nothing left to say once the unsupported tags are dropped")
        chunks = self._chunks(text)
        done = object()
        try:
            while True:
                chunk = await asyncio.to_thread(next, chunks, done)
                if chunk is done:
                    break
                yield chunk
        finally:
            try:
                chunks.close()
            except ValueError:
                pass     # cancelled while a worker thread is still reading

    async def synthesize(self, text: str) -> np.ndarray:
        parts = [chunk async for chunk in self.synthesize_stream(text)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    async def close(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None:
            await asyncio.to_thread(_stop, proc)
