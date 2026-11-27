"""Breeze TTS 2 (BreezeBlue) through the official `breeze-tts` repository code.

BreezeBlue ships no pip package: `repo_dir` must point at a git clone of
https://github.com/breezeblue-ai/breeze-tts, whose `breeze_infer` and `models`
packages are put on sys.path when the engine loads. The weights
(BreezeBlue/Breeze-TTS-2, ~7.7 GB, codec bundled) are read from the local
Hugging Face cache. torch, transformers and qwen_tts are imported inside
load(), so importing this module costs nothing.

LICENSE: the weights, anything derived from them and the audio they produce are
NON-COMMERCIAL (BreezeBlue Research and Non-Commercial License v1.1); the code
is Apache-2.0. Evaluation and publishing benchmark results are allowed.

Optional adapter: it is not wired into VCB's engine factory. Breeze streams
~160 ms chunks, so the benchmark's time-to-first-audio is real. The default
voice and reference-clip cloning are both supported; cloning needs the clip's
exact transcript. English and Chinese only; one request at a time.
"""

from __future__ import annotations

import asyncio
import logging
import os
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

import numpy as np

from ..config import TTSConfig, resolve_path
from .base import TTSEngine

log = logging.getLogger("vcb.tts.breeze")

# Breeze voices sounds that are written as plain-text words in parentheses. The
# open-weights README documents (laugh) and (sigh); breath and hesitation use the
# spellings of BreezeBlue's audio-tag guide for the hosted model, (inhales) and
# (pause), which the open weights are not confirmed to perform. Anything unmapped
# is dropped, or Breeze would read the word out loud.
_BREEZE_TAGS = {
    "laugh": "(laugh)",
    "sigh": "(sigh)",
    "breath": "(inhales)",
    "hesitation": "(pause)",
}
_ANGLE_TAG = re.compile(r"<\s*([a-zA-Z_]+)\s*>")


def to_breeze_tags(text: str) -> str:
    out = _ANGLE_TAG.sub(lambda m: _BREEZE_TAGS.get(m.group(1).lower()) or "\x00", text)
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
            "download it first: python tools/tts_benchmark.py download --engines breeze"
        ) from exc


def _repo_dir(value: str) -> str:
    if not value.strip():
        raise ValueError(
            "Breeze needs repo_dir: a git clone of https://github.com/breezeblue-ai/breeze-tts "
            "([engines.breeze.options] repo_dir in tools/tts_bench.toml)"
        )
    path = resolve_path(value)
    missing = [p for p in ("breeze_infer/runtime.py", "models/fast_streaming.py")
               if not (Path(path) / p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"repo_dir {path} is not a breeze-tts checkout (missing {', '.join(missing)})")
    return path


def _mono_float32(audio) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


class BreezeTTS(TTSEngine):
    sample_rate = 24_000
    supports_tags = True
    extra_tags = frozenset(_BREEZE_TAGS)

    @dataclass
    class Options:
        repo_dir: str = ""                           # git clone of breezeblue-ai/breeze-tts (required)
        checkpoint: str = "BreezeBlue/Breeze-TTS-2"  # Hub id (cached) or a local folder
        reference_text: str = ""                     # exact transcript of the clip (clone run)
        speaker: str = "S0"
        attn_implementation: str = "eager"           # the official CLI forces eager
        fast_all: bool = False                       # official fast path: Triton/torch.compile, ~14 GiB
        max_new_tokens: int = 1500
        max_seq_len: int = 2048
        repetition_penalty: float = 1.1
        cfg_scale: float = 1.0

    def __init__(self, cfg: TTSConfig, options: BreezeTTS.Options | None = None) -> None:
        self.cfg = cfg
        self.opt = options or self.Options()
        self._clone = bool(cfg.voice_wav)
        self._ref = ""
        self._runtime = None
        self._parts = None            # (tokenizer, model, audio_tokenizer)
        # The generator below is resumed once per chunk; one dedicated thread keeps
        # all GPU work (and any torch thread-local state) on the same thread.
        self._pool: ThreadPoolExecutor | None = None

    async def load(self) -> None:
        o = self.opt
        repo = _repo_dir(o.repo_dir)
        if self._clone:
            self._ref = resolve_path(self.cfg.voice_wav)
            if not Path(self._ref).is_file():
                raise FileNotFoundError(f"reference clip not found: {self._ref}")
            if not o.reference_text.strip():
                raise ValueError("Breeze voice cloning needs the exact transcript of the "
                                 "reference clip (reference_text)")
        if self.cfg.device == "cpu":
            raise RuntimeError("Breeze TTS 2 needs a CUDA GPU")
        device = "cuda:0" if self.cfg.device == "cuda" else self.cfg.device

        def _load():
            ckpt = Path(_snapshot(o.checkpoint))
            if not (ckpt / "audio_tokenizer").is_dir():
                raise FileNotFoundError(f"{ckpt} has no audio_tokenizer/ folder: not a Breeze-TTS-2 checkpoint")
            if repo not in sys.path:
                sys.path.insert(0, repo)
            from breeze_infer.runtime import load_runtime, update_generation_config_for_breeze
            from models.fast_streaming import FastBreezeStreamingRuntime, FastStreamingConfig

            tokenizer, model, audio_tokenizer = load_runtime(
                ckpt, device=device, attn_implementation=o.attn_implementation)
            update_generation_config_for_breeze(model)
            config = FastStreamingConfig(
                max_new_tokens=o.max_new_tokens, max_seq_len=o.max_seq_len, fast_all=o.fast_all,
                fast_text_encoder=False, fast_backbone_prefill=False, fast_backbone_decode=False,
                fast_depth_decoder=False, fast_codec=False, repetition_penalty=o.repetition_penalty,
            )
            runtime = FastBreezeStreamingRuntime(model, audio_tokenizer, config, tokenizer=tokenizer)
            return runtime, (tokenizer, model, audio_tokenizer)

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="breeze")
        self._runtime, self._parts = await asyncio.get_running_loop().run_in_executor(self._pool, _load)
        self.sample_rate = int(self._runtime.sample_rate)
        log.info("breeze loaded on %s (%s), voice: %s", device, o.checkpoint,
                 self._ref if self._clone else "default")

    def _begin(self, request: dict, seed: int):
        """Tokenise the request (clone mode also encodes the reference clip) and
        return the generator that yields audio chunks."""
        import torch
        from breeze_infer.runtime import set_all_seeds
        from breeze_infer.templates import get_template, prepare_inputs, select_template_name

        tokenizer, model, audio_tokenizer = self._parts
        template = get_template(select_template_name(request))
        set_all_seeds(seed)
        with torch.no_grad():
            inputs = prepare_inputs(
                tokenizer, audio_tokenizer, model, [request], template,
                guidance_scale=self.opt.cfg_scale, guidance_scale_ref=None, guidance_scale_ins=None,
            )
        return self._runtime.iter_audio_chunks(inputs, request_id=request["id"], seed=seed)

    @staticmethod
    def _pull(chunks, done):
        import torch

        with torch.no_grad():
            return next(chunks, done)

    async def synthesize_stream(self, text: str) -> AsyncIterator[np.ndarray]:
        assert self._runtime is not None and self._pool is not None, "call load() first"
        text = to_breeze_tags(text)
        if not text:
            raise ValueError("nothing left to say once the unsupported tags are dropped")
        request = {"id": "single-request", "text": text, "speaker": self.opt.speaker}
        if self._clone:
            request["ref_audio_path"] = self._ref
            request["ref_text"] = self.opt.reference_text.strip()
        # Drawn from the seeded `random` module: reproducible, but different per run.
        seed = random.randrange(2**31)
        loop = asyncio.get_running_loop()
        chunks = await loop.run_in_executor(self._pool, self._begin, request, seed)
        done = object()
        try:
            while True:
                chunk = await loop.run_in_executor(self._pool, self._pull, chunks, done)
                if chunk is done:
                    break
                audio = _mono_float32(chunk.audio)
                if len(audio):
                    yield audio
        finally:
            try:
                chunks.close()
            except ValueError:
                pass  # cancelled while the pool thread is still inside the generator

    async def synthesize(self, text: str) -> np.ndarray:
        parts = [chunk async for chunk in self.synthesize_stream(text)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    async def close(self) -> None:
        self._runtime = None
        self._parts = None
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
