"""Orpheus TTS (3B, 24 kHz) via orpheus-cpp: runs a local llama.cpp model that
emits SNAC audio tokens, decoded to waveform. Produces natural laughs, sighs,
and breaths from inline tags like <laugh>, <sigh>, <gasp>.

Uses preset voices (no wav cloning). Runs its own model in-process, separate
from the chat LLM server.
"""

import asyncio
import contextlib
import logging
import os
import sys
import threading
from pathlib import Path

import numpy as np

from ..config import TTSConfig
from .base import TTSEngine

log = logging.getLogger("vcb.tts.orpheus")


def _register_cuda_dlls() -> None:
    """ggml-cuda.dll (from llama-cpp-python) needs cudart64_12.dll / cublas64_12.dll.
    torch's cu124 build bundles them, plus the nvidia-* wheels. Add those folders
    to the DLL search path on Windows before importing llama_cpp.
    """
    if sys.platform != "win32":
        return
    site = Path(__file__).resolve().parents[2] / ".venv" / "Lib" / "site-packages"
    candidates = [
        site / "torch" / "lib",
        site / "nvidia" / "cublas" / "bin",
        site / "nvidia" / "cuda_runtime" / "bin",
        site / "nvidia" / "cuda_nvrtc" / "bin",
        site / "nvidia" / "cudnn" / "bin",
    ]
    for path in candidates:
        if path.is_dir():
            try:
                os.add_dll_directory(str(path))
            except OSError:
                pass
            # ggml-cuda.dll resolves cudart64_12.dll via the legacy PATH search,
            # so prepend as well (add_dll_directory alone isn't enough here).
            os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")


def _patch_snac_int64(model) -> None:
    """orpheus-cpp feeds the SNAC ONNX vocoder int32 tokens, but the model
    expects int64 -> InvalidArgument crash. Wrap the session to upcast.
    """
    sess = getattr(model, "_snac_session", None)
    if sess is None:
        return
    want_int64 = {i.name for i in sess.get_inputs() if "int64" in i.type}
    if not want_int64:
        return
    orig_run = sess.run

    def run(output_names, input_feed, run_options=None):
        fixed = {
            k: (v.astype(np.int64) if k in want_int64 and getattr(v, "dtype", None) == np.int32 else v)
            for k, v in input_feed.items()
        }
        return orig_run(output_names, fixed, run_options)

    sess.run = run


class OrpheusTTS(TTSEngine):
    sample_rate = 24_000
    supports_tags = True

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._model = None

    async def load(self) -> None:
        def _load():
            _register_cuda_dlls()
            import llama_cpp
            from orpheus_cpp import OrpheusCpp

            # orpheus-cpp hardcodes n_ctx=0, which makes llama.cpp allocate the
            # model's full 131072-token context (~14GB KV + ~6GB compute). Force
            # a small context during construction.
            n_ctx = self.cfg.orpheus_n_ctx
            orig_init = llama_cpp.Llama.__init__

            def patched_init(inner, *args, **kw):
                if not kw.get("n_ctx"):
                    kw["n_ctx"] = n_ctx
                return orig_init(inner, *args, **kw)

            llama_cpp.Llama.__init__ = patched_init
            try:
                model = OrpheusCpp(
                    n_gpu_layers=self.cfg.orpheus_n_gpu_layers,
                    verbose=False,
                    lang="en",
                )
            finally:
                llama_cpp.Llama.__init__ = orig_init
            _patch_snac_int64(model)
            return model

        self._model = await asyncio.to_thread(_load)
        log.info("orpheus loaded (voice=%s)", self.cfg.orpheus_voice)

    async def synthesize(self, text: str) -> np.ndarray:
        assert self._model is not None, "call load() first"
        options = {
            "voice_id": self.cfg.orpheus_voice,
            "pre_buffer_size": self.cfg.orpheus_pre_buffer_s,
            "temperature": self.cfg.orpheus_temperature,
            "top_p": self.cfg.orpheus_top_p,
            "top_k": self.cfg.orpheus_top_k,
            "min_p": self.cfg.orpheus_min_p,
        }

        def _run() -> np.ndarray:
            parts = [arr for _sr, arr in self._model.stream_tts_sync(text, options=options)]
            if not parts:
                return np.zeros(0, dtype=np.float32)
            audio = np.concatenate(parts, axis=1).reshape(-1)  # int16 (1, N) -> (N,)
            return audio.astype(np.float32) / 32768.0

        return await asyncio.to_thread(_run)

    async def synthesize_stream(self, text: str):
        """Yield audio as Orpheus produces it, so playback starts after the first
        ~0.1s chunk instead of waiting for the whole phrase.
        """
        assert self._model is not None, "call load() first"
        options = {
            "voice_id": self.cfg.orpheus_voice,
            "pre_buffer_size": self.cfg.orpheus_pre_buffer_s,
            "temperature": self.cfg.orpheus_temperature,
            "top_p": self.cfg.orpheus_top_p,
            "top_k": self.cfg.orpheus_top_k,
            "min_p": self.cfg.orpheus_min_p,
        }
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = threading.Event()
        DONE = object()

        def _producer() -> None:
            try:
                for _sr, arr in self._model.stream_tts_sync(text, options=options):
                    if stop.is_set():
                        break
                    audio = arr.reshape(-1).astype(np.float32) / 32768.0
                    loop.call_soon_threadsafe(queue.put_nowait, audio)
            except Exception as exc:  # surface to the consumer, don't hang
                loop.call_soon_threadsafe(queue.put_nowait, exc)
            finally:
                loop.call_soon_threadsafe(queue.put_nowait, DONE)

        fut = loop.run_in_executor(None, _producer)
        try:
            while True:
                item = await queue.get()
                if item is DONE:
                    break
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            stop.set()  # tell the worker to bail on barge-in / aclose
            with contextlib.suppress(Exception):
                await fut

    async def close(self) -> None:
        self._model = None

