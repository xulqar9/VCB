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
    """ggml-cuda.dll (from llama-cpp-python's CUDA 13 build) needs
    cublas64_13.dll from the nvidia-cublas wheel (nvidia\\cu13\\bin\\x86_64);
    the SNAC vocoder's onnxruntime-gpu needs the CUDA 12 cuBLAS/cuDNN DLLs
    from torch and the nvidia-*-cu12 wheels. Add all of those folders to the
    DLL search path on Windows before importing llama_cpp.
    """
    if sys.platform != "win32":
        return
    import site

    for base in map(Path, site.getsitepackages()):
        candidates = [
            base / "nvidia" / "cu13" / "bin" / "x86_64",
            base / "nvidia" / "cu13" / "bin",
            base / "torch" / "lib",
            base / "nvidia" / "cublas" / "bin",
            base / "nvidia" / "cuda_runtime" / "bin",
            base / "nvidia" / "cuda_nvrtc" / "bin",
            base / "nvidia" / "cudnn" / "bin",
        ]
        for path in candidates:
            if not path.is_dir():
                continue
            try:
                os.add_dll_directory(str(path))
            except OSError:
                pass
            # ggml-cuda.dll resolves its DLLs via the legacy PATH search, so
            # prepend as well (add_dll_directory alone isn't enough here).
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
            import onnxruntime as ort
            from orpheus_cpp import OrpheusCpp

            # The SNAC vocoder's ONNX graph has a few int64 shape ops the CUDA
            # provider lacks; onnxruntime runs those on the CPU and warns about
            # the 9 Memcpy nodes it inserts. Harmless (SNAC is tiny next to the
            # 3B model), so only show real errors.
            ort.set_default_logger_severity(3)
            # orpheus-cpp hardcodes n_ctx=0, which makes llama.cpp allocate the
            # model's full 131072-token context (~14GB KV + ~6GB compute). Force
            # a small context during construction. With several GPUs llama.cpp
            # splits the model over all of them (PCIe traffic every token);
            # keep it on the one VCB placed the voice on when that's known.
            n_ctx = self.cfg.orpheus_n_ctx
            gpu = None
            if self.cfg.device.startswith("cuda:"):
                gpu = int(self.cfg.device.split(":", 1)[1])
            orig_init = llama_cpp.Llama.__init__

            def patched_init(inner, *args, **kw):
                if not kw.get("n_ctx"):
                    kw["n_ctx"] = n_ctx
                if gpu is not None:
                    kw.setdefault("main_gpu", gpu)
                    kw.setdefault("split_mode", llama_cpp.LLAMA_SPLIT_MODE_NONE)
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
        log.info("orpheus loaded on %s (voice=%s)", self.cfg.device, self.cfg.orpheus_voice)

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

