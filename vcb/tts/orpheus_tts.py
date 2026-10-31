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
import re
import sys
import threading
from pathlib import Path

import numpy as np

from ..config import TTSConfig, resolve_path
from .base import TTSEngine

log = logging.getLogger("vcb.tts.orpheus")

# The only tags Orpheus was trained to perform (canopyai/Orpheus-TTS README).
# Anything else is read out or garbled, so near-synonyms are mapped and the
# rest dropped.
ORPHEUS_TAGS = frozenset(
    {"laugh", "chuckle", "sigh", "cough", "sniffle", "groan", "yawn", "gasp"})
_TAG_ALIASES = {"giggle": "chuckle"}
_ANGLE_TAG = re.compile(r"<\s*([a-zA-Z_]+)\s*>")


def to_orpheus_tags(text: str) -> str:
    def sub(m: "re.Match[str]") -> str:
        tag = m.group(1).lower()
        tag = _TAG_ALIASES.get(tag, tag)
        return f"<{tag}>" if tag in ORPHEUS_TAGS else ""

    return " ".join(_ANGLE_TAG.sub(sub, text).split())


def resolve_model(spec: str) -> str | None:
    """Path of the Orpheus .gguf to load: a local file, or "owner/repo/file.gguf"
    downloaded from HuggingFace (cached). Empty = orpheus-cpp's built-in model."""
    spec = spec.strip()
    if not spec:
        return None
    path = Path(resolve_path(spec))
    if path.is_file():
        return str(path)
    parts = spec.replace("\\", "/").split("/")
    if Path(spec).is_absolute() or len(parts) < 3 or not spec.lower().endswith(".gguf"):
        raise RuntimeError(
            f"Orpheus model {spec!r} not found. Set VCB_ORPHEUS_MODEL to a local .gguf "
            "file or 'owner/repo/file.gguf' on HuggingFace (empty = built-in model)."
        )
    from huggingface_hub import hf_hub_download

    log.info("orpheus: fetching %s (one-time download if not cached)", spec)
    return hf_hub_download(repo_id="/".join(parts[:2]), filename="/".join(parts[2:]))


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


class _StoppableLLM:
    """Wraps orpheus-cpp's llama.cpp model so every generation carries a
    stopping criterion: orpheus-cpp's stream only checks for cancellation
    between audio frames, and llama.cpp would otherwise run to max_tokens
    (seconds to tens of seconds) after a barge-in, blocking the next reply.
    Also adds the repetition penalty orpheus-cpp never passes."""

    def __init__(self, llm, stop: threading.Event, criteria_list,
                 repeat_penalty: float = 1.0) -> None:
        self._llm = llm
        self._criteria = criteria_list([lambda _ids, _logits: stop.is_set()])
        self._repeat_penalty = repeat_penalty

    def __call__(self, *args, **kwargs):
        kwargs.setdefault("stopping_criteria", self._criteria)
        kwargs.setdefault("repeat_penalty", self._repeat_penalty)
        return self._llm(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._llm, name)


class OrpheusTTS(TTSEngine):
    sample_rate = 24_000
    supports_tags = True
    # Orpheus streams within a chunk (first audio ~0.5 s whatever its length),
    # so a bigger first chunk costs little latency, while a lone "Hey," comes
    # out flat and every chunk boundary restarts the delivery.
    chunker_overrides = {"first_chunk_min_chars": 40, "min_chars": 100,
                         "first_chunk_clause_split": False}

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._model = None
        self._stop = threading.Event()       # set = abandon the current generation
        self._weights = ""

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
            # orpheus-cpp always downloads its Q4_K_M build; hand it ours instead.
            import orpheus_cpp.model as orpheus_model

            weights = resolve_model(self.cfg.orpheus_model)
            builtin_repo = OrpheusCpp.lang_to_model["en"]
            orig_download = orpheus_model.hf_hub_download

            def download(repo_id, *args, **kw):
                if weights and repo_id == builtin_repo:
                    return weights
                return orig_download(repo_id, *args, **kw)

            orpheus_model.hf_hub_download = download
            try:
                model = OrpheusCpp(
                    n_gpu_layers=self.cfg.orpheus_n_gpu_layers,
                    verbose=False,
                    lang="en",
                )
            finally:
                llama_cpp.Llama.__init__ = orig_init
                orpheus_model.hf_hub_download = orig_download
            _patch_snac_int64(model)
            model._llm = _StoppableLLM(model._llm, self._stop, llama_cpp.StoppingCriteriaList,
                                       self.cfg.orpheus_repeat_penalty)
            self._weights = Path(weights).name if weights else Path(builtin_repo).name
            # First generation pays llama.cpp's and the SNAC session's one-time
            # setup (seconds): do it now, not on the greeting the user would
            # otherwise sit through in silence -- or talk over.
            for _ in model.stream_tts_sync("Hello there, nice to meet you.",
                                           options={"voice_id": self.cfg.orpheus_voice,
                                                    "max_tokens": 600}):
                pass
            return model

        self._model = await asyncio.to_thread(_load)
        log.info("orpheus loaded on %s (voice=%s, model=%s, repeat_penalty=%.2f)",
                 self.cfg.device, self.cfg.orpheus_voice, self._weights,
                 self.cfg.orpheus_repeat_penalty)

    def _options(self, text: str) -> dict:
        return {
            "voice_id": self.cfg.orpheus_voice,
            "pre_buffer_size": self.cfg.orpheus_pre_buffer_s,
            "temperature": self.cfg.orpheus_temperature,
            "top_p": self.cfg.orpheus_top_p,
            "top_k": self.cfg.orpheus_top_k,
            "min_p": self.cfg.orpheus_min_p,
            # 7 tokens ~ 85 ms of audio and speech runs ~15 chars/s, so ~5.5
            # tokens per character; headroom on top. orpheus-cpp's default of
            # 2048 (~25 s) let a runaway on a short fragment stall for ages.
            "max_tokens": min(2048, 200 + 8 * len(text)),
        }

    async def synthesize(self, text: str) -> np.ndarray:
        assert self._model is not None, "call load() first"
        text = to_orpheus_tags(text)
        if not text:
            return np.zeros(0, dtype=np.float32)
        options = self._options(text)

        def _run() -> np.ndarray:
            self._stop.clear()
            parts = [arr for _sr, arr in self._model.stream_tts_sync(text, options=options)]
            if not parts:
                return np.zeros(0, dtype=np.float32)
            audio = np.concatenate(parts, axis=1).reshape(-1)  # int16 (1, N) -> (N,)
            return audio.astype(np.float32) / 32768.0

        return await asyncio.to_thread(_run)

    async def synthesize_stream(self, text: str):
        """Yield audio as Orpheus produces it, so playback starts after the first
        ~0.1s chunk instead of waiting for the whole phrase. Closing the
        generator early (barge-in) stops llama.cpp within one token.
        """
        assert self._model is not None, "call load() first"
        text = to_orpheus_tags(text)
        if not text:
            return
        options = self._options(text)
        loop = asyncio.get_running_loop()
        queue: asyncio.Queue = asyncio.Queue()
        stop = self._stop
        stop.clear()
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
            stop.set()  # barge-in / aclose: llama.cpp stops at its next token
            with contextlib.suppress(Exception):
                await fut

    async def close(self) -> None:
        self._stop.set()
        self._model = None

