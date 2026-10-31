"""Adapt the config to the machine VCB is running on, so a PC without an
NVIDIA GPU (an AMD Radeon, or someone using the cloud backend on a laptop)
still works."""

import logging
import os

from . import gpu
from .config import AppConfig, pick_stt_engine

log = logging.getLogger("vcb.hardware")

TTS_ENGINES = ("chatterbox-turbo", "kokoro", "indextts", "xtts", "orpheus", "csm")
_GPU_ONLY_TTS = {"chatterbox-turbo", "indextts", "xtts", "orpheus", "csm"}
# IndexTTS runs in its own CUDA-pinned environment, Orpheus inside a CUDA
# llama-cpp build: neither has an AMD (ROCm) path.
_NVIDIA_ONLY_TTS = {"indextts", "orpheus"}

BACKENDS = ("cuda", "rocm", "cpu")


def gpu_backend() -> str:
    """"cuda" (NVIDIA), "rocm" (AMD Radeon through torch's HIP build, which
    reuses the torch.cuda API and device names) or "cpu"."""
    try:
        import torch
    except ImportError:
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    return "rocm" if getattr(torch.version, "hip", None) else "cuda"


def bypass_miopen_rnn() -> None:
    """Run LSTM/GRU/RNN layers on torch's native GPU kernels instead of MIOpen.
    MIOpen's RNN path sets up a dropout descriptor whose HIP kernel is compiled
    at runtime and includes rocrand headers that AMD's Windows torch wheels
    don't ship ("'rocrand/rocrand_xorwow.h' file not found" ->
    miopenStatusUnknownError on Kokoro's first synthesis). Convolutions keep
    using MIOpen. Idempotent."""
    import functools

    import torch

    for cls in (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN):
        forward = cls.forward
        if getattr(forward, "_vcb_no_miopen", False):
            continue

        @functools.wraps(forward)
        def _forward(self, *args, _orig=forward, **kwargs):
            with torch.backends.cudnn.flags(enabled=False):
                return _orig(self, *args, **kwargs)

        _forward._vcb_no_miopen = True
        cls.forward = _forward


def _stt_to_cpu(cfg: AppConfig) -> None:
    cfg.stt.device = "cpu"
    if cfg.stt.engine != "parakeet":
        # Whisper needs a smaller model to keep up on a CPU; parakeet doesn't.
        cfg.stt.model = cfg.stt.cpu_model
        cfg.stt.compute_type = cfg.stt.cpu_compute_type


def _resolve_stt(cfg: AppConfig, backend: str) -> None:
    """Turn "auto" into a concrete speech recogniser and put it on a device
    this machine has. Parakeet (onnxruntime) uses the NVIDIA GPU or, just as
    happily, the CPU; the Whisper builds depend on the GPU vendor."""
    if cfg.stt.engine == "auto":
        cfg.stt.engine = pick_stt_engine(cfg.stt.language)
    if cfg.stt.engine == "parakeet":
        if backend != "cuda":
            cfg.stt.device = "cpu"
        return
    if backend == "cuda":
        return
    if backend == "rocm":
        # faster-whisper's CTranslate2 has no AMD build; Whisper as a plain
        # torch model runs on the GPU like the voice does.
        if cfg.stt.engine == "faster-whisper":
            cfg.stt.engine = "whisper-torch"
            return
        _stt_to_cpu(cfg)
        log.warning("%r speech recognition can't use an AMD GPU, so it runs on the CPU",
                    cfg.stt.engine)
        # AMD's torch links LLVM's OpenMP (libomp140) while CTranslate2 ships
        # Intel's (libiomp5md), which aborts the process when it finds another
        # runtime loaded ("OMP: Error #15") unless told to coexist. They never
        # share a parallel region, so this is safe.
        os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        return
    _stt_to_cpu(cfg)


def adapt_to_hardware(cfg: AppConfig, tts_engine: str, backend: str) -> str:
    """Pick settings that work on this machine's GPU (or none). Returns the
    TTS engine to actually use."""
    if tts_engine not in TTS_ENGINES:
        raise ValueError(
            f"unknown TTS engine {tts_engine!r} (use one of: {', '.join(TTS_ENGINES)})"
        )
    if backend not in BACKENDS:
        raise ValueError(f"unknown GPU backend {backend!r} (use one of: {', '.join(BACKENDS)})")
    _resolve_stt(cfg, backend)
    if backend == "cuda":
        return tts_engine
    if backend == "rocm":
        log.info("AMD GPU detected (ROCm): the voice runs on it; speech recognition: %s on %s",
                 cfg.stt.engine, cfg.stt.device)
        if tts_engine in _NVIDIA_ONLY_TTS:
            log.warning("TTS engine %r needs an NVIDIA GPU; using chatterbox-turbo "
                        "instead (it still clones your voice sample)", tts_engine)
            return "chatterbox-turbo"
        return tts_engine
    log.warning("no GPU usable by VCB detected -- running speech recognition and voice "
                "on the CPU (slower). An NVIDIA GPU (or a ROCm-capable AMD Radeon) "
                "is recommended.")
    cfg.tts.device = "cpu"
    if tts_engine in _GPU_ONLY_TTS:
        log.warning("TTS engine %r needs a GPU for real-time speech; using the kokoro "
                    "preset voice instead -- your voice sample is NOT used. To hear it, "
                    "VCB needs an NVIDIA GPU or a supported AMD Radeon.", tts_engine)
        return "kokoro"
    return tts_engine


# Engines whose GPU we can choose. Orpheus runs inside llama-cpp, not torch,
# but takes the CUDA index via main_gpu (same numbering as torch).
_PLACEABLE_TTS = {"chatterbox-turbo", "kokoro", "indextts", "xtts", "csm", "orpheus"}


def place_on_gpus(cfg: AppConfig, tts_engine: str, backend: str) -> None:
    """Spread speech recognition and the voice over the GPUs with the most
    free memory (measured after the LLM loaded), like Ollama does for the LLM.
    Skipped if the user pinned devices explicitly (STT/TTS device settings)
    or there's only one usable GPU. NVIDIA only (uses nvidia-smi)."""
    if backend != "cuda" or cfg.stt.device != "cuda" or cfg.tts.device != "cuda":
        return
    free = gpu.query_free_gb()
    usable = gpu.visible_gpus(os.environ.get("CUDA_VISIBLE_DEVICES"), list(free))
    if not usable or len(usable) < 2:
        return
    needs = {"stt": gpu.STT_VRAM_GB}
    if tts_engine in _PLACEABLE_TTS:
        needs["tts"] = gpu.TTS_VRAM_GB.get(tts_engine, gpu.TTS_VRAM_DEFAULT_GB)
    plan = gpu.plan_placement({i: free[i] for i in usable}, needs)
    logical = {phys: n for n, phys in enumerate(usable)}   # CUDA's index for it
    cfg.stt.device_index = logical[plan["stt"]]
    if "tts" in plan:
        cfg.tts.device = f"cuda:{logical[plan['tts']]}"
        cfg.tts.csm_device = cfg.tts.device
    log.info("GPU placement (free: %s): speech recognition -> GPU %d, voice -> %s",
             ", ".join(f"GPU {i} {free[i]:.1f} GB" for i in usable),
             plan["stt"], f"GPU {plan['tts']}" if "tts" in plan else "engine default")
