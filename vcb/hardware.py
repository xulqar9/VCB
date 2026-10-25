"""Adapt the config to the machine VCB is running on, so a PC without an
NVIDIA GPU (an AMD Radeon, or someone using the cloud backend on a laptop)
still works."""

import logging
import os

from . import gpu
from .config import AppConfig

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


def _stt_to_cpu(cfg: AppConfig) -> None:
    cfg.stt.device = "cpu"
    cfg.stt.model = cfg.stt.cpu_model
    cfg.stt.compute_type = cfg.stt.cpu_compute_type


def adapt_to_hardware(cfg: AppConfig, tts_engine: str, backend: str) -> str:
    """Pick settings that work on this machine's GPU (or none). Returns the
    TTS engine to actually use."""
    if tts_engine not in TTS_ENGINES:
        raise ValueError(
            f"unknown TTS engine {tts_engine!r} (use one of: {', '.join(TTS_ENGINES)})"
        )
    if backend not in BACKENDS:
        raise ValueError(f"unknown GPU backend {backend!r} (use one of: {', '.join(BACKENDS)})")
    if backend == "cuda":
        return tts_engine
    if backend == "rocm":
        # faster-whisper (CTranslate2) has no AMD build, so only the voice
        # gets the GPU; the CPU speech model keeps recognition responsive.
        log.warning("AMD GPU detected (ROCm): the voice runs on it; speech recognition "
                    "has no AMD GPU support and runs on the CPU.")
        _stt_to_cpu(cfg)
        if tts_engine in _NVIDIA_ONLY_TTS:
            log.warning("TTS engine %r needs an NVIDIA GPU; using chatterbox-turbo "
                        "instead (it still clones your voice sample)", tts_engine)
            return "chatterbox-turbo"
        return tts_engine
    log.warning("no GPU usable by VCB detected -- running speech recognition and voice "
                "on the CPU (slower). An NVIDIA GPU (or a ROCm-capable AMD Radeon) "
                "is recommended.")
    _stt_to_cpu(cfg)
    cfg.tts.device = "cpu"
    if tts_engine in _GPU_ONLY_TTS:
        log.warning("TTS engine %r needs a GPU for real-time speech; using the kokoro "
                    "preset voice instead -- your voice sample is NOT used. To hear it, "
                    "VCB needs an NVIDIA GPU or a supported AMD Radeon.", tts_engine)
        return "kokoro"
    return tts_engine


# Engines whose GPU we can choose (orpheus runs inside llama-cpp, not torch).
_PLACEABLE_TTS = {"chatterbox-turbo", "kokoro", "indextts", "xtts", "csm"}


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
