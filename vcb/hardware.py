"""Adapt the config to the machine VCB is running on, so a PC without an
NVIDIA GPU (e.g. someone using the cloud backend on a laptop) still works."""

import logging
import os

from . import gpu
from .config import AppConfig

log = logging.getLogger("vcb.hardware")

TTS_ENGINES = ("chatterbox-turbo", "kokoro", "xtts", "orpheus", "csm")
_GPU_ONLY_TTS = {"chatterbox-turbo", "xtts", "orpheus", "csm"}


def cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def adapt_to_hardware(cfg: AppConfig, tts_engine: str, has_cuda: bool) -> str:
    """Pick CPU-friendly settings when there's no CUDA GPU. Returns the TTS
    engine to actually use."""
    if tts_engine not in TTS_ENGINES:
        raise ValueError(
            f"unknown TTS engine {tts_engine!r} (use one of: {', '.join(TTS_ENGINES)})"
        )
    if has_cuda:
        return tts_engine
    log.warning("no CUDA GPU detected -- running speech recognition and voice "
                "on the CPU (slower). An NVIDIA GPU is recommended.")
    cfg.stt.device = "cpu"
    cfg.stt.model = cfg.stt.cpu_model
    cfg.stt.compute_type = cfg.stt.cpu_compute_type
    cfg.tts.device = "cpu"
    if tts_engine in _GPU_ONLY_TTS:
        log.warning("TTS engine %r needs a GPU for real-time speech; using kokoro "
                    "instead", tts_engine)
        return "kokoro"
    return tts_engine


# Engines whose GPU we can choose (orpheus runs inside llama-cpp, not torch).
_PLACEABLE_TTS = {"chatterbox-turbo", "kokoro", "xtts", "csm"}


def place_on_gpus(cfg: AppConfig, tts_engine: str, has_cuda: bool) -> None:
    """Spread speech recognition and the voice over the GPUs with the most
    free memory (measured after the LLM loaded), like Ollama does for the LLM.
    Skipped if the user pinned devices explicitly (STT/TTS device settings)
    or there's only one usable GPU."""
    if not has_cuda or cfg.stt.device != "cuda" or cfg.tts.device != "cuda":
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
