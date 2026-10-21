"""Adapt the config to the machine VCB is running on, so a PC without an
NVIDIA GPU (e.g. someone using the cloud backend on a laptop) still works."""

import logging

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
