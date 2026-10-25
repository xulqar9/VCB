from .base import STTEngine
from .faster_whisper_stt import FasterWhisperSTT
from .torch_whisper_stt import TorchWhisperSTT

STT_ENGINES = {"faster-whisper": FasterWhisperSTT, "whisper-torch": TorchWhisperSTT}

__all__ = ["STTEngine", "FasterWhisperSTT", "TorchWhisperSTT", "STT_ENGINES"]
