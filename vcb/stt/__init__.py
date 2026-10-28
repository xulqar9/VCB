from .base import STTEngine
from .faster_whisper_stt import FasterWhisperSTT
from .parakeet_stt import ParakeetSTT
from .torch_whisper_stt import TorchWhisperSTT

STT_ENGINES = {"parakeet": ParakeetSTT, "faster-whisper": FasterWhisperSTT,
               "whisper-torch": TorchWhisperSTT}

__all__ = ["STTEngine", "ParakeetSTT", "FasterWhisperSTT", "TorchWhisperSTT", "STT_ENGINES"]
