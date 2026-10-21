"""Per-turn latency tracking: STT delay, first token, first audio, buffer delay."""

import logging
import time
from dataclasses import dataclass, field

log = logging.getLogger("vcb.metrics")


@dataclass
class TurnMetrics:
    t_speech_end: float = 0.0
    t_stt_done: float = 0.0
    t_llm_first_token: float = 0.0
    t_first_chunk: float = 0.0
    t_first_audio: float = 0.0
    buffer_delay_s: float = 0.0          # playback backlog when first audio queued
    tts_synth_times: list[float] = field(default_factory=list)
    interrupted: bool = False

    def mark(self, name: str) -> None:
        setattr(self, name, time.perf_counter())

    def report(self) -> str:
        def d(a: float, b: float) -> str:
            return f"{(b - a) * 1000:6.0f}ms" if a and b else "   n/a"

        avg_tts = (
            f"{sum(self.tts_synth_times) / len(self.tts_synth_times) * 1000:.0f}ms"
            if self.tts_synth_times else "n/a"
        )
        return (
            f"stt={d(self.t_speech_end, self.t_stt_done)} | "
            f"first_token={d(self.t_stt_done, self.t_llm_first_token)} | "
            f"first_chunk={d(self.t_llm_first_token, self.t_first_chunk)} | "
            f"first_audio={d(self.t_first_chunk, self.t_first_audio)} | "
            f"voice_to_voice={d(self.t_speech_end, self.t_first_audio)} | "
            f"avg_tts_chunk={avg_tts} | buffer={self.buffer_delay_s * 1000:.0f}ms"
            + (" | INTERRUPTED" if self.interrupted else "")
        )

    def log_report(self) -> None:
        log.info("turn latency: %s", self.report())
