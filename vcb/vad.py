"""Silero VAD segmenter: mic frames in -> speech events out.

Emits SpeechStart the moment speech is confirmed (used for barge-in) and
SpeechEnd with the full utterance audio (pre-roll included) once trailing
silence is detected. The Silero model is tiny; it runs on CPU per 32 ms frame.
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass

import numpy as np
import torch

from .config import AudioConfig, VADConfig

log = logging.getLogger("vcb.vad")


@dataclass
class SpeechStart:
    pass


@dataclass
class SpeechEnd:
    audio: np.ndarray                    # float32 mono @ input_sample_rate


VADEvent = SpeechStart | SpeechEnd


class VADSegmenter:
    def __init__(self, vad_cfg: VADConfig, audio_cfg: AudioConfig) -> None:
        self.cfg = vad_cfg
        self.audio_cfg = audio_cfg
        self.events: asyncio.Queue[VADEvent] = asyncio.Queue()
        self._model = None               # loaded in load(), off the event loop

        frame_ms = audio_cfg.frame_ms
        self._min_speech_frames = max(1, vad_cfg.min_speech_ms // frame_ms)
        self._end_silence_frames = max(1, vad_cfg.end_silence_ms // frame_ms)
        self._max_frames = int(vad_cfg.max_utterance_s * 1000 / frame_ms)
        self._pre_roll: deque[np.ndarray] = deque(
            maxlen=max(1, vad_cfg.pre_roll_ms // frame_ms)
        )
        self._reset_state()

    async def load(self) -> None:
        """Fetch/build the Silero model off the event loop. torch.hub may hit
        disk or the network, so it must not run in the constructor."""
        if self._model is not None:
            return
        model, _ = await asyncio.to_thread(
            torch.hub.load, "snakers4/silero-vad", "silero_vad",
            trust_repo=True, onnx=False,
        )
        model.eval()
        self._model = model

    def _reset_state(self) -> None:
        self._in_speech = False
        self._started_emitted = False
        self._speech_frames = 0
        self._silence_frames = 0
        self._utterance: list[np.ndarray] = []

    @property
    def voiced_s(self) -> float:
        """Seconds of actual speech (not pauses) in the utterance so far; 0
        when the user isn't mid-utterance. Lets barge-in tell a quick "yeah"
        from someone really starting to talk."""
        if not self._in_speech:
            return 0.0
        return self._speech_frames * self.audio_cfg.frame_ms / 1000.0

    @torch.inference_mode()
    def _speech_prob(self, frame_f32: np.ndarray) -> float:
        tensor = torch.from_numpy(frame_f32)
        return float(self._model(tensor, self.audio_cfg.input_sample_rate).item())

    def process_frame(self, frame_i16: np.ndarray) -> None:
        """Feed one int16 mic frame. Events appear on self.events."""
        frame = frame_i16.astype(np.float32) / 32768.0
        prob = self._speech_prob(frame)
        is_speech = prob >= self.cfg.threshold

        if not self._in_speech:
            self._pre_roll.append(frame)
            if is_speech:
                self._in_speech = True
                self._speech_frames = 1
                self._silence_frames = 0
                self._utterance = list(self._pre_roll)
            return

        self._utterance.append(frame)
        if is_speech:
            self._speech_frames += 1
            self._silence_frames = 0
            if not self._started_emitted and self._speech_frames >= self._min_speech_frames:
                self._started_emitted = True
                self.events.put_nowait(SpeechStart())
        else:
            self._silence_frames += 1

        ended = self._silence_frames >= self._end_silence_frames
        too_long = len(self._utterance) >= self._max_frames
        if too_long and self._started_emitted:
            log.warning("heard %.0f s of sound without a pause -- if you weren't talking "
                        "that long, the microphone (%s) is picking up noise or the speakers; "
                        "pick another mic in the settings or lower its level in Windows",
                        self.cfg.max_utterance_s, self.audio_cfg.input_device or "default")
        if ended or too_long:
            if self._started_emitted:
                audio = np.concatenate(self._utterance)
                # Trim trailing silence, keep ~100 ms tail.
                keep = max(0, len(self._utterance) - self._silence_frames + 3)
                audio = audio[: keep * self.audio_cfg.frame_samples]
                self.events.put_nowait(SpeechEnd(audio=audio))
                log.debug("utterance ended: %.2fs", len(audio) / self.audio_cfg.input_sample_rate)
            self._reset_state()
