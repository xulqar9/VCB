"""Non-speech vocal sounds (laughs, gasps, sighs, moans, ...) from the raw
microphone. Silero VAD only passes on speech, so this runs alongside it:
every hop, the last second of audio is classified by an AudioSet model if
it's loud enough to be worth it. Overlapping windows of the same sound are
merged into one event."""

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("vcb.vocal.sounds")


@dataclass
class SoundEvent:
    label: str                       # AudioSet label, e.g. "Laughter"
    phrase: str                      # how it's told to the LLM, e.g. "laughs"
    score: float
    start: float                     # time.monotonic() bounds of the sound
    end: float
    claimed: bool = False            # attached to a user turn / handled


class SoundDetector:
    """AudioSet classifier (AST via transformers) restricted to vocal sounds."""

    def __init__(self, model_id: str, device: str, labels: dict[str, str]) -> None:
        self.model_id = model_id
        self.device = device
        self.labels = labels
        self._extractor = None
        self._model = None
        self._ids: list[int] = []
        self._names: list[str] = []

    def load(self) -> None:
        import torch
        from transformers import ASTFeatureExtractor, ASTForAudioClassification

        self._extractor = ASTFeatureExtractor.from_pretrained(self.model_id)
        model = ASTForAudioClassification.from_pretrained(self.model_id).eval()
        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        self._model = model.to(self.device, dtype)
        by_name = {name: i for i, name in model.config.id2label.items()}
        missing = [n for n in self.labels if n not in by_name]
        if missing:
            log.warning("sound model has no label(s) %s; they won't be detected", missing)
        self._names = [n for n in self.labels if n in by_name]
        self._ids = [by_name[n] for n in self._names]

    def classify(self, audio: np.ndarray, sr: int) -> dict[str, float]:
        """Probability of each vocal sound in this clip."""
        import torch

        feats = self._extractor(audio, sampling_rate=sr, return_tensors="pt")["input_values"]
        param = next(self._model.parameters())
        with torch.inference_mode():
            logits = self._model(feats.to(param.device, param.dtype)).logits[0]
        probs = torch.sigmoid(logits[self._ids].float()).cpu().tolist()
        return dict(zip(self._names, probs))


class SoundLane:
    """Windows the mic stream, classifies the loud windows off the event loop
    and keeps recent sound events."""

    def __init__(self, detector, labels: dict[str, str], sample_rate: int,
                 frame_samples: int, window_s: float, hop_s: float,
                 min_dbfs: float, threshold: float, keep_s: float = 30.0) -> None:
        self.detector = detector
        self.labels = labels
        self.sr = sample_rate
        frame_s = frame_samples / sample_rate
        self._window_frames = max(1, round(window_s / frame_s))
        self._hop_frames = max(1, round(hop_s / frame_s))
        self._min_dbfs = min_dbfs
        self.threshold = threshold
        self._keep_s = keep_s
        self._ring: deque[np.ndarray] = deque(maxlen=self._window_frames)
        self._since_hop = 0
        self._windows: asyncio.Queue[tuple[np.ndarray, float, float]] = asyncio.Queue(4)
        self.events: deque[SoundEvent] = deque()
        self.on_event = None             # callback(SoundEvent) for new/extended events

    def feed(self, frame_i16: np.ndarray, now: float | None = None) -> None:
        """One int16 mic frame (called for every frame; cheap)."""
        self._ring.append(frame_i16)
        self._since_hop += 1
        if self._since_hop < self._hop_frames or len(self._ring) < self._window_frames:
            return
        self._since_hop = 0
        audio = np.concatenate(self._ring).astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(audio ** 2)) + 1e-12)
        if 20 * np.log10(rms) < self._min_dbfs:
            return
        end = time.monotonic() if now is None else now
        start = end - len(audio) / self.sr
        if self._windows.full():         # analysis fell behind: keep it live
            self._windows.get_nowait()
        self._windows.put_nowait((audio, start, end))

    async def run(self) -> None:
        while True:
            audio, start, end = await self._windows.get()
            try:
                scores = await asyncio.to_thread(self.detector.classify, audio, self.sr)
            except Exception:
                log.exception("sound classification failed")
                continue
            self.add(scores, start, end)

    def add(self, scores: dict[str, float], start: float, end: float) -> None:
        """Record the strongest vocal sound in one window, merging it into the
        previous event when it's the same sound continuing."""
        label, score = max(scores.items(), key=lambda kv: kv[1], default=("", 0.0))
        if score < self.threshold:
            return
        last = self.events[-1] if self.events else None
        if last is not None and last.label == label and start <= last.end:
            last.end = max(last.end, end)
            last.score = max(last.score, score)
            event = last
        else:
            event = SoundEvent(label, self.labels[label], score, start, end)
            self.events.append(event)
            log.info("heard: %s (%.2f)", label, score)
        while self.events and self.events[0].end < end - self._keep_s:
            self.events.popleft()
        if self.on_event is not None:
            self.on_event(event)

    def claim(self, start: float, end: float) -> list[SoundEvent]:
        """Unclaimed events overlapping [start, end], now claimed."""
        found = [e for e in self.events
                 if not e.claimed and e.end >= start and e.start <= end]
        for e in found:
            e.claimed = True
        return found
