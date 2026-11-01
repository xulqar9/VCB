"""Hearing more than words (see VocalConfig): wires the sound lane, emotion
recognizer and voice-trait baseline together and phrases what they found
as a short note for the LLM's current turn."""

import asyncio
import logging
from dataclasses import dataclass

import numpy as np

from ..config import USER_NAME, VOCAL_SOUNDS, AudioConfig, VocalConfig
from .emotion import EMOTION_PHRASES, EmotionRecognizer
from .sounds import SoundDetector, SoundEvent, SoundLane
from .traits import VoiceBaseline, VoiceStats, measure_voice

log = logging.getLogger("vcb.vocal")


@dataclass(frozen=True)
class VoiceAnalysis:
    emotion: tuple[str, float] | None = None
    stats: VoiceStats | None = None


def sound_phrases(events: list[SoundEvent]) -> list[str]:
    """"laughs", "sighs", ... in order, without repeats."""
    return list(dict.fromkeys(e.phrase for e in sorted(events, key=lambda e: e.start)))


def lone_sound_text(events: list[SoundEvent]) -> str:
    """What a wordless sound reads like as the user's turn: "(laughs)"."""
    return "(" + " and ".join(sound_phrases(events)) + ")"


class VocalListener:
    def __init__(self, cfg: VocalConfig, audio_cfg: AudioConfig) -> None:
        self.cfg = cfg
        self.sr = audio_cfg.input_sample_rate
        self.sounds: SoundLane | None = None
        self.emotion: EmotionRecognizer | None = None
        self.baseline: VoiceBaseline | None = None
        if cfg.sounds:
            self.sounds = SoundLane(
                SoundDetector(cfg.sound_model, cfg.device, VOCAL_SOUNDS), VOCAL_SOUNDS,
                self.sr, audio_cfg.frame_samples, cfg.sound_window_s, cfg.sound_hop_s,
                cfg.sound_min_dbfs, cfg.sound_threshold)
        if cfg.emotion:
            self.emotion = EmotionRecognizer(cfg.emotion_model, cfg.device)
        if cfg.traits:
            self.baseline = VoiceBaseline(cfg.traits_baseline_utterances)

    @property
    def analyzes_utterances(self) -> bool:
        return self.emotion is not None or self.baseline is not None

    async def load(self) -> None:
        """Load the enabled models; one that can't load is switched off with
        a warning instead of stopping VCB."""
        async def _load(name: str, attr: str, hint: str) -> None:
            part = getattr(self, attr)
            try:
                await asyncio.to_thread(part.detector.load if attr == "sounds" else part.load)
                log.info("%s ready (%s)", name, self.cfg.device)
            except Exception as exc:
                setattr(self, attr, None)
                log.warning("%s is off: couldn't load it (%s). %s", name, exc, hint)

        jobs = []
        if self.sounds is not None:
            jobs.append(_load("sound detection", "sounds",
                              "Check your internet connection for the first download."))
        if self.emotion is not None:
            jobs.append(_load("voice emotion", "emotion",
                              "Start VCB with VCB.bat so it installs the voice emotion pack."))
        await asyncio.gather(*jobs)

    def feed(self, frame_i16: np.ndarray) -> None:
        if self.sounds is not None:
            self.sounds.feed(frame_i16)

    def _analyze(self, audio: np.ndarray) -> VoiceAnalysis:
        emotion = stats = None
        if self.emotion is not None and len(audio) >= self.cfg.emotion_min_s * self.sr:
            try:
                emotion = self.emotion.classify(audio)
            except Exception:
                log.exception("emotion recognition failed")
        if self.baseline is not None:
            stats = measure_voice(audio, self.sr)
        return VoiceAnalysis(emotion, stats)

    async def analyze(self, audio: np.ndarray | None) -> VoiceAnalysis:
        """Emotion + voice measurements of one utterance (runs alongside STT)."""
        if audio is None or not self.analyzes_utterances:
            return VoiceAnalysis()
        try:
            return await asyncio.to_thread(self._analyze, np.asarray(audio, np.float32))
        except Exception:
            log.exception("voice analysis failed")
            return VoiceAnalysis()

    def describe(self, analysis: VoiceAnalysis, text: str) -> list[str]:
        """Notes worth telling the LLM about this utterance."""
        notes = []
        if analysis.emotion is not None:
            label, score = analysis.emotion
            log.info("voice emotion: %s (%.2f)", label, score)
            if label in EMOTION_PHRASES and score >= self.cfg.emotion_min_score:
                notes.append(EMOTION_PHRASES[label])
        if analysis.stats is not None and self.baseline is not None:
            changes = self.baseline.compare(analysis.stats, len(text.split()))
            if changes:
                notes.append(_join(changes) + " than usual")
        return notes

    @staticmethod
    def cue(during_reply: list[SoundEvent], with_words: list[SoundEvent],
            notes: list[str]) -> str | None:
        """The per-turn note for the LLM, or None if nothing stood out."""
        lines = []
        if during_reply:
            lines.append("while you were still talking: " + _join(sound_phrases(during_reply)))
        if with_words:
            lines.append("with their words: " + _join(sound_phrases(with_words)))
        lines += notes
        if not lines:
            return None
        return (f"Voice cues from {USER_NAME}'s audio (automatic, may be wrong):\n"
                + "\n".join(f"- {line}" for line in lines)
                + "\nLet them shape your reply the way they would for someone who "
                "heard them. Never mention detecting or analyzing them.")


def _join(items: list[str]) -> str:
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]
