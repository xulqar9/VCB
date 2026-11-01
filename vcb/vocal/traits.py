"""How the user is speaking, compared with their own usual voice: pitch,
intonation, loudness and pace. Plain numpy (no extra install): an
autocorrelation pitch tracker over short frames, plus a running baseline so
only clear changes ("louder and faster than usual") are reported."""

from collections import deque
from dataclasses import dataclass
from statistics import median

import numpy as np

_FRAME_S = 0.04
_HOP_S = 0.01
_F0_MIN, _F0_MAX = 70.0, 500.0
_MIN_PERIODICITY = 0.5               # normalized autocorrelation peak = voiced
_SILENCE_BELOW_PEAK_DB = 35.0        # frames this far under the loudest are pauses
_ABS_FLOOR_DBFS = -55.0

# A change has to be at least this big to be mentioned.
PITCH_SHIFT_ST = 2.0                 # semitones
LOUDNESS_SHIFT_DB = 5.0
RANGE_RATIO = 1.5                    # intonation range, x or 1/x
RATE_RATIO = 1.3                     # words per second, x or 1/x
MIN_ACTIVE_S = 1.0                   # shorter utterances aren't measured
MIN_WORDS_FOR_RATE = 4


@dataclass(frozen=True)
class VoiceStats:
    pitch_st: float | None           # median pitch, semitones above 55 Hz
    range_st: float | None           # 10th-90th percentile pitch spread
    loudness_db: float               # median level of the active frames, dBFS
    active_s: float                  # time with sound (pauses excluded)


def _frames(audio: np.ndarray, sr: int) -> np.ndarray:
    size, hop = int(_FRAME_S * sr), int(_HOP_S * sr)
    if len(audio) < size:
        return np.empty((0, size), np.float32)
    count = 1 + (len(audio) - size) // hop
    idx = np.arange(size)[None, :] + hop * np.arange(count)[:, None]
    return audio[idx]


def measure_voice(audio: np.ndarray, sr: int) -> VoiceStats | None:
    """Pitch/loudness/active time of one utterance; None if there's too
    little sound to say anything."""
    audio = np.asarray(audio, np.float32)
    frames = _frames(audio, sr)
    if not len(frames):
        return None
    rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    active = db >= max(_ABS_FLOOR_DBFS, float(db.max()) - _SILENCE_BELOW_PEAK_DB)
    active_s = float(active.sum()) * _HOP_S
    if active_s < MIN_ACTIVE_S:
        return None

    voiced = frames[active] * np.hanning(frames.shape[1])
    voiced -= voiced.mean(axis=1, keepdims=True)
    n = voiced.shape[1]
    spec = np.fft.rfft(voiced, 2 * n, axis=1)
    ac = np.fft.irfft(np.abs(spec) ** 2, axis=1)[:, :n]
    ac /= ac[:, :1] + 1e-12
    lo, hi = int(sr / _F0_MAX), min(n - 1, int(sr / _F0_MIN))
    lags = lo + np.argmax(ac[:, lo:hi], axis=1)
    strength = ac[np.arange(len(lags)), lags]
    f0 = sr / lags[strength >= _MIN_PERIODICITY]
    pitch_st = range_st = None
    if len(f0) >= 20:
        st = 12 * np.log2(f0 / 55.0)
        pitch_st = float(np.median(st))
        range_st = float(np.percentile(st, 90) - np.percentile(st, 10))
    return VoiceStats(pitch_st, range_st, float(np.median(db[active])), active_s)


class VoiceBaseline:
    """The user's usual voice over recent utterances, and how a new one
    differs from it."""

    def __init__(self, min_samples: int = 3, keep: int = 30) -> None:
        self.min_samples = min_samples
        self._pitch: deque[float] = deque(maxlen=keep)
        self._range: deque[float] = deque(maxlen=keep)
        self._loud: deque[float] = deque(maxlen=keep)
        self._rate: deque[float] = deque(maxlen=keep)

    def compare(self, stats: VoiceStats, words: int) -> list[str]:
        """Plain-language differences from usual, then learn this utterance."""
        rate = words / stats.active_s if words >= MIN_WORDS_FOR_RATE else None
        notes: list[str] = []
        if len(self._loud) >= self.min_samples:
            notes = self._differences(stats, rate)
        if stats.pitch_st is not None:
            self._pitch.append(stats.pitch_st)
            self._range.append(stats.range_st)
        self._loud.append(stats.loudness_db)
        if rate is not None:
            self._rate.append(rate)
        return notes

    def _differences(self, stats: VoiceStats, rate: float | None) -> list[str]:
        notes = []
        loud = stats.loudness_db - median(self._loud)
        if loud >= LOUDNESS_SHIFT_DB:
            notes.append("louder")
        elif loud <= -LOUDNESS_SHIFT_DB:
            notes.append("quieter")
        if rate is not None and len(self._rate) >= self.min_samples:
            ratio = rate / median(self._rate)
            if ratio >= RATE_RATIO:
                notes.append("faster")
            elif ratio <= 1 / RATE_RATIO:
                notes.append("slower")
        if stats.pitch_st is not None and len(self._pitch) >= self.min_samples:
            shift = stats.pitch_st - median(self._pitch)
            if shift >= PITCH_SHIFT_ST:
                notes.append("higher-pitched")
            elif shift <= -PITCH_SHIFT_ST:
                notes.append("lower-pitched")
            usual = median(self._range)
            if usual > 0 and stats.range_st >= usual * RANGE_RATIO:
                notes.append("more animated")
            elif stats.range_st <= usual / RANGE_RATIO:
                notes.append("flatter")
        return notes
