"""Audio touch-ups applied just before playback: consistent loudness and
punctuation-driven pauses that give TTS a more human cadence."""

import numpy as np

from .config import ProsodyConfig


def apply_gain_limit(audio: np.ndarray, gain: float = 1.0, ceiling: float = 0.95) -> np.ndarray:
    """Optional makeup gain + a transparent peak limiter. Samples below `ceiling`
    pass through untouched (no coloring); only peaks above it are softly clipped,
    so normal speech keeps the engine's native quality.
    """
    if audio.size == 0:
        return audio
    y = audio if gain == 1.0 else (audio * gain).astype(np.float32)
    if ceiling and ceiling > 0 and float(np.max(np.abs(y))) > ceiling:
        over = np.abs(y) > ceiling
        y = y.copy()
        excess = (np.abs(y[over]) - ceiling) / (1.0 - ceiling)
        y[over] = np.sign(y[over]) * (ceiling + (1.0 - ceiling) * np.tanh(excess))
    return y.astype(np.float32, copy=False)


def trailing_pause_seconds(text: str, cfg: ProsodyConfig) -> float:
    """How long to pause after a chunk, based on its final punctuation."""
    t = text.rstrip().rstrip("\"')]}>")  # ignore trailing quotes/brackets/tags
    if not t:
        return cfg.pause_default_s
    if t.endswith("...") or t.endswith("\u2026"):
        return cfg.pause_ellipsis_s
    last = t[-1]
    if last == "?":
        return cfg.pause_question_s
    if last in ".!":
        return cfg.pause_sentence_s
    if last in ",;:":
        return cfg.pause_clause_s
    return cfg.pause_default_s


def silence(seconds: float, sample_rate: int) -> np.ndarray:
    return np.zeros(int(seconds * sample_rate), dtype=np.float32)
