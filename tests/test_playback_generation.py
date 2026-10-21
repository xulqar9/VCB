"""SpeakerPlayback generation counter: the primitive barge-in relies on.

flush() must bump the generation (so producers detect a barge-in), clear the
buffered audio instantly, and re-arm the pre-roll for the next reply. No audio
device is opened here -- only the in-memory buffer/flush contract is exercised.
"""

import numpy as np

from vcb.audio.playback import SpeakerPlayback
from vcb.config import AudioConfig


def _speaker():
    return SpeakerPlayback(AudioConfig())


def test_flush_bumps_generation():
    spk = _speaker()
    g0 = spk.generation
    spk.flush()
    assert spk.generation == g0 + 1


def test_flush_clears_buffer_and_rearms_priming():
    spk = _speaker()
    with spk._lock:
        spk._buffer.append(np.zeros(480, dtype=np.float32))
        spk._buffer_samples = 480
    spk._priming = False
    spk.flush()
    assert spk._buffer_samples == 0
    assert len(spk._buffer) == 0
    assert spk._priming is True


def test_is_playing_reflects_buffer():
    spk = _speaker()
    assert spk.is_playing is False
    with spk._lock:
        spk._buffer.append(np.zeros(100, dtype=np.float32))
        spk._buffer_samples = 100
    assert spk.is_playing is True
    spk.flush()
    assert spk.is_playing is False


def test_buffered_seconds():
    spk = _speaker()
    rate = AudioConfig().output_sample_rate
    with spk._lock:
        spk._buffer_samples = rate      # one second buffered
    assert abs(spk.buffered_seconds - 1.0) < 1e-6
