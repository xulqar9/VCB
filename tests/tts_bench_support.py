"""Shared helpers for the TTS benchmark tests (tools/tts_bench).

Imported by the test modules next to it. It puts tools/ on the import path,
and provides fake GPUs, tiny WAV files and stand-ins for nvidia-smi, so the
tests need no GPU, no model and no network.
"""

from __future__ import annotations

import math
import os
import struct
import sys
import time
import wave
from pathlib import Path

import pytest

if sys.version_info < (3, 11):      # the benchmark's config reader needs tomllib
    pytest.skip("the TTS benchmark needs Python 3.11 or newer", allow_module_level=True)

REPO_ROOT = Path(__file__).resolve().parent.parent
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.append(str(TOOLS_DIR))

from tts_bench.gpu import GpuInfo  # noqa: E402

# The machine the benchmark was designed for: GPU 0 runs the engine under test,
# GPU 1 is where the rest of VCB lives.
GPUS = [
    GpuInfo(0, "NVIDIA GeForce RTX 3090 Ti", 24564, 1000),
    GpuInfo(1, "NVIDIA GeForce RTX 5060 Ti", 16311, 5000),
]


class OsAs:
    """`runner.os` with another platform name, so the Windows-only and
    Linux-only paths can be tested anywhere (patching the real os.name would
    break pathlib)."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __getattr__(self, attr: str):
        return getattr(os, attr)


def write_wav(path: Path, seconds: float = 1.0, rate: int = 24000, amplitude: float = 0.2) -> Path:
    """A mono 16-bit sine tone."""
    frames = b"".join(struct.pack("<h", int(32767 * amplitude * math.sin(2 * math.pi * 220 * i / rate)))
                      for i in range(int(seconds * rate)))
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(frames)
    return path


class NoMonitor:
    """VramMonitor on a machine without nvidia-smi: nothing is measured."""

    available = False

    def start(self) -> bool:
        return False

    def wait_for_samples(self, timeout: float = 5.0) -> bool:
        return False

    def stop(self) -> None:
        pass

    def window(self, t0: float, t1: float) -> dict:
        return {}


class FakeGpuMonitor:
    """Stands in for nvidia-smi. Every GPU reports its idle memory, except that
    while an engine process is alive it reports the busy figures. `leak=True`
    keeps reporting them after the engine has exited, like a process that
    never gives its VRAM back."""

    available = True

    def __init__(self, idle: dict[int, int], busy: dict[int, int], leak: bool = False) -> None:
        self.idle, self.busy, self.leak = idle, busy, leak
        self._alive: list[list] = []            # [started, stopped or None] per engine process

    # What the runner calls.
    def start(self) -> bool:
        return True

    def wait_for_samples(self, timeout: float = 5.0) -> bool:
        return True

    def stop(self) -> None:
        pass

    def window(self, t0: float, t1: float) -> dict:
        busy = any(start <= t1 and (end is None or end >= t0) for start, end in self._alive)
        levels = self.busy if busy else self.idle
        return {g: {"min": mib, "median": mib, "max": mib, "n": 5} for g, mib in levels.items()}

    # What the tracked worker calls.
    def engine_started(self) -> None:
        self._alive.append([time.time(), None])

    def engine_stopped(self) -> None:
        if self.leak:
            return
        for span in self._alive:
            if span[1] is None:
                span[1] = time.time()


def install_fake_gpu(mp: pytest.MonkeyPatch, monitor: FakeGpuMonitor) -> None:
    """Use `monitor` instead of nvidia-smi, and tell it when engine processes
    start and stop."""
    from tts_bench import runner

    real = runner.WorkerProcess

    class Tracked(real):
        def start(self):
            super().start()
            monitor.engine_started()

        def kill(self):
            super().kill()
            monitor.engine_stopped()

        def close(self):
            super().close()
            monitor.engine_stopped()

    mp.setattr(runner, "VramMonitor", lambda: monitor)
    mp.setattr(runner, "WorkerProcess", Tracked)
