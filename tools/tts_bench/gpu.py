"""NVIDIA GPU helpers for the benchmark.

* listing the GPUs and picking the one an engine runs on (by index or name)
* a background sampler that records per-GPU memory use while engines run

Everything goes through nvidia-smi, so there is no extra dependency and the
benchmark itself never creates a CUDA context (which would add ~300 MB to the
GPU being measured). nvidia-smi's "memory.used" is the same number you see
yourself, so the figures are directly comparable with what you observe.
"""

from __future__ import annotations

import shutil
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True)
class GpuInfo:
    index: int          # nvidia-smi / PCI-bus order (what CUDA_DEVICE_ORDER=PCI_BUS_ID uses)
    name: str
    total_mib: int
    used_mib: int

    @property
    def label(self) -> str:
        return f"GPU {self.index} ({self.name}, {self.total_mib / 1024:.0f} GB)"


def nvidia_smi_path() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe:
        return exe
    fallback = Path(r"C:\Windows\System32\nvidia-smi.exe")
    return str(fallback) if fallback.is_file() else None


# -- listing and choosing GPUs -------------------------------------------------

def parse_gpu_list(text: str) -> list[GpuInfo]:
    """Parse `nvidia-smi --query-gpu=index,name,memory.total,memory.used
    --format=csv,noheader,nounits`."""
    gpus = []
    for line in text.strip().splitlines():
        try:
            idx, rest = line.split(",", 1)
            name, total, used = rest.rsplit(",", 2)
            gpus.append(GpuInfo(int(idx), name.strip(), int(total), int(used)))
        except ValueError:
            continue
    return gpus


def list_gpus() -> list[GpuInfo]:
    """The NVIDIA GPUs in this machine; [] without a driver / nvidia-smi."""
    exe = nvidia_smi_path()
    if exe is None:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,name,memory.total,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=15, creationflags=_NO_WINDOW,
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_gpu_list(out)


def resolve_gpu(spec: str | int, gpus: list[GpuInfo]) -> GpuInfo:
    """A GPU from a config value: an index ("1" / 1, nvidia-smi order) or a
    unique, case-insensitive part of its name ("3090 Ti", "5060"). A number
    is an index when a GPU has it and otherwise a model number."""
    text = str(spec).strip()
    available = ", ".join(g.label for g in gpus) or "none found"
    if text.isdigit():
        for g in gpus:
            if g.index == int(text):
                return g
    matches = [g for g in gpus if text.lower() in g.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if text.isdigit() and not matches:
        raise ValueError(f"no GPU with index {text} and no GPU name contains it (available: {available})")
    problem = "matches several GPUs" if matches else "matches no GPU"
    raise ValueError(f"GPU {text!r} {problem} (available: {available}). "
                     "Use an index or a more specific name.")


# -- memory sampler ------------------------------------------------------------

@dataclass(frozen=True)
class VramSample:
    t: float            # wall-clock seconds (same clock as time.time())
    index: int
    used_mib: int


def parse_sample(line: str) -> VramSample | None:
    """One line of `--query-gpu=timestamp,index,memory.used`:
    "2026/11/27 15:36:42.377, 0, 996"."""
    try:
        stamp, idx, used = (part.strip() for part in line.split(","))
        t = datetime.strptime(stamp, "%Y/%m/%d %H:%M:%S.%f").timestamp()
        return VramSample(t, int(idx), int(used))
    except ValueError:
        return None


def window_stats(samples: list[VramSample], t0: float, t1: float) -> dict[int, dict]:
    """min / median / max MiB per GPU among the samples taken in [t0, t1]."""
    per_gpu: dict[int, list[int]] = {}
    for s in samples:
        if t0 <= s.t <= t1:
            per_gpu.setdefault(s.index, []).append(s.used_mib)
    return {
        idx: {"min": min(v), "median": statistics.median(v), "max": max(v), "n": len(v)}
        for idx, v in per_gpu.items()
    }


class VramMonitor:
    """Streams `nvidia-smi -lms` in the background and keeps every timestamped
    sample, so any time window (model load, synthesis, idle) can be analysed
    afterwards. nvidia-smi stamps each sample itself, so pipe buffering cannot
    skew the timing. Samples arrive about every 100 ms per GPU."""

    def __init__(self, interval_ms: int = 100) -> None:
        self.interval_ms = interval_ms
        self.available = False
        self._proc: subprocess.Popen | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._samples: list[VramSample] = []

    def start(self) -> bool:
        exe = nvidia_smi_path()
        if exe is None:
            return False
        try:
            self._proc = subprocess.Popen(
                [exe, "--query-gpu=timestamp,index,memory.used",
                 "--format=csv,noheader,nounits", "-lms", str(self.interval_ms)],
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True,
                creationflags=_NO_WINDOW,
            )
        except OSError:
            return False
        self._thread = threading.Thread(target=self._read, daemon=True, name="vram-monitor")
        self._thread.start()
        self.available = True
        return True

    def _read(self) -> None:
        assert self._proc is not None and self._proc.stdout is not None
        for line in self._proc.stdout:
            sample = parse_sample(line)
            if sample is not None:
                with self._lock:
                    self._samples.append(sample)

    def stop(self) -> None:
        if self._proc is not None:
            self._proc.kill()
            self._proc.wait(timeout=5)
        if self._thread is not None:
            self._thread.join(timeout=5)

    def samples(self) -> list[VramSample]:
        with self._lock:
            return list(self._samples)

    def window(self, t0: float, t1: float) -> dict[int, dict]:
        return window_stats(self.samples(), t0, t1)

    def wait_for_samples(self, timeout: float = 5.0) -> bool:
        """Block until the first sample has arrived (the monitor is live)."""
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.samples():
                return True
            time.sleep(0.05)
        return False

    def __enter__(self) -> "VramMonitor":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()
