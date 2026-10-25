"""GPU memory budgeting for picking a local (Ollama) model that fits.

VCB's own speech models share the GPU with the LLM, and the user needs some
room left for other apps, so the LLM's budget is:

    total VRAM - speech models (STT + chosen voice) - reserve for other apps

Figures below were measured on an RTX 5060 Ti / 3090 Ti; the per-model VRAM
estimate is calibrated against Ollama's own reported usage at VCB's 8k context.
Uses nvidia-smi / Windows' adapter list only (no torch), so it works in the
settings window before the heavy dependencies are installed.
"""

import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

# Measured GPU memory of VCB's speech stack (GB).
STT_VRAM_GB = 2.2                          # faster-whisper large-v3-turbo, float16
TTS_VRAM_GB = {"chatterbox-turbo": 4.7, "kokoro": 0.7, "indextts": 6.4}
TTS_VRAM_DEFAULT_GB = 4.7                  # optional engines: assume a heavy one

DEFAULT_RESERVE_GB = 2.0                   # left free for the desktop, browser, games


@dataclass(frozen=True)
class GPU:
    name: str
    total_gb: float


@dataclass(frozen=True)
class ModelChoice:
    tag: str
    download_gb: float                     # weights (+ vision projector), from the registry
    note: str

    @property
    def vram_gb(self) -> float:
        return estimate_vram_gb(self.download_gb)


# Conversational, non-"thinking" models (thinking models stall the voice for
# seconds before the first word), smallest to largest = roughly worst to best.
OLLAMA_CHOICES: tuple[ModelChoice, ...] = (
    ModelChoice("llama3.2:3b", 1.88, "small and fast"),
    ModelChoice("gemma3:4b", 3.12, "small, natural chat"),
    ModelChoice("llama3.1:8b", 4.58, "good all-rounder"),
    ModelChoice("mistral-nemo:12b", 6.59, "great conversation"),
    ModelChoice("gemma3:12b", 7.61, "great conversation"),
    ModelChoice("mistral-small:24b", 13.35, "very capable"),
    ModelChoice("gemma3:27b", 16.22, "best quality"),
)


def estimate_vram_gb(download_gb: float) -> float:
    """GPU memory Ollama needs for a model at VCB's 8k context (weights + KV
    cache + compute buffers). Calibrated: llama3.2:3b 1.88 -> 2.9 GB,
    qwen2.5:32b 18.5 -> 21.3 GB."""
    return round(download_gb * 1.12 + 0.8, 1)


def _nvidia_smi() -> str | None:
    exe = shutil.which("nvidia-smi")
    if exe is None:
        fallback = Path(r"C:\Windows\System32\nvidia-smi.exe")
        exe = str(fallback) if fallback.is_file() else None
    return exe


def parse_nvidia_smi(output: str) -> list[GPU]:
    gpus = []
    for line in output.strip().splitlines():
        name, _, mib = line.rpartition(",")
        try:
            gpus.append(GPU(name.strip(), round(int(mib.strip()) / 1024, 1)))
        except ValueError:
            continue
    return gpus


def detect_gpus() -> list[GPU]:
    """NVIDIA GPUs and their total memory; [] if none / no driver."""
    exe = _nvidia_smi()
    if exe is None:
        return []
    try:
        out = subprocess.run(
            [exe, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return parse_nvidia_smi(out)


# -- AMD Radeon (ROCm) -------------------------------------------------------
# Radeon models AMD ships PyTorch-on-Windows (ROCm 7.2.1) builds for; the
# launcher installs that build for them so the voice runs on the GPU. gfx1200
# (RX 9060 XT), gfx1201 (RX 9070 / 9070 XT / AI PRO R9700), gfx1100 (RX 7900
# XTX / PRO W7900), gfx1101 (RX 7700). Windows 11 only.
_ROCM_WINDOWS_GPUS = re.compile(
    r"\bRX 9070( XT)?\b|\bRX 9060 XT\b|\bAI PRO R9700\b|\bRX 7900 XTX\b|"
    r"\bPRO W7900\b|\bRX 7700\b(?! XT)"
)
_DISPLAY_CLASS_KEY = (r"SYSTEM\CurrentControlSet\Control\Class"
                      r"\{4d36e968-e325-11ce-bfc1-08002be10318}")


def rocm_supported(gpu_name: str) -> bool:
    """Whether AMD's Windows PyTorch build supports this Radeon."""
    return "Radeon" in gpu_name and _ROCM_WINDOWS_GPUS.search(gpu_name) is not None


def _present_adapters() -> list[str]:
    """Display adapters Windows sees right now (Win32_VideoController)."""
    if sys.platform != "win32":
        return []
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "(Get-CimInstance Win32_VideoController).Name"],
            capture_output=True, text=True, timeout=20,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return []
    return [line.strip() for line in out.splitlines() if line.strip()]


def _registry_vram_gb() -> dict[str, float]:
    """Total memory per adapter name from the display driver's registry entry
    (WMI's AdapterRAM is a 32-bit field and caps at 4 GB). The registry also
    keeps entries of cards that were removed, so only use it for lookups."""
    if sys.platform != "win32":
        return {}
    import winreg

    vram: dict[str, float] = {}
    try:
        with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, _DISPLAY_CLASS_KEY) as cls:
            for i in range(64):
                try:
                    sub = winreg.EnumKey(cls, i)
                except OSError:
                    break
                try:
                    with winreg.OpenKey(cls, sub) as dev:
                        name, _ = winreg.QueryValueEx(dev, "DriverDesc")
                        size, _ = winreg.QueryValueEx(dev, "HardwareInformation.qwMemorySize")
                except OSError:
                    continue
                if isinstance(size, int) and size > 0:
                    vram[name] = round(size / 2**30, 1)
    except OSError:
        pass
    return vram


def detect_amd_gpus() -> list[GPU]:
    """AMD Radeon GPUs present (Windows only); total_gb is 0.0 if unknown."""
    names = [n for n in _present_adapters() if "Radeon" in n or n.startswith("AMD")]
    if not names:
        return []
    vram = _registry_vram_gb()
    return [GPU(n, vram.get(n, 0.0)) for n in names]


def speech_vram_gb(tts_engine: str, stt_on_gpu: bool = True) -> float:
    """GPU memory of VCB's speech stack. On AMD, speech recognition has no GPU
    build and runs on the CPU, so only the voice counts (stt_on_gpu=False)."""
    tts = TTS_VRAM_GB.get(tts_engine, TTS_VRAM_DEFAULT_GB)
    return STT_VRAM_GB + tts if stt_on_gpu else tts


def llm_budget_gb(gpus: list[GPU], tts_engine: str, reserve_gb: float,
                  stt_on_gpu: bool = True) -> float:
    """GPU memory left for the LLM. Ollama can split a model over several
    GPUs, so their memory adds up (one fast GPU is still better)."""
    if not gpus:
        return 0.0
    total = sum(g.total_gb for g in gpus)
    return round(max(0.0, total - speech_vram_gb(tts_engine, stt_on_gpu) - reserve_gb), 1)


def recommend(budget_gb: float,
              choices: tuple[ModelChoice, ...] = OLLAMA_CHOICES) -> ModelChoice | None:
    """Largest model that fits the budget, or None if even the smallest doesn't."""
    fitting = [c for c in choices if c.vram_gb <= budget_gb]
    return fitting[-1] if fitting else None


# -- placing VCB's own models on GPUs ----------------------------------------

def query_free_gb() -> dict[int, float]:
    """Free memory per GPU right now, keyed by nvidia-smi index (PCI order)."""
    exe = _nvidia_smi()
    if exe is None:
        return {}
    try:
        out = subprocess.run(
            [exe, "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    free = {}
    for line in out.strip().splitlines():
        idx, _, mib = line.partition(",")
        try:
            free[int(idx)] = round(int(mib.strip()) / 1024, 2)
        except ValueError:
            continue
    return free


def visible_gpus(cuda_visible_devices: str | None, physical: list[int]) -> list[int] | None:
    """Physical GPU indices this process can use, in CUDA's (logical) order.
    None if CUDA_VISIBLE_DEVICES uses a form we can't map (e.g. UUIDs)."""
    if cuda_visible_devices is None or not cuda_visible_devices.strip():
        return sorted(physical)
    try:
        ids = [int(x) for x in cuda_visible_devices.split(",") if x.strip()]
    except ValueError:
        return None
    return [i for i in ids if i in physical]


def plan_placement(free_gb: dict[int, float], needs_gb: dict[str, float]) -> dict[str, int]:
    """Assign each model (biggest first) to the GPU with the most free memory,
    accounting for what earlier models took. Returns name -> physical index."""
    remaining = dict(free_gb)
    plan = {}
    for name, need in sorted(needs_gb.items(), key=lambda kv: -kv[1]):
        best = max(remaining, key=lambda i: remaining[i])
        plan[name] = best
        remaining[best] -= need
    return plan
