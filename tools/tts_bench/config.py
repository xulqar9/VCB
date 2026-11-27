"""Benchmark settings, read from a TOML file (default: tools/tts_bench.toml).

Relative paths are resolved against the repository root, so the benchmark
behaves the same wherever it is started from.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

if sys.version_info < (3, 11):  # tomllib arrived in 3.11
    raise SystemExit("The TTS benchmark needs Python 3.11 or newer "
                     "(VCB's own interpreter: .runtime\\venv\\Scripts\\python.exe).")
import tomllib

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "tools" / "tts_bench.toml"

_RUN_KEYS = {"runs_per_sentence", "warmup_runs", "timeout_s", "load_timeout_s",
             "warmup_timeout_s", "max_restarts", "seed", "output_dir", "sentences"}
_VOICE_KEYS = {"reference_clip", "reference_text"}
_GPU_KEYS = {"default"}
_ENGINE_KEYS = {"gpu", "python", "options"}
_TOP_KEYS = {"run", "voice", "gpu", "python", "engines"}


@dataclass
class EngineSettings:
    gpu: str | int | None = None          # overrides [gpu].default for this engine
    python: str | None = None             # interpreter that has this engine's packages
    options: dict = field(default_factory=dict)   # passed to the adapter's Options


@dataclass
class BenchConfig:
    # [run]
    runs_per_sentence: int = 3
    warmup_runs: int = 1                  # uncounted runs per engine session
    timeout_s: float = 30.0               # hard limit per counted synthesis
    load_timeout_s: float = 1800.0        # loading a model may take minutes
    warmup_timeout_s: float = 300.0       # first call pays CUDA/JIT start-up costs
    max_restarts: int = 3                 # engine restarts after a timeout/crash
    seed: int = 1234
    output_dir: Path = REPO_ROOT / "output"
    sentences: Path = REPO_ROOT / "tools" / "tts_bench" / "sentences.txt"
    # [voice]
    reference_clip: Path | None = None    # clip for the voice-cloning pass
    reference_text: str = ""              # exact transcript of that clip
    # [gpu] / [python] / [engines.*]
    default_gpu: str | int = "3090 Ti"    # the engine under test; VCB's LLM stays on the 5060 Ti
    default_python: Path | None = None
    engines: dict[str, EngineSettings] = field(default_factory=dict)
    source: Path | None = None

    def settings_for(self, engine: str) -> EngineSettings:
        return self.engines.get(engine, EngineSettings())

    def gpu_for(self, engine: str) -> str | int:
        override = self.settings_for(engine).gpu
        return self.default_gpu if override in (None, "") else override

    def python_for(self, engine: str) -> Path:
        """Interpreter for an engine: its own entry, else [python].default, else
        VCB's runtime venv, else the interpreter running this tool."""
        own = self.settings_for(engine).python
        if own:
            return _resolve(own)
        if self.default_python:
            return self.default_python
        for rel in (".runtime/venv/Scripts/python.exe", ".runtime/venv/bin/python"):
            candidate = REPO_ROOT / rel
            if candidate.is_file():
                return candidate
        return Path(sys.executable)

    def validate(self) -> None:
        if self.runs_per_sentence < 1:
            raise ValueError("runs per sentence must be at least 1")
        if self.warmup_runs < 0:
            raise ValueError("warm-up runs cannot be negative")
        if self.timeout_s <= 0:
            raise ValueError("the timeout must be positive")


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else REPO_ROOT / p


def resolve_path(path: str | Path) -> Path:
    """A path written in the registry: "~" and %VARIABLES% are expanded, and a relative
    one is taken from the repository root."""
    return _resolve(os.path.expandvars(str(path)))


def _check_keys(section: str | None, data: dict, allowed: set[str]) -> None:
    unknown = sorted(set(data) - allowed)
    if not unknown:
        return
    if section is None:
        raise ValueError(f"unknown setting(s) outside any [section]: {', '.join(unknown)} "
                         f"(the sections are: {', '.join(sorted(allowed))})")
    raise ValueError(f"unknown key(s) in [{section}]: {', '.join(unknown)} "
                     f"(allowed: {', '.join(sorted(allowed))})")


def parse_config(data: dict, source: Path | None = None) -> BenchConfig:
    _check_keys(None, data, _TOP_KEYS)
    run = data.get("run", {})
    voice = data.get("voice", {})
    gpu = data.get("gpu", {})
    python = data.get("python", {})
    _check_keys("run", run, _RUN_KEYS)
    _check_keys("voice", voice, _VOICE_KEYS)
    _check_keys("gpu", gpu, _GPU_KEYS)
    _check_keys("python", python, {"default"})

    cfg = BenchConfig(source=source)
    for key in ("runs_per_sentence", "warmup_runs", "max_restarts", "seed"):
        if key in run:
            setattr(cfg, key, int(run[key]))
    for key in ("timeout_s", "load_timeout_s", "warmup_timeout_s"):
        if key in run:
            setattr(cfg, key, float(run[key]))
    if "output_dir" in run:
        cfg.output_dir = _resolve(run["output_dir"])
    if "sentences" in run:
        cfg.sentences = _resolve(run["sentences"])
    if voice.get("reference_clip"):
        cfg.reference_clip = _resolve(voice["reference_clip"])
    cfg.reference_text = str(voice.get("reference_text", "")).strip()
    if gpu.get("default") not in (None, ""):
        cfg.default_gpu = gpu["default"]
    if python.get("default"):
        cfg.default_python = _resolve(python["default"])

    for name, table in data.get("engines", {}).items():
        _check_keys(f"engines.{name}", table, _ENGINE_KEYS)
        cfg.engines[name] = EngineSettings(
            gpu=table.get("gpu"), python=table.get("python"),
            options=dict(table.get("options", {})),
        )

    cfg.validate()
    return cfg


def load_config(path: str | Path | None = None) -> BenchConfig:
    path = Path(path) if path else DEFAULT_CONFIG
    if not path.is_file():
        if path == DEFAULT_CONFIG:
            return BenchConfig()
        raise FileNotFoundError(f"config file not found: {path}")
    with open(path, "rb") as fh:
        return parse_config(tomllib.load(fh), source=path)
