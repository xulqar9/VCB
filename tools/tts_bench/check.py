"""`check`: is each engine ready to run? Read-only: nothing is loaded, installed
or downloaded.

For every distinct Python interpreter used by the selected engines this runs
tts_bench.probe once (see probe.py) and judges the answer against what each
engine needs:

  * the interpreter exists and has the engine's packages
  * its PyTorch build has kernels for the GPU the engine is pinned to, and the
    report names every other GPU too (for instance the RTX 5060 Ti)
  * the weights are in that interpreter's Hugging Face cache
  * the folders an engine needs (a git clone) exist
  * the files only VCB's own installer puts on disk (XTTS's model, IndexTTS) exist
  * the reference clip and its transcript exist
"""

from __future__ import annotations

import json
import os
import subprocess
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from .config import REPO_ROOT, BenchConfig, resolve_path
from .gpu import GpuInfo, list_gpus
from .registry import EngineSpec, install_hint, variant_name
from .runner import Target, resolve_target, worker_env

PROBE_TIMEOUT_S = 240       # the first `import torch` after boot can take a while
COMPLETE_FRACTION = 0.90    # a cached repo counts as complete from 90% of its Hub size
CLIP_MIN_S, CLIP_MAX_S = 3.0, 30.0   # reference clips outside this range clone badly

OK, WARN, FAIL = "ok", "WARN", "FAIL"
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


@dataclass(frozen=True)
class Finding:
    level: str          # OK, WARN or FAIL
    text: str


@dataclass
class EnvReport:
    """One Python interpreter and what the probe found in it."""
    python: Path
    engines: list[str]
    probe: dict                       # tts_bench.probe's answer, or {"error": ...}
    findings: list[Finding] = field(default_factory=list)


@dataclass
class VariantReport:
    variant: str
    findings: list[Finding]

    @property
    def ready(self) -> bool:
        return not any(f.level == FAIL for f in self.findings)


@dataclass
class EngineReport:
    spec: EngineSpec
    target: Target
    python: Path
    findings: list[Finding]
    variants: list[VariantReport]


@dataclass
class ClipReport:
    path: Path
    seconds: float | None
    rate: int | None


@dataclass
class CheckReport:
    gpus: list[GpuInfo]
    envs: list[EnvReport]
    engines: list[EngineReport]
    clip: ClipReport | None

    @property
    def ready(self) -> bool:
        return all(v.ready for e in self.engines for v in e.variants)


# -- running the probe ---------------------------------------------------------

def probe_env(base: Mapping[str, str]) -> dict[str, str]:
    """Like an engine's environment (offline, no VCB settings, no tokens), but
    every GPU stays visible so all of them are checked."""
    env = worker_env(base, Target("all", None))
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return env


def run_probe(python: Path, modules: list[str], repos: list[str],
              base_env: Mapping[str, str] | None = None) -> dict:
    cmd = [str(python), "-m", "tts_bench.probe", "--modules", ",".join(modules),
           "--repos", ",".join(repos)]
    try:
        done = subprocess.run(cmd, cwd=REPO_ROOT, env=probe_env(base_env or os.environ),
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=PROBE_TIMEOUT_S, creationflags=_NO_WINDOW)
    except subprocess.TimeoutExpired:
        return {"error": f"no answer after {PROBE_TIMEOUT_S} s"}
    except OSError as exc:
        return {"error": f"cannot start: {exc}"}
    for line in reversed(done.stdout.splitlines()):
        if line.startswith("{"):
            try:
                return json.loads(line)
            except ValueError:
                break
    tail = " | ".join((done.stderr or done.stdout).strip().splitlines()[-3:])
    return {"error": f"exit code {done.returncode}: {tail}"[:400]}


# -- judging the answers -------------------------------------------------------

def read_clip(path: Path) -> ClipReport:
    """Duration and sample rate of a reference clip (WAV through the stdlib,
    anything else through soundfile when it is installed)."""
    try:
        with wave.open(str(path), "rb") as fh:
            return ClipReport(path, fh.getnframes() / fh.getframerate(), fh.getframerate())
    except (wave.Error, EOFError):
        pass
    try:
        import soundfile
        info = soundfile.info(str(path))
        return ClipReport(path, info.duration, info.samplerate)
    except Exception:
        return ClipReport(path, None, None)


def _short_name(name: str) -> str:
    return name.replace("NVIDIA ", "").replace("GeForce ", "")


def _sm(capability: list[int]) -> str:
    return f"sm_{capability[0]}{capability[1]}"


def _device_line(dev: dict) -> tuple[str, str]:
    """(level, text) of one GPU as the probe's PyTorch sees it."""
    head = f"GPU {dev['index']} {_short_name(dev['name'])} ({_sm(dev['capability'])})"
    if not dev.get("matmul_ok"):
        why = dev.get("matmul_error") or "test calculation failed"
        return FAIL, f"{head}: PyTorch cannot run on it ({why})"
    if dev["support"] == "native":
        return OK, f"{head}: kernels built in, test calculation passed"
    if dev["support"] == "ptx":
        return WARN, f"{head}: only generic PTX code (compiled on first use: slow start)"
    return FAIL, f"{head}: this PyTorch build has no kernels for it"


def env_findings(probe: dict, python: Path) -> list[Finding]:
    """What the interpreter itself provides: Python, PyTorch and every GPU."""
    if "error" in probe:
        return [Finding(FAIL, f"the environment could not be inspected: {probe['error']}")]
    findings = [Finding(OK, f"Python {probe['python']} on {probe['platform']}")]
    torch = probe.get("torch")
    if torch is None:
        return findings + [Finding(WARN, "PyTorch was not inspected")]
    if "error" in torch:
        return findings + [Finding(FAIL, f"PyTorch does not import: {torch['error']}")]
    build = f"PyTorch {torch['version']}" + (f" (CUDA {torch['cuda_runtime']})" if torch["cuda_runtime"] else "")
    if not torch["cuda_available"]:
        return findings + [Finding(FAIL, f"{build} cannot see a CUDA GPU (CPU-only build or driver problem)")]
    findings.append(Finding(OK, build + "; kernels: " + " ".join(torch["arch_list"])))
    for dev in torch["devices"]:
        level, text = _device_line(dev)
        findings.append(Finding(WARN if level == FAIL else level, text))   # engine-specific verdict is below
    return findings


def _engine_gpu_finding(probe: dict, target: Target) -> Finding | None:
    """Can this engine's own interpreter use the GPU it is pinned to?"""
    if target.gpu is None:
        return Finding(OK, "runs on the CPU")
    torch = probe.get("torch") or {}
    for dev in torch.get("devices", []):
        if dev["index"] == target.gpu.index:
            level, text = _device_line(dev)
            return Finding(level, text)
    if torch.get("cuda_available") is False:
        return Finding(FAIL, f"GPU {target.gpu.index}: PyTorch sees no CUDA GPU")
    return Finding(FAIL, f"GPU {target.gpu.index}: PyTorch does not list it")


def _clip_findings(cfg: BenchConfig, spec: EngineSpec, clip: ClipReport | None) -> list[Finding]:
    if not cfg.reference_clip:
        return [Finding(FAIL, "no reference clip: set [voice] reference_clip")]
    if clip is None:
        return [Finding(FAIL, f"reference clip not found: {cfg.reference_clip}")]
    findings = []
    if clip.seconds is None:
        findings.append(Finding(WARN, f"reference clip {clip.path.name}: length unknown"))
    elif not CLIP_MIN_S <= clip.seconds <= CLIP_MAX_S:
        findings.append(Finding(WARN, f"reference clip is {clip.seconds:.1f} s; cloning works best with "
                                      f"{CLIP_MIN_S:g} to {CLIP_MAX_S:g} s"))
    else:
        findings.append(Finding(OK, f"reference clip {clip.path.name}: {clip.seconds:.1f} s, {clip.rate} Hz"))
    if spec.needs_reference_text:
        if cfg.reference_text:
            findings.append(Finding(OK, "reference transcript is set"))
        else:
            findings.append(Finding(FAIL, "needs the clip's exact transcript: set [voice] reference_text"))
    return findings


def _weights_findings(spec: EngineSpec, mode: str, probe: dict) -> list[Finding]:
    if "error" in probe:
        return [Finding(WARN, "weights were not checked: the environment could not be inspected")]
    cache = probe.get("hf_cache", {}).get("repos", {})
    # VCB's own engines are installed by VCB, never by this tool's `download` command.
    if spec.vcb_native:
        missing_fix = incomplete_fix = install_hint(spec)
    else:
        missing_fix = f"run `tts_benchmark.py download --engines {spec.key}`"
        incomplete_fix = "run the `download` command again"
    findings = []
    for weight in spec.weight_repos((mode,)):
        have = cache.get(weight.repo, {})
        got_gb = have.get("size_bytes", 0) / 1e9
        if not have.get("present"):
            findings.append(Finding(FAIL, f"weights {weight.repo} ({weight.size_gb:g} GB) are not "
                                          f"downloaded: {missing_fix}"))
        elif have.get("partial_files") or got_gb < COMPLETE_FRACTION * weight.size_gb:
            findings.append(Finding(FAIL, f"weights {weight.repo} are incomplete ({got_gb:.2f} of "
                                          f"{weight.size_gb:g} GB): {incomplete_fix}"))
        else:
            findings.append(Finding(OK, f"weights {weight.repo}: {got_gb:.2f} GB in the cache"))
    return findings


def _file_findings(spec: EngineSpec) -> list[Finding]:
    """The files an engine needs that no Hugging Face cache holds (XTTS's model, IndexTTS)."""
    findings = []
    for name in spec.requires_files:
        path = resolve_path(name)
        if not path.is_file():
            findings.append(Finding(FAIL, f"{path} is missing: {install_hint(spec)}"))
            continue
        size = path.stat().st_size
        findings.append(Finding(OK, f"{path}" + (f" ({size / 1e6:.0f} MB)" if size >= 1e6 else "")))
    return findings


def _engine_findings(cfg: BenchConfig, spec: EngineSpec, target: Target, env: EnvReport) -> list[Finding]:
    python, probe = env.python, env.probe
    if not python.is_file():
        return [Finding(FAIL, f"interpreter not found: {python} (set [engines.{spec.key}] python)")]
    if "error" in probe:
        return [Finding(FAIL, f"the environment could not be inspected: {probe['error']}")]
    findings: list[Finding] = []
    if spec.linux_only and probe["platform"].startswith("win"):
        findings.append(Finding(FAIL, "its runtime has no Windows build: run this tool inside WSL2 "
                                      "with a Linux interpreter"))
    if "torch" in spec.requires_modules:
        gpu = _engine_gpu_finding(probe, target)
        if gpu:
            findings.append(gpu)
    for name in spec.requires_modules:
        if name == "torch":
            if "error" in (probe.get("torch") or {}):
                findings.append(Finding(FAIL, "module torch does not import"))
            continue
        info = probe["modules"].get(name, {})
        if info.get("found"):
            findings.append(Finding(OK, f"module {name} {info.get('version') or ''}".rstrip()))
        else:
            findings.append(Finding(FAIL, f"module {name} is not installed in this interpreter"))
    options = cfg.settings_for(spec.key).options
    for name in spec.required_options:
        value = options.get(name)
        folder = Path(str(value)).expanduser() if value else None
        if folder is not None and not folder.is_absolute():
            folder = REPO_ROOT / folder
        if folder is not None and folder.is_dir():
            findings.append(Finding(OK, f"{name} = {folder}"))
        else:
            findings.append(Finding(FAIL, f"set [engines.{spec.key}.options] {name} to the engine's "
                                          "source folder (see `list` for the git clone)"))
    return findings + _file_findings(spec)


def run_check(cfg: BenchConfig, specs: list[EngineSpec], modes: list[str] | None = None,
              gpus: list[GpuInfo] | None = None,
              probe: Callable[..., dict] = run_probe) -> CheckReport:
    gpus = list_gpus() if gpus is None else gpus
    # One probe per interpreter, asked about everything its engines need.
    groups: dict[Path, list[EngineSpec]] = {}
    for spec in specs:
        groups.setdefault(cfg.python_for(spec.key), []).append(spec)
    envs: dict[Path, EnvReport] = {}
    for python, members in groups.items():
        modules = sorted({m for s in members for m in s.requires_modules})
        repos = sorted({w.repo for s in members for w in s.weights})
        answer = probe(python, modules, repos) if python.is_file() else {"error": "interpreter not found"}
        env = EnvReport(python, [s.key for s in members], answer)
        env.findings = env_findings(answer, python) if python.is_file() else \
            [Finding(FAIL, f"interpreter not found: {python}")]
        envs[python] = env

    clip = read_clip(cfg.reference_clip) if cfg.reference_clip and cfg.reference_clip.is_file() else None
    reports = []
    for spec in specs:
        env = envs[cfg.python_for(spec.key)]
        target = resolve_target(cfg.gpu_for(spec.key), gpus)
        shared = _engine_findings(cfg, spec, target, env)
        variants = []
        for mode in (m for m in spec.modes if modes is None or m in modes):
            own = list(shared)
            if mode == "clone":
                own += _clip_findings(cfg, spec, clip)
            own += _weights_findings(spec, mode, env.probe)
            variants.append(VariantReport(variant_name(spec.key, mode), own))
        reports.append(EngineReport(spec, target, env.python, shared, variants))
    return CheckReport(gpus, list(envs.values()), reports, clip)


# -- printing ------------------------------------------------------------------

def _lines(findings: list[Finding], indent: str) -> list[str]:
    return [f"{indent}{f.level:<4}  {f.text}" for f in findings]


def format_check(report: CheckReport, cfg: BenchConfig) -> str:
    out = ["Environment check (read-only: nothing is loaded, installed or downloaded)", ""]
    out.append("GPUs (nvidia-smi):")
    out.extend(f"  {g.label}, {g.used_mib / 1024:.1f} GB in use" for g in report.gpus)
    if not report.gpus:
        out.append("  none found: nvidia-smi is not available")
    if report.clip:
        clip = report.clip
        detail = f"{clip.seconds:.1f} s, {clip.rate} Hz" if clip.seconds is not None else "length unknown"
        out += ["", f"Reference clip: {clip.path} ({detail})"]
    out += ["", "Python environments:"]
    for env in report.envs:
        out.append(f"  {env.python}  (engines: {', '.join(env.engines)})")
        out.extend(_lines(env.findings, "    "))
    out += ["", "Engines:"]
    for eng in report.engines:
        out.append(f"  {eng.spec.key}: {eng.spec.title}, on {eng.target.label}")
        shared = {(f.level, f.text) for f in eng.findings}
        out.extend(_lines(eng.findings, "    "))
        for var in eng.variants:
            own = [f for f in var.findings if (f.level, f.text) not in shared]
            out.extend(_lines(own, "    "))
            state = "READY" if var.ready else "NOT READY"
            out.append(f"    -> {var.variant}: {state}")
        out.append("")
    ready = [v.variant for e in report.engines for v in e.variants if v.ready]
    blocked = [v.variant for e in report.engines for v in e.variants if not v.ready]
    out.append(f"Ready: {', '.join(ready) or 'none'}")
    if blocked:
        out.append(f"Not ready: {', '.join(blocked)}  (`tts_benchmark.py list` shows what each one needs)")
    return "\n".join(out)
