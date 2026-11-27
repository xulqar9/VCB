"""Runs the benchmark: one worker process per (engine, voice mode).

  1. The engine starts in its own interpreter, pinned to one GPU through
     CUDA_VISIBLE_DEVICES, and loads its model (worker.py).
  2. One warm-up synthesis that is not counted, then every sentence
     `runs_per_sentence` times, each under a hard timeout. A synthesis that
     hangs gets its process killed and restarted.
  3. nvidia-smi samples the memory of every GPU the whole time (gpu.VramMonitor).
  4. The worker is told to quit and we check that its VRAM really came back,
     so the next engine starts from the same idle state.

Results are written to output/<variant>/raw.json after every synthesis;
report.py turns them into the CSV, the JSON summary and the terminal tables.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping

from .config import REPO_ROOT, BenchConfig, resolve_path
from .gpu import GpuInfo, VramMonitor, list_gpus, resolve_gpu
from .registry import EngineSpec, install_hint, variant_name
from .sentences import Sentence, sentences_fingerprint

TOOLS_DIR = REPO_ROOT / "tools"
SCHEMA_VERSION = 1
RESULT_GRACE_S = 1.5        # time allowed beyond the timeout to receive/save the result
UNLOAD_TIMEOUT_S = 60.0     # how long a worker may take to unload on request
RELEASE_TOLERANCE_MIB = 300  # VRAM left over after exit that still counts as "released"
RELEASE_WAIT_S = 10.0
_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_STALE_FILES = re.compile(r"^[A-Za-z0-9]+_\d+\.wav$")


def _now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


# -- where an engine runs ------------------------------------------------------

@dataclass(frozen=True)
class Target:
    """The device one engine is pinned to."""
    spec: str                    # what the config said ("3090 Ti", "1", "cpu")
    gpu: GpuInfo | None          # None = CPU

    @property
    def device(self) -> str:
        # Inside the worker only the chosen GPU is visible, so it is always cuda:0.
        return "cuda:0" if self.gpu else "cpu"

    @property
    def visible_devices(self) -> str:
        return str(self.gpu.index) if self.gpu else ""

    @property
    def label(self) -> str:
        return self.gpu.label if self.gpu else "CPU"


def resolve_target(spec: str | int, gpus: list[GpuInfo]) -> Target:
    if str(spec).strip().lower() == "cpu":
        return Target("cpu", None)
    return Target(str(spec), resolve_gpu(spec, gpus))


# Everything an engine process must not inherit from VCB's own setup: its
# configuration (VCB_*), and access tokens (every model used here is public).
_SECRET_ENV = {"HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN",
               "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "MISTRAL_API_KEY"}

# Switches that keep an engine from talking to the network on its own.
_PRIVACY_ENV = {
    "HF_HUB_DISABLE_TELEMETRY": "1",
    "DO_NOT_TRACK": "1",
    "ANONYMIZED_TELEMETRY": "False",
    "GRADIO_ANALYTICS_ENABLED": "False",
    "VLLM_NO_USAGE_STATS": "1",
    "VLLM_DO_NOT_TRACK": "1",
}


def worker_env(base: Mapping[str, str], target: Target, allow_download: bool = False) -> dict[str, str]:
    """Environment of an engine process.

    * VCB's settings and .env never reach it, so VCB's configuration cannot
      change what is measured and its secrets are not exposed to engine code.
    * It sees exactly one GPU, numbered by PCI bus like nvidia-smi.
    * It is offline unless downloads were explicitly allowed.
    """
    env = {k: v for k, v in base.items()
           if not k.upper().startswith("VCB_") and k.upper() not in _SECRET_ENV}
    env["VCB_ENV_FILE"] = os.path.join(tempfile.gettempdir(), "vcb-tts-bench-no-env-file")  # does not exist
    env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    env["CUDA_VISIBLE_DEVICES"] = target.visible_devices
    env["PYTHONUTF8"] = "1"
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env.update(_PRIVACY_ENV)
    for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE"):
        if allow_download:
            env.pop(key, None)
        else:
            env[key] = "1"
    # `-m tts_bench.worker` finds the package through PYTHONPATH.
    extra = [p for p in (base.get("PYTHONPATH"),) if p]
    env["PYTHONPATH"] = os.pathsep.join([str(TOOLS_DIR), *extra])
    return env


# -- what to run ---------------------------------------------------------------

@dataclass(frozen=True)
class Job:
    sentence: Sentence
    run: int          # 0 = warm-up
    counted: bool


def build_jobs(sentences: list[Sentence], runs: int, warmup_runs: int) -> list[Job]:
    """Warm-up first, then run 1 of every sentence, run 2 of every sentence...
    Round-robin order keeps slow drift (thermals, clocks) spread over all
    sentences instead of piling it onto the last ones."""
    jobs = [Job(sentences[0], 0, False) for _ in range(warmup_runs)]
    jobs += [Job(s, r, True) for r in range(1, runs + 1) for s in sentences]
    return jobs


@dataclass
class VariantPlan:
    spec: EngineSpec
    mode: str                     # "default" (built-in voice) or "clone" (reference clip)
    target: Target
    python: Path
    options: dict
    voice_wav: str                # "" for the built-in voice
    reference_text: str
    skip: str | None = None       # why this variant cannot run

    @property
    def variant(self) -> str:
        return variant_name(self.spec.key, self.mode)


@dataclass
class Plan:
    variants: list[VariantPlan]
    gpus: list[GpuInfo]
    notes: list[str] = field(default_factory=list)

    def runnable(self) -> list[VariantPlan]:
        return [v for v in self.variants if not v.skip]


def _skip_reason(cfg: BenchConfig, spec: EngineSpec, mode: str, python: Path) -> str | None:
    if spec.linux_only and os.name == "nt":
        return "its runtime has no Windows build: run this tool inside WSL2 (see `list`)"
    if not python.is_file():
        return f"interpreter not found: {python} (set [engines.{spec.key}] python)"
    options = cfg.settings_for(spec.key).options
    for name in spec.required_options:
        folder = options.get(name)
        if not folder:
            return f"set [engines.{spec.key}.options] {name} (the engine's source folder; see `list`)"
        path = Path(str(folder)).expanduser()
        if not (path if path.is_absolute() else REPO_ROOT / path).is_dir():
            return f"[engines.{spec.key}.options] {name} is not a folder: {folder}"
    for name in spec.requires_files:
        path = resolve_path(name)
        if not path.is_file():
            return f"not installed: {path} is missing ({install_hint(spec)})"
    if mode == "clone":
        if not cfg.reference_clip:
            return "no reference clip: set [voice] reference_clip"
        if not cfg.reference_clip.is_file():
            return f"reference clip not found: {cfg.reference_clip}"
        if spec.needs_reference_text and not cfg.reference_text:
            return "needs the clip's exact transcript: set [voice] reference_text"
    return None


def build_plan(cfg: BenchConfig, specs: list[EngineSpec], modes: list[str] | None = None,
               gpus: list[GpuInfo] | None = None) -> Plan:
    """Decide which (engine, voice mode) variants run and where. Nothing is
    started and no engine code is imported."""
    gpus = list_gpus() if gpus is None else gpus
    plan = Plan([], gpus)
    for spec in specs:
        wanted = [m for m in spec.modes if modes is None or m in modes]
        if not wanted:
            plan.notes.append(f"{spec.key}: nothing to run (it supports only: {', '.join(spec.modes)})")
            continue
        target = resolve_target(cfg.gpu_for(spec.key), gpus)
        python = cfg.python_for(spec.key)
        for mode in wanted:
            plan.variants.append(VariantPlan(
                spec=spec, mode=mode, target=target, python=python,
                options=dict(cfg.settings_for(spec.key).options),
                voice_wav=str(cfg.reference_clip) if mode == "clone" and cfg.reference_clip else "",
                reference_text=cfg.reference_text if mode == "clone" else "",
                skip=_skip_reason(cfg, spec, mode, python),
            ))
    return plan


def conditions(cfg: BenchConfig, sentences: list[Sentence]) -> dict:
    """Everything that must match for two engines to be comparable."""
    return {
        "sentences_sha": sentences_fingerprint(sentences),
        "n_sentences": len(sentences),
        "runs_per_sentence": cfg.runs_per_sentence,
        "warmup_runs": cfg.warmup_runs,
        "timeout_s": cfg.timeout_s,
        "seed": cfg.seed,
    }


def voice_info(cfg: BenchConfig) -> dict | None:
    clip = cfg.reference_clip
    if not clip or not clip.is_file():
        return None
    return {"clip": clip.name, "clip_sha": hashlib.sha256(clip.read_bytes()).hexdigest()[:16],
            "text": cfg.reference_text}


# -- engine process ------------------------------------------------------------

class WorkerTimeout(Exception):
    pass


class WorkerDied(Exception):
    pass


def kill_tree(proc: subprocess.Popen) -> None:
    """Kill a process and everything it started (an engine may run its own
    server or helper processes), then reap it."""
    if os.name == "nt":
        if proc.poll() is None:
            subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)],
                           capture_output=True, creationflags=_NO_WINDOW)
    else:
        # The worker leads its own process group (start_new_session), so the group
        # is swept even after the worker has exited and left a server behind
        # (e.g. the watchdog's hard exit): such a server would keep holding VRAM.
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
    if proc.poll() is None:
        try:
            proc.kill()
        except OSError:
            pass
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass


class WorkerProcess:
    """One engine process and its JSON-lines pipe (protocol in worker.py)."""

    def __init__(self, cmd: list[str], env: dict[str, str], cwd: Path, log_path: Path) -> None:
        self.cmd, self.env, self.cwd, self.log_path = cmd, env, cwd, log_path
        self.proc: subprocess.Popen | None = None
        self._events: queue.Queue = queue.Queue()
        self._stderr_tail: collections.deque = collections.deque(maxlen=40)
        self._threads: list[threading.Thread] = []

    def start(self) -> None:
        extra = {} if os.name == "nt" else {"start_new_session": True}
        self.proc = subprocess.Popen(
            self.cmd, cwd=self.cwd, env=self.env, stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=_NO_WINDOW, **extra)
        for target in (self._pump_stdout, self._pump_stderr):
            thread = threading.Thread(target=target, daemon=True)
            thread.start()
            self._threads.append(thread)

    # Reader threads: the pipes are drained continuously so a chatty engine can
    # never block on a full pipe.
    def _pump_stdout(self) -> None:
        assert self.proc and self.proc.stdout
        for raw in self.proc.stdout:
            try:
                self._events.put(json.loads(raw))
            except ValueError:
                self._stderr_tail.append("[stdout] " + raw.decode("utf-8", "replace").rstrip())
        self._events.put(None)                   # end of stream: the process is gone

    def _pump_stderr(self) -> None:
        assert self.proc and self.proc.stderr
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "ab") as log:
            log.write(f"--- worker started {_now_iso()}: {' '.join(self.cmd)}\n".encode())
            for raw in self.proc.stderr:
                log.write(raw)
                log.flush()
                self._stderr_tail.append(raw.decode("utf-8", "replace").rstrip())

    def describe_death(self) -> str:
        rc = self.proc.poll() if self.proc else None
        tail = [line for line in self._stderr_tail if line.strip()][-1:] or ["(no output)"]
        return f"engine process exited (code {rc}); last output: {tail[0][:300]}"

    def send(self, message: dict) -> None:
        assert self.proc and self.proc.stdin
        try:
            self.proc.stdin.write(json.dumps(message).encode("utf-8") + b"\n")
            self.proc.stdin.flush()
        except OSError as exc:                   # includes BrokenPipeError
            raise WorkerDied(self.describe_death()) from exc

    def wait_event(self, accept: Callable[[dict], bool], timeout: float) -> dict:
        """The next event that `accept` approves; WorkerTimeout after `timeout`
        seconds, WorkerDied if the process ends first."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise WorkerTimeout()
            try:
                event = self._events.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if event is None:
                self._events.put(None)           # later waits must fail fast too
                raise WorkerDied(self.describe_death())
            if accept(event):
                return event

    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def kill(self) -> None:
        if self.proc is not None:
            kill_tree(self.proc)
        self._join()

    def close(self) -> dict | None:
        """Ask the engine to unload and exit; kill it if it does not. Returns its goodbye
        message (None when it never sent one)."""
        bye = None
        try:
            self.send({"op": "quit"})
            bye = self.wait_event(lambda e: e.get("event") == "bye", UNLOAD_TIMEOUT_S)
        except (WorkerDied, WorkerTimeout):
            pass
        try:
            assert self.proc
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            pass
        self.kill()
        return bye

    def _join(self) -> None:
        for thread in self._threads:
            thread.join(timeout=3)


# -- progress output -----------------------------------------------------------

class _Progress:
    """One status line per synthesis: rewritten in place on a terminal and
    appended line by line when the output is piped to a file."""

    def __init__(self, total: int) -> None:
        self.total = total
        self.n = 0
        self.failed = 0
        self.live = sys.stdout.isatty()
        self.width = max(40, shutil.get_terminal_size((100, 20)).columns - 1)

    def tick(self, row: dict) -> None:
        self.n += 1
        ok = row["status"] == "ok"
        self.failed += 0 if ok else 1
        label = "warm-up" if not row["counted"] else f"run {row['run']}"
        head = f"  {self.n:>3}/{self.total}  {row['sentence_id']:<4} {label:<7}"
        if ok:
            text = f"{head} {row['total_s']:6.2f} s   rtf {row['rtf']:.2f}"
        else:
            text = f"{head} FAILED  {row['error_type']}: {row['error']}"
        if self.failed and ok:
            text += f"   ({self.failed} failed)"
        if not self.live:
            print(text)
        elif ok:
            sys.stdout.write("\r" + text[: self.width].ljust(self.width))
            sys.stdout.flush()
        else:                                    # keep failures visible
            sys.stdout.write("\r" + " " * self.width + "\r")
            print(text[: self.width])

    def finish(self) -> None:
        if self.live and self.n:
            sys.stdout.write("\r" + " " * self.width + "\r")
            sys.stdout.flush()


# -- one engine, one voice mode ------------------------------------------------

class _Session:
    def __init__(self, plan: VariantPlan, cfg: BenchConfig, sentences: list[Sentence],
                 monitor: VramMonitor, allow_download: bool, gpus: list[GpuInfo] | None = None) -> None:
        self.plan, self.cfg, self.sentences, self.monitor = plan, cfg, sentences, monitor
        self.allow_download = allow_download
        self.gpus = list(gpus or [])             # every GPU the monitor watches, for the per-GPU report
        self.out_dir = cfg.output_dir / plan.variant
        self.env = worker_env(os.environ, plan.target, allow_download)
        self.cmd = [str(plan.python), "-m", "tts_bench.worker"]
        self.worker: WorkerProcess | None = None
        self.rows: list[dict] = []
        self.loads: list[dict] = []
        self.network_blocked: list[str] = []     # what the engine tried to reach and was refused
        self.kills = 0
        self.aborted: str | None = None
        self.baseline: dict[int, float] = {}
        self.vram: dict = {}
        self.started_at = _now_iso()
        self.finished_at: str | None = None
        self.versions: dict = {}
        self.progress = _Progress(0)
        self._req_id = 0

    # ... preparing the output folder
    def _prepare_dir(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        for path in self.out_dir.iterdir():     # leftovers of an earlier run of this variant
            if path.is_file() and (_STALE_FILES.match(path.name) or path.name in {"raw.json", "worker.log"}):
                path.unlink()

    def _tts_config(self) -> dict:
        """The registry's extra TTSConfig fields, with "$voice_wav" and "$reference_text"
        replaced by this variant's own values ("" for the built-in voice)."""
        plan = self.plan
        values = {"$voice_wav": plan.voice_wav, "$reference_text": plan.reference_text}
        return {key: values.get(value, value) if isinstance(value, str) else value
                for key, value in plan.spec.tts_config.items()}

    def _init_message(self) -> dict:
        plan = self.plan
        return {"op": "init", "module": plan.spec.module, "cls": plan.spec.cls,
                "tag_converter": plan.spec.tag_converter, "options": plan.options,
                "device": plan.target.device, "voice_wav": plan.voice_wav,
                "reference_text": plan.reference_text,
                "vcb_native": plan.spec.vcb_native, "tts_config": self._tts_config(),
                "offline": not self.allow_download,
                "repo_root": str(REPO_ROOT), "tools_dir": str(TOOLS_DIR)}

    def _note_blocked(self, attempts) -> list[str]:
        """Remember what the network guard refused (each attempt once) and return the new ones."""
        new = [a for a in attempts or [] if a not in self.network_blocked]
        self.network_blocked.extend(new)
        return new

    # ... VRAM helpers
    def _gpu_peaks(self, t0: float, t1: float) -> dict[int, int]:
        return {g: w["max"] for g, w in self.monitor.window(t0, t1).items()}

    def _measure_baseline(self) -> None:
        """Idle memory of every GPU just before this engine starts."""
        if not self.monitor.available:
            return
        time.sleep(1.5)                          # let the previous engine's memory settle
        now = time.time()
        self.baseline = {g: s["median"] for g, s in self.monitor.window(now - 1.4, now).items()}

    def _wait_for_release(self) -> None:
        """After the engine exited: has its VRAM gone back to the baseline?"""
        gpu = self.plan.target.gpu
        if not self.monitor.available or gpu is None or gpu.index not in self.baseline:
            return
        deadline = time.time() + RELEASE_WAIT_S
        while True:
            now = time.time()
            recent = self.monitor.window(now - 0.5, now).get(gpu.index)
            residual = (recent["max"] - self.baseline[gpu.index]) if recent else None
            if residual is not None and residual <= RELEASE_TOLERANCE_MIB:
                break
            if now >= deadline:
                break
            time.sleep(0.25)
        self.vram["released"] = residual is not None and residual <= RELEASE_TOLERANCE_MIB
        self.vram["residual_mib"] = residual

    # ... starting and stopping the engine
    def _spawn(self) -> bool:
        cfg = self.cfg
        t0 = time.time()
        attempt: dict = {"ok": False, "started_at": _now_iso()}
        worker = WorkerProcess(self.cmd, self.env, REPO_ROOT, self.out_dir / "worker.log")
        worker.start()
        self.worker = worker
        event = None
        try:
            worker.send(self._init_message())
            event = worker.wait_event(lambda e: e.get("event") in {"ready", "load_error"}, cfg.load_timeout_s)
        except WorkerTimeout:
            attempt.update(error_type="LoadTimeout",
                           error=f"model not ready after {cfg.load_timeout_s:.0f} s")
        except WorkerDied as exc:
            attempt.update(error_type="WorkerCrashed", error=str(exc))
        if event and event["event"] == "ready":
            attempt.update(ok=True, load_s=event["load_s"])
            self.versions = {k: event[k] for k in ("python", "torch", "cuda_runtime", "gpu_name",
                                                   "cuda_visible_devices", "sample_rate",
                                                   "supports_tags") if k in event}
        elif event:
            attempt.update(error_type=event.get("error_type", "LoadError"), error=event.get("error", ""))
        blocked = list(event.get("blocked_network") or []) if event else []
        if blocked:
            attempt["network_blocked"] = blocked
        attempt["wall_s"] = time.time() - t0
        attempt["vram_peak_mib"] = self._gpu_peaks(t0, time.time())
        self.loads.append(attempt)
        if attempt["ok"]:
            again = "engine restarted, " if len(self.loads) > 1 else ""
            print(f"   {again}ready in {attempt['load_s']:.1f} s")
        else:
            print(f"   load failed: {attempt['error_type']}: {attempt['error']}")
            worker.kill()
            self.worker = None
        for note in self._note_blocked(blocked):
            print(f"   kept off the network: {note}")
        return attempt["ok"]

    def _kill_worker(self) -> None:
        if self.worker is not None:
            self.worker.kill()
        self.worker = None
        self.kills += 1

    def _ensure_worker(self) -> bool:
        """A running engine, (re)started if needed. After a restart, one
        uncounted synthesis pays the start-up costs again."""
        if self.kills > self.cfg.max_restarts:
            self.aborted = f"engine restarted {self.kills - 1} times (max_restarts = {self.cfg.max_restarts})"
            return False
        restarted = bool(self.loads)
        if not self._spawn():
            return False
        if restarted:
            self._synthesize(Job(self.sentences[0], 0, False), record=False)
        return True

    # ... one synthesis
    def _row(self, job: Job, started: float, **fields) -> dict:
        row = {
            "engine": self.plan.spec.key, "variant": self.plan.variant, "mode": self.plan.mode,
            "sentence_id": job.sentence.id, "category": job.sentence.category,
            "intent": job.sentence.intent, "run": job.run, "counted": job.counted,
            "status": "ok", "error_type": "", "error": "",
            "ttfa_s": None, "total_s": None, "audio_s": None, "rtf": None,
            "n_chunks": None, "sample_rate": None, "text_sent": "",
            "tags_in": len(job.sentence.tags), "tags_dropped": None, "wav": "",
            "vram_peak_mib": {}, "torch_peak_alloc_mib": None, "torch_peak_reserved_mib": None,
            "started_at": datetime.fromtimestamp(started).astimezone().isoformat(timespec="milliseconds"),
        }
        row.update(fields)
        if row["status"] == "ok" and row["audio_s"]:
            row["rtf"] = row["total_s"] / row["audio_s"]
        return row

    @staticmethod
    def _from_result(event: dict, limit: float, counted: bool, rel_wav: str) -> dict:
        keep = ("ttfa_s", "total_s", "audio_s", "n_chunks", "sample_rate", "text_sent", "tags_in",
                "tags_dropped", "torch_peak_alloc_mib", "torch_peak_reserved_mib")
        fields = {k: event[k] for k in keep if k in event}
        if not event.get("ok"):
            return {**fields, "status": "error", "error_type": event.get("error_type", "Error"),
                    "error": event.get("error", "")}
        if event.get("wav"):
            fields["wav"] = rel_wav
        if counted and event["total_s"] > limit:       # finished, but too late
            return {**fields, "status": "timeout", "error_type": "TimeoutError",
                    "error": f"synthesis took {event['total_s']:.1f} s (limit {limit:g} s)"}
        return fields

    def _synthesize(self, job: Job, record: bool = True) -> None:
        cfg = self.cfg
        limit = cfg.timeout_s if job.counted else cfg.warmup_timeout_s
        self._req_id += 1
        req_id = self._req_id
        name = f"{job.sentence.id}_{job.run}.wav"
        request = {"op": "synth", "id": req_id, "text": job.sentence.text, "seed": cfg.seed + job.run,
                   "save_to": str(self.out_dir / name) if record else None, "timeout_s": limit}
        started = time.time()
        assert self.worker is not None
        try:
            self.worker.send(request)
            event = self.worker.wait_event(
                lambda e: e.get("event") == "result" and e.get("id") == req_id, limit + RESULT_GRACE_S)
            fields = self._from_result(event, limit, job.counted, f"{self.plan.variant}/{name}")
        except WorkerTimeout:
            self._kill_worker()
            fields = {"status": "timeout", "error_type": "TimeoutError", "total_s": limit,
                      "error": f"no audio after {limit:g} s; the engine process was killed and restarted"}
        except WorkerDied as exc:
            self._kill_worker()
            fields = {"status": "error", "error_type": "WorkerCrashed", "error": str(exc),
                      "total_s": time.time() - started}
        ended = time.time()
        if not record:
            return
        row = self._row(job, started, **fields)
        row["vram_peak_mib"] = self._gpu_peaks(started, ended)
        self._add(row)

    def _add(self, row: dict) -> None:
        self.rows.append(row)
        self.progress.tick(row)
        self._write_raw(complete=False)

    def _fail_remaining(self, jobs: collections.deque, error_type: str, error: str) -> None:
        while jobs:
            job = jobs.popleft()
            self._add(self._row(job, time.time(), status="error", error_type=error_type, error=error))

    # ... the whole session
    def run(self) -> Path:
        plan, cfg = self.plan, self.cfg
        self._prepare_dir()
        jobs = collections.deque(build_jobs(self.sentences, cfg.runs_per_sentence, cfg.warmup_runs))
        self.progress = _Progress(len(jobs))
        self._measure_baseline()
        session_start = time.time()
        print(f"\n== {plan.variant}: {plan.spec.title}, "
              f"{'reference clip' if plan.mode == 'clone' else 'built-in voice'}, on {plan.target.label}")
        print(f"   loading the model (engine log: {self.out_dir / 'worker.log'})")
        try:
            while jobs:
                if self.worker is None and not self._ensure_worker():
                    # Cannot (re)start the engine: every remaining run is a failure.
                    last = self.loads[-1]
                    if last["ok"]:
                        self._fail_remaining(jobs, "EngineAborted", self.aborted or "")
                    elif len(self.loads) == 1:
                        self._fail_remaining(jobs, f"LoadError:{last['error_type']}", last["error"])
                    else:
                        self._fail_remaining(jobs, f"EngineAborted:{last['error_type']}", last["error"])
                    break
                if self.worker is None:          # the re-warm-up killed the engine again
                    continue
                self._synthesize(jobs.popleft())
        finally:
            self.progress.finish()
            session_end = time.time()
            if self.worker is not None:
                bye = self.worker.close()        # unload, then exit
                self._note_blocked((bye or {}).get("blocked_network"))
                self.worker = None
            self._wait_for_release()
            self.finished_at = _now_iso()
            self._summarise_vram(session_start, session_end)
            # Every run, successful or not, leaves a row; missing rows mean an interrupted run.
            path = self._write_raw(complete=not jobs)
        self._print_outcome()
        return path

    def _summarise_vram(self, t0: float, t1: float) -> None:
        if not self.monitor.available:
            return
        peak = self._gpu_peaks(t0, t1)
        load_peak: dict[int, int] = {}
        for load in self.loads:
            for g, v in load.get("vram_peak_mib", {}).items():
                load_peak[g] = max(load_peak.get(g, 0), v)
        self.vram.update(
            baseline_mib=self.baseline, peak_mib=peak, load_peak_mib=load_peak,
            delta_mib={g: peak[g] - self.baseline[g] for g in peak if g in self.baseline},
        )

    def _print_outcome(self) -> None:
        counted = [r for r in self.rows if r["counted"]]
        ok = sum(r["status"] == "ok" for r in counted)
        text = f"   done: {ok} ok, {len(counted) - ok} failed of {len(counted)} counted runs"
        gpu = self.plan.target.gpu
        if gpu is not None and gpu.index in self.vram.get("delta_mib", {}):
            text += (f"; GPU {gpu.index} peak {self.vram['peak_mib'][gpu.index] / 1024:.1f} GB "
                     f"(+{self.vram['delta_mib'][gpu.index] / 1024:.1f} GB over idle)")
            if self.vram.get("released") is False:
                text += f"; WARNING: {self.vram['residual_mib']:.0f} MiB still in use after unloading"
            elif self.vram.get("released"):
                text += "; unloaded cleanly"
        print(text)

    # ... persistence
    def _write_raw(self, complete: bool) -> Path:
        plan = self.plan
        gpu = plan.target.gpu
        data = {
            "schema": SCHEMA_VERSION,
            "engine": plan.spec.key, "title": plan.spec.title,
            "variant": plan.variant, "mode": plan.mode,
            "gpu": ({"spec": plan.target.spec, "index": gpu.index, "name": gpu.name,
                     "total_mib": gpu.total_mib} if gpu else {"spec": "cpu"}),
            "gpus_seen": [{"index": g.index, "name": g.name, "total_mib": g.total_mib}
                          for g in self.gpus] if self.monitor.available else [],
            "python": str(plan.python), "versions": self.versions, "options": plan.options,
            "conditions": conditions(self.cfg, self.sentences),
            "voice": voice_info(self.cfg) if plan.mode == "clone" else None,
            "started_at": self.started_at, "finished_at": self.finished_at,
            "complete": complete, "restarts": max(0, self.kills), "aborted": self.aborted,
            "offline": not self.allow_download, "network_blocked": self.network_blocked,
            "loads": self.loads, "vram": self.vram, "rows": self.rows,
        }
        path = self.out_dir / "raw.json"
        tmp = path.with_suffix(".json.part")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        return path


# -- the whole benchmark -------------------------------------------------------

def preflight(plan: Plan) -> None:
    """Print the GPUs and what each engine will use, and flag busy GPUs."""
    print("GPUs:")
    for g in plan.gpus:
        print(f"  {g.label}: {g.used_mib / 1024:.1f} GB in use")
    for v in plan.runnable():
        gpu = v.target.gpu
        if gpu and gpu.used_mib > 0.25 * gpu.total_mib:
            print(f"  WARNING: {gpu.label} is already {gpu.used_mib / 1024:.1f} GB full; "
                  f"{v.variant} results will be affected. Close what is using it.")
            break


def run_plan(plan: Plan, cfg: BenchConfig, sentences: list[Sentence], *,
             allow_download: bool = False) -> list[Path]:
    """Run every runnable variant, one after the other (never in parallel, so
    engines cannot disturb each other's timings). Variants that cannot run
    (`VariantPlan.skip`) are left out; the caller reports why."""
    runnable = plan.runnable()
    if not runnable:
        return []
    preflight(plan)
    monitor = VramMonitor()
    if not (monitor.start() and monitor.wait_for_samples(5.0)):
        print("WARNING: nvidia-smi is not available, so VRAM will not be measured")
        monitor.available = False
    raw_files: list[Path] = []
    try:
        for variant in runnable:
            raw_files.append(_Session(variant, cfg, sentences, monitor, allow_download, plan.gpus).run())
    finally:
        monitor.stop()
    return raw_files
