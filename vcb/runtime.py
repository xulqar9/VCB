"""Process lifecycle: launch the llama.cpp server and a transcript window,
wait for readiness, and tear both down cleanly on exit.

The conversation transcript is written to a file (vcb.transcript); a second
console window shows it live, so the chat stays visually separate from the
system/STT logs in the main terminal.
"""

import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

from .config import AppConfig

log = logging.getLogger("vcb.runtime")

CREATE_NEW_CONSOLE = getattr(subprocess, "CREATE_NEW_CONSOLE", 0)


def _create_kill_on_close_job() -> int | None:
    """Windows Job Object that kills every assigned process when its last handle
    closes. We keep the handle open for the program's lifetime, so if this
    Python process dies for ANY reason (Ctrl+C, terminal kill, crash), the OS
    tears down llama-server and the transcript window instead of orphaning them.
    """
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    hjob = kernel32.CreateJobObjectW(None, None)
    if not hjob:
        return None

    class _BASIC(ctypes.Structure):
        _fields_ = [
            ("PerProcessUserTimeLimit", wintypes.LARGE_INTEGER),
            ("PerJobUserTimeLimit", wintypes.LARGE_INTEGER),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        ]

    class _IO(ctypes.Structure):
        _fields_ = [(n, ctypes.c_ulonglong) for n in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount",
        )]

    class _EXT(ctypes.Structure):
        _fields_ = [
            ("BasicLimitInformation", _BASIC),
            ("IoInfo", _IO),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        ]

    JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE = 0x2000
    JobObjectExtendedLimitInformation = 9
    info = _EXT()
    info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    ok = kernel32.SetInformationJobObject(
        hjob, JobObjectExtendedLimitInformation,
        ctypes.byref(info), ctypes.sizeof(info),
    )
    if not ok:
        kernel32.CloseHandle(hjob)
        return None
    return hjob


def _assign_to_job(job: int | None, proc: "subprocess.Popen") -> None:
    """Best-effort: put a child process under the kill-on-close job."""
    if job is None or sys.platform != "win32":
        return
    import ctypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    if not kernel32.AssignProcessToJobObject(job, int(proc._handle)):
        # Modern Windows supports nested jobs; a failure here just means we fall
        # back to the explicit shutdown() path, so log and carry on.
        log.debug("could not assign %s to job (err %d)", proc.pid, ctypes.get_last_error())


def setup_logging(level: str, transcript_path: str) -> Path:
    """Console gets system logs; the conversation goes to its own transcript
    file (written by vcb.transcript.Transcript), started fresh each session."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger("phonemizer").setLevel(logging.ERROR)

    path = Path(transcript_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("", encoding="utf-8")  # fresh transcript each session
    return path


def _resolve_binary(name: str) -> str:
    found = shutil.which(name)
    if found:
        return found
    local = Path(os.environ.get("LOCALAPPDATA", ""))
    # Default per-user install dir of the Ollama Windows installer.
    candidate = local / "Programs" / name.capitalize() / f"{name}.exe"
    if candidate.is_file():
        return str(candidate)
    # winget installs to a per-user Packages dir not always on a fresh PATH.
    winget = local / "Microsoft" / "WinGet" / "Packages"
    if winget.is_dir():
        for exe in winget.glob(f"**/{name}.exe"):
            return str(exe)
    return name  # let subprocess raise a clear error


def _http_ok(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status == 200
    except (urllib.error.URLError, ConnectionError, OSError):
        return False


def ollama_has_model(tags: dict, model: str) -> bool:
    """True if Ollama's /api/tags listing contains `model` (a bare name
    matches its ':latest' tag)."""
    want = model if ":" in model else f"{model}:latest"
    names = {m.get("name") or m.get("model") for m in tags.get("models", [])}
    return want in names or model in names


def _ollama_pull(base: str, model: str) -> None:
    log.info("downloading model %r through Ollama (first run only, this can "
             "take a while)...", model)
    req = urllib.request.Request(
        f"{base}/api/pull",
        data=json.dumps({"model": model, "stream": True}).encode(),
        headers={"Content-Type": "application/json"},
    )
    last_pct = -10
    with urllib.request.urlopen(req, timeout=None) as resp:
        for line in resp:
            if not line.strip():
                continue
            msg = json.loads(line)
            if "error" in msg:
                raise RuntimeError(f"Ollama could not pull {model!r}: {msg['error']}")
            total, done = msg.get("total"), msg.get("completed")
            if total and done is not None:
                pct = int(done * 100 / total)
                if pct >= last_pct + 10:
                    last_pct = pct
                    log.info("  %s: %d%% of %.1f GB", msg.get("status", "pulling"),
                             pct, total / 1e9)
    log.info("model %r ready", model)


def _ollama_post(base: str, path: str, body: dict, timeout: float | None = 30) -> bytes:
    req = urllib.request.Request(
        f"{base}{path}", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def ollama_sized_name(model: str, num_ctx: int) -> str:
    """Name of VCB's copy of `model` with a fixed context, e.g.
    'llama3.1:8b' -> 'llama3.1:8b-vcb8k' (a bare name gets ':latest')."""
    tagged = model if ":" in model else f"{model}:latest"
    return f"{tagged}-vcb{num_ctx // 1024}k"


def _ollama_sized_copy(base: str, model: str, num_ctx: int) -> str:
    """Create (or refresh) a copy of the model whose default context is
    num_ctx. The OpenAI-compatible endpoint can't pass num_ctx per request,
    and Ollama's default context can use several times the GPU memory VCB
    needs. The copy shares the original's files, so it costs no disk space."""
    name = ollama_sized_name(model, num_ctx)
    try:
        _ollama_post(base, "/api/create", {
            "model": name, "from": model,
            "parameters": {"num_ctx": num_ctx}, "stream": False,
        })
    except (urllib.error.URLError, OSError) as exc:
        log.warning("couldn't set Ollama's context size (%s); using %r as is -- "
                    "it may use more GPU memory than needed. Updating Ollama "
                    "usually fixes this.", exc, model)
        return model
    log.info("using %r (%r with a %d-token context)", name, model, num_ctx)
    return name


def ollama_fit_warning(entry: dict) -> str | None:
    """Warning text if a loaded model (an /api/ps entry) isn't fully on the GPU."""
    size, vram = entry.get("size") or 0, entry.get("size_vram") or 0
    if size <= 0 or vram >= size * 0.98:
        return None
    pct = int(vram * 100 / size)
    return (f"only {pct}% of model {entry.get('name')!r} fits in GPU memory; the "
            "rest runs on the CPU, so replies will be slow. Pick a smaller "
            "model in VCB settings (it recommends one that fits), use the "
            "Preset voice to free GPU memory, or close other GPU-heavy apps.")


def _ollama_preload_and_check(base: str, model: str, system_prompt: str = "") -> None:
    """Load the model now (so the first reply doesn't wait for it) and warn
    if it spilled over to the CPU. A one-token chat on the system prompt also
    pays the runner's first-inference setup and leaves the prompt's prefix in
    the KV cache, so the opening reply starts in milliseconds, not seconds."""
    log.info("loading %r into memory...", model)
    try:
        _ollama_post(base, "/api/generate", {"model": model}, timeout=600)
        with urllib.request.urlopen(f"{base}/api/ps", timeout=10) as resp:
            running = json.loads(resp.read()).get("models", [])
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log.warning("couldn't preload %r (%s); it will load on the first reply", model, exc)
        return
    for entry in running:
        if entry.get("name") == model or entry.get("model") == model:
            warning = ollama_fit_warning(entry)
            if warning:
                log.warning(warning)
            else:
                log.info("model loaded fully on the GPU (%.1f GB)",
                         entry.get("size_vram", 0) / 2**30)
    try:
        messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) \
            + [{"role": "user", "content": "Hi"}]
        _ollama_post(base, "/api/chat", {"model": model, "messages": messages, "stream": False,
                                         "options": {"num_predict": 1}}, timeout=120)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        log.debug("model warm-up skipped (%s)", exc)


class ManagedProcesses:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self._server: subprocess.Popen | None = None
        self._window: subprocess.Popen | None = None
        # Kept open for the program's lifetime so children die if we're hard-killed.
        self._job = _create_kill_on_close_job()

    def start_server(self) -> None:
        backend = self.cfg.backend
        if backend == "llama_cpp":
            self._start_llama_cpp()
        elif backend == "ollama":
            self._start_ollama()
        elif backend == "cloud":
            log.info("backend=cloud: no local process to launch (%s)",
                     self.cfg.llm.base_url)
            if not self.cfg.llm.api_key:
                raise RuntimeError(
                    f"backend=cloud but no API key is set ({self.cfg.cloud.api_key_env}). "
                    "Open VCB settings and paste your provider's API key."
                )
        else:
            raise ValueError(f"unknown backend {backend!r}")

    def _start_llama_cpp(self) -> None:
        scfg = self.cfg.server
        if not scfg.auto_launch:
            return
        if not Path(scfg.model_path).is_file():
            raise RuntimeError(f"llama.cpp model file not found: {scfg.model_path}")
        binary = _resolve_binary(scfg.binary)
        cmd = [
            binary,
            "-m", scfg.model_path,
            "--host", scfg.host,
            "--port", str(scfg.port),
            "-ngl", str(scfg.n_gpu_layers),
            "-c", str(scfg.ctx_size),
            *scfg.extra_args,
        ]
        # Pin the server to specific GPU(s) (VCB_LLM_CUDA_DEVICES) so it can
        # stay off a card reserved for TTS or rendering.
        env = None
        if scfg.cuda_visible_devices is not None:
            env = os.environ.copy()
            # Force PCI bus ordering so the index matches nvidia-smi (otherwise the
            # CUDA runtime may rank a newer card ahead of an older one).
            env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            env["CUDA_VISIBLE_DEVICES"] = scfg.cuda_visible_devices
            log.info("pinning llama-server to CUDA device(s) %s", scfg.cuda_visible_devices)
        log.info("launching llama-server: %s", " ".join(cmd))
        # New process group so we can signal the whole tree on shutdown.
        self._server = subprocess.Popen(
            cmd,
            creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            env=env,
        )
        _assign_to_job(self._job, self._server)
        self._wait_for_health(f"http://{scfg.host}:{scfg.port}/health", scfg.startup_timeout_s)

    def _start_ollama(self) -> None:
        ocfg = self.cfg.ollama
        # Ollama auto-detects and spreads across whatever GPU(s) are available --
        # no CUDA_VISIBLE_DEVICES pinning needed, unlike the llama.cpp path.
        base = ocfg.base_url.removesuffix("/v1").rstrip("/")
        health = f"{base}/api/version"
        # Normally Ollama already runs in the background (desktop app/service).
        # If not, start it ourselves when it's installed.
        if ocfg.auto_launch or not _http_ok(health):
            binary = _resolve_binary(ocfg.binary)
            if not ocfg.auto_launch and shutil.which(binary) is None \
                    and not Path(binary).is_file():
                raise RuntimeError(
                    "Ollama isn't running and isn't installed. Install it from "
                    "https://ollama.com/download, or open VCB settings and pick "
                    "the Cloud backend instead."
                )
            log.info("launching ollama serve")
            self._server = subprocess.Popen(
                [binary, "serve"],
                creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
            )
            _assign_to_job(self._job, self._server)
            self._wait_for_health(health, ocfg.startup_timeout_s)
        else:
            log.info("backend=ollama: using running Ollama at %s", base)
        with urllib.request.urlopen(f"{base}/api/tags", timeout=10) as resp:
            tags = json.loads(resp.read())
        if not ollama_has_model(tags, ocfg.model):
            _ollama_pull(base, ocfg.model)
        if ocfg.num_ctx > 0:
            self.cfg.llm.model = _ollama_sized_copy(base, ocfg.model, ocfg.num_ctx)
        _ollama_preload_and_check(base, self.cfg.llm.model, self.cfg.llm.system_prompt)

    def _wait_for_health(self, url: str, timeout_s: float, required: bool = True) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._server is not None and self._server.poll() is not None:
                raise RuntimeError(
                    f"server process exited early (code {self._server.returncode})"
                )
            try:
                with urllib.request.urlopen(url, timeout=2) as resp:
                    if resp.status == 200:
                        log.info("backend ready at %s", url)
                        return
            except (urllib.error.URLError, ConnectionError, OSError):
                time.sleep(0.5)
        msg = f"backend did not become healthy in time ({url})"
        if required:
            raise TimeoutError(msg)
        log.warning(msg)

    def start_transcript_window(self, transcript_path: Path, assistant_name: str = "") -> None:
        if not self.cfg.server.transcript_window or sys.platform != "win32":
            return
        cmd = [sys.executable, "-m", "vcb.transcript", str(transcript_path), assistant_name]
        self._window = subprocess.Popen(
            cmd, creationflags=CREATE_NEW_CONSOLE,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        _assign_to_job(self._job, self._window)
        log.info("transcript window opened (tailing %s)", transcript_path)

    def shutdown(self) -> None:
        for name, proc in (("transcript window", self._window), ("llm backend", self._server)):
            if proc is None or proc.poll() is not None:
                continue
            log.info("stopping %s", name)
            try:
                if proc is self._server:
                    proc.send_signal(signal.CTRL_BREAK_EVENT)
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                else:
                    proc.terminate()
            except Exception:
                proc.kill()
        self._server = self._window = None
