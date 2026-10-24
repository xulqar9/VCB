"""IndexTTS-2.5 (bilibili) voice cloning with real emotional delivery.

The most expressive engine: besides cloning the voice .wav it ACTS an emotion
per sentence (happy, sad, angry, ...). The LLM picks the mood with inline tags
(<excited>, <sad>, ...) that VCB teaches it via `prompt_instructions`; each tag
maps to IndexTTS's 8-emotion vector and stays in effect until the next one.

IndexTTS pins Python <3.12, torch 2.8 and an older transformers, so it can't
share VCB's environment. The launcher installs it into its own isolated runtime
(.runtime/indextts) when this engine is selected, and this class drives
indextts_worker.py there as a subprocess over pipes.
"""

import asyncio
import collections
import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
from concurrent.futures import Future
from pathlib import Path

import numpy as np

from .. import gpu
from ..config import TTSConfig, resolve_path
from .base import TTSEngine

log = logging.getLogger("vcb.tts.indextts")

# Copies of the model to run at most (one per GPU, ~6.4 GB each). Two already
# render faster than playback; a copy sharing a GPU with another barely helps.
MAX_COPIES = 2

WORKER = Path(__file__).with_name("indextts_worker.py")

# IndexTTS emotion vector order:
#   happy, angry, sad, afraid, disgusted, melancholic, surprised, calm
# A vector summing below 1 keeps some of the voice sample's own natural style.
# None = no emotion steering (the voice sample's own delivery).
MOODS: dict[str, list[float] | None] = {
    "happy":     [0.60, 0, 0, 0, 0, 0, 0.00, 0.20],
    "excited":   [0.68, 0, 0, 0, 0, 0, 0.28, 0.00],
    "playful":   [0.40, 0, 0, 0, 0, 0, 0.08, 0.32],
    "sad":       [0.00, 0, 0.48, 0, 0, 0.24, 0, 0.24],
    "angry":     [0.00, 0.56, 0, 0, 0.16, 0, 0.08, 0.00],
    "surprised": [0.08, 0, 0, 0, 0, 0, 0.48, 0.08],
    "afraid":    [0.00, 0, 0.08, 0.48, 0, 0, 0.08, 0.00],
    "calm":      None,
}

PROMPT_INSTRUCTIONS = (
    "\n\nYOUR VOICE\n"
    "Your voice carries real emotion, so let it show how you actually feel. Start "
    "every reply with ONE mood tag, and add a new one only when your mood genuinely "
    "shifts mid-reply. Moods: " + ", ".join(f"<{m}>" for m in MOODS) + ".\n"
    "The tag is silent - it only sets your tone of voice, it is never spoken. These "
    "mood tags are allowed in angle brackets in addition to any other tags you use.\n"
    'Example: "<excited> No way, you actually did it! <playful> Took you long enough, '
    'though."'
)

_MOOD_TAG = re.compile(r"<\s*(" + "|".join(MOODS) + r")\s*>", re.IGNORECASE)
_ANY_TAG = re.compile(r"<[^>\n]*>")


def split_moods(text: str, mood: str | None) -> tuple[list[tuple[str | None, str]], str | None]:
    """Split text at mood tags into (mood, words) segments. `mood` is the mood
    still in effect from earlier text; returns the segments and the new
    current mood (tags carry over to the following chunks)."""
    segments: list[tuple[str | None, str]] = []
    pos = 0
    for m in _MOOD_TAG.finditer(text):
        segments.append((mood, text[pos:m.start()]))
        mood, pos = m.group(1).lower(), m.end()
    segments.append((mood, text[pos:]))
    cleaned = []
    for seg_mood, words in segments:
        words = " ".join(_ANY_TAG.sub(" ", words).split())
        if any(c.isalnum() for c in words):
            cleaned.append((seg_mood, words))
    return cleaned, mood


def emotion_vector(mood: str | None, strength: float) -> list[float] | None:
    vec = MOODS.get(mood or "")
    if vec is None:
        return None
    scaled = [max(0.0, v * strength) for v in vec]
    total = sum(scaled)
    return [v / total for v in scaled] if total > 1.0 else scaled


def runtime_python(runtime_dir: str) -> Path:
    root = Path(resolve_path(runtime_dir))
    exe = "Scripts/python.exe" if sys.platform == "win32" else "bin/python"
    return root / "venv" / exe


def trim_voice_sample(src: Path, cache_dir: Path, max_s: float = 13.0,
                      min_s: float = 7.0) -> Path:
    """IndexTTS hard-cuts the voice sample at 15 s, often mid-word, which makes
    the cloned voice rougher. Cut long samples at the quietest moment between
    min_s and max_s instead (a pause between phrases); short ones pass through."""
    import soundfile as sf

    audio, sr = sf.read(str(src), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if len(audio) <= max_s * sr:
        return src
    hop = int(0.02 * sr)
    lo, hi = int(min_s * sr), int(max_s * sr)
    frames = audio[lo:hi][: (hi - lo) // hop * hop].reshape(-1, hop)
    quiet = int(np.argmin(np.sqrt((frames ** 2).mean(axis=1))))
    cut = lo + quiet * hop + hop // 2
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{src.stem}.{int(src.stat().st_mtime)}.trimmed.wav"
    if not out.is_file():
        sf.write(str(out), audio[:cut], sr)
    return out


def choose_gpus(setting: str, tts_device: str, free_gb: dict[int, float],
                cuda_visible: str | None, need_gb: float,
                max_copies: int = MAX_COPIES) -> list[int] | None:
    """Physical GPUs (nvidia-smi numbering) to run one IndexTTS copy on each.

    setting: "auto" = the GPU VCB placed the voice on, plus other visible GPUs
    with room for another copy (a second copy lets the next part of a reply
    render while the current one plays); or explicit numbers like "1" / "0,1".
    Returns None when GPUs can't be listed (no nvidia-smi): one copy on
    tts_device, as for any other engine."""
    if not free_gb:
        return None
    usable = gpu.visible_gpus(cuda_visible, list(free_gb)) or sorted(free_gb)
    picked = [int(x) for x in re.findall(r"\d+", setting or "")]
    picked = [g for g in dict.fromkeys(picked) if g in free_gb]
    if picked:
        return picked
    logical = int(tts_device.split(":")[1]) if tts_device.startswith("cuda:") else 0
    primary = usable[logical] if logical < len(usable) else usable[0]
    extras = sorted((g for g in usable if g != primary and free_gb[g] >= need_gb),
                    key=lambda g: -free_gb[g])
    return [primary, *extras][:max_copies]


class _Copy:
    """One IndexTTS worker process (one model copy on one GPU)."""

    def __init__(self, cmd: list[str], env: dict[str, str] | None, label: str) -> None:
        self.label = label
        self._lock = threading.Lock()    # one request/response at a time
        self._tail: collections.deque[str] = collections.deque(maxlen=40)
        flags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.PIPE, env=env, creationflags=flags)
        threading.Thread(target=self._drain, daemon=True, name=f"indextts-{label}").start()

    def _drain(self) -> None:
        assert self.proc.stderr is not None
        for raw in iter(self.proc.stderr.readline, b""):
            line = raw.decode("utf-8", "replace").rstrip()
            if line:
                self._tail.append(line)
                log.debug("worker %s: %s", self.label, line)

    def read_header(self) -> dict:
        assert self.proc.stdout is not None
        line = self.proc.stdout.readline()
        if not line:
            tail = "\n".join(self._tail) or "(no output)"
            raise RuntimeError(f"IndexTTS worker ({self.label}) exited unexpectedly. "
                               f"Last output:\n{tail}")
        return json.loads(line)

    def request(self, text: str, emo: list[float] | None) -> np.ndarray:
        """Blocking round trip, under the lock so the pipe protocol never desyncs."""
        with self._lock:
            assert self.proc.stdin is not None and self.proc.stdout is not None
            self.proc.stdin.write((json.dumps({"text": text, "emo": emo}) + "\n").encode())
            self.proc.stdin.flush()
            header = self.read_header()
            if "error" in header:
                log.warning("indextts (%s): %s", self.label, header["error"])
                return np.zeros(0, dtype=np.float32)
            data = self.proc.stdout.read(4 * int(header["n"]))
            return np.frombuffer(data, dtype="<f4").astype(np.float32)

    def stop(self) -> None:
        with self._lock:
            try:
                if self.proc.stdin:
                    self.proc.stdin.write(b'{"quit": true}\n')
                    self.proc.stdin.flush()
                self.proc.wait(timeout=5)
            except Exception:
                self.proc.kill()


class IndexTTSEngine(TTSEngine):
    sample_rate = 22_050
    supports_tags = False                # <laugh> etc. become spoken "Haha!"
    mood_tags = frozenset(MOODS)
    prompt_instructions = PROMPT_INSTRUCTIONS
    # Each request is generated on its own, so a lone "Hey!" or half a sentence
    # comes out flat and choppy. Whole sentences, several per chunk after the first.
    chunker_overrides = {"first_chunk_min_chars": 60, "min_chars": 140,
                         "first_chunk_clause_split": False}

    def __init__(self, cfg: TTSConfig) -> None:
        self.cfg = cfg
        self._copies: list[_Copy] = []
        self._jobs: queue.Queue = queue.Queue()
        self._alive = 0
        self._alive_lock = threading.Lock()
        # Chunks queued for speech, rendering ahead: (text, futures per mood segment).
        self._pending: collections.deque[tuple[str, list[Future]]] = collections.deque()
        self._mood: str | None = None

    def _command(self) -> list[str]:
        python = runtime_python(self.cfg.indextts_dir)
        if not python.is_file():
            raise RuntimeError(
                "the IndexTTS voice engine isn't installed yet. Start VCB with VCB.bat: "
                "it installs IndexTTS automatically when it's the selected voice "
                "(one-time, ~8 GB download)."
            )
        model_dir = Path(resolve_path(self.cfg.indextts_dir)) / "checkpoints"
        voice = resolve_path(self.cfg.voice_wav)
        if not (voice and Path(voice).is_file()):
            raise RuntimeError(
                f"IndexTTS clones a voice sample, but {self.cfg.voice_wav!r} wasn't found. "
                "Pick a .wav in the VCB settings (Voice sample)."
            )
        voice = trim_voice_sample(Path(voice), Path(resolve_path(self.cfg.indextts_dir)) / "voices")
        cmd = [str(python), str(WORKER), "--model-dir", str(model_dir),
               "--voice", str(voice), "--steps", str(self.cfg.indextts_steps)]
        if not self.cfg.indextts_bf16:
            cmd.append("--fp32")
        return cmd

    def _launch_plan(self) -> list[tuple[list[str], dict[str, str] | None, str]]:
        """(extra args, env, label) per copy. Each copy only sees its own GPU:
        IndexTTS puts some modules on the default GPU whatever device it's given,
        so two copies would otherwise end up sharing (and fighting over) one."""
        gpus = choose_gpus(self.cfg.indextts_gpus, self.cfg.device, gpu.query_free_gb(),
                           os.environ.get("CUDA_VISIBLE_DEVICES"),
                           gpu.TTS_VRAM_GB["indextts"] + 0.6)
        if gpus is None:
            device = self.cfg.device if self.cfg.device.startswith("cuda:") else "cuda:0"
            return [(["--device", device], None, device)]
        return [(["--device", "cuda:0"],
                 dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), CUDA_DEVICE_ORDER="PCI_BUS_ID"),
                 f"GPU {g}") for g in gpus]

    async def load(self) -> None:
        base = self._command()
        plan = self._launch_plan()
        log.info("starting IndexTTS on %s (first start after install takes a minute)",
                 " + ".join(label for _, _, label in plan))

        def _start(args, env, label):
            copy = _Copy([*base, *args], env, label)
            try:
                return copy, copy.read_header()
            except Exception:
                copy.proc.kill()
                raise

        results = await asyncio.gather(
            *(asyncio.to_thread(_start, *p) for p in plan), return_exceptions=True)
        for (_, _, label), res in zip(plan, results):
            if isinstance(res, BaseException):
                log.warning("IndexTTS copy on %s failed to start: %s", label, res)
                continue
            copy, ready = res
            self.sample_rate = int(ready["sr"])
            self._copies.append(copy)
            log.info("indextts ready on %s (sr=%d, ~%.1f GB GPU)", label,
                     self.sample_rate, ready.get("vram_gb", 0.0))
        if not self._copies:
            first = next(r for r in results if isinstance(r, BaseException))
            raise first
        self._alive = len(self._copies)
        for copy in self._copies:
            threading.Thread(target=self._serve, args=(copy,), daemon=True,
                             name=f"indextts-serve-{copy.label}").start()
        log.info("indextts voice: %s, %d cop%s", self.cfg.voice_wav, len(self._copies),
                 "y" if len(self._copies) == 1 else "ies")

    def _serve(self, copy: _Copy) -> None:
        """Feed one copy from the shared job queue: whichever copy is free takes
        the next part, so two GPUs render consecutive parts side by side."""
        while True:
            job = self._jobs.get()
            if job is None:
                break
            words, vec, fut = job
            if not fut.set_running_or_notify_cancel():
                continue                 # dropped by a barge-in before it started
            try:
                fut.set_result(copy.request(words, vec))
            except Exception as exc:     # this copy died: fail the job, stop serving
                fut.set_exception(exc)
                break
        with self._alive_lock:
            self._alive -= 1
            last = self._alive == 0
        if last:                         # nobody left to serve: fail what's waiting
            while True:
                try:
                    job = self._jobs.get_nowait()
                except queue.Empty:
                    break
                if job is not None and job[2].set_running_or_notify_cancel():
                    job[2].set_exception(RuntimeError("all IndexTTS workers stopped"))

    def _submit(self, text: str) -> list[Future]:
        if self._alive <= 0:
            raise RuntimeError("IndexTTS is not running")
        segments, self._mood = split_moods(text, self._mood)
        futures = []
        for mood, words in segments:
            fut: Future = Future()
            self._jobs.put((words, emotion_vector(mood, self.cfg.indextts_emotion_strength), fut))
            futures.append(fut)
        return futures

    def prepare(self, text: str) -> None:
        """Start rendering a chunk as soon as it's queued for speech (called in
        speaking order), so a free copy works ahead of playback."""
        self._pending.append((text, self._submit(text)))

    def cancel_pending(self) -> None:
        """Barge-in: drop every chunk rendered ahead that hasn't started yet."""
        while self._pending:
            for fut in self._pending.popleft()[1]:
                fut.cancel()

    def _take(self, text: str) -> list[Future]:
        while self._pending:
            queued, futures = self._pending.popleft()
            if queued == text:
                return futures
            for fut in futures:          # skipped by the speech loop (stale)
                fut.cancel()
        return self._submit(text)

    async def synthesize_stream(self, text: str):
        for fut in self._take(text):
            audio = await asyncio.wrap_future(fut)
            if len(audio):
                yield audio

    async def synthesize(self, text: str) -> np.ndarray:
        parts = [a async for a in self.synthesize_stream(text)]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    async def close(self) -> None:
        self.cancel_pending()
        copies, self._copies = self._copies, []
        for _ in copies:
            self._jobs.put(None)
        await asyncio.gather(*(asyncio.to_thread(c.stop) for c in copies))