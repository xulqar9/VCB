"""Speaker playback: async queue of float32 chunks -> sounddevice callback.

Internal ring-style buffer is fed from an asyncio queue by a pump task; the
audio callback only slices numpy arrays (no allocation-heavy work). flush()
discards everything instantly for barge-in. Device failures trigger reopen.
"""

import asyncio
import logging
import threading
import time
from collections import deque

import numpy as np
import sounddevice as sd

from ..config import AudioConfig
from .devices import resolve_device

log = logging.getLogger("vcb.audio.playback")


class SpeakerPlayback:
    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg
        self._queue: asyncio.Queue[tuple[int, np.ndarray]] = asyncio.Queue(maxsize=32)
        self._buffer: deque[np.ndarray] = deque()
        self._buffer_samples = 0
        self._lock = threading.Lock()
        self._generation = 0                      # bumped on flush; stale chunks dropped
        self._priming = True                      # hold output until pre-roll buffered
        self._prime_samples = int(
            cfg.output_sample_rate * cfg.playback_prime_ms / 1000
        )
        self._prime_grace = cfg.playback_prime_ms / 1000.0  # release if no new audio
        self._last_grow = 0.0
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: sd.OutputStream | None = None
        self._pump_task: asyncio.Task | None = None
        self._device_error = asyncio.Event()
        self._supervisor: asyncio.Task | None = None

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._supervisor = asyncio.create_task(self._supervise(), name="spk-supervisor")
        self._pump_task = asyncio.create_task(self._pump(), name="spk-pump")

    async def stop(self) -> None:
        for t in (self._pump_task, self._supervisor):
            if t:
                t.cancel()
                try:
                    await t
                except asyncio.CancelledError:
                    pass
        self._close_stream()

    async def play(self, chunk: np.ndarray) -> None:
        """Queue float32 mono audio at cfg.output_sample_rate."""
        await self._queue.put((self._generation, chunk))

    def flush(self) -> None:
        """Drop all pending audio immediately (barge-in)."""
        self._generation += 1
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        with self._lock:
            self._buffer.clear()
            self._buffer_samples = 0
            self._priming = True                  # re-arm pre-roll for the next reply

    @property
    def buffered_seconds(self) -> float:
        with self._lock:
            samples = self._buffer_samples
        return samples / self.cfg.output_sample_rate

    @property
    def generation(self) -> int:
        """Bumped by flush(); lets producers detect a barge-in and stop."""
        return self._generation

    @property
    def is_playing(self) -> bool:
        return self.buffered_seconds > 0 or not self._queue.empty()

    async def wait_until_drained(self, poll_s: float = 0.05) -> None:
        while self.is_playing:
            await asyncio.sleep(poll_s)

    # -- internals ------------------------------------------------------

    async def _pump(self) -> None:
        while True:
            gen, chunk = await self._queue.get()
            if gen != self._generation:
                continue                          # flushed while queued
            chunk = np.asarray(chunk, dtype=np.float32).reshape(-1)
            # Backpressure: don't hold more than ~3 s in the device buffer.
            while self.buffered_seconds > 3.0:
                await asyncio.sleep(0.05)
                if gen != self._generation:
                    break
            if gen != self._generation:
                continue
            with self._lock:
                self._buffer.append(chunk)
                self._buffer_samples += len(chunk)
                self._last_grow = time.monotonic()

    def _callback(self, outdata: np.ndarray, frames: int, _time, status) -> None:
        if status and status.output_underflow:
            log.debug("output underflow")
        out = outdata[:, 0]
        filled = 0
        with self._lock:
            # Pre-roll: stay silent until a lead buffer exists, so a streamed
            # laugh/sentence doesn't underrun and stutter on its first samples.
            if self._priming:
                enough = self._buffer_samples >= self._prime_samples
                # short reply that never reaches the lead: release once audio stops arriving
                stalled = (
                    self._buffer_samples > 0
                    and time.monotonic() - self._last_grow > self._prime_grace
                )
                if enough or stalled:
                    self._priming = False
                else:
                    out[:] = 0.0
                    return
            while filled < frames and self._buffer:
                head = self._buffer[0]
                take = min(len(head), frames - filled)
                out[filled:filled + take] = head[:take]
                filled += take
                if take == len(head):
                    self._buffer.popleft()
                else:
                    self._buffer[0] = head[take:]
                self._buffer_samples -= take
        out[filled:] = 0.0

    def _open_stream(self) -> None:
        self._stream = sd.OutputStream(
            samplerate=self.cfg.output_sample_rate,
            blocksize=0,                          # let PortAudio choose
            channels=1,
            dtype="float32",
            device=resolve_device(self.cfg.output_device, "output"),
            callback=self._callback,
            finished_callback=self._on_finished,
        )
        self._stream.start()
        log.info("speaker stream open (device=%s)", self.cfg.output_device or "default")

    def _on_finished(self) -> None:
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._device_error.set)

    def _close_stream(self) -> None:
        if self._stream is not None:
            try:
                self._stream.abort()
                self._stream.close()
            except Exception:
                pass
            self._stream = None

    async def _supervise(self) -> None:
        backoff = 0.5
        while True:
            try:
                self._device_error.clear()
                self._open_stream()
                backoff = 0.5
                await self._device_error.wait()
                log.warning("speaker stream ended unexpectedly, reopening")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("speaker open failed (%s), retry in %.1fs", exc, backoff)
            self._close_stream()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 5.0)
