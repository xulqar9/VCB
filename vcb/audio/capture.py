"""Microphone capture: sounddevice callback -> asyncio queue of int16 frames.

Runs the PortAudio callback on its own thread; frames are handed to the event
loop via call_soon_threadsafe. On device errors the stream is torn down and
reopened with backoff, so unplugging/replugging a mic never kills the session.
"""

import asyncio
import logging

import numpy as np
import sounddevice as sd

from ..config import AudioConfig
from .devices import resolve_device

log = logging.getLogger("vcb.audio.capture")


class MicCapture:
    def __init__(self, cfg: AudioConfig) -> None:
        self.cfg = cfg
        self.frames: asyncio.Queue[np.ndarray] = asyncio.Queue(cfg.mic_queue_max_frames)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stream: sd.InputStream | None = None
        self._task: asyncio.Task | None = None
        self._device_error = asyncio.Event()
        self.dropped_frames = 0

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._supervise(), name="mic-supervisor")

    async def stop(self) -> None:
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        self._close_stream()

    async def get_frame(self) -> np.ndarray:
        """Next int16 mono frame of cfg.frame_samples samples."""
        return await self.frames.get()

    # -- internals ------------------------------------------------------

    def _callback(self, indata: np.ndarray, _frames: int, _time, status) -> None:
        if status and status.input_overflow:
            log.debug("input overflow")
        frame = indata[:, 0].copy()
        loop = self._loop
        if loop is not None:
            loop.call_soon_threadsafe(self._enqueue, frame)

    def _enqueue(self, frame: np.ndarray) -> None:
        if self.frames.full():
            # Drop oldest so live audio stays current (never block the device).
            try:
                self.frames.get_nowait()
                self.dropped_frames += 1
            except asyncio.QueueEmpty:
                pass
        self.frames.put_nowait(frame)

    def _open_stream(self) -> None:
        self._stream = sd.InputStream(
            samplerate=self.cfg.input_sample_rate,
            blocksize=self.cfg.frame_samples,
            channels=1,
            dtype="int16",
            device=resolve_device(self.cfg.input_device, "input"),
            callback=self._callback,
            finished_callback=self._on_finished,
        )
        self._stream.start()
        log.info("mic stream open (device=%s)", self.cfg.input_device or "default")

    def _on_finished(self) -> None:
        # Called from PortAudio thread when the stream dies unexpectedly.
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
                log.warning("mic stream ended unexpectedly, reopening")
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("mic open failed (%s), retry in %.1fs", exc, backoff)
            self._close_stream()
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 5.0)
