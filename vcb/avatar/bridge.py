"""Avatar bridge: streams TTS audio + conversation state/emotion to an external
3D avatar app (Unreal MetaHuman) over a local TCP socket.

Wire format - every message is a 5-byte header followed by its payload:
    >B  msg_type   0 = JSON control (UTF-8 text), 1 = audio (float32 LE, mono)
    >I  length     payload length in bytes
Control payloads are small JSON objects: {"type": "...", ...}. On connect the
server sends {"type":"audio_format","sample_rate":..,"channels":1,"dtype":"float32"}
so the client knows how to play the audio frames that follow.

Sends are fire-and-forget: if the avatar client is absent or slow, frames are
dropped (oldest first) rather than stalling the real-time voice pipeline. Only
one avatar client is served at a time; a new connection replaces the old one.
"""

import asyncio
import contextlib
import json
import logging
import struct

import numpy as np

from ..config import AvatarConfig

log = logging.getLogger("vcb.avatar")

_JSON = 0
_AUDIO = 1
_HEADER = struct.Struct(">BI")


class AvatarBridge:
    def __init__(self, cfg: AvatarConfig) -> None:
        self.cfg = cfg
        self.sample_rate = 24_000            # overwritten with the TTS rate at start()
        self._server: asyncio.AbstractServer | None = None
        self._writer: asyncio.StreamWriter | None = None
        # Control frames (state/speech/interrupt) are rare and must never be
        # dropped, so they get an unbounded queue and priority over audio.
        self._ctrl: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue()
        # Audio is high-volume and real-time: bounded, drop-oldest when full.
        self._audio_q: asyncio.Queue[tuple[int, bytes]] = asyncio.Queue(cfg.queue_max)
        self._wake = asyncio.Event()
        self._sender: asyncio.Task | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(
            self._on_client, self.cfg.host, self.cfg.port
        )
        self._sender = asyncio.create_task(self._send_loop(), name="avatar-sender")
        log.info("avatar bridge listening on %s:%d", self.cfg.host, self.cfg.port)

    async def stop(self) -> None:
        if self._sender is not None:
            self._sender.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._sender
        if self._writer is not None:
            self._writer.close()
            with contextlib.suppress(Exception):
                await self._writer.wait_closed()
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()

    # -- public, fire-and-forget (safe to call from the hot voice loop) ----

    def state(self, name: str) -> None:
        """Announce the conversation state: idle | listening | thinking | speaking."""
        self._json({"type": "state", "state": name})

    def speech_start(self) -> None:
        self._json({"type": "speech_start"})

    def speech_chunk(self, text: str, emotions: list[str]) -> None:
        """Heads-up for the audio about to stream: the words and any emotion tags."""
        self._json({"type": "speech_chunk", "text": text, "emotions": emotions})

    def speech_end(self) -> None:
        self._json({"type": "speech_end"})

    def action(self, name: str) -> None:
        """Ask the avatar to perform a named body action (plays a montage in UE),
        e.g. "wave", "highfive", "backflip". Unknown names are ignored client-side."""
        self._json({"type": "action", "name": name})

    def interrupt(self) -> None:
        """Barge-in: the user cut in, so stop the mouth/gesture immediately."""
        # Drop the cancelled turn's audio still queued here, or it would stream
        # to the client right after the interrupt and restart the mouth.
        while not self._audio_q.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._audio_q.get_nowait()
        self._json({"type": "interrupt"})

    def speech_flush(self) -> None:
        """All audio for the turn has been shipped. Rides the AUDIO queue (not the
        priority control queue) so it arrives after the last audio frame and the
        client can flush its lip-sync pipeline tail immediately."""
        if self._writer is None:
            return
        payload = json.dumps({"type": "speech_flush"}).encode("utf-8")
        self._enqueue_audio((_JSON, payload))

    def audio(self, samples: np.ndarray) -> None:
        if self._writer is None:
            return
        arr = np.ascontiguousarray(samples, dtype=np.float32)
        # Split into short frames so the avatar gets a fresh loudness reading many
        # times a second and can drive lip-sync that tracks the speech as it plays.
        frame = max(1, self.sample_rate // 50)   # ~20 ms per frame
        for start in range(0, len(arr), frame):
            self._put_audio(arr[start:start + frame].tobytes())

    # -- internals --------------------------------------------------------

    def _json(self, obj: dict) -> None:
        if self._writer is None:
            return
        self._ctrl.put_nowait((_JSON, json.dumps(obj).encode("utf-8")))
        self._wake.set()

    def _put_audio(self, payload: bytes) -> None:
        self._enqueue_audio((_AUDIO, payload))

    def _enqueue_audio(self, item: tuple[int, bytes]) -> None:
        try:
            self._audio_q.put_nowait(item)
        except asyncio.QueueFull:
            # Stay real-time: drop the oldest audio frame, then enqueue the newest.
            with contextlib.suppress(asyncio.QueueEmpty):
                self._audio_q.get_nowait()
            with contextlib.suppress(asyncio.QueueFull):
                self._audio_q.put_nowait(item)
        self._wake.set()

    async def _send_loop(self) -> None:
        while True:
            # Control frames always take priority and are never dropped.
            if not self._ctrl.empty():
                msg_type, payload = self._ctrl.get_nowait()
            elif not self._audio_q.empty():
                msg_type, payload = self._audio_q.get_nowait()
            else:
                await self._wake.wait()
                self._wake.clear()
                continue
            writer = self._writer
            if writer is None:
                continue
            try:
                writer.write(_HEADER.pack(msg_type, len(payload)))
                writer.write(payload)
                await writer.drain()
            except Exception:
                log.warning("avatar client write failed; dropping connection")
                self._writer = None

    async def _on_client(self, reader: asyncio.StreamReader,
                         writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        log.info("avatar client connected: %s", peer)
        old = self._writer
        self._writer = writer
        if old is not None and old is not writer:
            with contextlib.suppress(Exception):
                old.close()
        # Fresh client starts clean: drop anything queued for the old one.
        for q in (self._ctrl, self._audio_q):
            while not q.empty():
                with contextlib.suppress(asyncio.QueueEmpty):
                    q.get_nowait()
        self._json({"type": "audio_format", "sample_rate": self.sample_rate,
                    "channels": 1, "dtype": "float32"})
        try:
            while await reader.read(1024):   # inbound is unused; read to spot EOF
                pass
        except Exception:
            pass
        finally:
            if self._writer is writer:
                self._writer = None
            with contextlib.suppress(Exception):
                writer.close()
            log.info("avatar client disconnected: %s", peer)
