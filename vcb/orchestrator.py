"""Pipeline orchestrator.

Task layout:
  mic loop        - mic frames -> VAD -> speech events (always running)
  event loop      - reacts to SpeechStart (barge-in) / SpeechEnd (new turn)
  turn task       - per utterance: STT -> LLM stream -> chunker -> tts queue
  tts loop        - text chunks -> audio -> playback (always running)

Barge-in cancels the turn task, flushes the TTS queue and playback buffer.
Cancellation also bumps the playback/turn generation so any work already queued
or in flight is superseded: queued TTS chunks are dropped before synthesis and a
transcript finishing after the barge-in is discarded. STT/TTS run in worker
threads that can't be interrupted, so the current in-flight synthesis and any
running transcription still finish in the background; their output is discarded
rather than voiced.
Errors inside a turn are logged and spoken past; the session never dies.
The always-on loops run under a supervisor that restarts them with backoff if
one crashes, so a single unexpected error can't end the session.
"""

import asyncio
import contextlib
import logging
import re
import time
from collections.abc import Awaitable, Callable
from typing import NamedTuple

from .audio import MicCapture, SpeakerPlayback
from .audiofx import apply_gain_limit, silence, trailing_pause_seconds
from .avatar import AvatarBridge
from .chunker import SentenceChunker
from .config import USER_NAME, AppConfig
from .conversation import ConversationManager
from .llm.base import LLMConfigError, LLMEngine
from .memory import Embedder, MemoryStore, consolidate
from .metrics import TurnMetrics
from .sanitize import (
    extract_action_tags,
    extract_emotion_tags,
    sanitize_for_speech,
    strip_speech_tags,
)
from .skills import (
    SkillContext,
    SkillRegistry,
    build_skills,
    looks_like_tool_call,
    parse_tool_call,
)
from .state import Phase, PipelineSnapshot
from .transcript import Transcript, WordPacer
from .stt.base import STTEngine
from .tts.base import TTSEngine
from .vad import SpeechEnd, SpeechStart, VADEvent, VADSegmenter

log = logging.getLogger("vcb.orchestrator")

_SENTINEL = None  # end-of-response marker on the tts queue


class _TTSItem(NamedTuple):
    """A queued text chunk. It carries its own playback generation and the
    metrics of the turn that produced it, so audio is never attributed to
    whichever turn happens to be current when it's finally synthesized."""
    gen: int
    text: str
    metrics: TurnMetrics | None = None

# Fragments that only appear in the instructions we inject, never in genuine
# spoken replies. A small model sometimes parrots its nudge/system text on
# AI-led turns; any chunk carrying one of these is dropped before it's voiced.
_ECHO_MARKERS = (
    "leading the conversation",
    "gone quiet",
    "one or two spoken sentences",
    "one or two natural spoken",
    "restate your instructions",
    "no tool names",
    "tool names or raw json",
    "read json",
    "convey emotion",
    "do not mention the tool",
    "do not mention them",
    "use the tools",
    "speaker label",
)


def _is_instruction_echo(text: str) -> bool:
    low = text.lower()
    return any(marker in low for marker in _ECHO_MARKERS)


_NAME_RE = re.compile(r"\s*You are ([A-Z][a-zA-Z]+)")


def _speaker_name(system_prompt: str) -> str:
    """Pull the persona's spoken name out of its prompt ("You are Iris, ...")."""
    m = _NAME_RE.match(system_prompt or "")
    return m.group(1) if m else "Assistant"


class Orchestrator:
    def __init__(
        self,
        cfg: AppConfig,
        stt: STTEngine,
        llm: LLMEngine,
        tts: TTSEngine,
        transcript_path: str | None = None,
    ) -> None:
        self.cfg = cfg
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.mic = MicCapture(cfg.audio)
        self.speaker = SpeakerPlayback(cfg.audio)
        self.avatar: AvatarBridge | None = (
            AvatarBridge(cfg.avatar) if cfg.avatar.enabled else None
        )
        self._avatar_speaking = False
        # UE buffers audio and plays it in real time; track when that buffered
        # audio will finish so we can time speech_end/idle to playback, not synth.
        self._avatar_play_end = 0.0
        # Same virtual playhead for local playback, used to time transcript words.
        self._play_end = 0.0
        self._avatar_end_task: asyncio.Task | None = None
        self.vad = VADSegmenter(cfg.vad, cfg.audio)
        self.convo = ConversationManager(cfg.llm)
        self._name = _speaker_name(cfg.llm.system_prompt)
        # Live chat transcript: words appear as they're heard (see transcript.py).
        self._transcript = Transcript(transcript_path, self._name)
        self._pacer = WordPacer(self._transcript)
        # Each queued chunk carries the playback generation it was created under;
        # a barge-in bumps that generation so stale chunks are dropped before we
        # waste synthesis on audio no one will hear (explicit supersede).
        self._tts_queue: asyncio.Queue[_TTSItem | None] = asyncio.Queue(cfg.tts.chunk_queue_max)
        self._turn_task: asyncio.Task | None = None
        # Bumped whenever a turn is cancelled/superseded. STT runs in a worker
        # thread that can't be cancelled, so a turn checks this after transcribe
        # to abandon a result the barge-in already made obsolete.
        self._turn_gen = 0
        # Metrics of the current turn, only used to flag it interrupted on
        # barge-in. The tts loop reads metrics from each queued item instead.
        self._metrics: TurnMetrics | None = None
        self._resume_pending = False       # last turn was cut off mid-sentence
        # Text of chunks the tts loop has actually played this turn (not just
        # generated), plus a signal fired once a reply's audio has fully drained.
        # Together these let a barge-in resume from what was heard.
        self._spoken: list[str] = []
        self._reply_drained = asyncio.Event()
        self._resume_budget = 0            # turns a cut-off thought stays resumable
        self._last_activity = time.monotonic()  # for AI-led idle detection
        self._user_speaking = False        # mic is mid-utterance right now
        self._session_id = time.strftime("%Y%m%d-%H%M%S")
        self._memory: MemoryStore | None = None
        self._embedder: Embedder | None = None
        if cfg.memory.enabled:
            self._memory = MemoryStore(cfg.memory.db_path, cfg.memory.ephemeral,
                                       cfg.memory.max_index_rows)
            self._embedder = Embedder(cfg.memory.embed_model, cfg.memory.query_instruction)
        self._consolidating = False
        self._consolidate_task: asyncio.Task | None = None
        self._shutdown_done = False
        self._tasks: list[asyncio.Task] = []
        self._skills: SkillRegistry | None = None
        self._tools_block: str | None = None
        if cfg.skills.enabled:
            self._skills = SkillRegistry(build_skills(cfg.skills.enabled_skills))
            self._tools_block = self._skills.prompt_block() or None

    async def run(self) -> None:
        log.info("loading models...")
        await asyncio.gather(
            self.stt.load(), self.llm.load(), self.tts.load(), self.vad.load()
        )
        if self._embedder is not None:
            await asyncio.to_thread(self._embedder.load)  # warm the embedder (CPU)
        # Playback must match whatever rate the chosen TTS engine produces.
        self.cfg.audio.output_sample_rate = self.tts.sample_rate
        await self.mic.start()
        await self.speaker.start()
        if self.avatar is not None:
            self.avatar.sample_rate = self.tts.sample_rate
            await self.avatar.start()
        loops: list[tuple[str, Callable[[], Awaitable[None]], Callable[[], None] | None]] = [
            ("mic-loop", self._mic_loop, None),
            ("event-loop", self._event_loop, None),
            # A dead tts loop would leave the current reply waiting forever on
            # its drain signal; release it so the turn can finish.
            ("tts-loop", self._tts_loop, self._reply_drained.set),
        ]
        if self.cfg.proactive.enabled:
            loops.append(("idle-loop", self._idle_loop, None))
        self._tasks = [
            asyncio.create_task(self._supervise(name, fn, on_crash), name=name)
            for name, fn, on_crash in loops
        ]
        log.info("ready - start talking")
        if self.cfg.proactive.enabled and self.cfg.proactive.greet_on_start:
            self._last_activity = time.monotonic()
            self._start_lead_turn(greeting=True)
        try:
            await asyncio.gather(*self._tasks)
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        if self._shutdown_done:        # run()'s finally and an external caller can both hit this
            return
        self._shutdown_done = True
        await self._cancel_turn()
        for t in self._tasks:
            t.cancel()
        for t in self._tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await t
        await self.mic.stop()
        await self.speaker.stop()
        if self.avatar is not None:
            await self.avatar.stop()
        # Final consolidation while the LLM is still up, so nothing is lost.
        if self._memory is not None:
            if self._consolidate_task and not self._consolidate_task.done():
                with contextlib.suppress(Exception):
                    await asyncio.wait_for(self._consolidate_task, timeout=20)
            with contextlib.suppress(Exception):
                await asyncio.wait_for(self._run_consolidation(), timeout=30)
        await asyncio.gather(
            self.stt.close(), self.llm.close(), self.tts.close(),
            return_exceptions=True,
        )
        if self._memory is not None:
            self._memory.close()
        self._transcript.close()

    # -- always-on loops --------------------------------------------------

    async def _supervise(self, name: str,
                         loop_fn: Callable[[], Awaitable[None]],
                         on_crash: Callable[[], None] | None = None) -> None:
        """Run an always-on loop, restarting it with exponential backoff if it
        crashes or exits. Repeated failures with no healthy run in between are
        treated as fatal so a permanently broken loop doesn't spin forever."""
        sup = self.cfg.supervision
        failures = 0
        while True:
            started = time.monotonic()
            try:
                await loop_fn()
                log.error("%s exited unexpectedly", name)
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("%s crashed", name)
            if on_crash is not None:
                with contextlib.suppress(Exception):
                    on_crash()
            if time.monotonic() - started >= sup.healthy_after_s:
                failures = 0                 # it ran fine for a while; fresh budget
            failures += 1
            if failures > sup.max_restarts:
                log.critical("%s failed %d times in a row, giving up", name, failures)
                raise RuntimeError(f"{name} kept failing")
            delay = min(sup.backoff_max_s,
                        sup.backoff_initial_s * 2 ** (failures - 1))
            log.warning("restarting %s in %.1fs (attempt %d/%d)",
                        name, delay, failures, sup.max_restarts)
            await asyncio.sleep(delay)

    def _snapshot(self) -> PipelineSnapshot:
        return PipelineSnapshot(
            user_speaking=self._user_speaking,
            turn_active=self._turn_active(),
            audio_playing=self.speaker.is_playing,
            avatar_speaking=self._avatar_speaking,
        )

    @property
    def phase(self) -> Phase:
        return self._snapshot().phase

    async def _mic_loop(self) -> None:
        while True:
            frame = await self.mic.get_frame()
            try:
                self.vad.process_frame(frame)
            except Exception:
                log.exception("VAD error, frame skipped")

    async def _event_loop(self) -> None:
        while True:
            event = await self.vad.events.get()
            try:
                await self._handle_speech_event(event)
            except asyncio.CancelledError:
                raise
            except Exception:
                # One malformed event must not tear down barge-in for the session.
                log.exception("event loop error, continuing")

    async def _handle_speech_event(self, event: VADEvent) -> None:
        if isinstance(event, SpeechStart):
            self._user_speaking = True
            self._last_activity = time.monotonic()
            if self.avatar is not None:
                self.avatar.state("listening")
            if self.cfg.barge_in and self._snapshot().assistant_active:
                log.info("barge-in: interrupting")
                await self._cancel_turn()
        elif isinstance(event, SpeechEnd):
            self._user_speaking = False
            self._last_activity = time.monotonic()
            await self._cancel_turn()
            if self.avatar is not None:
                self.avatar.state("thinking")
            self._metrics = TurnMetrics()
            self._metrics.mark("t_speech_end")
            self._turn_task = asyncio.create_task(
                self._run_turn(event.audio), name="turn"
            )

    async def _idle_loop(self) -> None:
        """AI-led mode: when nobody's talking and no reply is in flight, wait
        out the idle window, then let the assistant speak up on its own."""
        p = self.cfg.proactive
        while True:
            await asyncio.sleep(1.0)
            try:
                if self._snapshot().busy:
                    self._last_activity = time.monotonic()   # still busy; reset clock
                    continue
                if time.monotonic() - self._last_activity >= p.idle_seconds:
                    self._last_activity = time.monotonic()
                    self._start_lead_turn(greeting=False)
            except Exception:
                log.exception("idle loop error, continuing")

    async def _tts_loop(self) -> None:
        prosody = self.cfg.prosody
        play_local = self.avatar is None or self.cfg.avatar.local_playback
        pace_avatar = self.avatar is not None and not play_local
        while True:
            chunk = await self._tts_queue.get()
            if chunk is _SENTINEL:
                if self.avatar is not None and self._avatar_speaking:
                    # All audio is shipped: tell UE now so it can flush the tail
                    # still held inside Audio2Face (else the last word glitches).
                    self.avatar.speech_flush()
                    # UE is still playing buffered audio: schedule idle for when
                    # that playback drains (cancelled if the user barges in).
                    delay = max(0.0, self._avatar_play_end - time.monotonic()) \
                        if pace_avatar else 0.0
                    self._avatar_end_task = asyncio.create_task(
                        self._finish_avatar_speech(delay), name="avatar-end"
                    )
                # A turn isn't done when its text is queued -- only when its
                # audio has actually been heard. Block here until playback
                # drains, then release _speak_reply to commit the reply. A
                # barge-in cancels that wait instead, so only the spoken part
                # is recorded and the rest can be resumed.
                if play_local:
                    await self.speaker.wait_until_drained()
                elif pace_avatar:
                    remaining = self._avatar_play_end - time.monotonic()
                    if remaining > 0:
                        await asyncio.sleep(remaining)
                # Audio is done: show any words a slightly-off clock still holds.
                self._pacer.flush()
                self._transcript.end()
                self._reply_drained.set()
                continue
            chunk_gen, text, m = chunk
            gen = self.speaker.generation      # barge-in bumps this -> abandon chunk
            # Chunk queued before a barge-in: drop it without synthesizing.
            if chunk_gen != gen:
                continue
            t0 = time.perf_counter()
            if self.avatar is not None:
                self.avatar.speech_chunk(
                    strip_speech_tags(text), extract_emotion_tags(text)
                )
            agen = self.tts.synthesize_stream(text)
            completed = False
            chunk_start: float | None = None   # when this chunk's audio is heard
            chunk_secs = 0.0
            try:
                async for audio in agen:
                    if self.speaker.generation != gen:
                        break
                    if not len(audio):
                        continue
                    if prosody.enabled:
                        audio = apply_gain_limit(audio, prosody.gain, prosody.ceiling)
                    if m is not None and not m.t_first_audio:
                        m.buffer_delay_s = self.speaker.buffered_seconds
                        m.mark("t_first_audio")
                        m.log_report()
                    if self.avatar is not None and not self._avatar_speaking:
                        self._cancel_avatar_end()
                        self._avatar_speaking = True
                        self.avatar.state("speaking")
                        self.avatar.speech_start()
                    start = self._advance_clocks(len(audio), play_local, pace_avatar)
                    if chunk_start is None:
                        chunk_start = start
                    chunk_secs += len(audio) / self.tts.sample_rate
                    if play_local:
                        await self.speaker.play(audio)
                    if self.avatar is not None:
                        self.avatar.audio(audio)
                else:
                    completed = True
            except Exception:
                log.exception("TTS failed for chunk %r, skipping", text[:60])
            finally:
                with contextlib.suppress(Exception):
                    await agen.aclose()
                if m is not None:
                    m.tts_synth_times.append(time.perf_counter() - t0)
            # Record only what actually reached the speaker, so a later barge-in
            # resumes from what was heard, not from everything generated.
            if completed and self.speaker.generation == gen:
                self._spoken.append(text + " ")
                if chunk_start is not None:
                    self._pacer.schedule(strip_speech_tags(text).split(),
                                         chunk_start, chunk_secs)
            # Human cadence: pause between chunks sized by trailing punctuation.
            if completed and prosody.enabled and self.speaker.generation == gen:
                pause = trailing_pause_seconds(text, prosody)
                if pause > 0:
                    sil = silence(pause, self.tts.sample_rate)
                    self._advance_clocks(len(sil), play_local, pace_avatar)
                    if play_local:
                        await self.speaker.play(sil)
                    if self.avatar is not None:
                        self.avatar.audio(sil)

    def _advance_clocks(self, num_samples: int, play_local: bool,
                        pace_avatar: bool) -> float:
        """Push the virtual playheads forward by this audio's duration and
        return when it will start being heard (monotonic time)."""
        start = time.monotonic()
        if play_local:
            start = self._advance_clock("_play_end", num_samples)
        if pace_avatar:
            start = self._advance_clock("_avatar_play_end", num_samples)
        return start

    def _advance_clock(self, attr: str, num_samples: int) -> float:
        now = time.monotonic()
        start = max(now, getattr(self, attr))  # buffer had drained: restart from now
        setattr(self, attr, start + num_samples / self.tts.sample_rate)
        return start

    def _cancel_avatar_end(self) -> None:
        if self._avatar_end_task is not None:
            self._avatar_end_task.cancel()
            self._avatar_end_task = None

    async def _finish_avatar_speech(self, delay: float) -> None:
        """After the buffered audio finishes playing in UE, drop to idle."""
        # Extra guard so UE's passthrough buffer drains before the A2F stream is
        # closed by speech_end; otherwise the last fraction of a second is clipped.
        margin = max(0, self.cfg.avatar.speech_end_margin_ms) / 1000.0
        try:
            await asyncio.sleep(delay + margin)
        except asyncio.CancelledError:
            return
        if self.avatar is not None and self._avatar_speaking:
            self.avatar.speech_end()
            self.avatar.state("idle")
            self._avatar_speaking = False
        self._avatar_end_task = None

    # -- per-turn ----------------------------------------------------------

    def _turn_active(self) -> bool:
        return self._turn_task is not None and not self._turn_task.done()

    async def _cancel_turn(self) -> None:
        if self._metrics is not None and self._turn_active():
            self._metrics.interrupted = True
        self._turn_gen += 1                # supersede any in-flight STT/turn work
        if self._turn_active():
            self._turn_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._turn_task
        self._turn_task = None
        while not self._tts_queue.empty():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._tts_queue.get_nowait()
        self.speaker.flush()
        self._play_end = 0.0
        # Words not yet heard never reach the transcript; mark the cut.
        interrupted = self._pacer.has_pending
        self._pacer.cancel()
        self._transcript.end(interrupted=interrupted)
        if self.avatar is not None and self._avatar_speaking:
            self._cancel_avatar_end()
            self.avatar.interrupt()
            self._avatar_speaking = False

    async def _run_turn(self, audio) -> None:
        m = self._metrics
        turn_gen = self._turn_gen
        try:
            text = await self.stt.transcribe(audio, self.cfg.audio.input_sample_rate)
            # STT ran in an uncancellable worker thread; if a barge-in superseded
            # this turn while it ran, drop the now-obsolete transcript.
            if turn_gen != self._turn_gen:
                log.debug("transcript superseded by barge-in, discarding")
                return
            if m:
                m.mark("t_stt_done")
            if not text:
                log.debug("empty transcript, ignoring")
                return
            log.info("user: %s", text)
            self._transcript.user(text)
            self.convo.add_user(text)
            memory_block = await self._remember_user(text)

            # If we were cut off, keep the thought resumable across a few turns:
            # a brief backchannel shouldn't burn it, so it stays armed until the
            # model actually continues (a substantial reply) or the window runs out.
            hint = None
            resuming = False
            if self.cfg.resume.enabled and self._resume_pending:
                if self._resume_budget > 0:
                    self._resume_budget -= 1
                    hint = self.cfg.resume.hint
                    resuming = True
                    log.info("resume armed (%d turns left)", self._resume_budget)
                else:
                    self._resume_pending = False   # window expired without a resume

            messages = self.convo.messages(hint, memory_block, self._tools_block)
            await self._speak_reply(messages, m, resuming=resuming)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("turn failed")

    def _start_lead_turn(self, greeting: bool) -> None:
        """AI-led turn: the assistant speaks without any user input. Skipped if
        someone/something is already talking so we never talk over a real turn."""
        if self._snapshot().busy:
            return
        m = TurnMetrics()
        m.mark("t_speech_end")
        self._metrics = m
        p = self.cfg.proactive
        if greeting:
            cue = p.greeting_hint
        else:
            # If she just asked a question and is waiting on an answer, don't
            # barrel into a brand-new topic; ease off gently instead.
            last = self.convo.last_assistant()
            waiting = bool(last and last.rstrip().endswith("?"))
            cue = p.waiting_hint if waiting else p.hint
        # Deliver the nudge as a USER turn, not a trailing system message: this
        # model responds to user turns but tends to parrot trailing system text.
        # No tools_block: a lead turn is for chit-chat, not for firing tools
        # (the model would otherwise blurt the time or set random reminders).
        messages = self.convo.messages(None, None, None)
        messages.append({"role": "user", "content": cue})
        self._turn_task = asyncio.create_task(
            self._speak_reply(messages, m, temperature=p.lead_temperature,
                              max_tokens=p.lead_max_tokens),
            name="lead-turn",
        )

    async def _speak_reply(self, messages: list[dict[str, str]],
                           m: TurnMetrics | None,
                           temperature: float | None = None,
                           max_tokens: int | None = None,
                           resuming: bool = False) -> None:
        """Stream the model's reply, speak it, and record it. Shared by
        user-driven and AI-led turns; barge-in cancellation records the
        partial so the resume feature can pick it back up."""
        parts: list[str] = []
        self._spoken = []
        self._reply_drained.clear()
        try:
            try:
                await self._stream_reply(messages, m, parts,
                                         temperature=temperature,
                                         max_tokens=max_tokens)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                if isinstance(exc, LLMConfigError):
                    log.error("LLM request refused: %s", exc)   # no traceback: not a bug
                else:
                    log.exception("LLM stream failed")
                partial = "".join(parts).strip()
                if partial:
                    # A real mid-stream break: retries no longer restart once a
                    # token was voiced, so treat the partial as cut off. Keep it
                    # in context and let the next turn resume it -- don't bank it
                    # as a clean completed answer or feed it to long-term memory.
                    self.convo.add_assistant(partial, interrupted=True)
                    self._resume_pending = True
                    self._resume_budget = self.cfg.resume.window_turns
                else:
                    spoken = exc.spoken if isinstance(exc, LLMConfigError) \
                        else "Sorry, I hit a problem generating a response."
                    await self._enqueue_tts(spoken, m)
                await self._tts_queue.put(_SENTINEL)   # finalize avatar/playback
                return
            # Don't bank the reply until its audio has actually been heard; if
            # the LLM outran TTS, most of it is still only queued. A barge-in
            # cancels this wait and the spoken-only partial is recorded instead.
            await self._reply_drained.wait()
            spoken_text = "".join(parts).strip()
            if not spoken_text:
                return                       # nothing voiced (empty or all echo)
            log.info("assistant: %s", spoken_text)
            # A substantial reply on a resume turn means the cut-off thought was
            # actually continued, so the resume offer is spent; a short one (a
            # mere acknowledgement) leaves it armed for a later "continue".
            if resuming and len(spoken_text.split()) >= self.cfg.resume.resumed_min_words:
                self._resume_pending = False
            self.convo.add_assistant(spoken_text)
            await self._remember_assistant(spoken_text)
            self._maybe_consolidate()
        except asyncio.CancelledError:
            spoken_text = "".join(self._spoken).strip()
            self.convo.add_assistant(spoken_text, interrupted=True)
            self._resume_pending = bool(spoken_text)
            if self._resume_pending:            # fresh cut-off: reopen the window
                self._resume_budget = self.cfg.resume.window_turns
            raise

    async def _stream_reply(self, messages: list[dict[str, str]],
                            m: TurnMetrics | None, out: list[str],
                            temperature: float | None = None,
                            max_tokens: int | None = None) -> None:
        """Run the LLM with the skill/tool loop, pushing spoken chunks to the
        TTS queue and appending each to `out` so partial speech survives
        barge-in."""
        chunker = SentenceChunker(self.cfg.chunker)
        first_token = False
        spoke_final = False
        for _ in range(self.cfg.skills.max_tool_iters + 1):
            decided: str | None = None       # "tool" or "speak"
            head = ""                        # buffer until we can decide
            tool_text = ""
            raw = ""                         # everything streamed, for late tool detection
            async for delta in self.llm.stream(messages, temperature=temperature,
                                               max_tokens=max_tokens):
                raw += delta
                if not first_token:
                    first_token = True
                    if m:
                        m.mark("t_llm_first_token")
                if decided is None:
                    head += delta
                    if not head.lstrip():
                        continue
                    decided = ("tool" if self._skills
                               and looks_like_tool_call(head) else "speak")
                    if decided == "tool":
                        tool_text = head
                    else:
                        for chunk in chunker.push(head):
                            out.append(await self._push_spoken(chunk, m))
                    continue
                if decided == "tool":
                    tool_text += delta
                else:
                    for chunk in chunker.push(delta):
                        out.append(await self._push_spoken(chunk, m))

            # Tool call at the start (decided=tool) OR buried after prose the
            # model spoke first (decided=speak but raw still holds valid JSON).
            call = None
            if self._skills:
                call = parse_tool_call(tool_text if decided == "tool" else raw)
            if call is not None:
                log.info("tool: %s %s", call.name, call.args)
                result = await self._skills.dispatch(
                    call.name, call.args, self._skill_ctx()
                )
                log.info("tool result: %s", result)
                messages = messages + [
                    {"role": "assistant",
                     "content": (tool_text if decided == "tool" else raw).strip()},
                    {"role": "system", "content":
                        f"Tool '{call.name}' returned: {result}\n"
                        "Now answer the user in one or two natural spoken "
                        "sentences based on this. Do not mention the tool "
                        "or read JSON aloud."},
                ]
                chunker.reset()              # drop any buffered JSON, don't speak it
                continue                     # loop for the spoken reply
            if decided == "tool":            # looked like JSON but wasn't valid
                for chunk in chunker.push(tool_text):
                    out.append(await self._push_spoken(chunk, m))
            for chunk in chunker.flush():
                out.append(await self._push_spoken(chunk, m))
            spoke_final = True
            break
        else:
            log.warning("tool loop hit iteration cap without a spoken reply")
        if not spoke_final and not "".join(out).strip():
            await self._enqueue_tts("Sorry, I couldn't finish that.", m)
        await self._tts_queue.put(_SENTINEL)

    # -- skills ------------------------------------------------------------

    def _skill_ctx(self) -> SkillContext:
        return SkillContext(
            memory=self._memory,
            speak=self._speak,
            loop=asyncio.get_running_loop(),
        )

    async def _speak(self, text: str) -> None:
        """Enqueue text for immediate TTS (used by timers firing later)."""
        spoken = sanitize_for_speech(text, self.tts.supports_tags)
        if spoken:
            await self._enqueue_tts(spoken)

    async def _enqueue_tts(self, text: str, m: TurnMetrics | None = None) -> None:
        """Queue spoken text tagged with the current playback generation (so a
        later barge-in drops it before synthesis instead of voicing it late)
        and with the metrics of the turn it belongs to."""
        await self._tts_queue.put(_TTSItem(self.speaker.generation, text, m))

    async def _push_spoken(self, chunk: str, m: TurnMetrics | None) -> str:
        """Sanitize a chunk, enqueue it, and return it (with a trailing space)
        for accumulation. Empty after sanitizing -> skipped."""
        # Body-action tags are stripped by sanitize, so read them from the raw
        # chunk here and forward to the avatar before the words are voiced.
        if self.avatar is not None:
            for name in extract_action_tags(chunk):
                self.avatar.action(name)
        spoken = sanitize_for_speech(chunk, self.tts.supports_tags)
        if not spoken or _is_instruction_echo(spoken):
            return ""
        if m and not m.t_first_chunk:
            m.mark("t_first_chunk")
        await self._enqueue_tts(spoken, m)
        return spoken + " "

    # -- long-term memory --------------------------------------------------

    async def _remember_user(self, text: str) -> str | None:
        """Persist the user turn and return the memory context to inject:
        durable profile facts + rolling summary + relevant recalled turns."""
        if self._memory is None or self._embedder is None:
            return None
        try:
            qvec = await asyncio.to_thread(self._embedder.embed_query, text)
            exclude = self.cfg.llm.history_max_turns * 2  # already in live context
            facts, summary, lines = await asyncio.to_thread(
                lambda: (
                    self._memory.facts_block(),
                    self._memory.latest_summary(),
                    self._memory.recall(qvec, self.cfg.memory.top_k,
                                        self.cfg.memory.min_similarity, exclude),
                )
            )
            pvec = await asyncio.to_thread(self._embedder.embed_one, text)
            await asyncio.to_thread(
                self._memory.add_turn, self._session_id, "user", text, pvec
            )
        except Exception:
            log.exception("memory recall/store failed")
            return None
        parts: list[str] = []
        if facts:
            parts.append(f"What you know about {USER_NAME}:\n" + facts)
        if summary:
            parts.append("Summary of earlier conversation:\n" + summary)
        if lines:
            parts.append("Other things you remember that may be relevant:\n"
                         + "\n".join(f"- {ln}" for ln in lines))
        if not parts:
            return None
        block = "\n\n".join(parts)
        return block[: self.cfg.memory.max_recall_chars] + \
            "\n(Use naturally if relevant; don't recite it.)"

    async def _remember_assistant(self, text: str) -> None:
        if self._memory is None or self._embedder is None or not text:
            return
        try:
            vec = await asyncio.to_thread(self._embedder.embed_one, text)
            await asyncio.to_thread(
                self._memory.add_turn, self._session_id, "assistant", text, vec
            )
        except Exception:
            log.exception("memory store (assistant) failed")

    def _maybe_consolidate(self) -> None:
        """Kick off a background consolidation once enough new turns have piled up."""
        n = self.cfg.memory.consolidate_every_n
        if (self._memory is None or n <= 0 or self._consolidating
                or (self._consolidate_task and not self._consolidate_task.done())):
            return
        if self._memory.count_unconsolidated() >= n:
            self._consolidate_task = asyncio.create_task(
                self._run_consolidation(), name="consolidate"
            )

    async def _run_consolidation(self) -> None:
        if self._memory is None or self._consolidating:
            return
        self._consolidating = True
        try:
            turns = await asyncio.to_thread(self._memory.unconsolidated_turns)
            if not turns:
                return
            prev = await asyncio.to_thread(self._memory.latest_summary)
            summary, facts = await consolidate(self.llm, prev, turns)
            if summary:
                await asyncio.to_thread(
                    self._memory.add_summary, summary, turns[0][0], turns[-1][0]
                )
            for key, value in facts.items():
                await asyncio.to_thread(self._memory.upsert_fact, key, value)
            log.info("memory consolidated: %d turns -> summary=%s, +%d facts",
                     len(turns), bool(summary), len(facts))
        except Exception:
            log.exception("consolidation failed")
        finally:
            self._consolidating = False
