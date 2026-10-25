"""Orchestrator robustness: failed-but-partial replies, resume flag, idempotent
shutdown, and lazy VAD loading. Fake engines keep this GPU/device-free."""

import asyncio
import contextlib

from vcb.config import AppConfig
from vcb.orchestrator import Orchestrator, _SENTINEL


class _FakeSTT:
    def __init__(self, transcript=""):
        self._transcript = transcript

    async def load(self):
        pass

    async def transcribe(self, audio, sample_rate, context=None):
        self.contexts = getattr(self, "contexts", []) + [context]
        return self._transcript

    async def close(self):
        self.closed = getattr(self, "closed", 0) + 1


class _FakeLLM:
    def __init__(self, deltas, fail=False):
        self._deltas = deltas
        self._fail = fail
        self.closed = 0

    async def load(self):
        pass

    async def stream(self, messages, temperature=None, max_tokens=None):
        for d in self._deltas:
            yield d
        if self._fail:
            raise RuntimeError("stream broke mid-reply")

    async def close(self):
        self.closed += 1


class _FakeTTS:
    supports_tags = False
    sample_rate = 24_000

    async def load(self):
        pass

    async def close(self):
        self.closed = getattr(self, "closed", 0) + 1


def _orch(llm):
    cfg = AppConfig()
    cfg.memory.enabled = False
    cfg.skills.enabled = False
    cfg.proactive.enabled = False
    cfg.avatar.enabled = False
    return Orchestrator(cfg, _FakeSTT(), llm, _FakeTTS())


def _drain(orch):
    items = []
    while not orch._tts_queue.empty():
        items.append(orch._tts_queue.get_nowait())
    return items


def test_partial_failure_is_resumable_not_completed():
    async def _run():
        orch = _orch(_FakeLLM(["Hello there. ", "How are "], fail=True))
        await orch._speak_reply([{"role": "user", "content": "hi"}], None)
        return orch, _drain(orch)

    orch, items = asyncio.run(_run())
    # The partial is kept in live history and flagged for resume, not banked.
    assert orch.convo.last_assistant() == "Hello there."
    assert orch._resume_pending is True
    # No apology was spoken over the partial, but playback was finalized.
    texts = [t for t in items if t is not _SENTINEL]
    assert all("Sorry" not in item.text for item in texts)
    assert items[-1] is _SENTINEL


def test_clean_failure_apologizes():
    async def _run():
        orch = _orch(_FakeLLM([], fail=True))
        await orch._speak_reply([{"role": "user", "content": "hi"}], None)
        return orch, _drain(orch)

    orch, items = asyncio.run(_run())
    assert orch._resume_pending is False
    assert orch.convo.last_assistant() is None
    spoken = [item.text for item in items if item is not _SENTINEL]
    assert any("Sorry" in text for text in spoken)
    assert items[-1] is _SENTINEL


def test_config_error_speaks_its_explanation():
    from vcb.llm.base import LLMConfigError

    class _RefusingLLM(_FakeLLM):
        async def stream(self, messages, temperature=None, max_tokens=None):
            raise LLMConfigError("no credit", "My API account is out of credit.")
            yield  # pragma: no cover  (makes this an async generator)

    async def _run():
        orch = _orch(_RefusingLLM([]))
        await orch._speak_reply([{"role": "user", "content": "hi"}], None)
        return _drain(orch)

    items = asyncio.run(_run())
    assert [i.text for i in items if i is not _SENTINEL] == ["My API account is out of credit."]
    assert items[-1] is _SENTINEL


def test_successful_reply_is_completed():
    async def _run():
        orch = _orch(_FakeLLM(["All good here."]))
        task = asyncio.create_task(
            orch._speak_reply([{"role": "user", "content": "hi"}], None)
        )
        await asyncio.sleep(0.05)          # let the stream push its chunks + SENTINEL
        orch._reply_drained.set()          # stand in for the tts loop draining audio
        await task
        return orch, _drain(orch)

    orch, items = asyncio.run(_run())
    assert orch.convo.last_assistant() == "All good here."
    assert orch._resume_pending is False
    assert items[-1] is _SENTINEL


def test_bargein_records_only_what_was_heard():
    """The LLM often finishes generating before TTS finishes speaking. A barge-in
    must record only the spoken portion so resume continues from there -- not the
    whole generated reply."""
    async def _run():
        orch = _orch(_FakeLLM([
            "Sure. Once, a developer found a hidden path. ",
            "Then a long second half that was never actually heard. ",
        ]))
        task = asyncio.create_task(
            orch._speak_reply([{"role": "user", "content": "tell a story"}], None)
        )
        await asyncio.sleep(0.05)          # stream finishes; reply awaits drain
        # Only the first chunk reached the speaker before the user barged in.
        orch._spoken = ["Sure. Once, a developer found a hidden path. "]
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return orch

    orch = asyncio.run(_run())
    assert orch.convo.last_assistant() == "Sure. Once, a developer found a hidden path."
    assert orch._resume_pending is True


async def _drive_turn(orch, transcript):
    """Run one full turn to completion, standing in for the tts loop's drain."""
    orch.stt = _FakeSTT(transcript)
    task = asyncio.create_task(orch._run_turn(b"audio"))
    await asyncio.sleep(0.05)
    orch._reply_drained.set()
    await task


def test_resume_survives_short_backchannel():
    """A brief backchannel that the model just acknowledges must not burn the
    resume offer -- a later 'continue' should still resume."""
    async def _run():
        orch = _orch(_FakeLLM(["Yeah?"]))   # short reply, not an actual continuation
        orch._resume_pending = True
        orch._resume_budget = 3
        await _drive_turn(orch, "wait wait")
        return orch

    orch = asyncio.run(_run())
    assert orch._resume_pending is True
    assert orch._resume_budget == 2


def test_substantial_reply_clears_resume():
    async def _run():
        orch = _orch(_FakeLLM([
            "Of course. The developer kept going, past the login screen and "
            "into a back-end dashboard that needed no password at all."
        ]))
        orch._resume_pending = True
        orch._resume_budget = 3
        await _drive_turn(orch, "continue please")
        return orch

    orch = asyncio.run(_run())
    assert orch._resume_pending is False


def test_resume_expires_after_window():
    async def _run():
        orch = _orch(_FakeLLM(["Yeah?"]))
        orch._resume_pending = True
        orch._resume_budget = 1              # only one armed turn left
        await _drive_turn(orch, "hmm")       # short reply -> still pending, budget 0
        await _drive_turn(orch, "you there") # no budget -> resume expires
        return orch

    orch = asyncio.run(_run())
    assert orch._resume_pending is False
    assert orch._resume_budget == 0




def test_shutdown_is_idempotent():
    async def _run():
        llm = _FakeLLM(["hi"])
        orch = _orch(llm)
        await orch.shutdown()
        await orch.shutdown()          # second call must be a no-op
        return orch, llm

    orch, llm = asyncio.run(_run())
    assert orch._shutdown_done is True
    assert llm.closed == 1             # engines torn down exactly once


def test_vad_model_loads_lazily():
    # Constructing the segmenter must not touch torch.hub (no network/disk).
    from vcb.vad import VADSegmenter
    from vcb.config import AudioConfig, VADConfig

    seg = VADSegmenter(VADConfig(), AudioConfig())
    assert seg._model is None


def test_vad_warns_when_sound_never_pauses(caplog):
    """A mic feeding constant 'speech' (noise, speakers, a controller mic) makes
    every utterance hit the length cap; that must be visible at WARNING level,
    and a normal utterance must not trigger it."""
    import logging

    import numpy as np

    from vcb.config import AudioConfig, VADConfig
    from vcb.vad import SpeechEnd, SpeechStart, VADSegmenter

    audio = AudioConfig()
    seg = VADSegmenter(VADConfig(max_utterance_s=2.0), audio)
    seg._speech_prob = lambda frame: 1.0            # mic never goes quiet
    frame = np.zeros(audio.frame_samples, dtype=np.int16)
    with caplog.at_level(logging.WARNING, logger="vcb.vad"):
        for _ in range(seg._max_frames):
            seg.process_frame(frame)
    assert "without a pause" in caplog.text
    assert isinstance(seg.events.get_nowait(), SpeechStart)
    assert isinstance(seg.events.get_nowait(), SpeechEnd)

    caplog.clear()
    seg = VADSegmenter(VADConfig(max_utterance_s=2.0), audio)
    probs = iter([1.0] * 20 + [0.0] * 40)           # speech, then silence
    seg._speech_prob = lambda frame: next(probs, 0.0)
    with caplog.at_level(logging.WARNING, logger="vcb.vad"):
        for _ in range(60):
            seg.process_frame(frame)
    assert "without a pause" not in caplog.text
    assert isinstance(seg.events.get_nowait(), SpeechStart)
    assert isinstance(seg.events.get_nowait(), SpeechEnd)


def test_queued_chunks_carry_their_turns_metrics():
    """Each queued chunk is tagged with the metrics of the turn that produced
    it, so a later turn can't be credited with an earlier turn's audio."""
    from vcb.metrics import TurnMetrics

    async def _run():
        orch = _orch(_FakeLLM(["First reply here."]))
        m = TurnMetrics()
        task = asyncio.create_task(
            orch._speak_reply([{"role": "user", "content": "hi"}], m)
        )
        await asyncio.sleep(0.05)
        orch._reply_drained.set()
        await task
        orch._metrics = TurnMetrics()      # a newer turn becomes current
        return m, _drain(orch)

    m, items = asyncio.run(_run())
    chunks = [item for item in items if item is not _SENTINEL]
    assert chunks and all(item.metrics is m for item in chunks)


class _OneChunkTTS(_FakeTTS):
    async def synthesize_stream(self, text):
        import numpy as np
        yield np.zeros(240, dtype=np.float32)


def test_tts_loop_attributes_audio_to_item_metrics():
    from vcb.metrics import TurnMetrics
    from vcb.orchestrator import _TTSItem

    async def _run():
        orch = _orch(_FakeLLM([]))
        orch.tts = _OneChunkTTS()
        orch.cfg.prosody.enabled = False
        played = []

        async def _play(audio):
            played.append(audio)

        orch.speaker.play = _play
        old, new = TurnMetrics(), TurnMetrics()
        orch._metrics = new                # current turn differs from the chunk's
        await orch._tts_queue.put(_TTSItem(orch.speaker.generation, "hi", old))
        loop = asyncio.create_task(orch._tts_loop())
        await asyncio.sleep(0.05)
        loop.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop
        return old, new, played

    old, new, played = asyncio.run(_run())
    assert played
    assert old.t_first_audio and len(old.tts_synth_times) == 1
    assert not new.t_first_audio and not new.tts_synth_times


def _fast_supervision(orch, max_restarts=2):
    sup = orch.cfg.supervision
    sup.backoff_initial_s = 0.0
    sup.backoff_max_s = 0.0
    sup.max_restarts = max_restarts


def test_supervisor_restarts_crashed_loop():
    async def _run():
        orch = _orch(_FakeLLM([]))
        _fast_supervision(orch, max_restarts=3)
        calls = {"n": 0, "crash_hooks": 0}
        done = asyncio.Event()

        async def _flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise RuntimeError("boom")
            done.set()
            await asyncio.Event().wait()   # healthy: run until cancelled

        def _hook():
            calls["crash_hooks"] += 1

        task = asyncio.create_task(orch._supervise("flaky", _flaky, _hook))
        await asyncio.wait_for(done.wait(), 1)
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        return calls

    calls = asyncio.run(_run())
    assert calls["n"] == 3
    assert calls["crash_hooks"] == 2


def test_supervisor_gives_up_after_max_restarts():
    import pytest

    async def _run():
        orch = _orch(_FakeLLM([]))
        _fast_supervision(orch, max_restarts=2)
        calls = {"n": 0}

        async def _broken():
            calls["n"] += 1
            raise RuntimeError("always")

        with pytest.raises(RuntimeError, match="kept failing"):
            await orch._supervise("broken", _broken)
        return calls

    calls = asyncio.run(_run())
    assert calls["n"] == 3                 # first run + 2 restarts


def test_tts_loop_crash_releases_waiting_reply():
    """If the tts loop dies, the supervisor's crash hook must release a reply
    waiting on the drain signal instead of hanging the turn forever."""
    async def _run():
        orch = _orch(_FakeLLM([]))
        _fast_supervision(orch, max_restarts=1)
        orch._reply_drained.clear()

        async def _dies():
            raise RuntimeError("tts loop died")

        with contextlib.suppress(RuntimeError):
            await orch._supervise("tts-loop", _dies, orch._reply_drained.set)
        return orch._reply_drained.is_set()

    assert asyncio.run(_run()) is True


def test_phase_reflects_pipeline_signals():
    from vcb.state import Phase

    async def _run():
        orch = _orch(_FakeLLM([]))
        phases = [orch.phase]
        orch._user_speaking = True
        phases.append(orch.phase)
        orch._user_speaking = False
        orch._turn_task = asyncio.create_task(asyncio.sleep(1))
        phases.append(orch.phase)
        orch._avatar_speaking = True
        phases.append(orch.phase)
        orch._turn_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await orch._turn_task
        return phases

    assert asyncio.run(_run()) == [
        Phase.IDLE, Phase.LISTENING, Phase.THINKING, Phase.SPEAKING
    ]


def test_lead_turn_skipped_while_busy():
    async def _run():
        orch = _orch(_FakeLLM(["hello"]))
        orch._avatar_speaking = True       # avatar still playing a reply
        orch._start_lead_turn(greeting=False)
        return orch._turn_task

    assert asyncio.run(_run()) is None


def _lead_cues(orch, greeting_first=True, nudges=2):
    """Start lead turns and capture the cue (last user message) each sends."""
    cues = []

    async def _fake_speak(messages, m, **_kw):
        cues.append(messages[-1]["content"])
        orch.convo.add_assistant("How's your day going?")

    orch._speak_reply = _fake_speak

    async def _run():
        if greeting_first:
            orch._start_lead_turn(greeting=True)
            await orch._turn_task
        for _ in range(nudges):
            if orch._lead_wait() is None:
                break
            orch._start_lead_turn(greeting=False)
            await orch._turn_task

    asyncio.run(_run())
    return cues


def test_ai_led_backs_off_then_goes_quiet_when_ignored():
    orch = _orch(_FakeLLM([]))
    p = orch.cfg.proactive
    assert orch._lead_wait() == p.idle_seconds
    cues = _lead_cues(orch, greeting_first=True, nudges=5)
    # Greeting, then one "no rush" line, then silence -- not endless questions.
    assert len(cues) == 2
    assert cues[0] == p.greeting_hint
    assert cues[1].endswith(p.give_space_hint)
    assert orch._lead_wait() is None


def test_nudge_tells_model_it_was_ignored():
    import time

    orch = _orch(_FakeLLM([]))
    orch.convo.add_user("hi")
    orch.convo.add_assistant("What do you do for fun?")
    orch.convo.add_assistant("I like hiking, personally.")
    orch._last_user_spoke = time.monotonic() - 42
    note = orch._silence_note()
    assert "42 seconds pass in silence" in note
    assert "your last 2 messages" in note
    assert '"I like hiking, personally."' in note        # quotes what went unanswered
    orch._last_user_spoke = time.monotonic() - 150
    assert "3 minutes" in orch._silence_note() or "2 minutes" in orch._silence_note()


def test_welcome_back_note_only_after_ignored_nudges():
    import time

    orch = _orch(_FakeLLM([]))
    assert orch._welcome_back_note() is None          # normal turn: no note
    orch._unanswered_leads = 2
    orch._last_user_spoke = time.monotonic() - 95
    note = orch._welcome_back_note()
    assert "about 2 minutes" in note or "about a minute" in note
    assert "speaking again" in note


def test_welcome_back_note_reaches_the_model():
    import time

    async def _run():
        orch = _orch(_FakeLLM(["Hey, welcome back!"]))
        seen = []
        real = orch.convo.messages

        def _spy(hint=None, *a, **kw):
            seen.append(hint)
            return real(hint, *a, **kw)

        orch.convo.messages = _spy
        orch._unanswered_leads = 2
        orch._last_user_spoke = time.monotonic() - 70
        await _drive_turn(orch, "Sorry, I was making coffee.")
        return seen, orch

    seen, orch = asyncio.run(_run())
    assert seen and seen[-1] and "speaking again" in seen[-1]
    assert orch._unanswered_leads == 0


def test_ai_led_without_greeting_nudges_then_gives_space():
    orch = _orch(_FakeLLM([]))
    p = orch.cfg.proactive
    orch.convo.add_assistant("I love rainy days.")
    cues = []

    async def _fake_speak(messages, m, **_kw):
        cues.append(messages[-1]["content"])

    orch._speak_reply = _fake_speak

    async def _run():
        waits = []
        while (w := orch._lead_wait()) is not None:
            waits.append(w)
            orch._start_lead_turn(greeting=False)
            await orch._turn_task
        return waits

    waits = asyncio.run(_run())
    assert waits == [p.idle_seconds, p.idle_seconds * p.idle_backoff]   # 10 s, then 20 s
    assert len(cues) == 2
    assert cues[0].startswith("[Stage direction") and cues[0].endswith(p.hint)
    assert cues[1].endswith(p.give_space_hint)


def test_user_reply_rearms_ai_led():
    async def _run():
        orch = _orch(_FakeLLM(["Nice."]))
        orch._unanswered_leads = orch.cfg.proactive.max_unanswered
        assert orch._lead_wait() is None
        await _drive_turn(orch, "I was out walking.")
        return orch

    orch = asyncio.run(_run())
    assert orch._unanswered_leads == 0
    assert orch._lead_wait() == orch.cfg.proactive.idle_seconds

def test_strip_reaction_opener():
    from vcb.orchestrator import strip_reaction_opener as s

    # Real outputs from mistral-nemo after being ignored:
    assert s("That's great to hear! So, what kind of projects?") == "So, what kind of projects?"
    assert s("Oh, I get it. No big plans then.") == "No big plans then."
    assert s("Oh, that's interesting! What's that project about?") == "What's that project about?"
    assert s("Oh, sorry, I guess you stepped away.") == "Sorry, I guess you stepped away."
    assert s("Oh, I get it.") == ""                       # nothing left -> drop chunk
    # Natural openers are kept.
    assert s("Hmm, maybe that was a big question.") == "Hmm, maybe that was a big question."
    assert s("No pressure, by the way.") == "No pressure, by the way."
    assert s("Well, I've been reading a lot lately.") == "Well, I've been reading a lot lately."
    assert s("I see you're busy, no worries!") == "You're busy, no worries!" or \
        s("I see you're busy, no worries!") == "I see you're busy, no worries!"


def test_nudge_drops_reaction_but_normal_reply_keeps_it():
    async def _run(unprompted):
        orch = _orch(_FakeLLM(["Oh, that's great to hear! ", "What else is new?"]))
        task = asyncio.create_task(orch._speak_reply(
            [{"role": "user", "content": "x"}], None, unprompted=unprompted))
        await asyncio.sleep(0.05)
        orch._reply_drained.set()
        await task
        return orch.convo.last_assistant(), [i.text for i in _drain(orch) if i is not _SENTINEL]

    said, chunks = asyncio.run(_run(unprompted=True))
    assert said == "What else is new?"
    assert all("great to hear" not in c for c in chunks)

    said, _ = asyncio.run(_run(unprompted=False))         # a real answer to the user
    assert said.startswith("Oh, that's great to hear!")

def test_strip_reaction_opener_edge_cases():
    from vcb.orchestrator import strip_reaction_opener as s

    # Not a stand-alone reaction: keep the whole sentence intact.
    assert s("That's great to see you've been busy.") == "That's great to see you've been busy."
    # Quoted lines (models sometimes wrap speech in quotes).
    assert s('"Oh, hey, by the way, I\'m Iris."') == '"Hey, by the way, I\'m Iris."'
    assert s('"Oh, I get it." ') == ""
