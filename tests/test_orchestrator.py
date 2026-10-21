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

    async def transcribe(self, audio, sample_rate):
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
