"""Turn-taking: backchannels, unfinished sentences, busy detection, and the
talk-over behavior in the orchestrator. Examples are from a real session log."""

import asyncio
import time

import pytest

from vcb.turns import is_backchannel, looks_incomplete, mentions_busy


@pytest.mark.parametrize("text", [
    "Yeah.", "Yeah, yeah, yeah.", "Yes.", "Mhm.", "Uh-huh", "Right.", "Okay.",
    "Hmm.", "I see.", "Got it.", "Oh, nice.", "Yeah, right.",
])
def test_backchannels(text):
    assert is_backchannel(text)


@pytest.mark.parametrize("text", [
    "No.", "Wait.", "Stop.", "What?", "No, no, I'm gonna keep playing.",
    "It.", "I.", "Yes, baby, yes.", "Nice, we scored!", "Fuck.", "",
    "I see you're busy.",
])
def test_not_backchannels(text):
    assert not is_backchannel(text)


@pytest.mark.parametrize("text", [
    "I just shot on the goal and", "I'm playing with...", "His 50s is...",
    "Only 16 seconds left and...", "I strongly believe we are really close to",
    "my teammates defense is ass do you know what rank I am,",
])
def test_incomplete(text):
    assert looks_incomplete(text)


@pytest.mark.parametrize("text", [
    "It was not in. I hit the bar.", "Four minutes.", "How did you know I'm Diamond?",
    "Fuck!", "Yeah.", "",
])
def test_complete(text):
    assert not looks_incomplete(text)


@pytest.mark.parametrize("text", [
    "Sometimes I have to focus.", "No, I'm in the game.", "I'm playing right now",
    "give me a sec", "hold on", "I'm busy", "can't talk right now",
])
def test_busy(text):
    assert mentions_busy(text)


@pytest.mark.parametrize("text", [
    "I like to play games", "Pretty fine, I'm doing alright", "We are pressuring.",
])
def test_not_busy(text):
    assert not mentions_busy(text)


# -- orchestrator behavior ----------------------------------------------------

from tests.test_orchestrator import _FakeLLM, _FakeSTT, _drain, _orch  # noqa: E402
from vcb.vad import SpeechEnd, SpeechStart  # noqa: E402


class _FakeSpeaker:
    def __init__(self):
        self.is_playing = True
        self.generation = 0

    def flush(self):
        self.is_playing = False
        self.generation += 1


class _FakeVAD:
    voiced_s = 0.0


def _speaking_orch(transcript):
    """An orchestrator in the middle of saying a reply."""
    orch = _orch(_FakeLLM(["Sure thing."]))
    orch.stt = _FakeSTT(transcript)
    orch.speaker = _FakeSpeaker()
    orch.vad = _FakeVAD()
    orch._turn_task = asyncio.create_task(asyncio.sleep(30))   # reply in progress
    return orch


def test_backchannel_while_speaking_does_not_interrupt():
    async def _run():
        orch = _speaking_orch("Yeah.")
        reply = orch._turn_task
        await orch._handle_speech_event(SpeechStart())
        orch.vad.voiced_s = 0.4
        await asyncio.sleep(0.1)
        await orch._handle_speech_event(SpeechEnd(audio=b"x"))
        await orch._side_task
        alive = not reply.done() and orch.speaker.is_playing
        reply.cancel()
        return alive, orch

    alive, orch = asyncio.run(_run())
    assert alive                                   # she kept talking
    assert orch.convo.messages()[-1]["role"] == "system"   # "Yeah." not recorded


def test_short_real_interjection_interrupts_and_gets_reply():
    async def _run():
        orch = _speaking_orch("Wait, what?")
        reply = orch._turn_task
        await orch._handle_speech_event(SpeechStart())
        orch.vad.voiced_s = 0.5
        await orch._handle_speech_event(SpeechEnd(audio=b"x"))
        await orch._side_task
        stopped = reply.cancelled() or reply.done()
        new_turn = orch._turn_task
        await asyncio.sleep(0.05)
        orch._reply_drained.set()
        await new_turn
        return stopped, orch

    stopped, orch = asyncio.run(_run())
    assert stopped
    assert orch.convo.last_assistant() == "Sure thing."
    users = [m["content"] for m in orch.convo.messages() if m["role"] == "user"]
    assert users == ["Wait, what?"]


def test_sustained_talk_over_interrupts_without_waiting_for_the_end():
    async def _run():
        orch = _speaking_orch("Never mind.")
        reply = orch._turn_task
        await orch._handle_speech_event(SpeechStart())
        await asyncio.sleep(0.05)
        assert not reply.done()                    # not yet: could be a "yeah"
        orch.vad.voiced_s = orch.cfg.turn_taking.barge_in_min_speech_s + 0.1
        await asyncio.sleep(0.1)
        return reply.done(), orch.speaker.is_playing

    done, playing = asyncio.run(_run())
    assert done and not playing


def test_unfinished_sentence_waits_before_replying():
    async def _run():
        orch = _orch(_FakeLLM(["Nice."]))
        orch.cfg.turn_taking.incomplete_wait_s = 0.3
        orch.stt = _FakeSTT("I just shot on the goal and")
        task = asyncio.create_task(orch._run_turn(b"audio"))
        await asyncio.sleep(0.15)
        early = list(_drain(orch))                 # nothing yet: still waiting
        await asyncio.sleep(0.4)
        orch._reply_drained.set()
        await task
        return early, orch

    early, orch = asyncio.run(_run())
    assert early == []
    assert orch.convo.last_assistant() == "Nice."


def test_continuation_joins_the_unfinished_message():
    async def _run():
        orch = _orch(_FakeLLM(["Ouch, unlucky."]))
        orch.cfg.turn_taking.incomplete_wait_s = 0.5
        orch.stt = _FakeSTT("I just shot on the goal and")
        orch._turn_task = asyncio.create_task(orch._run_turn(b"audio"))
        await asyncio.sleep(0.1)
        await orch._handle_speech_event(SpeechStart())      # they keep talking
        orch.stt = _FakeSTT("I hit the crossbar.")
        await orch._handle_speech_event(SpeechEnd(audio=b"x"))
        await asyncio.sleep(0.1)
        orch._reply_drained.set()
        await orch._turn_task
        return orch

    orch = asyncio.run(_run())
    users = [m["content"] for m in orch.convo.messages() if m["role"] == "user"]
    assert users == ["I just shot on the goal and I hit the crossbar."]


def test_busy_user_gets_no_check_ins():
    orch = _orch(_FakeLLM([]))
    assert orch._lead_wait() is not None
    orch._accept_user_text("Sometimes I have to focus.")
    assert orch._lead_wait() is None
    orch._quiet_until = time.monotonic() - 1                 # quiet period over
    assert orch._lead_wait() is not None


def test_goodbye_stops_check_ins_until_the_user_speaks_again():
    orch = _orch(_FakeLLM([]))
    orch._accept_user_text("Uh let me let me go. I will uh talk to you later, right?")
    assert orch._lead_wait() is None
    orch._last_activity = time.monotonic() - 3600            # however long it's quiet
    assert orch._lead_wait() is None
    orch._accept_user_text("Hey, I'm back.")
    assert orch._lead_wait() is not None


@pytest.mark.parametrize("text, expected", [
    ("Bye!", True), ("Okay, goodbye.", True), ("Good night Sophia", True),
    ("See you tomorrow.", True), ("I gotta go now.", True), ("ttyl", True),
    ("I'm going to bed.", True), ("Talk to you soon.", True),
    ("I saw you yesterday.", False), ("Let's go!", False),
    ("Tell me a bedtime story.", False), ("What's on the byeline?", False),
])
def test_mentions_goodbye(text, expected):
    from vcb.turns import mentions_goodbye
    assert mentions_goodbye(text) is expected


def test_check_ins_get_rarer_the_more_often_they_happen():
    orch = _orch(_FakeLLM([]))
    p = orch.cfg.proactive
    base = orch._lead_wait()
    now = time.monotonic()
    orch._nudge_times.extend([now - 30, now - 20, now - 10, now - 5])
    assert orch._lead_wait() == base * p.idle_backoff ** 2   # 4 recent -> 2 doublings
    orch._nudge_times.clear()
    orch._nudge_times.extend([now - p.nudge_window_s - 5] * 6)   # old ones don't count
    assert orch._lead_wait() == base


def test_no_welcome_back_after_a_short_pause():
    orch = _orch(_FakeLLM([]))
    orch._unanswered_leads = 1
    orch._last_user_spoke = time.monotonic() - 20
    assert orch._welcome_back_note() is None
    orch._last_user_spoke = time.monotonic() - 90
    assert orch._welcome_back_note() is not None


def test_stt_gets_conversation_context():
    async def _run():
        orch = _orch(_FakeLLM(["Ok."]))
        orch._stt_vocab = "Diamond, Rocket League"
        orch.convo.add_assistant("Are you still in Plat 3, or back to Diamond?")
        orch.stt = _FakeSTT("Plat 3.")
        task = asyncio.create_task(orch._run_turn(b"audio"))
        await asyncio.sleep(0.05)
        orch._reply_drained.set()
        await task
        return orch.stt.contexts[0]

    ctx = asyncio.run(_run())
    assert "Rocket League" in ctx and "Plat 3, or back to Diamond?" in ctx
    assert len(ctx) <= 200
