"""Hearing more than words (vcb.vocal): voice measurements, the sound lane,
the per-turn cue and how the orchestrator uses them. No models are loaded:
detectors are faked, so this runs on any machine."""

import asyncio
import time

import numpy as np

from vcb.config import VOCAL_SOUNDS, AppConfig
from vcb.orchestrator import Orchestrator
from vcb.vocal import SoundEvent, VocalListener, VoiceAnalysis, lone_sound_text
from vcb.vocal.sounds import SoundLane
from vcb.vocal.traits import VoiceBaseline, VoiceStats, measure_voice

SR = 16_000


def _tone(freq, seconds=2.0, amp=0.1, glide=0.0):
    t = np.arange(int(seconds * SR)) / SR
    f = freq * 2 ** (glide * t / seconds / 12)          # glide: semitones over the clip
    phase = 2 * np.pi * np.cumsum(f) / SR
    return (amp * np.sin(phase)).astype(np.float32)


# -- voice traits ----------------------------------------------------------

def test_measure_voice_tracks_pitch_and_loudness():
    low, high = measure_voice(_tone(120), SR), measure_voice(_tone(240), SR)
    assert abs(high.pitch_st - low.pitch_st - 12) < 0.5      # an octave
    assert abs(low.loudness_db - 20 * np.log10(0.1 / np.sqrt(2))) < 1.5
    assert low.active_s > 1.5
    loud = measure_voice(_tone(120, amp=0.4), SR)
    assert abs(loud.loudness_db - low.loudness_db - 12) < 1.0


def test_measure_voice_needs_enough_sound():
    assert measure_voice(_tone(120, seconds=0.5), SR) is None
    assert measure_voice(np.zeros(SR * 2, np.float32), SR) is None


def test_measure_voice_sees_intonation():
    flat = measure_voice(_tone(150), SR)
    sung = measure_voice(_tone(150, glide=8), SR)
    assert sung.range_st > flat.range_st + 4


def test_baseline_reports_clear_changes_only():
    base = VoiceBaseline(min_samples=3)
    usual = VoiceStats(pitch_st=20.0, range_st=4.0, loudness_db=-30.0, active_s=2.0)
    for _ in range(3):
        assert base.compare(usual, words=6) == []           # still learning
    assert base.compare(usual, words=6) == []               # same as usual
    excited = VoiceStats(pitch_st=23.0, range_st=7.0, loudness_db=-23.0, active_s=2.0)
    assert base.compare(excited, words=9) == [
        "louder", "faster", "higher-pitched", "more animated"]
    subdued = VoiceStats(pitch_st=17.0, range_st=2.0, loudness_db=-37.0, active_s=2.0)
    assert base.compare(subdued, words=4) == [
        "quieter", "slower", "lower-pitched", "flatter"]


def test_baseline_ignores_rate_of_short_replies():
    base = VoiceBaseline(min_samples=1)
    base.compare(VoiceStats(None, None, -30.0, 2.0), words=6)
    # "no" over 2 s isn't "slower"; no pitch measured isn't compared either.
    assert base.compare(VoiceStats(None, None, -30.0, 2.0), words=1) == []


# -- sound lane --------------------------------------------------------------

def _lane(threshold=0.3):
    return SoundLane(None, VOCAL_SOUNDS, SR, frame_samples=512, window_s=1.0,
                     hop_s=0.5, min_dbfs=-48, threshold=threshold)


def test_sound_lane_windows_only_loud_audio():
    async def _run():
        lane = _lane()
        quiet = np.full(512, 10, np.int16)                    # about -70 dBFS
        for _ in range(100):
            lane.feed(quiet)
        assert lane._windows.empty()
        loud = (np.sin(np.arange(512) / 3) * 8000).astype(np.int16)
        for _ in range(100):                                  # ~3.2 s of audio
            lane.feed(loud)
        assert lane._windows.qsize() == 4                     # capped; oldest dropped
        audio, start, end = lane._windows.get_nowait()
        assert len(audio) == 31 * 512 and abs((end - start) - 31 * 512 / SR) < 1e-6

    asyncio.run(_run())


def test_sound_lane_merges_a_continuing_sound():
    lane = _lane()
    seen = []
    lane.on_event = seen.append
    lane.add({"Laughter": 0.9, "Sigh": 0.1}, 0.0, 1.0)
    lane.add({"Laughter": 0.6}, 0.5, 1.5)                     # same laugh, overlapping
    lane.add({"Laughter": 0.2}, 1.0, 2.0)                     # below threshold
    lane.add({"Gasp": 0.5}, 1.5, 2.5)
    assert [(e.label, e.start, e.end, e.score) for e in lane.events] == [
        ("Laughter", 0.0, 1.5, 0.9), ("Gasp", 1.5, 2.5, 0.5)]
    assert [e.phrase for e in seen] == ["laughs", "laughs", "gasps"]


def test_sound_lane_claims_each_event_once():
    lane = _lane()
    lane.add({"Laughter": 0.9}, 0.0, 1.0)
    lane.add({"Sigh": 0.9}, 5.0, 6.0)
    assert [e.label for e in lane.claim(0.5, 2.0)] == ["Laughter"]
    assert lane.claim(0.0, 2.0) == []
    assert [e.label for e in lane.claim(0.0, 10.0)] == ["Sigh"]


def test_sound_lane_forgets_old_events():
    lane = _lane()
    lane.add({"Laughter": 0.9}, 0.0, 1.0)
    lane.add({"Sigh": 0.9}, 40.0, 41.0)
    assert [e.label for e in lane.events] == ["Sigh"]


# -- the cue -----------------------------------------------------------------

def _event(label, start=0.0):
    return SoundEvent(label, VOCAL_SOUNDS[label], 0.9, start, start + 1)


def test_lone_sound_text():
    assert lone_sound_text([_event("Laughter")]) == "(laughs)"
    assert lone_sound_text([_event("Sigh", 2), _event("Laughter", 0),
                            _event("Laughter", 4)]) == "(laughs and sighs)"


def test_cue_lists_what_was_heard():
    assert VocalListener.cue([], [], []) is None
    cue = VocalListener.cue([_event("Laughter")], [_event("Gasp")],
                            ["sounds sad", "louder and faster than usual"])
    lines = cue.splitlines()
    assert "may be wrong" in lines[0]
    assert lines[1:5] == ["- while you were still talking: laughs",
                          "- with their words: gasps", "- sounds sad",
                          "- louder and faster than usual"]
    assert "Never mention" in lines[-1]


def test_describe_applies_the_emotion_threshold():
    cfg = AppConfig()
    listener = VocalListener(cfg.vocal, cfg.audio)
    assert listener.describe(VoiceAnalysis(emotion=("sad", 0.9)), "hi") != []
    assert listener.describe(VoiceAnalysis(emotion=("sad", 0.3)), "hi") == []
    assert listener.describe(VoiceAnalysis(emotion=("neutral", 0.99)), "hi") == []


def test_analyze_is_a_no_op_when_off():
    cfg = AppConfig()
    listener = VocalListener(cfg.vocal, cfg.audio)
    assert asyncio.run(listener.analyze(np.zeros(SR, np.float32))) == VoiceAnalysis()


# -- orchestrator ------------------------------------------------------------

class _STT:
    def __init__(self, transcript=""):
        self.transcript = transcript

    async def load(self):
        pass

    async def transcribe(self, audio, sample_rate, context=None):
        return self.transcript

    async def close(self):
        pass


class _LLM:
    """Records the prompt each turn would send (see _orch)."""

    def __init__(self):
        self.calls = []


def _orch(transcript="", react=True, sounds=True):
    cfg = AppConfig()
    cfg.memory.enabled = False
    cfg.skills.enabled = False
    cfg.proactive.enabled = False
    cfg.avatar.enabled = False
    cfg.turn_taking.incomplete_wait_s = 0
    cfg.vad.end_silence_ms = 100
    cfg.vocal.react_to_lone_sounds = react
    llm = _LLM()
    orch = Orchestrator(cfg, _STT(transcript), llm, _TTS())
    if sounds:
        orch.vocal.sounds = _lane()      # sounds "on", without the model

    async def _speak(messages, m, **_kw):
        llm.calls.append(messages)

    orch._speak_reply = _speak
    return orch, llm


class _TTS:
    supports_tags = False
    sample_rate = 24_000

    async def load(self):
        pass

    async def close(self):
        pass


def _prompt(messages):
    return "\n".join(str(m.get("content", "")) for m in messages)


def _user_turns(orch):
    return [m["content"] for m in orch.convo.messages(None) if m["role"] == "user"]


def test_a_laugh_without_words_gets_a_reply():
    async def _run():
        orch, llm = _orch(transcript="")
        await orch._run_turn(b"x", sounds=[_event("Laughter")])
        return orch, llm

    orch, llm = asyncio.run(_run())
    assert _user_turns(orch) == ["(laughs)"]
    assert len(llm.calls) == 1


def test_a_laugh_with_words_is_a_cue_not_history():
    async def _run():
        orch, llm = _orch(transcript="oh no, really")
        await orch._run_turn(b"x", sounds=[_event("Laughter")])
        return orch, llm

    orch, llm = asyncio.run(_run())
    assert _user_turns(orch) == ["oh no, really"]
    assert "- with their words: laughs" in _prompt(llm.calls[0])
    assert "laughs" not in _prompt(orch.convo.messages(None))   # transient
    assert orch._pending_sounds == []


def test_sounds_during_a_reply_reach_the_next_turn_once():
    async def _run():
        orch, llm = _orch(transcript="so anyway")
        orch._keep_reply_sounds([_event("Laughter")])
        await orch._run_turn(b"x")
        await orch._run_turn(b"x")
        return llm

    llm = asyncio.run(_run())
    assert "- while you were still talking: laughs" in _prompt(llm.calls[0])
    assert "Voice cues" not in _prompt(llm.calls[1])


def test_reacting_off_keeps_a_lone_laugh_for_later():
    async def _run():
        orch, llm = _orch(transcript="", react=False)
        await orch._run_turn(b"x", sounds=[_event("Laughter")])
        assert llm.calls == []
        orch.stt.transcript = "that was funny"
        await orch._run_turn(b"x")
        return llm

    llm = asyncio.run(_run())
    assert "- with their words: laughs" in _prompt(llm.calls[0])


def test_a_lone_sound_while_idle_starts_a_turn():
    async def _run():
        orch, llm = _orch()
        now = time.monotonic()
        orch.vocal.sounds.add({"Gasp": 0.8}, now - 2.0, now - 1.0)
        await orch._settle_sounds()
        await orch._turn_task
        return orch, llm

    orch, llm = asyncio.run(_run())
    assert _user_turns(orch) == ["(gasps)"]
    assert len(llm.calls) == 1


def test_a_lone_sound_while_the_assistant_talks_waits_for_the_next_turn():
    async def _run():
        orch, llm = _orch(transcript="yes")
        orch._turn_active = lambda: True
        now = time.monotonic()
        orch.vocal.sounds.add({"Sigh": 0.8}, now - 2.0, now - 1.0)
        await orch._settle_sounds()
        assert orch._turn_task is None
        orch._turn_active = lambda: False
        await orch._run_turn(b"x")
        return llm

    llm = asyncio.run(_run())
    assert "- while you were still talking: sighs" in _prompt(llm.calls[0])


def test_no_cue_when_hearing_is_off():
    async def _run():
        orch, llm = _orch(transcript="hello", sounds=False)
        assert orch._claim_utterance_sounds(b"x") == []
        await orch._run_turn(b"x")
        return llm

    llm = asyncio.run(_run())
    assert "Voice cues" not in _prompt(llm.calls[0])
