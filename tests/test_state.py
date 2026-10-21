"""PipelineSnapshot: phase derivation and the guard predicates built on it."""

import pytest

from vcb.state import Phase, PipelineSnapshot


@pytest.mark.parametrize("kwargs, phase", [
    ({}, Phase.IDLE),
    ({"user_speaking": True}, Phase.LISTENING),
    ({"user_speaking": True, "audio_playing": True}, Phase.LISTENING),
    ({"turn_active": True}, Phase.THINKING),
    ({"turn_active": True, "audio_playing": True}, Phase.SPEAKING),
    ({"audio_playing": True}, Phase.SPEAKING),
    ({"avatar_speaking": True}, Phase.SPEAKING),
])
def test_phase(kwargs, phase):
    assert PipelineSnapshot(**kwargs).phase is phase


def test_assistant_active_ignores_user_speech():
    # Barge-in only cares whether the assistant has something to cut.
    assert not PipelineSnapshot(user_speaking=True).assistant_active
    for flag in ("turn_active", "audio_playing", "avatar_speaking"):
        assert PipelineSnapshot(**{flag: True}).assistant_active


def test_busy_includes_user_speech():
    assert not PipelineSnapshot().busy
    for flag in ("user_speaking", "turn_active", "audio_playing", "avatar_speaking"):
        assert PipelineSnapshot(**{flag: True}).busy
