"""sanitize_for_speech and the tag extractors: nothing that shouldn't be voiced
should reach TTS, and the tag/action extractors must report what the avatar needs.
"""

from vcb.sanitize import (
    extract_action_tags,
    extract_emotion_tags,
    sanitize_for_speech,
    strip_speech_tags,
)


def test_strips_leading_speaker_label():
    assert sanitize_for_speech("Iris: hello there") == "hello there"


def test_drops_leaked_tool_json():
    out = sanitize_for_speech('Sure thing. {"tool": "calculator", "args": {}}')
    assert "{" not in out and "tool" not in out
    assert out.startswith("Sure thing.")


def test_stray_close_brace_removed():
    assert "}" not in sanitize_for_speech('done "expression": 2}')


def test_emotion_tag_kept_when_engine_supports_it():
    out = sanitize_for_speech("that's funny <laugh>", keep_emotion_tags=True)
    assert "<laugh>" in out


def test_emotion_tag_becomes_onomatopoeia_without_support():
    out = sanitize_for_speech("that's funny <laugh>", keep_emotion_tags=False)
    assert "<laugh>" not in out
    assert "Haha" in out


def test_action_stage_directions_removed():
    assert sanitize_for_speech("Hello *waves happily* there") == "Hello there"


def test_emoji_removed():
    assert "\U0001f600" not in sanitize_for_speech("Hi \U0001f600 there")


def test_math_less_than_is_not_treated_as_tag():
    out = sanitize_for_speech("is a < b always true?")
    assert "a" in out and "b" in out


def test_invented_angle_directive_removed():
    out = sanitize_for_speech("Okay <waits for your reply> sure")
    assert "waits for your reply" not in out


def test_strip_speech_tags_for_transcript():
    assert strip_speech_tags("Hi <laugh> there") == "Hi there"


def test_extract_emotion_tags_in_order():
    assert extract_emotion_tags("<laugh> ok <sigh> hmm <bogus>") == ["laugh", "sigh"]


def test_extract_action_tags():
    assert extract_action_tags("hey <action:wave> and <action:Dance>") == ["wave", "dance"]
