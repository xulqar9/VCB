"""sanitize_for_speech and the tag extractors: nothing that shouldn't be voiced
should reach TTS, and the tag/action extractors must report what the avatar needs.
"""

from vcb.sanitize import (
    extract_action_tags,
    extract_emotion_tags,
    sanitize_for_speech,
    soften_shouting,
    strip_speech_tags,
)


def test_shouted_sentence_is_normal_cased():
    assert soften_shouting("LET'S GOOOOO! What a game.") == "Let's gooooo! What a game."


def test_shouted_sentences_and_i_contractions():
    assert soften_shouting("HELL YES! Nice. WHAT?! I'M SO PROUD.") == (
        "Hell yes! Nice. What?! I'm so proud."
    )


def test_shouting_keeps_acronyms_and_leading_tags():
    assert soften_shouting("<laugh> GG, THE GPU CARRIED!") == (
        "<laugh> GG, the GPU carried!"
    )


def test_isolated_caps_word_softened_but_acronyms_kept():
    assert soften_shouting("That was AMAZING on the GPU, NASA level.") == (
        "That was amazing on the GPU, NASA level."
    )
    assert soften_shouting("Go get that W!") == "Go get that W!"
    assert soften_shouting("I think AI is OK.") == "I think AI is OK."


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


def test_short_parenthetical_stage_direction_removed():
    assert sanitize_for_speech("Hello (laughs) there") == "Hello there"
    assert sanitize_for_speech("(sighs) Fine.") == "Fine."


def test_informative_parenthetical_kept():
    out = sanitize_for_speech(
        "From today (Saturday, October 24, 2026) until November 20, 2026, is **27 days**. "
        "(That means you have a little under four weeks!)")
    assert out == ("From today, Saturday, October 24, 2026, until November 20, 2026, "
                   "is 27 days. That means you have a little under four weeks!")


def test_bold_keeps_its_words():
    assert sanitize_for_speech("That's **so** cool") == "That's so cool"


def test_list_items_are_spoken_without_markers():
    assert sanitize_for_speech("* Monday at 8 AM: go to the office.") == \
        "Monday at 8 AM: go to the office."
    assert sanitize_for_speech("Here's your list:\n- buy milk\n- call mom\n1. first") == \
        "Here's your list: buy milk call mom first"
    assert sanitize_for_speech("I'll do it *gently* now") == "I'll do it now"   # still an action


def test_chunker_never_splits_inside_parentheses():
    from vcb.chunker import SentenceChunker
    from vcb.config import ChunkerConfig

    raw = "From today (Saturday, October 24, 2026) until November 20, it's 27 days. Okay!"
    ch = SentenceChunker(ChunkerConfig())
    chunks = []
    for i in range(0, len(raw), 5):
        chunks += ch.push(raw[i:i + 5])
    chunks += ch.flush()
    assert all(c.count("(") == c.count(")") for c in chunks)
    assert "27 days" in " ".join(sanitize_for_speech(c) for c in chunks)
