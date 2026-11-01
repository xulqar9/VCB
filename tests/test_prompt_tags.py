"""Engine prompt instructions own the tag vocabulary, not persona prompts."""

import re

from vcb.config import PROMPTS_DIR, USER_PROMPTS_DIR

_INLINE_TAG = re.compile(
    r"<(?:laugh|chuckle|giggle|sigh|gasp|yawn|groan|whisper|happy|angry|"
    r"surprised|sarcastic|dramatic|fear|crying|shush|clear_throat|nod)>",
    re.IGNORECASE,
)


def test_persona_prompts_leave_tag_vocabulary_to_the_tts_engine():
    paths = [*PROMPTS_DIR.glob("*.txt"), *USER_PROMPTS_DIR.glob("*.txt")]
    assert paths
    for path in paths:
        assert not _INLINE_TAG.search(path.read_text(encoding="utf-8")), path
