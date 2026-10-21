"""Strip anything that shouldn't be spoken aloud before text reaches TTS.

Guarantees TTS never voices stage directions (*sips tea*), narration,
markdown, emojis, or emoticons, regardless of what the LLM emits.
"""

import re

# *action*, _action_, (aside), [narration]
_ACTIONS = re.compile(r"\*[^*\n]+\*|_[^_\n]+_|\([^)\n]*\)|\[[^\]\n]*\]")
# leftover half of an action split across streaming chunks: "[pauses," or "smiles]"
_OPEN_FRAGMENT = re.compile(r"[\[(*_][^\])*_\n]*$")
_CLOSE_FRAGMENT = re.compile(r"^[^\[(*_\n]*[\])*_]")
_MARKDOWN = re.compile(r"[*_`#>~|]+")
# common text emoticons: :) :-D ;) ^_^ :3 <3 :P xD o/
_EMOTICONS = re.compile(
    r"(?<!\w)(?:[:;=xX8]['`\-^]?[)(\][DPpOo3><|/\\]+|<3|\^[_.]?\^|[oO]_[oO]|;_;|:3)(?!\w)"
)
_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"   # emoji, pictographs, symbols, regional flags
    "\U00002190-\U000021ff"   # arrows
    "\U00002300-\U000023ff"   # misc technical (⌚ ⏰ ⏳)
    "\U000025a0-\U000027bf"   # geometric shapes, misc symbols, dingbats
    "\U00002b00-\U00002bff"   # stars & extra arrows (⭐ ⬆)
    "\U0000fe00-\U0000fe0f"   # variation selectors
    "\U0000200d"              # zero-width joiner (compound emoji)
    "\U000020e3"              # combining enclosing keycap
    "]+",
    flags=re.UNICODE,
)
_WS = re.compile(r"\s{2,}")

# The model sometimes prefixes its speech with its own name ("Iris: hi") or
# spills a tool call as raw JSON ("...help. {\"tool\": ..."). Neither should ever
# be voiced: strip a leading speaker label and everything from the first brace.
_SPEAKER_LABEL = re.compile(r"^\s*[A-Z][a-zA-Z]{1,15}:\s+")
_BRACE_RUN = re.compile(r"\{.*", re.DOTALL)

# Emotion tags Orpheus can voice; kept only when the engine supports them.
_EMOTION_TAGS = {
    "laugh", "chuckle", "sigh", "cough", "sniffle", "groan", "yawn", "gasp",
    "giggle", "gulp",
}
# Engines without tag support can't voice <laugh>; approximate the vocal ones
# with onomatopoeia they WILL pronounce, so the reply isn't dropped to silence.
_ONOMATOPOEIA = {
    "laugh": "Haha!", "chuckle": "Heh heh.", "giggle": "Hehe!",
    "sigh": "Haaah.", "gasp": "Ah!", "groan": "Ugh.", "yawn": "Yaawn.",
    "cough": "Ahem.", "sniffle": "Sniff.", "gulp": "Gulp.",
    "cry": "Boohoo.", "sob": "Boohoo.", "moan": "Mmmm.",
}
_ANGLE_TAG = re.compile(r"<\s*([a-zA-Z_]+)\s*>")
# Body action directives the LLM emits to make the avatar move: "<action:wave>",
# "<action:highfive>". The colon keeps them distinct from voiced emotion <tags>,
# and _ANGLE_DIRECTIVE below strips them from anything that reaches TTS.
_ACTION_TAG = re.compile(r"<\s*action\s*:\s*([a-zA-Z0-9_]+)\s*>", re.IGNORECASE)
# Stage directions the model invents in angle brackets: "<waits for your prompt>"
# or an unclosed "<smiles". Real emotion tags are handled/shielded first; this
# clears whatever <...> remains so it never reaches TTS or the transcript.
# The lookahead requires a non-space right after "<" so math like "a < b" is safe.
_ANGLE_DIRECTIVE = re.compile(r"<(?=\S)[^>\n]*>?")
# Any inline tag, for producing a clean human transcript.
_DISPLAY_TAG = re.compile(r"<[^>\n]*>")

# Smart punctuation -> ASCII so TTS pronounces it and transcripts don't mojibake.
# Em-dash / horizontal bar join clauses -> comma pause (avoids "store-got").
_SMART = {
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": ", ", "\u2026": "...", "\u00a0": " ",
    "\u2011": "-", "\u2012": "-", "\u2015": ", ", "\u2032": "'", "\u2033": '"',
}
_SMART_RE = re.compile("|".join(map(re.escape, _SMART)))
# Collapse artifacts from the dash->comma swap: " ," / ",," -> ",".
_COMMA_FIX = re.compile(r"\s+,|,\s*,+")


def sanitize_for_speech(text: str, keep_emotion_tags: bool = False) -> str:
    text = _SMART_RE.sub(lambda m: _SMART[m.group()], text)
    text = _SPEAKER_LABEL.sub("", text)     # drop a leading "Iris:" style label
    text = _BRACE_RUN.sub(" ", text)        # drop leaked tool JSON (never spoken)
    text = text.replace("}", " ")           # stray close from a split JSON chunk

    # Handle <tag>s first: keep known emotion tags (shielded from later passes
    # by a null-byte placeholder), drop everything else in angle brackets.
    placeholders: dict[str, str] = {}

    def _angle(m: "re.Match[str]") -> str:
        tag = m.group(1).lower()
        if keep_emotion_tags and tag in _EMOTION_TAGS:
            key = f"\x00{len(placeholders)}\x00"
            placeholders[key] = f"<{tag}>"
            return " " + key + " "
        # Tag-less engine: voice a spoken approximation instead of dropping it.
        if not keep_emotion_tags and tag in _ONOMATOPOEIA:
            return " " + _ONOMATOPOEIA[tag] + " "
        return " "

    text = _ANGLE_TAG.sub(_angle, text)
    text = _ANGLE_DIRECTIVE.sub(" ", text)  # drop invented <stage directions>
    text = _ACTIONS.sub(" ", text)
    text = _OPEN_FRAGMENT.sub(" ", text)   # drop dangling "[pauses," half
    text = _CLOSE_FRAGMENT.sub(" ", text)  # drop dangling "laughs]" half
    text = _EMOTICONS.sub(" ", text)
    text = _EMOJI.sub(" ", text)
    text = _MARKDOWN.sub(" ", text)
    text = _WS.sub(" ", text)
    text = _COMMA_FIX.sub(",", text)
    text = text.strip()
    for key, val in placeholders.items():
        text = text.replace(key, val)
    return _WS.sub(" ", text).strip()


def strip_speech_tags(text: str) -> str:
    """Remove inline <tags> so the transcript reads as plain spoken words, even
    when the TTS engine keeps them to perform laughs/sighs."""
    return _WS.sub(" ", _DISPLAY_TAG.sub("", text)).strip()


def extract_emotion_tags(text: str) -> list[str]:
    """Return the voiced emotion tags (<laugh>, <sigh>, ...) present in the text,
    in order, for driving an avatar's facial expression. Only tags the engines
    actually perform are reported; tag-less engines lose them at sanitize time."""
    return [m.group(1).lower() for m in _ANGLE_TAG.finditer(text)
            if m.group(1).lower() in _EMOTION_TAGS]


def extract_action_tags(text: str) -> list[str]:
    """Return the body-action names the LLM requested via <action:name> markers,
    in order, for driving avatar montages (wave, highfive, backflip, ...). These
    never reach TTS - sanitize strips them as stage directions."""
    return [m.group(1).lower() for m in _ACTION_TAG.finditer(text)]
