"""Text heuristics for natural turn-taking.

All pure functions on a transcript, so they're cheap and easy to test:
- is_backchannel: "yeah", "mhm", "right" said while the assistant talks --
  listener feedback, not a request to stop.
- looks_incomplete: the utterance stopped mid-sentence ("...and", "I hit the")
  so the user is probably about to continue.
- mentions_busy: the user says they're focused/playing/busy, so unprompted
  check-ins would be unwelcome.
"""

import re

_WORD = re.compile(r"[a-z']+")

_BACKCHANNEL_WORDS = {
    "yeah", "yea", "ya", "yes", "yep", "yup", "ok", "okay", "alright", "right",
    "sure", "mhm", "mm", "mmm", "hmm", "hm", "uh", "huh", "uhhuh", "ah", "aha",
    "oh", "wow", "nice", "cool", "true", "exactly", "totally", "indeed", "i",
    "see", "got", "it", "gotcha",
}
# Whole-utterance fillers that are only backchannels when alone ("i", "it",
# "see" etc. must appear as part of a phrase like "I see" / "got it").
_STANDALONE_ONLY = {"i", "see", "got", "it"}


def _words(text: str) -> list[str]:
    return _WORD.findall(text.lower().replace("-", ""))


def is_backchannel(text: str, max_words: int = 4) -> bool:
    """True for listener feedback like "Yeah.", "Mhm, right.", "Uh-huh", "I see"."""
    words = _words(text)
    if not words or len(words) > max_words:
        return False
    if not all(w in _BACKCHANNEL_WORDS for w in words):
        return False
    return not all(w in _STANDALONE_ONLY for w in words) or " ".join(words) in (
        "i see", "got it")


_TRAILING_WORDS = {
    "and", "but", "or", "so", "because", "cause", "the", "a", "an", "to", "of",
    "in", "on", "at", "with", "for", "from", "about", "into", "my", "your",
    "his", "her", "their", "our", "its", "that", "which", "who", "is", "was",
    "are", "were", "i", "if", "when", "like", "um", "uh", "then", "than",
    "just", "very", "really", "gonna", "wanna", "too",
}


def looks_incomplete(text: str) -> bool:
    """The user stopped mid-sentence: trailing "...", a comma, or a last word
    that can't end a sentence ("and", "the", "my")."""
    stripped = text.strip()
    if not stripped:
        return False
    if stripped.endswith(("...", "…", ",", "-", "—")):
        return True
    if stripped[-1] in "?!":
        return False
    words = _words(stripped)
    return bool(words) and words[-1] in _TRAILING_WORDS


_BUSY = re.compile(
    r"\b(?:focus(?:ing)?|concentrat\w*|busy|"
    r"(?:in|into|playing|play)\s+(?:a|the|my|another|this)\s+(?:game|match|round|meeting|call)|"
    r"in\s+game|mid[- ]game|i'?m\s+playing|i\s+am\s+playing|"
    r"can'?t\s+(?:talk|speak)|(?:give|gimme)\s+me\s+a\s+(?:sec|second|minute|moment)|"
    r"one\s+(?:sec|second|moment|minute)|hold\s+on|brb)\b",
    re.IGNORECASE,
)


def mentions_busy(text: str) -> bool:
    """The user says they're busy, focusing, or in the middle of a game."""
    return bool(_BUSY.search(text))


_GOODBYE = re.compile(
    r"\b(?:(?:good)?\s*bye(?:\s*bye)?|bye\s+for\s+now|good\s*night|"
    r"(?:talk|speak|chat)\s+(?:to|with)\s+you\s+(?:later|soon|tomorrow)|ttyl|"
    r"see\s+(?:you|ya)(?:\s+(?:later|soon|tomorrow))?|catch\s+you\s+later|later\s+then|"
    r"(?:i\s+)?(?:gotta|got\s+to|have\s+to|need\s+to|must)\s+go|i'?m\s+(?:off|leaving|heading\s+out)|"
    r"(?:going|off|heading)\s+to\s+(?:bed|sleep)|signing\s+off|logging\s+off)\b",
    re.IGNORECASE,
)


def mentions_goodbye(text: str) -> bool:
    """The user is wrapping up the conversation ("bye", "talk to you later")."""
    return bool(_GOODBYE.search(text))
