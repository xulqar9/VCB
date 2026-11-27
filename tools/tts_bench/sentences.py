"""The fixed benchmark sentences (tools/tts_bench/sentences.txt) and tag helpers."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

CATEGORIES = ("neutral", "emotion", "human")

# Canonical tags the sentence file may use. Adapters map them to each engine's
# own syntax, or drop them when the engine has no such sound.
CANONICAL_TAGS = ("laugh", "sigh", "hesitation", "breath")

ANGLE_TAG = re.compile(r"<\s*([a-zA-Z_]+)\s*>")


@dataclass(frozen=True)
class Sentence:
    id: str
    category: str
    intent: str
    text: str

    @property
    def tags(self) -> list[str]:
        return [m.lower() for m in ANGLE_TAG.findall(self.text)]


def parse_sentences(text: str) -> list[Sentence]:
    """Parse "id | category | intent | text" lines. Blank lines and lines
    starting with # are ignored; the text may itself contain '|'."""
    sentences: list[Sentence] = []
    seen: set[str] = set()
    for lineno, raw in enumerate(text.splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = [p.strip() for p in line.split("|", 3)]
        if len(parts) != 4 or not all(parts):
            raise ValueError(f"sentences line {lineno}: expected 'id | category | intent | text'")
        sid, category, intent, body = parts
        if not re.fullmatch(r"[A-Za-z0-9]+", sid):
            raise ValueError(f"sentences line {lineno}: id {sid!r} must be letters/digits only "
                             "(it becomes part of the WAV file name)")
        if sid in seen:
            raise ValueError(f"sentences line {lineno}: duplicate id {sid!r}")
        if category not in CATEGORIES:
            raise ValueError(f"sentences line {lineno}: category {category!r} "
                             f"is not one of {', '.join(CATEGORIES)}")
        seen.add(sid)
        sentences.append(Sentence(sid, category, intent, body))
    if not sentences:
        raise ValueError("no sentences found")
    return sentences


def load_sentences(path: str | Path) -> list[Sentence]:
    return parse_sentences(Path(path).read_text(encoding="utf-8"))


def sentences_fingerprint(sentences: list[Sentence]) -> str:
    """Hash of everything that defines the test input, so reports can tell
    when two result sets were produced from different sentences."""
    blob = "\n".join(f"{s.id}|{s.category}|{s.intent}|{s.text}" for s in sentences)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]
