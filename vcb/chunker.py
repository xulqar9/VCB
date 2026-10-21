"""Streaming text chunker: LLM token deltas in, TTS-ready phrases out.

The first chunk is released at the first clause boundary after a small
minimum so audio starts as early as possible; later chunks wait for full
sentence boundaries for better prosody.
"""

import re

from .config import ChunkerConfig

_SENTENCE_END = re.compile(r"[.!?…]['\")\]]?\s")
_CLAUSE_END = re.compile(r"[,;:\u2014]\s")
_OPEN = {"*": "*", "(": ")", "[": "]", "<": ">"}


def _inside_action(buf: str) -> bool:
    """True if buf has an unclosed *, ( or [ span (an in-progress action)."""
    for opener, closer in _OPEN.items():
        if buf.count(opener) > buf.count(closer):
            return True
    return False



class SentenceChunker:
    def __init__(self, cfg: ChunkerConfig) -> None:
        self.cfg = cfg
        self._buf = ""
        self._emitted_first = False

    def reset(self) -> None:
        self._buf = ""
        self._emitted_first = False

    def push(self, delta: str) -> list[str]:
        """Feed a token delta; returns zero or more complete chunks."""
        self._buf += delta
        out: list[str] = []
        while True:
            chunk = self._try_extract()
            if chunk is None:
                break
            out.append(chunk)
        return out

    def flush(self) -> list[str]:
        """Call at end of stream; returns whatever remains."""
        rest = self._buf.strip()
        self._buf = ""
        self._emitted_first = False
        return [rest] if rest else []

    def _try_extract(self) -> str | None:
        min_chars = (
            self.cfg.first_chunk_min_chars if not self._emitted_first else self.cfg.min_chars
        )
        if len(self._buf) < min_chars:
            return None

        # Hold the chunk until an action span closes, so *...* / [...] stay whole
        # for the sanitizer (unless we hit the hard max, then flush anyway).
        if _inside_action(self._buf) and len(self._buf) < self.cfg.max_chars:
            return None

        m = _SENTENCE_END.search(self._buf, min_chars - 1)
        if m is None and not self._emitted_first:
            m = _CLAUSE_END.search(self._buf, min_chars - 1)  # eager first chunk
        if m is not None:
            return self._take(m.end())
        if len(self._buf) >= self.cfg.max_chars:
            space = self._buf.rfind(" ", self.cfg.min_chars, self.cfg.max_chars)
            return self._take(space if space > 0 else self.cfg.max_chars)
        return None

    def _take(self, end: int) -> str:
        chunk, self._buf = self._buf[:end].strip(), self._buf[end:]
        self._emitted_first = True
        return chunk
