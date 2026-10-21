"""Live conversation transcript (logs/chat.log) and its console viewer.

The assistant's words are written as they are *heard*, not when the LLM
generates them: each spoken chunk's words are spread over its audio duration
starting when that audio reaches the speaker (karaoke-style; the TTS engines
don't report word timings, so within a chunk the pacing is even). A barge-in
drops the words that were never played, so the transcript matches what you
actually heard.

    python -m vcb.transcript logs\\chat.log     # the viewer (opened by VCB)
"""

import asyncio
import codecs
import sys
import time
from collections import deque
from pathlib import Path


class Transcript:
    """Appends the conversation to a text file, flushing every write so the
    viewer shows it immediately. path=None makes it a no-op (tests)."""

    def __init__(self, path: str | Path | None, assistant_name: str) -> None:
        self._f = open(path, "a", encoding="utf-8") if path else None
        self._name = assistant_name
        self._line_open = False          # an assistant line is being written

    def _write(self, text: str) -> None:
        if self._f is not None:
            self._f.write(text)
            self._f.flush()

    def user(self, text: str) -> None:
        self.end(interrupted=False)
        self._write(f"{time.strftime('%H:%M:%S')}  You:  {text}\n")

    def word(self, word: str) -> None:
        if not self._line_open:
            self._write(f"{time.strftime('%H:%M:%S')}  {self._name}:  {word}")
            self._line_open = True
        else:
            self._write(f" {word}")

    def end(self, interrupted: bool = False) -> None:
        if self._line_open:
            self._write(" --\n" if interrupted else "\n")
            self._line_open = False

    def close(self) -> None:
        self.end()
        if self._f is not None:
            self._f.close()
            self._f = None


class WordPacer:
    """Reveals words on the transcript at scheduled (monotonic) times, always
    in order. cancel() drops unrevealed words (barge-in); flush() reveals the
    rest immediately (reply finished)."""

    def __init__(self, transcript: Transcript) -> None:
        self.transcript = transcript
        self._pending: deque[list] = deque()     # [word, timer handle]

    def schedule(self, words: list[str], start: float, duration: float) -> None:
        if not words:
            return
        loop = asyncio.get_running_loop()
        now = time.monotonic()
        step = duration / len(words)
        for i, word in enumerate(words):
            item = [word, None]
            item[1] = loop.call_later(max(0.0, start + i * step - now), self._reveal, item)
            self._pending.append(item)

    def _reveal(self, item: list) -> None:
        while self._pending:
            head = self._pending.popleft()
            head[1].cancel()
            self.transcript.word(head[0])
            if head is item:
                break

    @property
    def has_pending(self) -> bool:
        return bool(self._pending)

    def flush(self) -> None:
        while self._pending:
            word, handle = self._pending.popleft()
            handle.cancel()
            self.transcript.word(word)

    def cancel(self) -> None:
        while self._pending:
            self._pending.popleft()[1].cancel()


def view(path: str) -> None:
    """Console viewer: print the transcript file as it grows, character by
    character (unlike `Get-Content -Wait`, which only shows whole lines)."""
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW("VCB chat")
    sys.stdout.reconfigure(encoding="utf-8", newline="")   # file already has CRLF
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pos = 0
    while True:
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size < pos:               # truncated: a new session started
                    pos = 0
                    print("\n" + "-" * 40)
                f.seek(pos)
                data = f.read()
                pos = f.tell()
        except FileNotFoundError:
            data = b""
        if data:
            sys.stdout.write(decoder.decode(data))
            sys.stdout.flush()
        time.sleep(0.05)


if __name__ == "__main__":
    try:
        view(sys.argv[1])
    except KeyboardInterrupt:
        pass
