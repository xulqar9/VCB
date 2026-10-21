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


class _Renderer:
    """Pretty-prints the transcript stream for a console: word wrap with a
    hanging indent under the message, dim timestamps, colored speaker names.
    Input arrives in arbitrary pieces (often single words), so it's processed
    incrementally; a trailing partial word is held until its end is seen."""

    DIM, USER, BOT, RESET = "\x1b[90m", "\x1b[1;36m", "\x1b[1;35m", "\x1b[0m"

    def __init__(self, out, width_fn, color: bool = True) -> None:
        self.out = out
        self.width_fn = width_fn
        self.color = color
        self.col = 0
        self.indent = 0
        self.field = 0               # 0 = timestamp, 1 = speaker, 2+ = message
        self.pending_space = False
        self.held = ""
        self.name_width = len("You:")

    def _w(self, s: str) -> None:
        self.out.write(s)

    def _paint(self, code: str, s: str) -> str:
        return f"{code}{s}{self.RESET}" if self.color else s

    def feed(self, text: str, final: bool = False) -> None:
        text = self.held + text.replace("\r", "")
        self.held = ""
        i = 0
        while i < len(text):
            ch = text[i]
            if ch == "\n":
                self._w("\n")
                self.col = self.indent = self.field = 0
                self.pending_space = False
                i += 1
                continue
            if ch == " ":
                j = i
                while j < len(text) and text[j] == " ":
                    j += 1
                if self.field < 2:
                    self._w(text[i:j])              # keep the header spacing
                    self.col += j - i
                else:
                    self.pending_space = self.col > self.indent
                i = j
                continue
            j = i
            while j < len(text) and text[j] not in " \n":
                j += 1
            word = text[i:j]
            if j == len(text) and not final:
                self.held = word                    # may continue in the next read
                return
            self._word(word)
            i = j

    def _word(self, word: str) -> None:
        if self.field == 0:
            self._w(self._paint(self.DIM, word))
            self.col += len(word)
            self.field = 1 if ":" in word else 2
            return
        if self.field == 1:
            code = self.USER if word == "You:" else self.BOT
            # Pad names to a common width so every message starts in one column.
            self.name_width = max(self.name_width, len(word))
            pad = self.name_width - len(word) + 2
            self._w(self._paint(code, word) + " " * pad)
            self.col += len(word) + pad
            self.field = 2
            self.indent = self.col           # wrapped lines align under the message
            return
        width = max(20, self.width_fn() - 1)
        extra = 1 if self.pending_space else 0
        if self.col + extra + len(word) > width and self.col > self.indent:
            self._w("\n" + " " * self.indent)
            self.col = self.indent
            extra = 0
        if extra:
            self._w(" ")
        self._w(word)
        self.col += extra + len(word)
        self.pending_space = False

    def flush(self) -> None:
        if self.held:
            self.feed("", final=True)


def _enable_ansi() -> bool:
    if sys.platform != "win32":
        return True
    import ctypes
    k32 = ctypes.windll.kernel32
    handle = k32.GetStdHandle(-11)
    mode = ctypes.c_uint32()
    if not k32.GetConsoleMode(handle, ctypes.byref(mode)):
        return False
    return bool(k32.SetConsoleMode(handle, mode.value | 0x0004))  # VT processing


def view(path: str, assistant_name: str = "") -> None:
    """Console viewer: print the transcript file as it grows, word by word
    (unlike `Get-Content -Wait`, which only shows whole lines)."""
    import shutil
    if sys.platform == "win32":
        import ctypes
        ctypes.windll.kernel32.SetConsoleTitleW("VCB chat")
    sys.stdout.reconfigure(encoding="utf-8")
    renderer = _Renderer(sys.stdout, lambda: shutil.get_terminal_size().columns,
                         color=_enable_ansi())
    if assistant_name:
        renderer.name_width = max(renderer.name_width, len(assistant_name) + 1)
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    pos = 0
    idle = 0
    while True:
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                if size < pos:               # truncated: a new session started
                    pos = 0
                    renderer.flush()
                    print("\n" + "-" * 40)
                f.seek(pos)
                data = f.read()
                pos = f.tell()
        except FileNotFoundError:
            data = b""
        if data:
            renderer.feed(decoder.decode(data))
            idle = 0
        else:
            idle += 1
            if idle == 4:                    # ~0.2 s quiet: show a held word
                renderer.flush()
        sys.stdout.flush()
        time.sleep(0.05)


if __name__ == "__main__":
    try:
        view(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "")
    except KeyboardInterrupt:
        pass
