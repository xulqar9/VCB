"""Live transcript: words appear paced over the audio, in order; a barge-in
drops unheard words and marks the cut; the viewer streams partial lines."""

import asyncio
import contextlib
import time

import numpy as np

from vcb.transcript import Transcript, WordPacer


def _lines(path):
    return path.read_text(encoding="utf-8")


def test_words_are_revealed_over_time_in_order(tmp_path):
    path = tmp_path / "chat.log"
    t = Transcript(path, "Iris")

    async def _run():
        pacer = WordPacer(t)
        start = time.monotonic()
        pacer.schedule(["one", "two", "three", "four"], start, 0.2)
        await asyncio.sleep(0.01)
        early = _lines(path)
        await asyncio.sleep(0.3)
        t.end()
        return early, _lines(path)

    early, final = asyncio.run(_run())
    assert early.endswith("Iris:  one")                     # line starts as audio starts
    assert final.endswith("Iris:  one two three four\n")


def test_cancel_drops_unheard_words_and_marks_cut(tmp_path):
    path = tmp_path / "chat.log"
    t = Transcript(path, "Iris")

    async def _run():
        pacer = WordPacer(t)
        pacer.schedule(["heard", "never", "heard"], time.monotonic(), 3.0)
        await asyncio.sleep(0.05)
        assert pacer.has_pending
        pacer.cancel()
        t.end(interrupted=True)
        t.user("wait")
        await asyncio.sleep(1.2)                 # cancelled timers must not fire

    asyncio.run(_run())
    text = _lines(path)
    assert "Iris:  heard --\n" in text
    assert "never" not in text
    assert text.rstrip().endswith("You:  wait")


def test_flush_reveals_the_rest_immediately(tmp_path):
    path = tmp_path / "chat.log"
    t = Transcript(path, "Iris")

    async def _run():
        pacer = WordPacer(t)
        pacer.schedule(["a", "b", "c"], time.monotonic() + 5, 1.0)
        pacer.flush()
        t.end()

    asyncio.run(_run())
    assert _lines(path).endswith("Iris:  a b c\n")


def test_user_line_closes_an_open_assistant_line(tmp_path):
    path = tmp_path / "chat.log"
    t = Transcript(path, "Iris")
    t.word("hello")
    t.user("hi")
    assert _lines(path).splitlines()[-2:][0].endswith("Iris:  hello")
    assert _lines(path).splitlines()[-1].endswith("You:  hi")


def test_renderer_wraps_on_words_with_hanging_indent():
    import io
    from vcb.transcript import _Renderer

    out = io.StringIO()
    r = _Renderer(out, lambda: 41, color=False)
    # Arrives word by word, like the live transcript.
    r.feed("14:02:12  Iris:  I")
    for w in " can set you a reminder to check before you head out.".split(" ")[1:]:
        r.feed(" " + w)
    r.feed("\n14:02:19  You:  ok\n")
    r.flush()
    lines = out.getvalue().splitlines()
    assert lines[0] == "14:02:12  Iris:  I can set you a"
    assert lines[1] == "                 reminder to check"
    assert lines[2] == "                 before you head out."
    assert lines[3] == "14:02:19  You:   ok"           # padded to align with "Iris:"
    assert all(len(line) <= 40 for line in lines)


def test_renderer_holds_partial_word_until_complete():
    import io
    from vcb.transcript import _Renderer

    out = io.StringIO()
    r = _Renderer(out, lambda: 80, color=False)
    r.feed("10:00:00  Iris:  hel")
    assert "hel" not in out.getvalue()
    r.feed("lo there")
    r.flush()
    assert out.getvalue().endswith("hello there")


def test_null_transcript_writes_nothing():
    t = Transcript(None, "Iris")
    t.user("x"), t.word("y"), t.end(), t.close()     # must not raise


def test_tts_loop_shows_words_as_audio_plays(tmp_path):
    """End to end through the tts loop: a chunk's words appear on the
    transcript paced across its audio, not all at once when queued."""
    from tests.test_orchestrator import _FakeLLM, _orch
    from vcb.orchestrator import _SENTINEL, _TTSItem

    class _HalfSecondTTS:
        supports_tags = False
        sample_rate = 24_000

        async def synthesize_stream(self, text):
            yield np.zeros(12_000, dtype=np.float32)   # 0.5 s of audio

    path = tmp_path / "chat.log"

    async def _run():
        orch = _orch(_FakeLLM([]))
        orch._transcript = Transcript(path, "Iris")
        orch._pacer = WordPacer(orch._transcript)
        orch.tts = _HalfSecondTTS()
        orch.cfg.prosody.enabled = False

        async def _play(_audio):
            pass

        async def _drained():
            pass

        orch.speaker.play = _play
        orch.speaker.wait_until_drained = _drained
        loop = asyncio.create_task(orch._tts_loop())
        await orch._tts_queue.put(_TTSItem(orch.speaker.generation, "one two three four"))
        await asyncio.sleep(0.1)
        partial = _lines(path)
        await asyncio.sleep(0.6)
        await orch._tts_queue.put(_SENTINEL)
        await asyncio.sleep(0.05)
        loop.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await loop
        return partial, _lines(path)

    partial, final = asyncio.run(_run())
    assert "one" in partial and "four" not in partial
    assert final.endswith("Iris:  one two three four\n")
