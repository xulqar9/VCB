"""SentenceChunker: eager first chunk, action-span holding, flush, hard split."""

from vcb.chunker import SentenceChunker
from vcb.config import ChunkerConfig


def _chunker():
    return SentenceChunker(ChunkerConfig())


def test_first_chunk_released_eagerly_at_clause():
    c = _chunker()
    out = c.push("Honestly, this opening clause keeps going for a while now.")
    assert out
    assert out[0] == "Honestly,"


def test_short_input_holds_until_flush():
    c = _chunker()
    assert c.push("Hey") == []          # below min, no boundary
    assert c.flush() == ["Hey"]


def test_unclosed_action_is_held_then_emitted_whole():
    c = _chunker()
    assert c.push("Hi *waves") == []    # action span not yet closed, held
    out = c.push(" a lot* and that is the end. ")
    joined = " ".join(out)
    assert "*waves a lot*" in joined


def test_flush_clears_buffer_and_resets():
    c = _chunker()
    c.push("leftover words with no terminator")
    first = c.flush()
    assert first == ["leftover words with no terminator"]
    assert c.flush() == []              # nothing left after flushing


def test_hard_split_when_no_punctuation():
    c = _chunker()
    out = c.push("word " * 100)         # 500 chars, no sentence end
    assert out                          # forced to split at a space
    assert all(len(chunk) <= ChunkerConfig().max_chars for chunk in out)
