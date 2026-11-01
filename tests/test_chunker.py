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


def _chunks(cfg, text):
    c = SentenceChunker(cfg)
    return [x for tok in text.split(" ") for x in c.push(tok + " ")] + c.flush()


LONG_TAIL = ("<playful> Exactly, a cheat code for life! Press reset, all the bugs disappear, "
             "you get a fresh start. Would you use it to redo a project, or maybe just hit "
             "pause on the job hunt for a bit and see where things go? <excited> Honestly, I'd "
             "use mine to replay that last Rocket League match, because that aerial was so "
             "close, and I swear the ball clipped the crossbar before it bounced out.")


def test_big_chunks_never_cut_mid_sentence_when_a_sentence_ends_earlier():
    # A sentence ending before min_chars, then one too long to fit: the chunk
    # used to be cut at a space ("...and I swear the ball" | "clipped the...").
    cfg = ChunkerConfig(first_chunk_min_chars=60, min_chars=140, first_chunk_clause_split=False)
    chunks = _chunks(cfg, LONG_TAIL)
    assert chunks[1].endswith("see where things go?")
    assert chunks[2].startswith("<excited> Honestly")
    assert all(c[-1] in ".!?" for c in chunks)


def test_overlong_sentence_splits_at_a_clause_not_a_word():
    cfg = ChunkerConfig(first_chunk_min_chars=60, min_chars=140, first_chunk_clause_split=False)
    sentence = ("So I was thinking about the whole thing for a really long time last night, "
                "and honestly the more I turned it over in my head while lying there in the "
                "dark the less sure I became about any of it, because every single option "
                "seemed to lead somewhere that I did not really want to end up going at all.")
    chunks = _chunks(cfg, "Okay. " + sentence)
    assert chunks[0] == "Okay."                        # the sentence end in reach
    assert chunks[1].endswith("any of it,")            # then the last clause, not a word
    assert chunks[2].startswith("because every single option")
