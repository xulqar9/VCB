"""IndexTTS engine: mood tags -> emotion vectors, and the pipe protocol to the
isolated worker process (exercised with a tiny fake worker, no GPU needed)."""

import asyncio
import dataclasses
import sys
import textwrap
import time

import numpy as np
import pytest
import soundfile as sf

from vcb import gpu
from vcb.chunker import SentenceChunker
from vcb.config import AppConfig, ChunkerConfig, TTSConfig
from vcb.hardware import adapt_to_hardware
from vcb.sanitize import sanitize_for_speech
from vcb.tts.indextts_tts import (
    IndexTTSEngine, choose_gpus, emotion_vector, split_moods, trim_voice_sample,
)


def test_split_moods_switches_and_carries_over():
    segs, mood = split_moods("<excited> No way! <playful> Took you long enough.", None)
    assert segs == [("excited", "No way!"), ("playful", "Took you long enough.")]
    assert mood == "playful"
    # The next chunk has no tag: the last mood stays in effect.
    segs, mood = split_moods("Anyway, what now?", mood)
    assert segs == [("playful", "Anyway, what now?")] and mood == "playful"


def test_split_moods_tag_only_chunk_and_other_tags():
    segs, mood = split_moods("<SAD>", None)
    assert segs == [] and mood == "sad"
    segs, _ = split_moods("Haha! <laugh> okay", None)
    assert segs == [(None, "Haha! okay")]


def test_emotion_vector_scaling():
    assert emotion_vector("calm", 1.0) is None
    assert emotion_vector(None, 1.0) is None
    assert emotion_vector("sad", 0.0) == [0.0] * 8
    strong = emotion_vector("excited", 5.0)
    assert abs(sum(strong) - 1.0) < 1e-9          # never above a full emotion


def test_sanitize_keeps_mood_tags_only_for_mood_engines():
    moods = IndexTTSEngine.mood_tags
    assert sanitize_for_speech("<excited> We won!", False, moods) == "<excited> We won!"
    assert sanitize_for_speech("<excited> We won!") == "We won!"


FAKE_WORKER = textwrap.dedent("""
    import json, os, struct, sys, time
    out = sys.stdout.buffer
    out.write(json.dumps({"ready": True, "sr": 16000, "vram_gb": 1.0}).encode() + b"\\n")
    out.flush()
    gpu = float(os.environ.get("CUDA_VISIBLE_DEVICES", "-1"))
    for line in sys.stdin:
        req = json.loads(line)
        if req.get("quit"):
            break
        if req["text"] == "boom":
            out.write(b'{"error": "kaput"}\\n'); out.flush(); continue
        if req["text"].startswith("slow"):
            time.sleep(0.3)
        # Encode what was received in the audio: n = len(text), value = emo[0]
        # (or, without a mood, the GPU this copy was given).
        n = len(req["text"])
        v = req["emo"][0] if req["emo"] else gpu
        out.write(json.dumps({"n": n}).encode() + b"\\n" + struct.pack("<%df" % n, *[v] * n))
        out.flush()
""")


def _fake_engine(tmp_path, monkeypatch, free_gb):
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER)
    monkeypatch.setattr(gpu, "query_free_gb", lambda: free_gb)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    eng = IndexTTSEngine(TTSConfig())
    monkeypatch.setattr(eng, "_command", lambda: [sys.executable, str(worker)])
    asyncio.run(eng.load())
    return eng


@pytest.fixture
def engine(tmp_path, monkeypatch):
    eng = _fake_engine(tmp_path, monkeypatch, {})       # no nvidia-smi: one copy
    yield eng
    asyncio.run(eng.close())


def test_worker_round_trip_with_moods(engine):
    assert engine.sample_rate == 16000

    async def run():
        first = [a async for a in engine.synthesize_stream("<excited> Yes! <sad> Oh.")]
        carried = await engine.synthesize("Still sad.")
        return first, carried

    first, carried = asyncio.run(run())
    assert [len(a) for a in first] == [len("Yes!"), len("Oh.")]
    assert first[0][0] == pytest.approx(0.68) and first[1][0] == 0.0   # excited / sad
    assert len(carried) == len("Still sad.") and carried[0] == 0.0     # sad carried over


def test_worker_error_returns_silence_and_keeps_serving(engine):
    async def run():
        bad = await engine.synthesize("boom")
        good = await engine.synthesize("fine")
        return bad, good

    bad, good = asyncio.run(run())
    assert len(bad) == 0 and len(good) == 4 and good[0] == -1.0        # no mood = None


def test_two_copies_render_ahead_in_order(tmp_path, monkeypatch):
    eng = _fake_engine(tmp_path, monkeypatch, {0: 20.0, 1: 20.0})
    try:
        chunks = [f"slow part {i}." for i in range(4)]
        t0 = time.monotonic()
        for c in chunks:                   # queued for speech -> rendering starts
            eng.prepare(c)

        async def speak():
            return [await eng.synthesize(c) for c in chunks]

        audio = asyncio.run(speak())
        elapsed = time.monotonic() - t0
        assert [len(a) for a in audio] == [len(c) for c in chunks]     # in order
        assert {float(a[0]) for a in audio} == {0.0, 1.0}             # both GPUs worked
        assert elapsed < 4 * 0.3                                      # side by side
    finally:
        asyncio.run(eng.close())


def test_barge_in_cancels_rendering_ahead(engine):
    engine.prepare("<sad> slow one.")
    engine.prepare("slow two.")
    engine.prepare("slow three.")
    engine.cancel_pending()
    assert not engine._pending

    async def run():
        return await engine.synthesize("after barge-in")

    assert len(asyncio.run(run())) == len("after barge-in")


def test_choose_gpus():
    free = {0: 10.0, 1: 12.0}
    assert choose_gpus("auto", "cuda", free, None, 7.0) == [0, 1]
    assert choose_gpus("auto", "cuda:1", free, None, 7.0) == [1, 0]
    assert choose_gpus("auto", "cuda", {0: 10.0, 1: 3.0}, None, 7.0) == [0]  # no room
    assert choose_gpus("1", "cuda", free, None, 7.0) == [1]                   # pinned
    assert choose_gpus("0, 1", "cuda", free, None, 7.0) == [0, 1]
    assert choose_gpus("7", "cuda", free, None, 7.0) == [0, 1]   # unknown GPU -> auto
    assert choose_gpus("auto", "cuda:0", free, "1", 7.0) == [1]  # VCB_CUDA_DEVICES=1
    assert choose_gpus("auto", "cuda", {}, None, 7.0) is None    # can't list GPUs


def test_missing_runtime_explains_how_to_install(tmp_path):
    cfg = TTSConfig()
    cfg.indextts_dir = str(tmp_path / "nothing-here")
    with pytest.raises(RuntimeError, match="VCB.bat"):
        asyncio.run(IndexTTSEngine(cfg).load())


def test_no_gpu_falls_back_to_kokoro():
    cfg = AppConfig()
    assert adapt_to_hardware(cfg, "indextts", backend="cuda") == "indextts"
    assert adapt_to_hardware(cfg, "indextts", backend="cpu") == "kokoro"
    # An AMD GPU can't run IndexTTS' CUDA runtime, but can still clone the voice.
    assert adapt_to_hardware(AppConfig(), "indextts", backend="rocm") == "chatterbox-turbo"


def test_chunker_overrides_keep_whole_sentences():
    cfg = dataclasses.replace(ChunkerConfig(), **IndexTTSEngine.chunker_overrides)
    chunker = SentenceChunker(cfg)
    reply = ("<happy> Hey! So good to see you. Are you thinking about firing up "
             "Rocket League today, or are you just relaxing for now? Either way, "
             "I'm here. Tell me how your last match went.")
    chunks = [c for tok in reply.split(" ") for c in chunker.push(tok + " ")] + chunker.flush()
    assert chunks[0].startswith("<happy> Hey! So good to see you. Are you thinking")
    assert chunks[0].endswith("for now?")                 # not cut at the comma
    assert all(c[-1] in ".!?" for c in chunks)


def test_trim_voice_sample_cuts_at_a_pause(tmp_path):
    sr = 16000
    t = np.arange(int(20 * sr)) / sr
    audio = 0.5 * np.sin(2 * np.pi * 220 * t).astype(np.float32)
    audio[int(10.0 * sr):int(10.3 * sr)] = 0.0            # a pause at 10 s
    src = tmp_path / "voice.wav"
    sf.write(src, audio, sr)
    out = trim_voice_sample(src, tmp_path / "cache")
    assert 10.0 <= sf.info(out).duration <= 10.3
    short = tmp_path / "short.wav"
    sf.write(short, audio[: 5 * sr], sr)
    assert trim_voice_sample(short, tmp_path / "cache") == short
