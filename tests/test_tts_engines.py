"""Orpheus and Chatterbox-Turbo engine glue: tag handling, sampling options and
model selection (exercised with fakes, no GPU or model download needed)."""

import asyncio
import dataclasses
import threading

import huggingface_hub
import numpy as np
import pytest
import torch

from vcb.chunker import SentenceChunker
from vcb.config import ChunkerConfig, TTSConfig
from vcb.sanitize import sanitize_for_speech
from vcb.tts.chatterbox_turbo_tts import ChatterboxTurboTTS, _to_turbo_tags
from vcb.tts.orpheus_tts import (
    OrpheusTTS,
    _StoppableLLM,
    resolve_model,
    to_orpheus_tags,
)


def test_orpheus_tags_map_or_drop_what_it_cannot_voice():
    assert to_orpheus_tags("<giggle> Stop it!") == "<chuckle> Stop it!"
    assert to_orpheus_tags("Oh. <gulp> Okay <sigh> fine.") == "Oh. Okay <sigh> fine."
    assert to_orpheus_tags("<LAUGH> ha") == "<laugh> ha"
    assert to_orpheus_tags("<gulp>") == ""


def test_orpheus_skips_text_with_nothing_to_voice():
    eng = OrpheusTTS(TTSConfig())
    eng._model = object()                # would raise if Orpheus were called

    async def run():
        whole = await eng.synthesize("<gulp>")
        streamed = [a async for a in eng.synthesize_stream("<gulp>")]
        return whole, streamed

    whole, streamed = asyncio.run(run())
    assert len(whole) == 0 and streamed == []


def test_stoppable_llm_adds_repeat_penalty_and_stop():
    seen = {}

    def llm(prompt, **kw):
        seen.update(kw)
        return "ok"

    stop = threading.Event()
    wrapped = _StoppableLLM(llm, stop, list, repeat_penalty=1.1)
    assert wrapped("p", max_tokens=5) == "ok"
    assert seen["repeat_penalty"] == 1.1 and seen["max_tokens"] == 5
    (criterion,) = seen["stopping_criteria"]
    assert not criterion(None, None)
    stop.set()
    assert criterion(None, None)
    wrapped("p", repeat_penalty=1.3)     # an explicit value still wins
    assert seen["repeat_penalty"] == 1.3


def test_orpheus_defaults():
    cfg = TTSConfig()
    assert cfg.orpheus_repeat_penalty >= 1.1                 # upstream: needed for stability
    assert cfg.orpheus_model == ""                           # built-in Q4_K_M: Q8 too slow


def test_resolve_orpheus_model(tmp_path, monkeypatch):
    assert resolve_model("") is None                         # orpheus-cpp's built-in
    local = tmp_path / "orpheus.gguf"
    local.write_bytes(b"gguf")
    assert resolve_model(str(local)) == str(local)

    calls = []
    monkeypatch.setattr(huggingface_hub, "hf_hub_download",
                        lambda repo_id, filename: calls.append((repo_id, filename)) or "x")
    assert resolve_model("owner/repo/sub/model-Q8_0.gguf") == "x"
    assert calls == [("owner/repo", "sub/model-Q8_0.gguf")]

    for bad in ("owner/model.gguf", str(tmp_path / "missing.gguf"), "owner/repo/file.bin"):
        with pytest.raises(RuntimeError, match="VCB_ORPHEUS_MODEL"):
            resolve_model(bad)


def test_orpheus_first_chunk_is_a_whole_sentence():
    cfg = dataclasses.replace(ChunkerConfig(), **OrpheusTTS.chunker_overrides)
    chunker = SentenceChunker(cfg)
    reply = ("I hear you, feeling drained can make everything feel heavier. If you "
             "could take a tiny break, what would be the most relaxing thing?")
    chunks = [c for tok in reply.split(" ") for c in chunker.push(tok + " ")] + chunker.flush()
    assert chunks[0] == "I hear you, feeling drained can make everything feel heavier."


def test_turbo_extra_tags_survive_sanitizing():
    keep = ChatterboxTurboTTS.extra_tags
    spoken = sanitize_for_speech("<whisper> It's a secret. <giggle> <sarcastic> Sure.",
                                 True, keep)
    assert _to_turbo_tags(spoken) == "[whispering] It's a secret. [chuckle] [sarcastic] Sure."
    # Without the engine's extra tags the sanitizer drops them, as before.
    assert sanitize_for_speech("<whisper> Hi.", True) == "Hi."


def test_turbo_advertises_only_tags_it_keeps():
    import re
    advertised = set(re.findall(r"<([a-z_]+)>", ChatterboxTurboTTS.prompt_instructions))
    assert advertised and advertised <= ChatterboxTurboTTS.extra_tags


def test_turbo_passes_sampling_settings():
    cfg = TTSConfig()
    cfg.chatterbox_temperature, cfg.chatterbox_repetition_penalty = 0.6, 1.4
    seen = {}

    class FakeTurbo:
        def generate(self, text, **kw):
            seen.update(kw, text=text)
            return torch.zeros(1, 10)

    eng = ChatterboxTurboTTS(cfg)
    eng._model = FakeTurbo()
    audio = asyncio.run(eng.synthesize("<laugh> Hi."))
    assert audio.dtype == np.float32 and len(audio) == 10
    assert seen["text"] == "[laugh] Hi."
    assert seen["temperature"] == 0.6 and seen["repetition_penalty"] == 1.4
    assert seen["top_p"] == cfg.chatterbox_top_p and seen["top_k"] == cfg.chatterbox_top_k
