"""Tests of the four optional TTS adapters: Breeze TTS 2, Voxtral, Step-Audio-EditX and Qwen3-TTS.

No model and none of their libraries is loaded. Each test puts a stand-in where the real library would be (for
Voxtral, a small HTTP server on 127.0.0.1 where vLLM-Omni would be) and checks what the adapter asks of it: the call
shapes, the validation and its messages, the tag handling and the cleanup. Whether the models then sound right is
only found out by running them."""

from __future__ import annotations

import asyncio
import contextlib
import importlib
import io
import json
import os
import random
import subprocess
import sys
import textwrap
import threading
import time
import types
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from vcb.config import TTSConfig, resolve_path
from vcb.tts import breeze_tts, qwen3_tts, step_audio_editx_tts, voxtral_tts

try:
    import soundfile        # imported up front: its loader reads sys.platform, which some tests pretend to change
except ImportError:         # pragma: no cover
    soundfile = None

needs_soundfile = pytest.mark.skipif(soundfile is None, reason="soundfile is not installed")
REPO_ROOT = Path(__file__).resolve().parents[1]
TOOLS_DIR = REPO_ROOT / "tools"


# -- helpers -------------------------------------------------------------------

def config(clip="", device="cuda") -> TTSConfig:
    """What the benchmark hands an engine: a clip means clone mode, no clip the built-in voice."""
    return TTSConfig(voice_wav=str(clip), device=device)


async def speak(engine, *sentences):
    """Load an engine, say the sentences, unload it."""
    await engine.load()
    try:
        return [await engine.synthesize(s) for s in sentences]
    finally:
        await engine.close()


def write_tree(root: Path, files: dict) -> Path:
    for name, source in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(source), encoding="utf-8")
    importlib.invalidate_caches()
    return root


def reference_clip(folder: Path) -> Path:
    clip = folder / "ref.wav"
    clip.write_bytes(b"RIFF")           # the adapters only check that it exists; the engine reads it
    return clip


CHECKOUT_MODULES = {"breeze_infer", "models", "tokenizer", "tts", "model_loader"}


@pytest.fixture(autouse=True)
def clean_imports(monkeypatch):
    """Breeze and Step-Audio-EditX put a repository checkout on sys.path and import its modules by plain names
    (`tts`, `models`, ...). Whatever a test imported that way goes away afterwards, and sys.path is restored."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in [n for n in sys.modules if n.split(".")[0] in CHECKOUT_MODULES]:
        monkeypatch.delitem(sys.modules, name)
    yield
    for name in [n for n in sys.modules if n.split(".")[0] in CHECKOUT_MODULES]:
        del sys.modules[name]


class Hub(types.ModuleType):
    """Stand-in for huggingface_hub that remembers what was asked of it."""

    def __init__(self):
        super().__init__("huggingface_hub")
        self.calls, self.error = [], None

    def snapshot_download(self, repo, **kwargs):
        self.calls.append((repo, kwargs))
        if self.error is not None:
            raise self.error
        return f"/cache/{repo}"


@pytest.fixture()
def hub(monkeypatch):
    stand_in = Hub()
    monkeypatch.setitem(sys.modules, "huggingface_hub", stand_in)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    return stand_in


class FakeTensor:
    """The part of a torch tensor the adapters use to get numpy audio (bfloat16 needs .float() before .numpy())."""

    def __init__(self, values):
        self.values, self.steps = np.asarray(values, dtype=np.float32), []

    def detach(self):
        self.steps.append("detach")
        return self

    def float(self):
        self.steps.append("float")
        return self

    def cpu(self):
        self.steps.append("cpu")
        return self

    def numpy(self):
        self.steps.append("numpy")
        return self.values


# -- every engine's own sound-tag syntax ---------------------------------------

CONVERTERS = {
    "breeze": (breeze_tts.to_breeze_tags,
               {"laugh": "(laugh)", "sigh": "(sigh)", "breath": "(inhales)", "hesitation": "(pause)"}),
    "step-audio-editx": (step_audio_editx_tts.to_step_audio_tags,
                         {"laugh": "[laugh]", "sigh": "[sigh]", "breath": "[breath]", "hesitation": "[uhm]"}),
    "qwen3-tts": (qwen3_tts.to_qwen3_text, {}),       # no inline tags: dropped, never read aloud
    "voxtral": (voxtral_tts.to_voxtral_text, {}),
}
ENGINE_TAGS = [pytest.param(key, tag, id=f"{key}-{tag}")
               for key, (_, table) in CONVERTERS.items() for tag in ("laugh", "sigh", "breath", "hesitation")]


@pytest.mark.parametrize("key, tag", ENGINE_TAGS)
def test_each_canonical_tag_becomes_the_engines_own_tag_or_is_dropped(key, tag):
    convert, table = CONVERTERS[key]
    assert convert(f"Okay <{tag}> then.") == (f"Okay {table[tag]} then." if tag in table else "Okay then.")


@pytest.mark.parametrize("key", list(CONVERTERS))
def test_tag_conversion_keeps_spacing_and_punctuation_tidy(key):
    convert, table = CONVERTERS[key]
    laugh = table.get("laugh")
    assert convert("That is funny <laugh>.") == (f"That is funny {laugh}." if laugh else "That is funny.")
    assert convert("Hello  <whisper>  there") == "Hello there"                     # unknown tags are never spoken
    assert convert("<whisper>Hello") == "Hello"
    assert convert("Well, <HESITATION> maybe.") == ("Well, " + (table.get("hesitation", "") + " ").lstrip() + "maybe.")
    assert convert("Plain text, no tags.") == "Plain text, no tags."
    assert convert("if a < b and b > c") == "if a < b and b > c"                  # not a tag


def test_a_sentence_of_nothing_but_unsupported_tags_is_left_empty():
    assert breeze_tts.to_breeze_tags("<whisper>") == ""
    assert qwen3_tts.to_qwen3_text("<laugh>") == "" and voxtral_tts.to_voxtral_text("<laugh>") == ""
    assert breeze_tts.to_breeze_tags("<laugh>") == "(laugh)"                       # Breeze can voice it alone


# -- where the weights come from -----------------------------------------------

ADAPTERS = [
    pytest.param(breeze_tts, "breeze", id="breeze"),
    pytest.param(qwen3_tts, "qwen3-tts", id="qwen3"),
    pytest.param(step_audio_editx_tts, "step-audio-editx", id="step-audio"),
    pytest.param(voxtral_tts, "voxtral", id="voxtral"),
]


@pytest.mark.parametrize("module, key", ADAPTERS)
def test_a_model_folder_is_used_as_it_is_and_never_looked_up_on_the_hub(module, key, hub, tmp_path):
    assert module._snapshot(str(tmp_path)) == resolve_path(str(tmp_path))
    assert hub.calls == []


@pytest.mark.parametrize("module, key", ADAPTERS)
@pytest.mark.parametrize("value, offline", [("1", True), ("TRUE", True), (" yes ", True), ("On", True),
                                            ("", False), ("0", False), ("false", False)])
def test_a_hub_id_comes_from_the_local_cache_when_the_environment_is_offline(module, key, hub, monkeypatch,
                                                                              value, offline):
    monkeypatch.setenv("HF_HUB_OFFLINE", value)
    assert module._snapshot("owner/model") == "/cache/owner/model"
    assert hub.calls == [("owner/model", {"local_files_only": offline})]


@pytest.mark.parametrize("module, key", ADAPTERS)
def test_missing_weights_say_how_to_download_them(module, key, hub, monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    hub.error = OSError("not cached")
    with pytest.raises(FileNotFoundError) as caught:
        module._snapshot("owner/model")
    assert str(caught.value) == ("could not get owner/model from the Hugging Face cache (OSError); "
                                 f"download it first: python tools/tts_benchmark.py download --engines {key}")
    assert caught.value.__cause__ is hub.error


# -- Qwen3-TTS -----------------------------------------------------------------

@pytest.fixture()
def qwen(monkeypatch, hub):
    """Stand-ins for torch and qwen_tts. `calls` lists what the adapter asked of the model, in order."""
    state = types.SimpleNamespace(calls=[], audio=np.full(2400, 0.25, dtype=np.float32), rate=24000)

    class Qwen3TTSModel:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            state.calls.append(("from_pretrained", path, kwargs))
            return cls()

        def create_voice_clone_prompt(self, **kwargs):
            state.calls.append(("create_voice_clone_prompt", kwargs))
            return "PROMPT"

        def generate_voice_clone(self, **kwargs):
            state.calls.append(("generate_voice_clone", kwargs))
            return [state.audio], state.rate

        def generate_custom_voice(self, **kwargs):
            state.calls.append(("generate_custom_voice", kwargs))
            return [state.audio], state.rate

    library = types.ModuleType("qwen_tts")
    library.Qwen3TTSModel = Qwen3TTSModel
    torch = types.ModuleType("torch")
    torch.bfloat16, torch.float16, torch.float32 = "bf16", "f16", "f32"
    monkeypatch.setitem(sys.modules, "qwen_tts", library)
    monkeypatch.setitem(sys.modules, "torch", torch)
    return state


def load_args(repo, **extra):
    return ("from_pretrained", f"/cache/{repo}",
            {"device_map": "cuda:0", "dtype": "bf16", "attn_implementation": "sdpa", **extra})


def test_qwen3_default_voice_uses_the_custom_voice_model_and_a_built_in_speaker(qwen):
    engine = qwen3_tts.Qwen3TTS(config())
    (audio,) = asyncio.run(speak(engine, "Hello <laugh> there."))
    assert qwen.calls == [
        load_args("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"),
        ("generate_custom_voice", {"text": "Hello there.", "language": "English", "speaker": "Ryan", "instruct": None}),
    ]
    assert audio.dtype == np.float32 and audio.shape == (2400,)


def test_qwen3_clone_uses_the_base_model_and_analyses_the_clip_only_once(qwen, tmp_path):
    clip = reference_clip(tmp_path)
    options = qwen3_tts.Qwen3TTS.Options(reference_text="  Hello there.  ")
    asyncio.run(speak(qwen3_tts.Qwen3TTS(config(clip), options), "One.", "Two."))
    prompt = {"ref_audio": resolve_path(str(clip)), "ref_text": "Hello there.", "x_vector_only_mode": False}
    assert qwen.calls == [
        load_args("Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        ("create_voice_clone_prompt", prompt),
        ("generate_voice_clone", {"text": "One.", "language": "English", "voice_clone_prompt": "PROMPT"}),
        ("generate_voice_clone", {"text": "Two.", "language": "English", "voice_clone_prompt": "PROMPT"}),
    ]


def test_qwen3_can_clone_from_the_voice_alone_without_a_transcript(qwen, tmp_path):
    options = qwen3_tts.Qwen3TTS.Options(x_vector_only=True)
    asyncio.run(speak(qwen3_tts.Qwen3TTS(config(reference_clip(tmp_path)), options), "One."))
    assert qwen.calls[1][1]["ref_text"] is None and qwen.calls[1][1]["x_vector_only_mode"] is True


def test_qwen3_settings_reach_the_model(qwen):
    options = qwen3_tts.Qwen3TTS.Options(speaker="Aiden", language="Chinese", instruct="Calm.", dtype="float16",
                                         attn_implementation="eager", custom_voice_model="Qwen/Other")
    asyncio.run(speak(qwen3_tts.Qwen3TTS(config(device="cuda:1"), options), "Hi."))
    assert qwen.calls[0] == ("from_pretrained", "/cache/Qwen/Other",
                             {"device_map": "cuda:1", "dtype": "f16", "attn_implementation": "eager"})
    assert qwen.calls[1][1] == {"text": "Hi.", "language": "Chinese", "speaker": "Aiden", "instruct": "Calm."}


def test_qwen3_follows_the_sample_rate_the_model_reports(qwen):
    qwen.rate = 22050
    engine = qwen3_tts.Qwen3TTS(config())
    asyncio.run(speak(engine, "Hi."))
    assert engine.sample_rate == 22050 and qwen3_tts.Qwen3TTS.sample_rate == 24000


def test_qwen3_converts_tensors_to_float32_numpy(qwen):
    qwen.audio = FakeTensor([[0.5, -0.5, 0.25]])
    (audio,) = asyncio.run(speak(qwen3_tts.Qwen3TTS(config()), "Hi."))
    assert qwen.audio.steps == ["detach", "float", "cpu", "numpy"]
    assert audio.dtype == np.float32 and audio.tolist() == [0.5, -0.5, 0.25]


@pytest.mark.parametrize("make, error, message", [
    (lambda tmp: qwen3_tts.Qwen3TTS(config(), qwen3_tts.Qwen3TTS.Options(dtype="int8")),
     ValueError, "dtype must be one of bfloat16, float16, float32, not 'int8'"),
    (lambda tmp: qwen3_tts.Qwen3TTS(config(tmp / "missing.wav"), qwen3_tts.Qwen3TTS.Options(reference_text="x")),
     FileNotFoundError, "reference clip not found"),
    (lambda tmp: qwen3_tts.Qwen3TTS(config(reference_clip(tmp))),
     ValueError, "exact transcript of the reference clip"),
])
def test_qwen3_refuses_settings_it_cannot_work_with(qwen, tmp_path, make, error, message):
    with pytest.raises(error, match=message):
        asyncio.run(make(tmp_path).load())


def test_qwen3_says_so_when_only_unsupported_tags_are_left(qwen):
    with pytest.raises(ValueError, match="nothing left to say"):
        asyncio.run(speak(qwen3_tts.Qwen3TTS(config()), "<whisper>"))


def test_qwen3_must_be_loaded_again_after_close(qwen):
    engine = qwen3_tts.Qwen3TTS(config())

    async def scenario():
        await engine.load()
        await engine.close()
        await engine.synthesize("Hi.")

    with pytest.raises(AssertionError, match="call load"):
        asyncio.run(scenario())


# -- Breeze TTS 2 --------------------------------------------------------------

BREEZE_FILES = {
    "breeze_infer/__init__.py": "",
    "breeze_infer/runtime.py": """
        CALLS = []

        def load_runtime(ckpt, device, attn_implementation):
            CALLS.append(("load_runtime", str(ckpt), device, attn_implementation))
            return "tokenizer", "model", "codec"

        def update_generation_config_for_breeze(model):
            CALLS.append(("update_generation_config", model))

        def set_all_seeds(seed):
            CALLS.append(("set_all_seeds", seed))
        """,
    "breeze_infer/templates.py": """
        from breeze_infer.runtime import CALLS

        def select_template_name(request):
            return "clone" if "ref_audio_path" in request else "default"

        def get_template(name):
            return "template:" + name

        def prepare_inputs(tokenizer, audio_tokenizer, model, requests, template, guidance_scale,
                           guidance_scale_ref, guidance_scale_ins):
            CALLS.append(("prepare_inputs", requests, template, guidance_scale, guidance_scale_ref,
                          guidance_scale_ins))
            return "inputs"
        """,
    "models/__init__.py": "",
    "models/fast_streaming.py": """
        import numpy as np
        from breeze_infer.runtime import CALLS

        KEEP = []

        class FastStreamingConfig:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

        class Chunk:
            def __init__(self, audio):
                self.audio = audio

        class FastBreezeStreamingRuntime:
            sample_rate = 22050

            def __init__(self, model, audio_tokenizer, config, tokenizer=None):
                CALLS.append(("runtime", model, audio_tokenizer, config.kwargs, tokenizer))

            def iter_audio_chunks(self, inputs, request_id, seed):
                CALLS.append(("iter_audio_chunks", inputs, request_id, seed))
                generator = self._parts()
                KEEP.append(generator)      # kept alive: only an explicit close() runs its cleanup, not garbage collection
                return generator

            def _parts(self):
                try:
                    for part in ([0.1] * 4, [], [0.2] * 2):
                        CALLS.append(("chunk", len(part)))
                        yield Chunk(np.array(part, dtype=np.float32))
                finally:
                    CALLS.append(("generator_closed",))
        """,
}


@pytest.fixture()
def breeze(tmp_path, monkeypatch):
    """A fake breeze-tts checkout, a fake Breeze-TTS-2 checkpoint and a stand-in for torch."""
    repo = write_tree(tmp_path / "breeze-tts", BREEZE_FILES)
    checkpoint = tmp_path / "Breeze-TTS-2"
    (checkpoint / "audio_tokenizer").mkdir(parents=True)
    torch = types.ModuleType("torch")
    torch.no_grad = contextlib.nullcontext
    monkeypatch.setitem(sys.modules, "torch", torch)

    def calls():
        runtime = sys.modules.get("breeze_infer.runtime")
        return runtime.CALLS if runtime else []

    options = breeze_tts.BreezeTTS.Options(repo_dir=str(repo), checkpoint=str(checkpoint),
                                           reference_text="  Hello there.  ")
    return types.SimpleNamespace(repo=repo, checkpoint=checkpoint, options=options, calls=calls, tmp=tmp_path)


def test_breeze_loads_the_checkpoint_through_the_checkouts_own_code(breeze):
    engine = breeze_tts.BreezeTTS(config(), breeze.options)
    asyncio.run(speak(engine))
    assert breeze.calls() == [
        ("load_runtime", resolve_path(str(breeze.checkpoint)), "cuda:0", "eager"),
        ("update_generation_config", "model"),
        ("runtime", "model", "codec",
         {"max_new_tokens": 1500, "max_seq_len": 2048, "fast_all": False, "fast_text_encoder": False,
          "fast_backbone_prefill": False, "fast_backbone_decode": False, "fast_depth_decoder": False,
          "fast_codec": False, "repetition_penalty": 1.1}, "tokenizer"),
    ]
    assert engine.sample_rate == 22050                          # taken from the runtime, not assumed
    assert sys.path[0] == resolve_path(str(breeze.repo))


def test_breeze_speaks_the_request_it_builds_and_drops_empty_chunks(breeze):
    engine = breeze_tts.BreezeTTS(config(), breeze.options)

    async def scenario():
        await engine.load()
        return [part async for part in engine.synthesize_stream("Hello <laugh> there.")]

    parts = asyncio.run(scenario())
    assert [p.tolist() for p in parts] == [pytest.approx([0.1] * 4), pytest.approx([0.2] * 2)]
    calls = breeze.calls()
    request = {"id": "single-request", "text": "Hello (laugh) there.", "speaker": "S0"}
    assert ("prepare_inputs", [request], "template:default", 1.0, None, None) in calls
    (seeded,) = [c for c in calls if c[0] == "set_all_seeds"]
    (started,) = [c for c in calls if c[0] == "iter_audio_chunks"]
    assert started[:3] == ("iter_audio_chunks", "inputs", "single-request") and started[3] == seeded[1]
    assert 0 <= seeded[1] < 2**31
    assert calls[-1] == ("generator_closed",)


def test_breeze_clone_sends_the_clip_and_its_transcript(breeze):
    clip = reference_clip(breeze.tmp)
    audio = asyncio.run(speak(breeze_tts.BreezeTTS(config(clip), breeze.options), "Hi there."))[0]
    request = {"id": "single-request", "text": "Hi there.", "speaker": "S0",
               "ref_audio_path": resolve_path(str(clip)), "ref_text": "Hello there."}
    assert ("prepare_inputs", [request], "template:clone", 1.0, None, None) in breeze.calls()
    assert audio.dtype == np.float32 and audio.tolist() == pytest.approx([0.1] * 4 + [0.2] * 2)


def test_breeze_streams_chunk_by_chunk_instead_of_computing_ahead(breeze):
    engine = breeze_tts.BreezeTTS(config(), breeze.options)

    async def scenario():
        await engine.load()
        stream = engine.synthesize_stream("Hello there.")
        first = await stream.__anext__()
        pulled = [c for c in breeze.calls() if c[0] == "chunk"]
        await stream.aclose()                                   # the caller gives up after the first chunk
        return first, pulled

    first, pulled = asyncio.run(scenario())
    assert len(first) == 4 and pulled == [("chunk", 4)]
    assert ("generator_closed",) in breeze.calls()              # and the engine's generator is closed, not leaked


def test_breeze_seeds_are_reproducible_but_differ_from_run_to_run(breeze):
    engine = breeze_tts.BreezeTTS(config(), breeze.options)

    async def scenario():
        await engine.load()
        seen = []
        for seed in (1234, 1234, 99):
            random.seed(seed)                                   # the benchmark seeds `random` before each run
            breeze.calls().clear()
            await engine.synthesize("One.")
            await engine.synthesize("Two.")
            seen.append([c[1] for c in breeze.calls() if c[0] == "set_all_seeds"])
        return seen

    state = random.getstate()
    try:
        first, again, other = asyncio.run(scenario())
    finally:
        random.setstate(state)
    assert first == again and first != other and first[0] != first[1]


@pytest.mark.parametrize("settings, build, error, message", [
    ("no repo_dir", lambda b: b.options.__setattr__("repo_dir", "") or config(), ValueError, "needs repo_dir"),
    ("not a checkout", lambda b: b.options.__setattr__("repo_dir", str(b.tmp)) or config(),
     FileNotFoundError, r"not a breeze-tts checkout \(missing breeze_infer/runtime.py, models/fast_streaming.py\)"),
    ("no clip file", lambda b: config(b.tmp / "missing.wav"), FileNotFoundError, "reference clip not found"),
    ("no transcript", lambda b: b.options.__setattr__("reference_text", " ") or config(reference_clip(b.tmp)),
     ValueError, "exact transcript of the reference clip"),
    ("cpu", lambda b: config(device="cpu"), RuntimeError, "Breeze TTS 2 needs a CUDA GPU"),
    ("wrong checkpoint", lambda b: (b.checkpoint / "audio_tokenizer").rmdir() or config(),
     FileNotFoundError, "has no audio_tokenizer/ folder"),
])
def test_breeze_refuses_settings_it_cannot_work_with(breeze, settings, build, error, message):
    cfg = build(breeze)
    with pytest.raises(error, match=message):
        asyncio.run(breeze_tts.BreezeTTS(cfg, breeze.options).load())


def test_breeze_says_so_when_only_unsupported_tags_are_left(breeze):
    with pytest.raises(ValueError, match="nothing left to say"):
        asyncio.run(speak(breeze_tts.BreezeTTS(config(), breeze.options), "<whisper>"))


def test_breeze_must_be_loaded_again_after_close(breeze):
    engine = breeze_tts.BreezeTTS(config(), breeze.options)

    async def scenario():
        await engine.load()
        await engine.close()
        return [c async for c in engine.synthesize_stream("Hi.")]

    with pytest.raises(AssertionError, match="call load"):
        asyncio.run(scenario())


# -- Step-Audio-EditX ----------------------------------------------------------

STEP_FILES = {
    "model_loader.py": "",
    "tokenizer.py": """
        class StepAudioTokenizer:
            def __init__(self, tokenizer_dir, model_source):
                self.args = (tokenizer_dir, model_source)
        """,
    "tts.py": """
        import os
        import numpy as np

        EVENTS = []

        class StepAudioTTS:
            fail_edit = False

            def __init__(self, model_dir, audio_tokenizer, **kwargs):
                EVENTS.append(("init", model_dir, audio_tokenizer.args, kwargs, os.getcwd(),
                               os.environ.get("VLLM_ATTENTION_BACKEND")))

            def clone(self, prompt_wav_path, prompt_text, target_text):
                EVENTS.append(("clone", prompt_wav_path, prompt_text, target_text))
                return np.full((1, 2400), 0.1, dtype=np.float32), 24000

            def edit(self, prompt_wav_path, prompt_text, edit_type, target_text):
                import soundfile
                first_pass, rate = soundfile.read(prompt_wav_path, dtype="float32")
                EVENTS.append(("edit", prompt_wav_path, prompt_text, edit_type, target_text, first_pass, rate))
                if self.fail_edit:
                    raise RuntimeError("edit failed")
                return np.full((1, 4800), 0.2, dtype=np.float32), 22050
        """,
}


@pytest.fixture()
def step(tmp_path, monkeypatch):
    """A fake Step-Audio-EditX checkout and fake model folders. The platform is Linux, which vLLM needs."""
    repo = write_tree(tmp_path / "Step-Audio-EditX", STEP_FILES)
    (tmp_path / "model").mkdir()
    (tmp_path / "tokenizer").mkdir()
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "-")            # so that the test's own change is undone afterwards
    monkeypatch.delenv("VLLM_ATTENTION_BACKEND")
    options = step_audio_editx_tts.StepAudioEditXTTS.Options(
        repo_dir=str(repo), model=str(tmp_path / "model"), tokenizer=str(tmp_path / "tokenizer"),
        reference_text="  Hello there.  ")

    def events():
        module = sys.modules.get("tts")
        return module.EVENTS if module else []

    return types.SimpleNamespace(repo=repo, options=options, events=events, tmp=tmp_path,
                                 clip=reference_clip(tmp_path))


def test_step_audio_loads_the_official_classes_with_the_cli_defaults(step):
    engine = step_audio_editx_tts.StepAudioEditXTTS(config(step.clip), step.options)
    asyncio.run(speak(engine))
    ((_, model_dir, tokenizer_args, kwargs, _, backend),) = step.events()
    assert (model_dir, tokenizer_args) == (resolve_path(str(step.tmp / "model")),
                                           (resolve_path(str(step.tmp / "tokenizer")), "local"))
    assert kwargs == {"model_source": "local", "gpu_memory_utilization": 0.5, "max_model_len": 3072,
                      "max_num_seqs": 1, "dtype": "bfloat16", "enforce_eager": False,
                      "cosyvoice_dtype": "bfloat16", "cosyvoice_cuda_graph": True}
    assert backend == "TRITON_ATTN"                              # as the official CLI sets it
    assert sys.path[0] == resolve_path(str(step.repo))


def test_step_audio_keeps_an_attention_backend_the_user_chose(step, monkeypatch):
    monkeypatch.setenv("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    asyncio.run(speak(step_audio_editx_tts.StepAudioEditXTTS(config(step.clip), step.options)))
    assert step.events()[0][-1] == "FLASH_ATTN"


def test_step_audio_can_run_from_inside_the_checkout(step, monkeypatch):
    monkeypatch.chdir(step.tmp)
    step.options.chdir_repo = True
    asyncio.run(speak(step_audio_editx_tts.StepAudioEditXTTS(config(step.clip), step.options)))
    assert Path(step.events()[0][4]).resolve() == step.repo.resolve()


def test_step_audio_clones_plain_text_in_a_single_pass(step):
    engine = step_audio_editx_tts.StepAudioEditXTTS(config(step.clip), step.options)
    (audio,) = asyncio.run(speak(engine, "The meeting starts at nine."))
    assert step.events()[1:] == [("clone", resolve_path(str(step.clip)), "Hello there.",
                                  "The meeting starts at nine.")]
    assert audio.shape == (2400,) and audio.dtype == np.float32 and engine.sample_rate == 24000


@needs_soundfile
def test_step_audio_adds_sound_tags_by_editing_what_it_just_cloned(step):
    engine = step_audio_editx_tts.StepAudioEditXTTS(config(step.clip), step.options)
    (audio,) = asyncio.run(speak(engine, "That was funny <laugh> and you know it <hesitation> really."))
    _, clone, edit = step.events()
    assert clone == ("clone", resolve_path(str(step.clip)), "Hello there.", "That was funny and you know it really.")
    _, first_pass, prompt_text, edit_type, target, heard, rate = edit
    assert (prompt_text, edit_type) == ("That was funny and you know it really.", "paralinguistic")
    assert target == "That was funny [laugh] and you know it [uhm] really."
    assert (len(heard), rate) == (2400, 24000) and heard == pytest.approx(np.full(2400, 0.1), abs=1e-3)
    assert not Path(first_pass).parent.exists()                  # the temporary folder is gone again
    assert audio.shape == (4800,) and engine.sample_rate == 22050 and audio[0] == pytest.approx(0.2)


@needs_soundfile
def test_step_audio_cleans_up_when_the_edit_fails(step):
    engine = step_audio_editx_tts.StepAudioEditXTTS(config(step.clip), step.options)

    async def scenario():
        await engine.load()
        sys.modules["tts"].StepAudioTTS.fail_edit = True
        await engine.synthesize("Oh <sigh> well.")

    with pytest.raises(RuntimeError, match="edit failed"):
        asyncio.run(scenario())
    first_pass = step.events()[-1][1]
    assert not Path(first_pass).parent.exists()


def test_step_audio_says_so_when_only_unsupported_tags_are_left(step):
    engine = step_audio_editx_tts.StepAudioEditXTTS(config(step.clip), step.options)
    with pytest.raises(ValueError, match="nothing left to say"):
        asyncio.run(speak(engine, "<whisper>"))


@pytest.mark.parametrize("what, build, error, message", [
    ("no repo_dir", lambda s: s.options.__setattr__("repo_dir", "") or config(s.clip), ValueError, "needs repo_dir"),
    ("not a checkout", lambda s: s.options.__setattr__("repo_dir", str(s.tmp)) or config(s.clip),
     FileNotFoundError, r"not a Step-Audio-EditX checkout \(missing tts.py, tokenizer.py, model_loader.py\)"),
    ("no clip", lambda s: config(), ValueError, "has no built-in voice"),
    ("no clip file", lambda s: config(s.tmp / "missing.wav"), FileNotFoundError, "reference clip not found"),
    ("no transcript", lambda s: s.options.__setattr__("reference_text", "") or config(s.clip),
     ValueError, "exact transcript of the reference clip"),
    ("cpu", lambda s: config(s.clip, device="cpu"), RuntimeError, "needs a CUDA GPU"),
])
def test_step_audio_refuses_settings_it_cannot_work_with(step, what, build, error, message):
    cfg = build(step)
    with pytest.raises(error, match=message):
        asyncio.run(step_audio_editx_tts.StepAudioEditXTTS(cfg, step.options).load())


def test_step_audio_refuses_to_start_on_windows_where_vllm_does_not_exist(step, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="no Windows build: run the benchmark inside WSL2"):
        asyncio.run(step_audio_editx_tts.StepAudioEditXTTS(config(step.clip), step.options).load())
    assert step.events() == []                                   # nothing of the checkout was imported


# -- Voxtral -------------------------------------------------------------------

SAMPLES = np.array([0, 16384, -16384, 32767, -32768, 1000], dtype="<i2")
EXPECTED = (SAMPLES / 32768.0).astype(np.float32)


def wav_bytes(samples, rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(rate)
        fh.writeframes(np.asarray(samples, dtype="<i2").tobytes())
    return buffer.getvalue()


class Speech(BaseHTTPRequestHandler):
    """The two endpoints of the vLLM-Omni server that the adapter uses."""

    def log_message(self, *args):
        pass

    def do_GET(self):
        status = self.server.health.pop(0) if self.server.health else 200
        self.server.requests.append(("GET", self.path, None, None))
        self.send_response(status)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.requests.append(("POST", self.path, body, self.headers["Content-Type"]))
        if self.server.mode == "error":
            payload = json.dumps({"error": "unknown voice 'nobody'"}).encode()
            self.send_response(400)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
        elif self.server.mode == "wav":
            self.send_response(200)
            self.send_header("Content-Length", str(len(self.server.wav)))
            self.end_headers()
            self.wfile.write(self.server.wav)
        else:                                   # raw PCM in small odd-sized pieces; the body ends with the connection
            self.send_response(200)
            self.end_headers()
            data = SAMPLES.tobytes()
            for start in range(0, len(data), 3):
                self.wfile.write(data[start:start + 3])
                time.sleep(0.005)


@pytest.fixture()
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Speech)
    httpd.daemon_threads = True
    httpd.requests, httpd.health, httpd.mode = [], [], "pcm"
    httpd.wav = wav_bytes(SAMPLES, 22050)
    httpd.port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    yield httpd
    httpd.shutdown()
    httpd.server_close()
    thread.join(5)


class FakeProc:
    """Stands in for the vLLM-Omni server process."""

    def __init__(self, exit_code=None, ignores_terminate=False):
        self.exit_code, self.ignores_terminate = exit_code, ignores_terminate
        self.terminated = self.killed = False

    def poll(self):
        return self.exit_code

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed, self.exit_code = True, -9

    def wait(self, timeout=None):
        if self.terminated and not self.killed and self.ignores_terminate:
            raise subprocess.TimeoutExpired("vllm", timeout)
        if self.exit_code is None:
            self.exit_code = -15
        return self.exit_code


@pytest.fixture()
def vllm(monkeypatch):
    """Nothing is launched: `started` records each command line, `proc` is the process that is returned. The
    platform is Linux, and time only moves when the adapter sleeps (`slept` lists the pauses), so waiting for the
    server takes no real time and does not depend on how fine the machine's clock is."""
    state = types.SimpleNamespace(started=[], proc=FakeProc(), now=0.0, slept=[])

    def popen(cmd, **kwargs):
        state.started.append((cmd, kwargs))
        return state.proc

    def sleep(seconds):
        state.now += seconds
        state.slept.append(seconds)

    monkeypatch.setattr(voxtral_tts, "subprocess", types.SimpleNamespace(
        Popen=popen, DEVNULL=subprocess.DEVNULL, TimeoutExpired=subprocess.TimeoutExpired))
    monkeypatch.setattr(voxtral_tts, "time", types.SimpleNamespace(monotonic=lambda: state.now, sleep=sleep))
    monkeypatch.setattr(sys, "platform", "linux")
    for name in ("VLLM_NO_USAGE_STATS", "DO_NOT_TRACK"):
        monkeypatch.delenv(name, raising=False)
    return state


def voxtral(server, **options) -> voxtral_tts.VoxtralTTS:
    return voxtral_tts.VoxtralTTS(config(), voxtral_tts.VoxtralTTS.Options(port=server.port, **options))


def posts(server):
    return [r for r in server.requests if r[0] == "POST"]


def test_voxtral_starts_vllm_omni_on_loopback_and_waits_until_it_is_healthy(hub, server, vllm):
    server.health = [503, 503]                                   # not ready for the first two polls
    asyncio.run(voxtral(server).load())
    ((cmd, kwargs),) = vllm.started
    assert cmd == [sys.executable, "-m", "vllm.entrypoints.cli.main", "serve", "mistralai/Voxtral-4B-TTS-2603",
                   "--omni", "--host", "127.0.0.1", "--port", str(server.port)]
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert (kwargs["env"]["VLLM_NO_USAGE_STATS"], kwargs["env"]["DO_NOT_TRACK"]) == ("1", "1")   # telemetry stays off
    assert [r[:2] for r in server.requests] == [("GET", "/health")] * 3
    assert vllm.slept == [2, 2]                                  # it polls every two seconds


def test_voxtral_serves_a_local_model_folder_under_its_configured_name(server, vllm, tmp_path):
    folder = tmp_path / "voxtral"
    folder.mkdir()
    asyncio.run(voxtral(server, model=str(folder), trust_remote_code=True, server_args=["--max-model-len", 4096]).load())
    ((cmd, _),) = vllm.started
    assert cmd[3:5] == ["serve", resolve_path(str(folder))]
    assert cmd[-5:] == ["--served-model-name", str(folder), "--trust-remote-code", "--max-model-len", "4096"]


def test_voxtral_leaves_a_telemetry_choice_of_the_user_alone(hub, server, vllm, monkeypatch):
    monkeypatch.setenv("DO_NOT_TRACK", "0")
    asyncio.run(voxtral(server).load())
    env = vllm.started[0][1]["env"]
    assert (env["DO_NOT_TRACK"], env["VLLM_NO_USAGE_STATS"]) == ("0", "1")


def test_voxtral_streams_pcm_and_decodes_samples_split_across_network_reads(hub, server, vllm):
    engine = voxtral(server)

    async def scenario():
        await engine.load()
        return [part async for part in engine.synthesize_stream("Hello <laugh> there.")]

    parts = asyncio.run(scenario())
    assert np.concatenate(parts).tolist() == EXPECTED.tolist()
    ((_, path, body, content_type),) = posts(server)
    assert (path, content_type) == ("/v1/audio/speech", "application/json")
    assert body == {"model": "mistralai/Voxtral-4B-TTS-2603", "input": "Hello there.", "voice": "casual_female",
                    "language": "English", "stream": True, "stream_format": "audio", "response_format": "pcm"}


@needs_soundfile
def test_voxtral_can_fetch_one_wav_instead_of_streaming(hub, server, vllm):
    server.mode = "wav"
    engine = voxtral(server, stream=False, voice="neutral_male", language="French")
    (audio,) = asyncio.run(speak(engine, "Bonjour."))
    assert audio.tolist() == pytest.approx(EXPECTED.tolist(), abs=1e-4) and engine.sample_rate == 22050
    assert posts(server)[0][2] == {"model": "mistralai/Voxtral-4B-TTS-2603", "input": "Bonjour.",
                                   "voice": "neutral_male", "language": "French", "response_format": "wav"}


def test_voxtral_hands_over_pcm_as_it_arrives_without_reading_ahead(hub, server, vllm, monkeypatch):
    data = SAMPLES.tobytes()
    pieces = [data[:3], data[3:6], data[6:8], data[8:]]          # every boundary but one falls inside a sample

    class Response:
        reads, closed = 0, False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            Response.closed = True
            return False

        def read1(self, n=-1):
            Response.reads += 1
            return pieces.pop(0) if pieces else b""

    engine = voxtral(server)

    async def scenario():
        await engine.load()
        monkeypatch.setattr(voxtral_tts, "_LOCAL", types.SimpleNamespace(open=lambda request, timeout=None: Response()))
        stream = engine.synthesize_stream("Hello.")
        first = await stream.__anext__()
        reads_after_first_chunk = Response.reads
        rest = [part async for part in stream]
        return first, reads_after_first_chunk, rest

    first, reads, rest = asyncio.run(scenario())
    assert reads == 1 and first.tolist() == EXPECTED[:1].tolist()
    assert np.concatenate([first, *rest]).tolist() == EXPECTED.tolist()


def test_voxtral_closes_the_connection_when_the_caller_stops_listening(hub, server, vllm, monkeypatch):
    class Response:
        closed = False

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            Response.closed = True
            return False

        def read1(self, n=-1):
            return SAMPLES.tobytes()

    engine = voxtral(server)
    kept = []                                                    # holds the generator, so only an explicit close ends it
    original = engine._chunks

    def keep(text):
        kept.append(original(text))
        return kept[-1]

    async def scenario():
        await engine.load()
        monkeypatch.setattr(engine, "_chunks", keep)
        monkeypatch.setattr(voxtral_tts, "_LOCAL", types.SimpleNamespace(open=lambda request, timeout=None: Response()))
        stream = engine.synthesize_stream("Hello.")
        await stream.__anext__()
        await stream.aclose()                                    # what a timed-out benchmark run does

    asyncio.run(scenario())
    assert kept and Response.closed


PROXY_PROBE = """
import sys, urllib.request
sys.path.insert(0, {root!r})
from vcb.tts import voxtral_tts
request = urllib.request.Request("http://127.0.0.1:{port}/health")
with voxtral_tts._LOCAL.open(request, timeout=20) as response:
    print(response.status)
"""


def test_voxtral_requests_never_go_through_a_proxy(server):
    """The opener is built when the module is imported, so this runs in a fresh interpreter whose environment
    already names a proxy (a dead one: if it were used, the request could not succeed)."""
    env = {name: value for name, value in os.environ.items()
           if name.lower() not in ("http_proxy", "https_proxy", "all_proxy", "no_proxy")}
    for name in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY"):
        env[name] = "http://127.0.0.1:9"
    env["VCB_ENV_FILE"] = str(REPO_ROOT / "tests" / "_no_such.env")
    done = subprocess.run([sys.executable, "-c", PROXY_PROBE.format(root=str(REPO_ROOT), port=server.port)],
                          cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "200"
    assert [r[:2] for r in server.requests] == [("GET", "/health")]


def test_voxtral_explains_an_error_answer_of_the_server(hub, server, vllm):
    server.mode = "error"
    with pytest.raises(RuntimeError, match=r"the Voxtral server answered HTTP 400: .*unknown voice 'nobody'"):
        asyncio.run(speak(voxtral(server, voice="nobody"), "Hello."))


def test_voxtral_sends_nothing_when_only_unsupported_tags_are_left(hub, server, vllm):
    with pytest.raises(ValueError, match="nothing left to say"):
        asyncio.run(speak(voxtral(server), "<laugh>"))
    assert posts(server) == []


def test_voxtral_stops_a_server_that_never_becomes_healthy(hub, server, vllm):
    server.health = [503] * 50
    with pytest.raises(TimeoutError, match="was not ready after 7 s"):
        asyncio.run(voxtral(server, startup_timeout_s=7).load())
    assert len(server.requests) == 5 and vllm.slept == [2, 2, 2, 2]     # polls at 0, 2, 4, 6 and 8 s
    assert vllm.proc.terminated                                  # never leave a half-started server holding VRAM


def test_voxtral_reports_a_server_that_exits_before_it_is_ready(hub, server, vllm):
    vllm.proc = FakeProc(exit_code=3)
    with pytest.raises(RuntimeError, match=r"exited with code 3 before it was ready \(see worker.log\)"):
        asyncio.run(voxtral(server).load())


def test_voxtral_close_stops_the_server_and_kills_one_that_ignores_the_request(hub, server, vllm):
    engine = voxtral(server)
    asyncio.run(speak(engine))
    assert vllm.proc.terminated and not vllm.proc.killed
    vllm.proc = FakeProc(ignores_terminate=True)
    asyncio.run(speak(engine))
    assert vllm.proc.terminated and vllm.proc.killed


def test_voxtral_must_be_loaded_again_after_close(hub, server, vllm):
    engine = voxtral(server)

    async def scenario():
        await engine.load()
        await engine.close()
        return [c async for c in engine.synthesize_stream("Hi.")]

    with pytest.raises(AssertionError, match="call load"):
        asyncio.run(scenario())


def test_voxtral_has_no_voice_cloning_and_starts_nothing_when_asked_for_it(server, vllm, tmp_path):
    engine = voxtral_tts.VoxtralTTS(config(reference_clip(tmp_path)))
    with pytest.raises(ValueError, match="no voice cloning"):
        asyncio.run(engine.load())
    assert vllm.started == []


def test_voxtral_refuses_to_start_on_windows_where_vllm_omni_does_not_exist(server, vllm, monkeypatch):
    monkeypatch.setattr(sys, "platform", "win32")
    with pytest.raises(RuntimeError, match="no Windows build: run the benchmark inside WSL2"):
        asyncio.run(voxtral(server).load())
    assert vllm.started == []


# -- nothing is imported until an engine is loaded -----------------------------

HEAVY = ("torch", "torchaudio", "transformers", "accelerate", "vllm", "vllm_omni", "huggingface_hub", "soundfile",
         "qwen_tts", "librosa", "onnxruntime", "funasr", "funasr_detach", "mistral_common", "chatterbox")

LAZY_PROBE = """
import sys
sys.path.insert(0, {tools!r})
if sys.version_info >= (3, 11):
    import tts_bench.registry, tts_bench.catalog, tts_bench.cli, tts_bench.review
    assert not [m for m in sys.modules if m == "vcb" or m.startswith("vcb.")], "the benchmark tool imported VCB"
import vcb.tts
from vcb.tts import breeze_tts, qwen3_tts, step_audio_editx_tts, voxtral_tts
print(",".join(sorted(m for m in {heavy!r} if m in sys.modules)))
"""


def test_importing_the_adapters_and_the_benchmark_tool_loads_no_engine_library():
    env = {**os.environ, "VCB_ENV_FILE": str(REPO_ROOT / "tests" / "_no_such.env"), "HF_HUB_OFFLINE": "1"}
    done = subprocess.run([sys.executable, "-c", LAZY_PROBE.format(tools=str(TOOLS_DIR), heavy=HEAVY)],
                          cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=180)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == ""                             # none of the heavy libraries was imported
