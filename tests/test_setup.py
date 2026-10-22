"""Publishing/setup helpers: .env editing, user overrides (prompts, model
presets), hardware fallback, Ollama model detection, audio device lookup."""

import json

import pytest

from vcb import config
from vcb.config import AppConfig, ModelPreset, load_user_presets, resolve_model_preset
from vcb.envfile import read_env, update_env
from vcb.hardware import adapt_to_hardware
from vcb.runtime import ollama_has_model


# -- .env editing -------------------------------------------------------------

def test_update_env_preserves_unmanaged_lines(tmp_path):
    env = tmp_path / ".env"
    env.write_text(
        "# my comment\n"
        "HF_TOKEN=abc\n"
        "VCB_BACKEND=llama_cpp            # inline note\n"
        "VCB_AI_LED=true\n",
        encoding="utf-8",
    )
    update_env(env, {"VCB_BACKEND": "cloud", "VCB_AI_LED": None,
                     "VCB_USER_NAME": "Alex Smith"})
    text = env.read_text(encoding="utf-8")
    assert "# my comment" in text and "HF_TOKEN=abc" in text
    assert "VCB_AI_LED" not in text
    assert read_env(env) == {"HF_TOKEN": "abc", "VCB_BACKEND": "cloud",
                             "VCB_USER_NAME": "Alex Smith"}


def test_update_env_creates_file_and_round_trips_tricky_values(tmp_path):
    env = tmp_path / ".env"
    update_env(env, {"A": r"C:\My Voices\me.wav", "B": "x#y", "C": ""})
    assert read_env(env) == {"A": r"C:\My Voices\me.wav", "B": "x#y", "C": ""}


def test_read_env_handles_bom(tmp_path):
    env = tmp_path / ".env"
    env.write_bytes("\ufeffVCB_BACKEND=cloud\n".encode("utf-8"))
    assert read_env(env) == {"VCB_BACKEND": "cloud"}


# -- user overrides -----------------------------------------------------------

def test_user_prompt_overrides_builtin(tmp_path, monkeypatch):
    builtin, user = tmp_path / "prompts", tmp_path / "user_prompts"
    builtin.mkdir(), user.mkdir()
    (builtin / "assistant.txt").write_text("public", encoding="utf-8")
    (builtin / "coach.txt").write_text("coach", encoding="utf-8")
    (user / "assistant.txt").write_text("mine", encoding="utf-8")
    (user / "secret.txt").write_text("secret", encoding="utf-8")
    monkeypatch.setattr(config, "PROMPTS_DIR", builtin)
    monkeypatch.setattr(config, "USER_PROMPTS_DIR", user)
    assert config.load_system_prompt("assistant") == "mine"
    assert config.load_system_prompt("coach") == "coach"
    assert config.available_prompts() == ["assistant", "coach", "secret"]
    with pytest.raises(FileNotFoundError):
        config.load_system_prompt("nope")


def test_load_user_presets(tmp_path):
    path = tmp_path / "models.json"
    path.write_text(json.dumps({"mine": {
        "label": "my-model", "model_path": "models\\m.gguf",
        "extra_args": ["--no-jinja"]}}), encoding="utf-8-sig")
    presets = load_user_presets(path)
    assert presets["mine"] == ModelPreset("my-model", "models\\m.gguf",
                                          extra_args=("--no-jinja",))
    assert load_user_presets(tmp_path / "missing.json") == {}


def test_active_model_may_be_a_gguf_path():
    preset = resolve_model_preset(r"D:\llms\Some-Model.Q4_K_M.gguf")
    assert preset.model_path.endswith("Some-Model.Q4_K_M.gguf")
    assert preset.label == "Some-Model.Q4_K_M"
    with pytest.raises(ValueError):
        resolve_model_preset("not-a-preset")


def test_public_defaults_are_generic():
    cfg = AppConfig()
    assert cfg.backend == "ollama"
    assert cfg.server.cuda_visible_devices is None
    assert config.USER_NAME == "the user"
    for text in (cfg.resume.hint, cfg.proactive.hint, cfg.proactive.greeting_hint,
                 cfg.proactive.waiting_hint):
        assert "the user" in text


# -- hardware fallback ----------------------------------------------------------

def test_cpu_fallback_switches_to_light_models():
    cfg = AppConfig()
    engine = adapt_to_hardware(cfg, "chatterbox-turbo", has_cuda=False)
    assert engine == "kokoro"
    assert cfg.stt.device == "cpu" and cfg.stt.model == cfg.stt.cpu_model
    assert cfg.tts.device == "cpu"


def test_gpu_keeps_requested_engine():
    cfg = AppConfig()
    assert adapt_to_hardware(cfg, "chatterbox-turbo", has_cuda=True) == "chatterbox-turbo"
    assert cfg.stt.device == "cuda"
    with pytest.raises(ValueError):
        adapt_to_hardware(cfg, "nope", has_cuda=True)


# -- ollama -------------------------------------------------------------------

def test_ollama_has_model_matches_tags():
    tags = {"models": [{"name": "llama3.1:8b-instruct-q4_K_M"}, {"name": "gemma3:latest"}]}
    assert ollama_has_model(tags, "llama3.1:8b-instruct-q4_K_M")
    assert ollama_has_model(tags, "gemma3")             # bare name -> :latest
    assert not ollama_has_model(tags, "gemma3:12b")
    assert not ollama_has_model({}, "llama3.1")


# -- audio devices ------------------------------------------------------------

_DEVICES = [
    {"name": "Microphone (USB Mic)", "hostapi": 0, "max_input_channels": 1, "max_output_channels": 0},
    {"name": "Speakers (Realtek)", "hostapi": 0, "max_input_channels": 0, "max_output_channels": 2},
    {"name": "Microphone (USB Mic)", "hostapi": 1, "max_input_channels": 1, "max_output_channels": 0},
]


@pytest.fixture
def fake_sd(monkeypatch):
    from vcb.audio import devices

    def query_devices(kind=None):
        if kind == "input":
            return _DEVICES[0]
        return _DEVICES

    monkeypatch.setattr(devices.sd, "query_devices", query_devices)
    return devices


def test_resolve_device_by_name(fake_sd):
    assert fake_sd.resolve_device("Microphone (USB Mic)", "input") == 0   # default host API
    assert fake_sd.resolve_device("realtek", "output") == 1               # fragment
    assert fake_sd.resolve_device("Gone Device", "input") is None          # -> system default
    assert fake_sd.resolve_device(None, "input") is None
    assert fake_sd.resolve_device(5, "input") == 5


def test_list_devices_dedupes_host_apis(fake_sd):
    assert fake_sd.list_devices("input") == ["Microphone (USB Mic)"]
    assert fake_sd.list_devices("output") == ["Speakers (Realtek)"]
