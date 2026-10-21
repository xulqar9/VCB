"""Backend selection: llama_cpp / ollama / cloud all resolve llm.base_url,
llm.api_key, and llm.model correctly. No network or subprocess involved."""

import os

import pytest

from vcb.config import AppConfig


def test_default_backend_is_ollama():
    cfg = AppConfig()
    assert cfg.backend == "ollama"
    assert cfg.llm.base_url == cfg.ollama.base_url
    assert cfg.llm.model == cfg.ollama.model


def test_llama_cpp_backend_uses_active_preset():
    cfg = AppConfig()
    cfg.backend = "llama_cpp"
    cfg.configure_backend()
    assert cfg.llm.base_url == f"http://{cfg.server.host}:{cfg.server.port}/v1"
    assert cfg.llm.model                     # set from the active MODEL_PRESET
    assert cfg.server.model_path.endswith(".gguf")


def test_ollama_backend_uses_ollama_config():
    cfg = AppConfig()
    cfg.backend = "ollama"
    cfg.ollama.model = "llama3.1:8b"
    cfg.ollama.base_url = "http://127.0.0.1:11434/v1"
    cfg.configure_backend()
    assert cfg.llm.base_url == "http://127.0.0.1:11434/v1"
    assert cfg.llm.model == "llama3.1:8b"
    assert cfg.llm.api_key == "ollama"


def test_cloud_backend_reads_api_key_from_env(monkeypatch):
    monkeypatch.setenv("VCB_CLOUD_API_KEY", "sk-test-123")
    cfg = AppConfig()
    cfg.backend = "cloud"
    cfg.cloud.model = "gpt-4o-mini"
    cfg.cloud.reasoning_effort = "minimal"
    cfg.configure_backend()
    assert cfg.llm.api_key == "sk-test-123"
    assert cfg.llm.model == "gpt-4o-mini"
    assert cfg.llm.base_url == "https://api.openai.com/v1"
    assert cfg.llm.reasoning_effort == "minimal"
    cfg.backend = "ollama"                 # thinking level is a cloud-only setting
    cfg.configure_backend()
    assert cfg.llm.reasoning_effort is None


def test_cloud_backend_missing_key_is_empty_not_crash(monkeypatch):
    monkeypatch.delenv("VCB_CLOUD_API_KEY", raising=False)
    cfg = AppConfig()
    cfg.backend = "cloud"
    cfg.configure_backend()
    assert cfg.llm.api_key == ""


def test_unknown_backend_raises():
    cfg = AppConfig()
    cfg.backend = "not-a-backend"
    with pytest.raises(ValueError):
        cfg.configure_backend()


def test_switching_backend_rederives_llm_config():
    cfg = AppConfig()
    original_model = cfg.llm.model
    cfg.backend = "llama_cpp"
    cfg.configure_backend()
    assert cfg.llm.model != original_model
    cfg.backend = "ollama"
    cfg.configure_backend()
    assert cfg.llm.model == original_model
