"""GPU budgeting and the Ollama model recommendation."""

from vcb.gpu import (
    GPU, OLLAMA_CHOICES, ModelChoice, estimate_vram_gb, llm_budget_gb,
    parse_nvidia_smi, recommend, speech_vram_gb,
)
from vcb.runtime import ollama_fit_warning, ollama_sized_name
from vcb.settings import model_label, model_tag


def test_parse_nvidia_smi():
    out = "NVIDIA GeForce RTX 3090 Ti, 24564\nNVIDIA GeForce RTX 5060 Ti, 16311\n"
    assert parse_nvidia_smi(out) == [GPU("NVIDIA GeForce RTX 3090 Ti", 24.0),
                                     GPU("NVIDIA GeForce RTX 5060 Ti", 15.9)]
    assert parse_nvidia_smi("") == []
    assert parse_nvidia_smi("garbage line\n") == []


def test_estimate_matches_measured_ollama_usage():
    # Measured with Ollama at an 8k context: 2.90 GB and 21.28 GB.
    assert abs(estimate_vram_gb(1.88) - 2.9) <= 0.2
    assert abs(estimate_vram_gb(18.49) - 21.28) <= 0.5


def test_budget_subtracts_speech_and_reserve():
    gpus = [GPU("8GB card", 8.0)]
    assert speech_vram_gb("kokoro") < speech_vram_gb("chatterbox-turbo")
    assert llm_budget_gb(gpus, "chatterbox-turbo", 2) == 0.0      # never negative
    assert llm_budget_gb(gpus, "kokoro", 2) == round(8 - speech_vram_gb("kokoro") - 2, 1)
    assert llm_budget_gb([], "kokoro", 2) == 0.0
    # Several GPUs add up (Ollama splits a model across them).
    two = [GPU("a", 24.0), GPU("b", 16.0)]
    assert llm_budget_gb(two, "kokoro", 2) == round(40 - speech_vram_gb("kokoro") - 2, 1)


def test_recommend_picks_largest_model_that_fits():
    assert recommend(0.0) is None
    assert recommend(3.0).tag == "llama3.2:3b"
    assert recommend(6.0).tag == "llama3.1:8b"
    assert recommend(100.0).tag == OLLAMA_CHOICES[-1].tag
    for budget in (3.0, 6.0, 10.0, 20.0):
        assert recommend(budget).vram_gb <= budget


def test_typical_cards():
    def pick(gb, engine):
        return recommend(llm_budget_gb([GPU("x", gb)], engine, 2.0))
    assert pick(8, "chatterbox-turbo") is None          # cloned voice fills an 8 GB card
    assert pick(8, "kokoro").tag == "llama3.2:3b"
    assert pick(12, "kokoro").tag == "llama3.1:8b"
    assert pick(24, "chatterbox-turbo").tag == "gemma3:12b"


def test_model_label_round_trips_to_tag():
    c = ModelChoice("gemma3:12b", 7.61, "great conversation")
    label = model_label(c, budget_gb=20.0)
    assert label.startswith("gemma3:12b ") and "fits" in label
    assert "too big" in model_label(c, budget_gb=1.0)
    assert model_tag(label) == "gemma3:12b"
    assert model_tag("my-custom:tag") == "my-custom:tag"
    assert model_tag("  ") == ""


def test_sized_copy_name():
    assert ollama_sized_name("llama3.1:8b", 8192) == "llama3.1:8b-vcb8k"
    assert ollama_sized_name("gemma3", 4096) == "gemma3:latest-vcb4k"


def test_fit_warning_only_when_spilling_to_cpu():
    gib = 2**30
    assert ollama_fit_warning({"name": "m", "size": 6 * gib, "size_vram": 6 * gib}) is None
    warning = ollama_fit_warning({"name": "m", "size": 10 * gib, "size_vram": 6 * gib})
    assert warning and "60%" in warning and "smaller model" in warning


# -- spreading STT / TTS over GPUs ----------------------------------------------

from vcb.gpu import plan_placement, visible_gpus  # noqa: E402


def test_plan_puts_biggest_model_on_emptiest_gpu_then_spreads():
    # After the LLM loaded: GPU 0 has 9.6 GB free, GPU 1 7.2 GB.
    plan = plan_placement({0: 9.6, 1: 7.2}, {"stt": 2.2, "tts": 4.7})
    assert plan == {"tts": 0, "stt": 1}           # 9.6-4.7=4.9 < 7.2 -> STT on GPU 1
    # Plenty of room on one card: both go there.
    assert plan_placement({0: 20.0, 1: 3.0}, {"stt": 2.2, "tts": 4.7}) == {"tts": 0, "stt": 0}


def test_visible_gpus_maps_cuda_visible_devices():
    assert visible_gpus(None, [1, 0]) == [0, 1]
    assert visible_gpus("", [0, 1]) == [0, 1]
    assert visible_gpus("1", [0, 1]) == [1]
    assert visible_gpus("1,0", [0, 1]) == [1, 0]          # CUDA order follows the list
    assert visible_gpus("GPU-abc", [0, 1]) is None


def test_place_on_gpus_sets_devices(monkeypatch):
    from vcb import gpu, hardware
    from vcb.config import AppConfig

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpu, "query_free_gb", lambda: {0: 9.6, 1: 7.2})
    cfg = AppConfig()
    hardware.place_on_gpus(cfg, "chatterbox-turbo", has_cuda=True)
    assert cfg.tts.device == "cuda:0" and cfg.stt.device_index == 1

    # With CUDA_VISIBLE_DEVICES=1,0 the physical GPU 1 is CUDA's index 0.
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1,0")
    cfg = AppConfig()
    hardware.place_on_gpus(cfg, "chatterbox-turbo", has_cuda=True)
    assert cfg.tts.device == "cuda:1" and cfg.stt.device_index == 0


def test_place_on_gpus_respects_single_gpu_and_explicit_devices(monkeypatch):
    from vcb import gpu, hardware
    from vcb.config import AppConfig

    monkeypatch.setattr(gpu, "query_free_gb", lambda: {0: 9.6, 1: 7.2})
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")        # user pinned one GPU
    cfg = AppConfig()
    hardware.place_on_gpus(cfg, "chatterbox-turbo", has_cuda=True)
    assert cfg.tts.device == "cuda" and cfg.stt.device_index == 0

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    cfg = AppConfig()
    cfg.tts.device = "cuda:1"                                # explicit choice wins
    hardware.place_on_gpus(cfg, "chatterbox-turbo", has_cuda=True)
    assert cfg.tts.device == "cuda:1"

    cfg = AppConfig()
    hardware.place_on_gpus(cfg, "chatterbox-turbo", has_cuda=False)
    assert cfg.tts.device == "cuda"
