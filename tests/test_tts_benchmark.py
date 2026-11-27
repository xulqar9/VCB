"""Tests for the TTS benchmark's planning logic (tools/tts_bench): statistics, the test
sentences, GPU selection, the engines' environment, the plan, the environment check, the
download plan, and that the registry agrees with the adapters it describes. Reports and
the review page are in test_tts_benchmark_report.py, real runs against a fake engine in
test_tts_benchmark_run.py, the four optional adapters in test_tts_bench_adapters.py."""

from __future__ import annotations

import dataclasses
import importlib
import os
import re
import shlex
import socket
import sys
import types
from collections import Counter
from pathlib import Path

import pytest

from tts_bench_support import GPUS, REPO_ROOT, TOOLS_DIR, NoMonitor, OsAs, write_wav

from tts_bench import catalog, check, cli, probe, runner, stats, worker  # noqa: E402
from tts_bench import gpu as gpu_mod  # noqa: E402
from tts_bench import sentences as sent  # noqa: E402
from tts_bench.config import (BenchConfig, EngineSettings, load_config, parse_config,  # noqa: E402
                              resolve_path)
from tts_bench.registry import (BENCH_ENGINES, ENGINES, OPTIONAL_ENGINES, RETIRED_INSTALL_HINT,  # noqa: E402
                                RETIRED_VCB_ENGINES, VCB_ENGINES, VCB_INSTALL_HINT, get_engine, select_engines,
                                variant_name)
from vcb.config import TTSConfig  # noqa: E402
from vcb.hardware import RETIRED_TTS_ENGINES  # noqa: E402
from vcb.tts.base import TTSEngine  # noqa: E402

SENTENCES_FILE = TOOLS_DIR / "tts_bench" / "sentences.txt"
PYTHON = Path(sys.executable)


# -- statistics ----------------------------------------------------------------

def test_percentile_matches_numpy():
    np = pytest.importorskip("numpy")
    data = [0.4, 1.3, 2.2, 2.9, 3.1, 7.5, 8.8, 9.0, 12.5, 30.0, 0.9]
    for q in (0, 5, 25, 50, 75, 95, 99, 100):
        assert stats.percentile(data, q) == pytest.approx(float(np.percentile(data, q)))
    assert stats.percentile([1, 3], 95) == pytest.approx(2.9)


def test_describe_gives_mean_median_p95_and_skips_missing_values():
    got = stats.describe([1, 2, 3, 4, 5, None, float("nan")])
    assert got["n"] == 5
    assert (got["mean"], got["median"]) == (3, 3)
    assert got["p95"] == pytest.approx(4.8)
    assert stats.describe([None]) == {"n": 0, "mean": None, "median": None, "p95": None}


# -- the test sentences --------------------------------------------------------

def test_shipped_sentences_follow_the_brief():
    items = sent.load_sentences(SENTENCES_FILE)
    assert Counter(s.category for s in items) == {"neutral": 5, "emotion": 5, "human": 5}
    assert len({s.id for s in items}) == 15
    assert [s.intent for s in items if s.category == "emotion"] == \
        ["excited", "sad", "annoyed", "whispering", "warm"]
    assert {t for s in items if s.category == "human" for t in s.tags} == set(sent.CANONICAL_TAGS)
    assert all(s.tags for s in items if s.category == "human")
    assert not any(s.tags for s in items if s.category != "human")


@pytest.mark.parametrize("text, message", [
    ("n01 | neutral | plain", "expected 'id | category | intent | text'"),
    ("n-1 | neutral | plain | Hello there.", "must be letters/digits only"),
    ("n01 | neutral | plain | A.\nn01 | neutral | plain | B.", "duplicate id"),
    ("n01 | angry | plain | Hello.", "is not one of"),
    ("# only a comment\n\n", "no sentences found"),
])
def test_bad_sentence_files_are_rejected_with_a_reason(text, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        sent.parse_sentences(text)


def test_sentence_text_may_contain_a_pipe():
    (item,) = sent.parse_sentences("a1 | neutral | plain | Either this | or that.")
    assert item.text == "Either this | or that."


def test_fingerprint_follows_the_content_not_the_layout():
    one = sent.parse_sentences("a1 | neutral | plain | One.")
    assert sent.sentences_fingerprint(one) == \
        sent.sentences_fingerprint(sent.parse_sentences("# note\n\na1 | neutral | plain | One.\n"))
    assert sent.sentences_fingerprint(one) != \
        sent.sentences_fingerprint(sent.parse_sentences("a1 | neutral | plain | Two."))


# -- choosing engines ----------------------------------------------------------

def test_engine_selection():
    assert [s.key for s in select_engines(None)] == ["chatterbox"]            # the baseline by default
    assert [s.key for s in select_engines("all")] == list(BENCH_ENGINES)
    assert [s.key for s in select_engines("vcb")] == list(VCB_ENGINES)
    assert [s.key for s in select_engines("vcb,breeze,kokoro")] == [*VCB_ENGINES, "breeze"]    # no engine twice
    assert [s.key for s in select_engines("qwen3-tts,breeze,breeze")] == ["qwen3-tts", "breeze"]
    with pytest.raises(ValueError, match="unknown engine 'nope'"):
        select_engines("breeze,nope")
    assert (variant_name("breeze", "default"), variant_name("breeze", "clone")) == ("breeze", "breeze_clone")


def test_the_engine_groups_name_every_engine_once():
    shown = [key for key, spec in ENGINES.items() if not spec.hidden]
    assert sorted(BENCH_ENGINES) == sorted(shown) and len(set(BENCH_ENGINES)) == len(BENCH_ENGINES)
    assert not set(VCB_ENGINES) & set(OPTIONAL_ENGINES) and not set(VCB_ENGINES) & set(RETIRED_VCB_ENGINES)
    assert {key for key in shown if ENGINES[key].vcb_native} == {*VCB_ENGINES, *RETIRED_VCB_ENGINES}
    assert {key for key in shown if ENGINES[key].retired} == set(RETIRED_VCB_ENGINES) == set(RETIRED_TTS_ENGINES)


def test_weights_to_download_depend_on_the_voice_mode():
    qwen = get_engine("qwen3-tts")
    assert [w.repo for w in qwen.weight_repos(("default",))] == ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"]
    assert [w.repo for w in qwen.weight_repos(("clone",))] == ["Qwen/Qwen3-TTS-12Hz-1.7B-Base"]
    assert qwen.download_gb() == pytest.approx(4.52 + 4.54)


# -- GPUs ----------------------------------------------------------------------

def test_nvidia_smi_gpu_list_is_parsed_and_bad_lines_are_skipped():
    text = ("0, NVIDIA GeForce RTX 3090 Ti, 24564, 1021\n"
            "1, NVIDIA GeForce RTX 5060 Ti, 16311, 5012\n"
            "not a gpu line\n"
            "2, Name, With Comma, 8192, 10\n")
    gpus = gpu_mod.parse_gpu_list(text)
    assert [(g.index, g.name, g.total_mib, g.used_mib) for g in gpus] == [
        (0, "NVIDIA GeForce RTX 3090 Ti", 24564, 1021),
        (1, "NVIDIA GeForce RTX 5060 Ti", 16311, 5012),
        (2, "Name, With Comma", 8192, 10)]
    assert gpus[0].label == "GPU 0 (NVIDIA GeForce RTX 3090 Ti, 24 GB)"
    assert gpus[1].label == "GPU 1 (NVIDIA GeForce RTX 5060 Ti, 16 GB)"


@pytest.mark.parametrize("spec, index", [("3090 Ti", 0), ("3090", 0), ("RTX 5060 TI", 1), ("5060", 1),
                                         (1, 1), ("0", 0)])
def test_a_gpu_is_chosen_by_name_or_index(spec, index):
    assert gpu_mod.resolve_gpu(spec, GPUS).index == index


@pytest.mark.parametrize("spec, message", [
    ("2", "no GPU with index 2"),
    ("4090", "no GPU with index 4090 and no GPU name contains it"),
    ("RTX", "matches several GPUs"),
    ("GTX 1080", "matches no GPU"),
])
def test_an_unusable_gpu_choice_says_what_is_available(spec, message):
    with pytest.raises(ValueError, match=message) as err:
        gpu_mod.resolve_gpu(spec, GPUS)
    assert "RTX 3090 Ti" in str(err.value) and "RTX 5060 Ti" in str(err.value)


def test_memory_samples_are_parsed_and_summarised_per_window():
    sample = gpu_mod.parse_sample("2026/11/27 15:36:42.377, 1, 996")
    assert (sample.index, sample.used_mib) == (1, 996)
    assert gpu_mod.parse_sample("garbage") is None
    samples = [gpu_mod.VramSample(t, 0, mib) for t, mib in [(1.0, 100), (2.0, 300), (3.0, 200), (9.0, 999)]]
    assert gpu_mod.window_stats(samples, 0.5, 3.5) == {0: {"min": 100, "median": 200, "max": 300, "n": 3}}
    assert gpu_mod.window_stats(samples, 50, 60) == {}


# -- what an engine process gets to see ----------------------------------------

def test_engine_environment_is_isolated_from_vcb():
    base = {"PATH": "/bin", "VCB_TTS_ENGINE": "kokoro", "vcb_voice_wav": "x.wav", "HF_TOKEN": "t",
            "hf_token": "t2", "Mistral_Api_Key": "k", "PYTHONPATH": "/old"}
    env = runner.worker_env(base, runner.Target("3090 Ti", GPUS[0]))
    assert env["PATH"] == "/bin"
    assert not [k for k in env if k.upper().startswith("VCB_") and k != "VCB_ENV_FILE"]
    assert not Path(env["VCB_ENV_FILE"]).exists()                 # VCB's .env is never read
    assert not {"HF_TOKEN", "hf_token", "Mistral_Api_Key"} & set(env)
    assert (env["CUDA_VISIBLE_DEVICES"], env["CUDA_DEVICE_ORDER"]) == ("0", "PCI_BUS_ID")
    assert (env["HF_HUB_OFFLINE"], env["TRANSFORMERS_OFFLINE"]) == ("1", "1")
    assert env["HF_HUB_DISABLE_TELEMETRY"] == "1" and env["DO_NOT_TRACK"] == "1"
    assert env["PYTHONPATH"].split(os.pathsep) == [str(runner.TOOLS_DIR), "/old"]


def test_engine_may_download_only_when_asked_to():
    base = {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    env = runner.worker_env(base, runner.Target("cpu", None), allow_download=True)
    assert "HF_HUB_OFFLINE" not in env and "TRANSFORMERS_OFFLINE" not in env
    assert env["CUDA_VISIBLE_DEVICES"] == ""


# -- the run protocol ----------------------------------------------------------

def test_jobs_start_with_an_uncounted_warmup_then_go_round_robin():
    items = sent.parse_sentences("a | neutral | plain | One.\nb | neutral | plain | Two.\nc | neutral | plain | Three.")
    jobs = runner.build_jobs(items, runs=2, warmup_runs=1)
    assert [(j.sentence.id, j.run, j.counted) for j in jobs] == [
        ("a", 0, False), ("a", 1, True), ("b", 1, True), ("c", 1, True),
        ("a", 2, True), ("b", 2, True), ("c", 2, True)]
    assert all(j.counted for j in runner.build_jobs(items, runs=1, warmup_runs=0))


def test_fifteen_sentences_three_runs_make_45_counted_syntheses():
    jobs = runner.build_jobs(sent.load_sentences(SENTENCES_FILE), runs=3, warmup_runs=1)
    assert (sum(j.counted for j in jobs), sum(not j.counted for j in jobs)) == (45, 1)


# -- settings ------------------------------------------------------------------

def test_shipped_config_sets_the_benchmark_protocol():
    cfg = load_config(TOOLS_DIR / "tts_bench.toml")
    assert (cfg.runs_per_sentence, cfg.warmup_runs, cfg.timeout_s) == (3, 1, 30.0)
    assert cfg.default_gpu == "3090 Ti"                       # the engine under test; VCB stays on the 5060 Ti
    assert cfg.reference_clip == REPO_ROOT / "desired_voice.wav"
    assert cfg.reference_text


def test_unknown_config_keys_are_rejected():
    with pytest.raises(ValueError, match=r"unknown key\(s\) in \[run\]: runs_per_senence"):
        parse_config({"run": {"runs_per_senence": 5}})
    with pytest.raises(ValueError, match=r"unknown key\(s\) in \[engines.breeze\]: gpus"):
        parse_config({"engines": {"breeze": {"gpus": 1}}})
    with pytest.raises(ValueError, match=r"unknown setting\(s\) outside any \[section\]: sentences"):
        parse_config({"sentences": "x.txt"})


@pytest.mark.parametrize("run, message", [
    ({"runs_per_sentence": 0}, "at least 1"),
    ({"warmup_runs": -1}, "cannot be negative"),
    ({"timeout_s": 0}, "must be positive"),
])
def test_impossible_run_settings_are_rejected(run, message):
    with pytest.raises(ValueError, match=message):
        parse_config({"run": run})


def test_missing_config_file_is_an_error(tmp_path):
    with pytest.raises(FileNotFoundError, match="config file not found"):
        load_config(tmp_path / "nope.toml")


# -- the plan ------------------------------------------------------------------

def test_each_engine_can_be_pinned_to_its_own_gpu(tmp_path):
    toml = tmp_path / "bench.toml"
    toml.write_text(f"""
[run]
runs_per_sentence = 5
[gpu]
default = "3090 Ti"
[python]
default = '{PYTHON}'
[engines.qwen3-tts]
gpu = 1
options = {{ speaker = "Aiden" }}
""", encoding="utf-8")
    cfg = load_config(toml)
    plan = runner.build_plan(cfg, [get_engine("breeze"), get_engine("qwen3-tts")], gpus=GPUS)
    assert {v.variant: v.target.gpu.index for v in plan.variants} == \
        {"breeze": 0, "breeze_clone": 0, "qwen3-tts": 1, "qwen3-tts_clone": 1}
    qwen = next(v for v in plan.variants if v.variant == "qwen3-tts")
    assert qwen.options == {"speaker": "Aiden"} and qwen.python == PYTHON
    assert qwen.target.device == "cuda:0"                     # the engine sees only its own GPU


def test_a_missing_default_gpu_is_reported_instead_of_guessing():
    with pytest.raises(ValueError, match="matches no GPU"):
        runner.build_plan(BenchConfig(), [get_engine("chatterbox")], gpus=[GPUS[1]])
    plan = runner.build_plan(BenchConfig(default_gpu="cpu", default_python=PYTHON),
                             [get_engine("chatterbox")], modes=["default"], gpus=[])
    assert plan.variants[0].target.device == "cpu"


def _skip(cfg, key, mode):
    plan = runner.build_plan(cfg, [get_engine(key)], modes=[mode], gpus=GPUS)
    return plan.variants[0].skip


def test_cloning_engines_get_a_second_voice_and_presets_do_not(tmp_path):
    clip = write_wav(tmp_path / "ref.wav")
    cfg = BenchConfig(default_python=PYTHON, reference_clip=clip, reference_text="Hello there.")
    cfg.engines["breeze"] = EngineSettings(options={"repo_dir": str(tmp_path)})
    cfg.engines["step-audio-editx"] = EngineSettings(options={"repo_dir": str(tmp_path)})
    plan = runner.build_plan(cfg, select_engines("all"), gpus=GPUS)
    assert [v.variant for v in plan.variants] == [
        "chatterbox", "chatterbox_clone", "kokoro", "orpheus", "xtts_clone", "csm", "csm_clone", "indextts_clone",
        "breeze", "breeze_clone", "voxtral", "step-audio-editx_clone", "qwen3-tts", "qwen3-tts_clone"]
    assert all(v.voice_wav == str(clip) for v in plan.variants if v.mode == "clone")
    assert all(v.voice_wav == "" for v in plan.variants if v.mode == "default")
    assert runner.build_plan(cfg, [get_engine("voxtral")], modes=["clone"], gpus=GPUS).notes == \
        ["voxtral: nothing to run (it supports only: default)"]


def test_linux_only_engines_are_skipped_on_windows(monkeypatch):
    cfg = BenchConfig(default_python=PYTHON)
    monkeypatch.setattr(runner, "os", OsAs("nt"))
    assert "no Windows build" in _skip(cfg, "voxtral", "default")
    monkeypatch.setattr(runner, "os", OsAs("posix"))
    assert _skip(cfg, "voxtral", "default") is None


def test_engines_with_a_source_checkout_need_it_set(tmp_path):
    cfg = BenchConfig(default_python=PYTHON)
    assert "set [engines.breeze.options] repo_dir" in _skip(cfg, "breeze", "default")
    cfg.engines["breeze"] = EngineSettings(options={"repo_dir": str(tmp_path / "missing")})
    assert "repo_dir is not a folder" in _skip(cfg, "breeze", "default")
    cfg.engines["breeze"] = EngineSettings(options={"repo_dir": str(tmp_path)})
    assert _skip(cfg, "breeze", "default") is None


def test_clone_variants_need_the_clip_and_for_some_engines_its_transcript(tmp_path):
    clip = write_wav(tmp_path / "ref.wav")

    def clone_skip(**kw):
        return _skip(BenchConfig(default_python=PYTHON, **kw), "qwen3-tts", "clone")

    assert "no reference clip" in clone_skip()
    assert "reference clip not found" in clone_skip(reference_clip=tmp_path / "gone.wav")
    assert "exact transcript" in clone_skip(reference_clip=clip)
    assert clone_skip(reference_clip=clip, reference_text="Hello there.") is None
    assert _skip(BenchConfig(default_python=PYTHON, reference_clip=clip), "chatterbox", "clone") is None


def test_a_missing_interpreter_is_reported(tmp_path):
    cfg = BenchConfig(default_python=tmp_path / "no-python.exe")
    assert "interpreter not found" in _skip(cfg, "chatterbox", "default")


def test_registry_paths_expand_variables_and_relative_ones_start_at_the_repository(monkeypatch, tmp_path):
    monkeypatch.setenv("BENCH_TEST_DIR", str(tmp_path))
    token = "%BENCH_TEST_DIR%" if os.name == "nt" else "$BENCH_TEST_DIR"
    assert resolve_path(f"{token}/tts/model.pth") == tmp_path / "tts" / "model.pth"
    assert resolve_path(".runtime/indextts/installed.txt") == REPO_ROOT / ".runtime" / "indextts" / "installed.txt"
    assert resolve_path(tmp_path / "a") == tmp_path / "a"


@pytest.mark.parametrize("key, hint", [("orpheus", VCB_INSTALL_HINT), ("xtts", RETIRED_INSTALL_HINT)])
def test_a_vcb_engine_that_is_not_installed_is_skipped_and_told_how_to_install_it(tmp_path, key, hint):
    clip = write_wav(tmp_path / "ref.wav")
    cfg = BenchConfig(default_python=PYTHON, reference_clip=clip)
    model = tmp_path / "tts" / "model.pth"
    spec = dataclasses.replace(get_engine(key), requires_files=(str(model),))

    def skip():
        return runner.build_plan(cfg, [spec], gpus=GPUS).variants[0].skip

    assert skip() == f"not installed: {model} is missing ({hint})"
    model.parent.mkdir()
    model.write_bytes(b"weights")
    assert skip() is None


def test_the_registry_settings_of_a_vcb_engine_get_this_variants_voice(tmp_path):
    clip = write_wav(tmp_path / "ref.wav")
    cfg = BenchConfig(default_python=PYTHON, reference_clip=clip, reference_text="Hello there.",
                      output_dir=tmp_path / "out")
    plan = runner.build_plan(cfg, [get_engine("csm")], gpus=GPUS)
    got = {v.mode: runner._Session(v, cfg, [], NoMonitor(), allow_download=False)._tts_config()
           for v in plan.variants}
    assert got["clone"] == {"csm_ref_audio": str(clip), "csm_ref_text": "Hello there."}
    assert got["default"] == {"csm_ref_audio": "", "csm_ref_text": ""}      # CSM then invents a voice per line


def test_the_engine_is_told_to_stay_offline_unless_downloads_are_allowed(tmp_path):
    cfg = BenchConfig(default_python=PYTHON, output_dir=tmp_path / "out")
    kokoro, voxtral = (runner.build_plan(cfg, [get_engine(key)], gpus=GPUS).variants[0]
                       for key in ("kokoro", "voxtral"))

    def init(variant, allow_download):
        return runner._Session(variant, cfg, [], NoMonitor(), allow_download=allow_download)._init_message()

    assert (init(kokoro, False)["offline"], init(kokoro, True)["offline"]) == (True, False)
    assert (init(kokoro, False)["vcb_native"], init(voxtral, False)["vcb_native"]) == (True, False)


def test_the_worker_installs_no_network_guard_when_downloads_are_allowed():
    # Installing the guard patches the whole process, so the offline case is tested in a worker process
    # (native_run in test_tts_benchmark_run.py) and in a fresh interpreter (test_tts_bench_netguard.py).
    before = (socket.getaddrinfo, socket.socket.connect)
    assert worker._network_guard({"offline": False})() == []
    assert (socket.getaddrinfo, socket.socket.connect) == before


# -- the environment check -----------------------------------------------------

def _device(index, capability, support="native", matmul_ok=True):
    dev = {"index": index, "name": GPUS[index].name, "capability": capability, "total_mib": GPUS[index].total_mib,
           "support": support, "matmul_ok": matmul_ok}
    if not matmul_ok:
        dev["matmul_error"] = "no kernel image is available for execution on the device"
    return dev


def _probe_answer(devices, arch_list, modules=None, repos=None, platform="win32"):
    return {"python": "3.12.1", "executable": "python", "platform": platform,
            "torch": {"version": "2.7.1+cu128", "cuda_runtime": "12.8", "cuda_available": True,
                      "arch_list": arch_list, "devices": devices},
            "modules": modules or {},
            "hf_cache": {"dir": "C:/hf", "repos": repos or {}}}


@pytest.mark.parametrize("archs, capability, expected", [
    (["sm_50", "sm_86", "sm_90", "sm_120"], (12, 0), "native"),     # a build with RTX 50 series kernels
    (["sm_90", "compute_90"], (12, 0), "ptx"),
    (["sm_50", "sm_86"], (12, 0), "no"),                            # an older build cannot run on the 5060 Ti
    (["sm_80"], (8, 6), "native"),                                  # Ampere kernels run on newer minor versions
    (["sm_90"], (8, 6), "no"),
])
def test_arch_support_tells_whether_a_torch_build_runs_on_a_gpu(archs, capability, expected):
    assert probe.arch_support(archs, *capability) == expected


def test_check_reports_what_each_engine_is_missing(tmp_path):
    clip = write_wav(tmp_path / "ref.wav", seconds=5)
    cfg = BenchConfig(default_python=PYTHON, reference_clip=clip, reference_text="Hello there.")
    asked = []

    def fake_probe(python, modules, repos):
        asked.append((python, modules, repos))
        return _probe_answer(
            [_device(0, [8, 6]), _device(1, [12, 0])], ["sm_86", "sm_120"],
            modules={"chatterbox": {"found": True, "version": "0.1.7"}, "qwen_tts": {"found": False}},
            repos={"ResembleAI/chatterbox-turbo": {"present": True, "size_bytes": 4_044_000_000, "partial_files": 0}})

    report = check.run_check(cfg, [get_engine("chatterbox"), get_engine("qwen3-tts")], gpus=GPUS, probe=fake_probe)
    assert asked == [(PYTHON, ["chatterbox", "qwen_tts", "torch"],
                      ["Qwen/Qwen3-TTS-12Hz-1.7B-Base", "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                       "ResembleAI/chatterbox-turbo"])]           # one probe per interpreter
    assert {v.variant: v.ready for e in report.engines for v in e.variants} == {
        "chatterbox": True, "chatterbox_clone": True, "qwen3-tts": False, "qwen3-tts_clone": False}
    assert not report.ready
    text = check.format_check(report, cfg)
    assert "GPU 1 RTX 5060 Ti (sm_120): kernels built in, test calculation passed" in text
    assert "module qwen_tts is not installed in this interpreter" in text
    assert "are not downloaded: run `tts_benchmark.py download --engines qwen3-tts`" in text
    assert "reference clip ref.wav: 5.0 s, 24000 Hz" in text
    assert "Ready: chatterbox, chatterbox_clone" in text
    assert "Not ready: qwen3-tts, qwen3-tts_clone" in text


def test_check_says_when_pytorch_cannot_use_the_5060_ti(tmp_path):
    answer = _probe_answer([_device(0, [8, 6]), _device(1, [12, 0], support="no", matmul_ok=False)],
                           ["sm_50", "sm_86"],
                           modules={"chatterbox": {"found": True}},
                           repos={"ResembleAI/chatterbox-turbo": {"present": True, "size_bytes": 4_044_000_000,
                                                                  "partial_files": 0}})
    spec = [get_engine("chatterbox")]

    def verdict(gpu):
        cfg = BenchConfig(default_python=PYTHON, default_gpu=gpu)
        rep = check.run_check(cfg, spec, modes=["default"], gpus=GPUS, probe=lambda *a: answer)
        return rep, rep.engines[0].variants[0]

    rep, on_3090 = verdict("3090 Ti")
    assert on_3090.ready                                           # the engine under test is unaffected...
    gpu_lines = [f for f in rep.envs[0].findings if "5060 Ti" in f.text]
    assert gpu_lines and gpu_lines[0].level == check.WARN          # ...but the problem is still shown
    assert "PyTorch cannot run on it" in gpu_lines[0].text

    _, on_5060 = verdict("5060 Ti")
    assert not on_5060.ready
    assert any(f.level == check.FAIL and "PyTorch cannot run on it" in f.text for f in on_5060.findings)


def test_check_marks_linux_only_engines_on_windows_and_incomplete_downloads(tmp_path):
    cfg = BenchConfig(default_python=PYTHON)
    answer = _probe_answer([_device(0, [8, 6]), _device(1, [12, 0])], ["sm_86", "sm_120"],
                           modules={"vllm": {"found": True, "version": "0.18.1"}, "vllm_omni": {"found": True},
                                    "mistral_common": {"found": True}},
                           repos={"mistralai/Voxtral-4B-TTS-2603": {"present": True, "size_bytes": 5_000_000_000,
                                                                    "partial_files": 1}})
    rep = check.run_check(cfg, [get_engine("voxtral")], gpus=GPUS, probe=lambda *a: answer)
    text = check.format_check(rep, cfg)
    assert "its runtime has no Windows build" in text
    assert "weights mistralai/Voxtral-4B-TTS-2603 are incomplete" in text
    assert not rep.ready


def test_check_without_an_interpreter_does_not_probe(tmp_path):
    cfg = BenchConfig(default_python=tmp_path / "no-python.exe")
    rep = check.run_check(cfg, [get_engine("chatterbox")], modes=["default"], gpus=GPUS,
                          probe=lambda *a: pytest.fail("must not be called"))
    assert "interpreter not found" in check.format_check(rep, cfg)


def test_check_sends_a_missing_vcb_voice_to_vcb_and_a_missing_optional_one_to_the_download_command():
    cfg = BenchConfig(default_python=PYTHON)
    answer = _probe_answer([_device(0, [8, 6]), _device(1, [12, 0])], ["sm_86", "sm_120"],
                           modules={"kokoro": {"found": True}, "qwen_tts": {"found": True}})

    def weights_fix(key):
        rep = check.run_check(cfg, [get_engine(key)], modes=["default"], gpus=GPUS, probe=lambda *a: answer)
        (finding,) = [f for f in rep.engines[0].variants[0].findings if "are not downloaded" in f.text]
        return finding.text

    assert "VCB.bat --extras" in weights_fix("kokoro") and "download --engines" not in weights_fix("kokoro")
    assert "download --engines qwen3-tts" in weights_fix("qwen3-tts")


def test_check_looks_for_the_files_a_vcb_engine_keeps_outside_the_hugging_face_cache(tmp_path):
    model = tmp_path / "xtts" / "model.pth"
    spec = dataclasses.replace(get_engine("xtts"), requires_files=(str(model),))
    cfg = BenchConfig(default_python=PYTHON, reference_clip=write_wav(tmp_path / "ref.wav", seconds=5))
    answer = _probe_answer([_device(0, [8, 6]), _device(1, [12, 0])], ["sm_86", "sm_120"],
                           modules={"TTS": {"found": True, "version": "0.27.1"}})

    def variant():
        return check.run_check(cfg, [spec], gpus=GPUS, probe=lambda *a: answer).engines[0].variants[0]

    gone = variant()
    assert not gone.ready
    assert [f.text for f in gone.findings if f.level == check.FAIL] == [
        f"{model} is missing: {RETIRED_INSTALL_HINT}"]
    model.parent.mkdir()
    model.write_bytes(b"x" * 3_000_000)
    there = variant()
    assert there.ready and any(f.level == check.OK and f.text == f"{model} (3 MB)" for f in there.findings)


# -- what is in the Hugging Face cache -----------------------------------------

def _fake_hf_repo(cache, repo, sizes, incomplete=0):
    folder = cache / ("models--" + repo.replace("/", "--"))
    blobs, snap = folder / "blobs", folder / "snapshots" / "rev1"
    blobs.mkdir(parents=True)
    snap.mkdir(parents=True)
    for i, size in enumerate(sizes):
        (blobs / f"blob{i}").write_bytes(b"x" * size)
        (snap / f"file{i}.bin").write_bytes(b"x" * size)
    for i in range(incomplete):
        (blobs / f"part{i}.incomplete").write_bytes(b"x" * 10)


def test_cache_status_sees_complete_partial_and_missing_downloads(tmp_path):
    _fake_hf_repo(tmp_path, "org/whole", [100, 200])
    _fake_hf_repo(tmp_path, "org/half", [100], incomplete=2)
    assert probe.cache_status("org/whole", tmp_path) == {"present": True, "size_bytes": 300, "partial_files": 0}
    assert probe.cache_status("org/half", tmp_path) == {"present": True, "size_bytes": 100, "partial_files": 2}
    assert probe.cache_status("org/none", tmp_path)["present"] is False


# -- the download plan ---------------------------------------------------------

def _present(size_gb):
    return {"present": True, "size_bytes": int(size_gb * 1e9), "partial_files": 0}


def test_download_plan_lists_only_what_is_missing_with_licenses():
    cfg = BenchConfig(default_python=PYTHON)
    asked = []

    def fake_probe(python, modules, repos):
        asked.append((python, modules, repos))
        return {"hf_cache": {"dir": "C:/hf", "repos": {
            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": _present(4.52),       # already complete
            "BreezeBlue/Breeze-TTS-2": {"present": True, "size_bytes": 2_000_000_000, "partial_files": 1}}}}

    plan = catalog.plan_downloads(cfg, [get_engine("breeze"), get_engine("qwen3-tts")], probe=fake_probe)
    assert asked == [(PYTHON, [], ["BreezeBlue/Breeze-TTS-2", "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                                   "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"])]
    assert [i.repo for i in plan.items] == ["BreezeBlue/Breeze-TTS-2", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"]
    assert plan.present == ["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"]
    assert plan.total_gb == pytest.approx(7.68 + 4.54)
    text = catalog.format_download_plan(plan)
    assert "nothing is uploaded" in text
    assert re.search(r"BreezeBlue/Breeze-TTS-2\s+7\.68 GB\s+breeze\s+NON-COMMERCIAL", text)
    assert re.search(r"Qwen/Qwen3-TTS-12Hz-1.7B-Base\s+4\.54 GB\s+qwen3-tts\s+ok to use commercially", text)
    assert "total 12.22 GB" in text
    assert "license of BreezeBlue/Breeze-TTS-2:" in text and "license of Qwen" not in text
    assert "Already downloaded: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice" in text


def test_nothing_is_planned_for_complete_downloads_and_unreadable_caches_are_reported(tmp_path):
    cfg = BenchConfig(default_python=PYTHON)
    qwen = [get_engine("qwen3-tts")]
    done = catalog.plan_downloads(cfg, qwen, modes=["default"],
                                  probe=lambda *a: {"hf_cache": {"dir": "C:/hf", "repos": {
                                      "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": _present(4.52)}}})
    assert catalog.format_download_plan(done).splitlines()[-1] == "Nothing to download."
    broken = catalog.plan_downloads(cfg, qwen, modes=["default"], probe=lambda *a: {"error": "exit code 1: boom"})
    assert broken.problems == [f"{PYTHON}: exit code 1: boom"] and not broken.items
    missing = catalog.plan_downloads(BenchConfig(default_python=tmp_path / "gone.exe"), qwen,
                                     probe=lambda *a: pytest.fail("must not be called"))
    assert missing.problems and "interpreter not found" in missing.problems[0]


def test_vcb_engines_are_left_to_vcb_and_never_downloaded_by_this_tool():
    cfg = BenchConfig(default_python=PYTHON)
    only_vcb = catalog.plan_downloads(cfg, select_engines("vcb"),
                                      probe=lambda *a: pytest.fail("nothing to look up for VCB's own engines"))
    assert only_vcb.native == list(VCB_ENGINES) and not only_vcb.items
    text = catalog.format_download_plan(only_vcb)
    assert f"Left to VCB, which installs its own voices: {', '.join(VCB_ENGINES)}" in text
    assert "VCB.bat --extras" in text and text.splitlines()[-1] == "Nothing to download."
    assert "Retired" not in text

    retired = catalog.plan_downloads(cfg, select_engines("kokoro,xtts,csm,indextts"),
                                     probe=lambda *a: pytest.fail("nothing to look up for VCB's own engines"))
    text = catalog.format_download_plan(retired)
    assert "Left to VCB, which installs its own voices: kokoro\n" in text
    assert f"Retired from VCB, which no longer installs them: xtts, csm, indextts\n  {RETIRED_INSTALL_HINT}" in text

    asked = []

    def fake_probe(python, modules, repos):
        asked.append(repos)
        return {"hf_cache": {"dir": "C:/hf", "repos": {r: {"present": False} for r in repos}}}

    mixed = catalog.plan_downloads(cfg, [get_engine("kokoro"), get_engine("qwen3-tts")], modes=["default"],
                                   probe=fake_probe)
    assert asked == [["Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"]]        # nothing of Kokoro's is looked up
    assert mixed.native == ["kokoro"] and [i.engine for i in mixed.items] == ["qwen3-tts"]
    assert "hexgrad/Kokoro-82M" not in catalog.format_download_plan(mixed)


def test_engine_list_shows_sizes_and_license_restrictions():
    text = catalog.format_engine_list(BenchConfig())

    def row(key):
        return re.search(rf"^\s+{re.escape(key)}\s{{2,}}.*$", text, re.M).group(0).rstrip()

    assert row("breeze").endswith("NON-COMMERCIAL") and "7.68 GB" in row("breeze")
    assert row("voxtral").endswith("NON-COMMERCIAL") and "8.04 GB" in row("voxtral")
    assert row("step-audio-editx").endswith("NOT STATED")
    assert row("qwen3-tts").endswith("commercial use OK")
    assert row("chatterbox").endswith("commercial use OK") and "4.04 GB" in row("chatterbox")
    assert row("kokoro").endswith("commercial use OK") and "0.33 GB" in row("kokoro")
    assert row("xtts").endswith("NON-COMMERCIAL") and "retired from VCB" in row("xtts")  # Coqui Public Model License
    assert row("indextts").endswith("commercial use OK") and "retired from VCB" in row("indextts")
    assert not re.search(r"^\s+fake(-native)?\s", text, re.M)                  # the test engines are hidden
    assert "VCB's own voices are installed by VCB" in text


# -- the registry agrees with the adapters -------------------------------------
# The registry (tools/tts_bench/registry.py) describes each engine; the adapter (vcb/tts/) is what actually runs.
# Nothing else would notice if one of them were edited without the other.

OPTIONAL_ADAPTERS = ("breeze", "voxtral", "step-audio-editx", "qwen3-tts")


def _adapter(key):
    spec = get_engine(key)
    module = importlib.import_module(spec.module)
    return spec, module, getattr(module, spec.cls)


def _option_fields(cls):
    options = getattr(cls, "Options", None)
    return {f.name: f for f in dataclasses.fields(options)} if options else {}


@pytest.mark.parametrize("key", list(ENGINES))
def test_each_engine_has_the_class_and_tag_converter_the_worker_imports(key):
    spec, module, cls = _adapter(key)
    assert issubclass(cls, TTSEngine)
    assert callable(worker._resolve_converter(module, spec.tag_converter))


@pytest.mark.parametrize("key", list(ENGINES))
def test_the_registered_tag_coverage_is_what_the_converter_really_keeps(key):
    spec, module, _ = _adapter(key)
    convert = worker._resolve_converter(module, spec.tag_converter)
    kept = {tag: bool(convert(f"<{tag}>").strip()) for tag in sent.CANONICAL_TAGS}
    assert kept == spec.tag_coverage


@pytest.mark.parametrize("key", list(ENGINES))
def test_the_registered_options_exist_in_the_adapter(key):
    spec, _, cls = _adapter(key)
    names = set(_option_fields(cls))
    assert set(spec.required_options) <= names
    # The transcript reaches the engine as an option, or (VCB's own engines) as a TTSConfig field.
    takes_transcript = "reference_text" in names or "$reference_text" in spec.tts_config.values()
    assert spec.needs_reference_text == takes_transcript
    assert "gpu" not in names and "device" not in names              # the benchmark pins GPUs, not the adapter


@pytest.mark.parametrize("key", list(ENGINES))
def test_the_extra_config_fields_of_an_engine_are_real_and_not_the_ones_the_worker_sets(key):
    fields = {f.name for f in dataclasses.fields(TTSConfig)}
    assert set(get_engine(key).tts_config) <= fields - {"voice_wav", "device"}


@pytest.mark.parametrize("key", list(ENGINES))
def test_an_engine_performs_sound_tags_exactly_when_it_says_it_does(key):
    spec, _, cls = _adapter(key)
    assert bool(cls.supports_tags) == any(spec.tag_coverage.values())


@pytest.mark.parametrize("key", VCB_ENGINES + RETIRED_VCB_ENGINES)
def test_vcb_engines_are_built_from_the_config_alone_and_the_others_are_not(key):
    spec, _, cls = _adapter(key)
    assert spec.vcb_native and not _option_fields(cls) and not spec.required_options
    assert not any(get_engine(other).vcb_native for other in OPTIONAL_ADAPTERS)


def test_the_weights_listed_for_vcb_engines_are_the_ones_vcb_loads():
    from vcb.tts import kokoro_tts
    assert [w.repo for w in get_engine("kokoro").weights] == [kokoro_tts.REPO_ID]
    assert [w.repo for w in get_engine("csm").weights] == [TTSConfig().csm_model_id]


@pytest.mark.parametrize("key", OPTIONAL_ADAPTERS)
def test_the_weights_to_download_are_the_ones_the_adapter_loads(key):
    spec, _, cls = _adapter(key)
    hub_ids = {f.default for f in _option_fields(cls).values()
               if isinstance(f.default, str) and re.fullmatch(r"[\w.-]+/[\w.-]+", f.default)}
    assert {w.repo for w in spec.weights} == hub_ids


def test_qwen3_downloads_the_model_that_each_voice_mode_loads():
    spec, _, cls = _adapter("qwen3-tts")
    options = cls.Options()
    assert [w.repo for w in spec.weight_repos(("default",))] == [options.custom_voice_model]
    assert [w.repo for w in spec.weight_repos(("clone",))] == [options.base_model]


@pytest.mark.parametrize("key", OPTIONAL_ADAPTERS)
def test_the_download_hint_of_each_adapter_is_a_command_the_tool_accepts(key, monkeypatch):
    spec, module, _ = _adapter(key)
    hub = types.ModuleType("huggingface_hub")

    def snapshot_download(repo, **kwargs):
        raise OSError("not cached")

    hub.snapshot_download = snapshot_download
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    with pytest.raises(FileNotFoundError) as caught:
        module._snapshot("owner/model")
    command = shlex.split(re.search(r"download it first: (.*)$", str(caught.value)).group(1))
    assert command[:2] == ["python", "tools/tts_benchmark.py"] and (REPO_ROOT / command[1]).is_file()
    args = cli.build_parser().parse_args(command[2:])
    assert args.command == "download" and [s.key for s in select_engines(args.engines)] == [spec.key]
