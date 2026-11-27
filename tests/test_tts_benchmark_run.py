"""End-to-end tests of the TTS benchmark: the real runner, worker process and command line, driving the
built-in fake engine (it makes sine waves) instead of a real model. No GPU, model or network is involved."""

from __future__ import annotations

import ast
import contextlib
import csv
import io
import json
import os
import re
import shutil
import subprocess
import sys
import textwrap
import time
import wave
from collections import namedtuple
from pathlib import Path

import pytest

from tts_bench_support import GPUS, REPO_ROOT, TOOLS_DIR, FakeGpuMonitor, NoMonitor, OsAs, install_fake_gpu, write_wav

from tts_bench import cli, runner  # noqa: E402
from tts_bench.catalog import DownloadItem, DownloadPlan  # noqa: E402
from tts_bench.sentences import load_sentences  # noqa: E402

SHIPPED_SENTENCES = TOOLS_DIR / "tts_bench" / "sentences.txt"
PYTHON = Path(sys.executable)
Result = namedtuple("Result", "code out err")


# -- helpers -------------------------------------------------------------------

def run_cli(*argv) -> Result:
    """The command line, with its output captured."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main([str(a) for a in argv])
    return Result(code, out.getvalue(), err.getvalue())


def write_config(path: Path, *, run: str = "", voice: str = "", engines: str = "") -> Path:
    """A settings file: this Python for every engine, and the 3090 Ti as the default GPU."""
    path.write_text(f'[run]\n{run}\n[voice]\n{voice}\n[gpu]\ndefault = "3090 Ti"\n'
                    f"[python]\ndefault = '{PYTHON}'\n{engines}\n", encoding="utf-8")
    return path


def live_workers():
    """PIDs of engine workers still running below this process (None without psutil)."""
    try:
        import psutil
    except ImportError:
        return None
    deadline = time.time() + 5
    while True:
        found = []
        for child in psutil.Process().children(recursive=True):
            try:
                if "tts_bench.worker" in " ".join(child.cmdline()):
                    found.append(child.pid)
            except psutil.Error:
                pass
        if not found or time.time() > deadline:
            return found
        time.sleep(0.2)


class Finished:
    """A finished run: what the command printed and what it left in the output folder."""

    def __init__(self, result, out, config, orphans=None, envs=None):
        self.result, self.out, self.config, self.orphans, self.envs = result, out, config, orphans, envs

    def raw(self, variant="fake"):
        return json.loads((self.out / variant / "raw.json").read_text(encoding="utf-8"))

    @property
    def summary(self):
        return json.loads((self.out / "benchmark_summary.json").read_text(encoding="utf-8"))

    @property
    def csv_rows(self):
        with open(self.out / "benchmark_raw.csv", newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))


def counted(raw):
    return [r for r in raw["rows"] if r["counted"]]


def wav_info(path: Path):
    """(channels, bytes per sample, sample rate, seconds) of a WAV file."""
    with wave.open(str(path), "rb") as fh:
        return fh.getnchannels(), fh.getsampwidth(), fh.getframerate(), fh.getnframes() / fh.getframerate()


# -- killing an engine ---------------------------------------------------------

def test_kill_tree_stops_an_engine_and_everything_it_started(tmp_path):
    (tmp_path / "child.py").write_text(textwrap.dedent("""\
        import sys, time
        end = time.time() + 60
        while time.time() < end:
            with open(sys.argv[1], "a") as fh:
                fh.write(".")
            time.sleep(0.05)
        """), encoding="utf-8")
    (tmp_path / "parent.py").write_text(textwrap.dedent("""\
        import subprocess, sys, time
        subprocess.Popen([sys.executable, sys.argv[1], sys.argv[2]])
        time.sleep(3600)
        """), encoding="utf-8")
    beat = tmp_path / "beat.txt"
    proc = subprocess.Popen([sys.executable, str(tmp_path / "parent.py"), str(tmp_path / "child.py"), str(beat)],
                            start_new_session=sys.platform != "win32")
    try:
        deadline = time.time() + 30
        while time.time() < deadline and not (beat.exists() and beat.stat().st_size >= 3):
            time.sleep(0.05)
        assert beat.exists() and beat.stat().st_size >= 3, "the helper process did not start"
        runner.kill_tree(proc)
        assert proc.poll() is not None
        size = beat.stat().st_size
        time.sleep(0.5)
        assert beat.stat().st_size == size              # the process it started is gone too
    finally:
        if proc.poll() is None:
            proc.kill()


# -- a run with every kind of failure ------------------------------------------

FAILURE_SENTENCES = """\
a01 | neutral | plain | The meeting starts at nine tomorrow.
a02 | human | laugh | That was funny <laugh> and you know it <hesitation> really.
f01 | neutral | plain | [[FAIL]] This one raises an error.
f02 | neutral | plain | [[SILENT]] This one comes back silent.
f03 | neutral | plain | [[EMPTY]] This one returns no audio.
f04 | neutral | plain | [[HANG]] This one never finishes.
a03 | neutral | plain | The last sentence works again.
"""


@pytest.fixture(scope="module")
def failure_run(tmp_path_factory):
    """One run in which sentences fail in every way: an exception, silence, no audio and a hang."""
    work = tmp_path_factory.mktemp("failure_run")
    sentences, out = work / "sentences.txt", work / "out"
    sentences.write_text(FAILURE_SENTENCES, encoding="utf-8")
    (out / "fake").mkdir(parents=True)
    (out / "fake" / "zz99_7.wav").write_bytes(b"left over from an earlier run")
    (out / "fake" / "notes.txt").write_text("not ours", encoding="utf-8")
    config = write_config(
        work / "bench.toml", engines="[engines.fake]\noptions = { latency_s = 0.05 }",
        run=f"runs_per_sentence = 1\nwarmup_runs = 1\ntimeout_s = 1.5\nwarmup_timeout_s = 20\n"
            f"load_timeout_s = 60\nmax_restarts = 2\nsentences = '{sentences}'\noutput_dir = '{out}'")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runner, "list_gpus", lambda: GPUS)
        mp.setattr(runner, "VramMonitor", NoMonitor)
        result = run_cli("run", "--engines", "fake", "--modes", "default", "--no-review", "--config", config)
    return Finished(result, out, config, orphans=live_workers())


def test_failed_sentences_are_recorded_one_by_one_and_the_run_goes_on(failure_run):
    rows = {r["sentence_id"]: r for r in counted(failure_run.raw())}
    assert list(rows) == ["a01", "a02", "f01", "f02", "f03", "f04", "a03"]          # the order of the file
    assert [r["status"] for r in rows.values()] == ["ok", "ok", "error", "error", "error", "timeout", "ok"]
    assert (rows["f01"]["error_type"], "injected failure" in rows["f01"]["error"]) == ("RuntimeError", True)
    assert rows["f02"]["error_type"] == "SilentAudio"                                # silence is not a result
    assert rows["f03"]["error_type"] == "EmptyAudio"
    assert (rows["f04"]["error_type"], rows["f04"]["total_s"]) == ("TimeoutError", 1.5)
    assert failure_run.result.code == 0                                              # some runs succeeded


def test_the_warm_up_is_recorded_but_not_counted(failure_run):
    rows = failure_run.raw()["rows"]
    assert len(rows) == 8
    (warm,) = [r for r in rows if not r["counted"]]
    assert (warm["run"], warm["status"]) == (0, "ok")
    assert failure_run.summary["variants"]["fake"]["runs"] == {"expected": 7, "ok": 3, "failed": 4}


def test_a_synthesis_that_hangs_is_cut_off_and_the_engine_is_restarted(failure_run):
    raw = failure_run.raw()
    assert raw["restarts"] == 1 and len(raw["loads"]) == 2 and all(load["ok"] for load in raw["loads"])
    assert raw["complete"] is True and raw["aborted"] is None
    assert (failure_run.out / "fake" / "worker.log").is_file()                       # for finding out why
    if failure_run.orphans is None:
        pytest.skip("psutil is not installed")
    assert failure_run.orphans == []                                                 # no engine process is left


def test_a_successful_row_has_timing_audio_and_tag_data(failure_run):
    a02 = next(r for r in counted(failure_run.raw()) if r["sentence_id"] == "a02")
    assert (a02["tags_in"], a02["tags_dropped"]) == (2, 1)                           # the fake engine has no (pause)
    assert a02["text_sent"] == "That was funny (laugh) and you know it really."
    assert 0 < a02["ttfa_s"] < a02["total_s"] and a02["n_chunks"] == 2               # streamed: first chunk is early
    assert a02["rtf"] == pytest.approx(a02["total_s"] / a02["audio_s"])
    assert a02["sample_rate"] == 22050


def test_clips_exist_for_successful_runs_only_and_old_ones_are_cleaned_up(failure_run):
    folder = failure_run.out / "fake"
    assert sorted(p.name for p in folder.glob("*.wav")) == ["a01_0.wav", "a01_1.wav", "a02_1.wav", "a03_1.wav"]
    assert (folder / "notes.txt").read_text(encoding="utf-8") == "not ours"          # other files are left alone
    for row in failure_run.raw()["rows"]:
        if row["status"] == "ok":
            assert row["wav"] == f"fake/{row['sentence_id']}_{row['run']}.wav"
            channels, width, rate, seconds = wav_info(failure_run.out / row["wav"])
            assert (channels, width, rate) == (1, 2, 22050) and seconds == pytest.approx(row["audio_s"], abs=0.01)
        else:
            assert row["wav"] == ""                                                  # no clip, no path


def test_the_summary_and_the_terminal_report_the_failures(failure_run):
    s = failure_run.summary["variants"]["fake"]
    assert s["failures"] == {"RuntimeError": 1, "SilentAudio": 1, "EmptyAudio": 1, "TimeoutError": 1}
    assert s["restarts"] == 1 and s["complete"] is True
    out = failure_run.result.out
    assert "Failures:" in out and "fake: 4 of 7 failed" in out and "engine restarted 1x" in out
    assert "done: 3 ok, 4 failed of 7 counted runs" in out
    assert re.search(r"f01\s+.*FAILED\s+RuntimeError: injected failure", out)
    assert "WARNING: nvidia-smi is not available, so VRAM will not be measured" in out
    assert len(failure_run.csv_rows) == 8


def test_the_engine_ran_on_the_default_gpu_the_3090_ti(failure_run):
    raw = failure_run.raw()
    assert raw["gpu"]["index"] == 0 and "3090 Ti" in raw["gpu"]["name"]
    assert raw["versions"]["cuda_visible_devices"] == "0"                            # it sees only that GPU


# -- a clean run of both voice modes -------------------------------------------

@pytest.fixture(scope="module")
def clean_run(tmp_path_factory):
    """Both voice modes on five sentences, two runs each, on the 5060 Ti, with fake VRAM readings."""
    work = tmp_path_factory.mktemp("clean_run")
    wanted = ("n01", "e01", "h01", "h03", "h04")
    sentences, out = work / "sentences.txt", work / "out"
    sentences.write_text("".join(f"{s.id} | {s.category} | {s.intent} | {s.text}\n"
                                 for s in load_sentences(SHIPPED_SENTENCES) if s.id in wanted), encoding="utf-8")
    clip = write_wav(work / "ref.wav", seconds=2)
    config = write_config(
        work / "bench.toml", engines="[engines.fake]\noptions = { latency_s = 0.05 }",
        voice=f"reference_clip = '{clip}'\nreference_text = 'Hello there.'",
        run=f"runs_per_sentence = 2\nwarmup_runs = 1\ntimeout_s = 10\nsentences = '{sentences}'\noutput_dir = '{out}'")
    monitor = FakeGpuMonitor(idle={0: 1000, 1: 5000}, busy={0: 1000, 1: 9000})
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runner, "list_gpus", lambda: GPUS)
        install_fake_gpu(mp, monitor)
        result = run_cli("run", "--engines", "fake", "--gpu", "5060 Ti", "--config", config)
    return Finished(result, out, config)


def test_both_voice_modes_run_and_save_every_clip(clean_run):
    assert clean_run.result.code == 0
    assert sorted(p.name for p in clean_run.out.iterdir() if p.is_dir()) == ["fake", "fake_clone", "review"]
    for variant in ("fake", "fake_clone"):
        raw = clean_run.raw(variant)
        assert len(counted(raw)) == 10 and all(r["status"] == "ok" for r in counted(raw))
        assert raw["complete"] is True and raw["restarts"] == 0
        names = sorted(p.name for p in (clean_run.out / variant).glob("*.wav"))
        assert names == sorted(["n01_0.wav"] + [f"{s}_{n}.wav" for s in ("n01", "e01", "h01", "h03", "h04")
                                                for n in (1, 2)])
    assert clean_run.result.out.count("done: 10 ok, 0 failed of 10 counted runs") == 2


def test_the_clone_variant_records_which_reference_clip_it_used(clean_run):
    voice = clean_run.raw("fake_clone")["voice"]
    assert (clean_run.raw("fake_clone")["mode"], voice["clip"], voice["text"]) == ("clone", "ref.wav", "Hello there.")
    assert re.fullmatch(r"[0-9a-f]{16}", voice["clip_sha"])
    assert clean_run.raw("fake")["mode"] == "default"


def test_the_engine_sees_only_the_gpu_it_was_given(clean_run):
    for variant in ("fake", "fake_clone"):
        raw = clean_run.raw(variant)
        assert raw["gpu"]["index"] == 1 and "5060 Ti" in raw["gpu"]["name"]
        assert raw["versions"]["cuda_visible_devices"] == "1"


def test_vram_is_recorded_for_every_gpu_and_given_back_after_unloading(clean_run):
    raw = clean_run.raw("fake")
    assert [g["index"] for g in raw["gpus_seen"]] == [0, 1]
    assert raw["vram"]["baseline_mib"] == {"0": 1000, "1": 5000}
    assert raw["vram"]["peak_mib"] == {"0": 1000, "1": 9000}
    assert raw["vram"]["delta_mib"] == {"0": 0, "1": 4000}                           # what the engine added
    assert raw["vram"]["released"] is True and raw["vram"]["residual_mib"] == 0
    assert all(r["vram_peak_mib"] == {"0": 1000, "1": 9000} for r in counted(raw))
    assert "unloaded cleanly" in clean_run.result.out


def test_terminal_summary_and_files_show_peak_vram_per_gpu(clean_run):
    out = clean_run.result.out
    assert "peak GPU 0" in out and "peak GPU 1" in out
    assert re.search(r"^\s+fake(\s+\S+){4}\s+1\.0 GB \(\+0\.0\)\s+8\.8 GB \(\+3\.9\)\s+yes$", out, re.M)
    rows = clean_run.csv_rows
    assert len(rows) == 22 and {r["variant"] for r in rows} == {"fake", "fake_clone"}   # 2 x (10 + the warm-up)
    assert {"vram_peak_gpu0_mib", "vram_peak_gpu1_mib"} <= set(rows[0])
    assert {r["vram_peak_gpu1_mib"] for r in rows} == {"9000"} and {r["gpu"] for r in rows} == {"GPU 1 RTX 5060 Ti"}
    summary = clean_run.summary
    assert set(summary["variants"]) == {"fake", "fake_clone"} and summary["warnings"] == []
    assert summary["variants"]["fake"]["vram"]["peak_mib"]["1"] == 9000
    assert [g["index"] for g in summary["gpus"]] == [0, 1]


def test_the_run_builds_the_blind_listening_page(clean_run):
    assert "blind listening" in clean_run.result.out
    assert "tts_benchmark.py serve" in clean_run.result.out                          # how to listen in VS Code's browser
    html = (clean_run.out / "review.html").read_text(encoding="utf-8")
    clips = list((clean_run.out / "review" / "clips").glob("*.wav"))
    assert len(clips) == 10 and all(re.fullmatch(r"[0-9a-f]{12}", c.stem) for c in clips)
    assert "fake_clone" not in html and "Fake engine" not in html and "http" not in html.replace("https?", "")
    radios = lambda field: len(re.findall(rf'<input type="radio"[^>]*data-field="{field}"', html))  # noqa: E731
    assert radios("naturalness") == radios("emotion") == 10 * 5
    assert radios("human_sounds") == 3 * 2 * 5                                       # h01, h03 and h04 only


def test_report_and_review_commands_rebuild_from_the_saved_results(clean_run, tmp_path):
    copy = tmp_path / "copy"
    shutil.copytree(clean_run.out, copy)
    for name in ("benchmark_summary.json", "benchmark_raw.csv", "review.html"):
        (copy / name).unlink()
    result = run_cli("report", "--output", copy, "--config", clean_run.config)
    assert result.code == 0 and "TTS benchmark: 5 sentences x 2 runs" in result.out
    assert (copy / "benchmark_summary.json").is_file() and (copy / "benchmark_raw.csv").is_file()
    result = run_cli("review", "--output", copy, "--config", clean_run.config, "--seed", 5)
    assert result.code == 0 and "Blind listening page:" in result.out and (copy / "review.html").is_file()
    assert "tts_benchmark.py serve" in result.out


def test_report_and_review_say_when_there_is_nothing_to_use(clean_run, tmp_path):
    result = run_cli("report", "--output", tmp_path / "nothing", "--config", clean_run.config)
    assert result.code == 1 and "No results folder at" in result.out
    (tmp_path / "empty").mkdir()
    result = run_cli("review", "--output", tmp_path / "empty", "--config", clean_run.config)
    assert result.code == 1 and "nothing to review" in result.out


# -- VRAM that is not given back, and the engine's environment -----------------

@pytest.fixture(scope="module")
def leak_run(tmp_path_factory):
    """One sentence, with an engine whose VRAM never comes back after unloading."""
    work = tmp_path_factory.mktemp("leak_run")
    sentences, out = work / "sentences.txt", work / "out"
    sentences.write_text("n01 | neutral | plain | The meeting starts at nine tomorrow.\n", encoding="utf-8")
    config = write_config(
        work / "bench.toml", engines="[engines.fake]\noptions = { latency_s = 0.05 }",
        run=f"runs_per_sentence = 1\nwarmup_runs = 0\ntimeout_s = 10\nsentences = '{sentences}'\noutput_dir = '{out}'")
    monitor = FakeGpuMonitor(idle={0: 1000, 1: 5000}, busy={0: 1000, 1: 9000}, leak=True)
    envs = []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runner, "list_gpus", lambda: GPUS)
        mp.setattr(runner, "RELEASE_WAIT_S", 0.5)
        install_fake_gpu(mp, monitor)
        tracked = runner.WorkerProcess

        class Recording(tracked):
            def __init__(self, cmd, env, cwd, log_path):
                envs.append(dict(env))
                super().__init__(cmd, env, cwd, log_path)

        mp.setattr(runner, "WorkerProcess", Recording)
        result = run_cli("run", "--engines", "fake", "--modes", "default", "--gpu", "5060 Ti", "--no-review",
                         "--config", config)
    return Finished(result, out, config, envs=envs)


def test_vram_that_is_not_given_back_is_reported(leak_run):
    assert leak_run.result.code == 0                                                 # the run itself worked
    vram = leak_run.raw()["vram"]
    assert (vram["released"], vram["residual_mib"]) == (False, 4000)
    assert "WARNING: 4000 MiB still in use after unloading" in leak_run.result.out
    assert "NO (4000 MiB left)" in leak_run.result.out


def test_the_engine_process_gets_an_isolated_offline_environment(leak_run):
    (env,) = leak_run.envs
    assert env["HF_HUB_OFFLINE"] == "1" and env["TRANSFORMERS_OFFLINE"] == "1"
    assert (env["CUDA_VISIBLE_DEVICES"], env["CUDA_DEVICE_ORDER"]) == ("1", "PCI_BUS_ID")
    assert not [k for k in env if k.upper().startswith("VCB_") and k != "VCB_ENV_FILE"]
    assert env["PYTHONPATH"].split(os.pathsep)[0] == str(TOOLS_DIR)


# -- an engine that VCB loads itself, kept offline -----------------------------

NATIVE_SENTENCES = """\
v01 | human | laugh | Oh no zzecho <laugh> that is funny <hesitation> really.
v02 | emotion | excited | <excited> It was AMAZING, <breath> zzecho truly!
v03 | neutral | plain | zzlookup The meeting starts at nine tomorrow.
"""


@pytest.fixture(scope="module")
def native_run(tmp_path_factory):
    """The hidden engine "fake-native", which stands in for Kokoro, XTTS and the other engines VCB builds from a
    TTSConfig, in both voice modes. It fails with what it was given, and it looks up host names the way a model
    manager does (once while loading, once for the sentence v03)."""
    work = tmp_path_factory.mktemp("native_run")
    sentences, out = work / "sentences.txt", work / "out"
    sentences.write_text(NATIVE_SENTENCES, encoding="utf-8")
    clip = write_wav(work / "ref.wav", seconds=2)
    config = write_config(
        work / "bench.toml", voice=f"reference_clip = '{clip}'\nreference_text = 'Hello there.'",
        run=f"runs_per_sentence = 1\nwarmup_runs = 0\ntimeout_s = 10\nsentences = '{sentences}'\noutput_dir = '{out}'")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runner, "list_gpus", lambda: GPUS)
        mp.setattr(runner, "VramMonitor", NoMonitor)
        mp.setenv("TTS_BENCH_FAKE_LOOKUP", "1")
        result = run_cli("run", "--engines", "fake-native", "--no-review", "--config", config)
    return Finished(result, out, config)


NATIVE_VARIANTS = ("fake-native", "fake-native_clone")


def native_rows(run, variant):
    return {r["sentence_id"]: r for r in counted(run.raw(variant))}


def test_a_vcb_engine_hears_each_line_the_way_vcb_prepares_it(native_run):
    assert native_run.result.code == 0
    for variant in NATIVE_VARIANTS:
        rows = native_rows(native_run, variant)
        # <laugh> is spoken as a word, other tags go and a shouted long word is lower-cased; a mood tag the engine
        # performs stays
        assert "heard 'Oh no zzecho Haha! that is funny really.'" in rows["v01"]["error"]
        assert "heard '<excited> It was amazing, zzecho truly!'" in rows["v02"]["error"]
        v01 = rows["v01"]
        assert (v01["text_sent"], v01["tags_in"], v01["tags_dropped"]) == (
            "Oh no zzecho Haha! that is funny really.", 2, 2)            # tags are counted in the sentence as written
    assert 'fake-native: performs no sound tags; as in VCB, <laugh> and <sigh> were spoken as the words "Haha!"' \
        in native_run.result.out


def test_a_vcb_engine_gets_the_voice_settings_of_its_variant(native_run):
    clip = str(native_run.config.parent / "ref.wav")
    assert native_rows(native_run, "fake-native")["v01"]["error"].endswith(
        "voice_wav=''; csm_ref_audio=''; csm_ref_text=''")                # the built-in voice
    assert native_rows(native_run, "fake-native_clone")["v01"]["error"].endswith(
        f"voice_wav={clip!r}; csm_ref_audio={clip!r}; csm_ref_text='Hello there.'")


def test_a_lookup_that_the_library_swallows_does_not_fail_the_synthesis(native_run):
    for variant in NATIVE_VARIANTS:
        assert [r["status"] for r in native_rows(native_run, variant).values()] == ["error", "error", "ok"]
        assert (native_run.out / variant / "v03_1.wav").is_file()


def test_a_vcb_engine_is_kept_offline_and_the_run_says_what_was_refused(native_run):
    looked_up = ["look up bench-load.invalid", "look up bench-synth.invalid"]
    for variant in NATIVE_VARIANTS:
        raw = native_run.raw(variant)
        assert raw["offline"] is True
        assert raw["loads"][0]["network_blocked"] == ["look up bench-load.invalid"]     # while it loaded
        assert raw["network_blocked"] == looked_up                                      # and while it synthesized
        assert native_run.summary["variants"][variant]["network_blocked"] == looked_up
        log = (native_run.out / variant / "worker.log").read_text(encoding="utf-8", errors="replace")
        assert "[netguard] blocked: look up bench-load.invalid" in log
    out = native_run.result.out
    assert out.count("   kept off the network: look up bench-load.invalid") == 2        # once per engine start
    assert "Network: engines run offline; these connection attempts were refused (nothing was sent):" in out
    for variant in NATIVE_VARIANTS:
        assert f"  {variant}: look up bench-load.invalid; look up bench-synth.invalid" in out


# -- the command line without any engine ---------------------------------------

def _plan_config(tmp_path):
    clip = write_wav(tmp_path / "ref.wav")
    return write_config(tmp_path / "bench.toml", voice=f"reference_clip = '{clip}'\nreference_text = 'Hello there.'")


def test_dry_run_prints_the_plan_and_starts_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "list_gpus", lambda: GPUS)
    monkeypatch.setattr(cli, "run_plan", lambda *a, **k: pytest.fail("no engine may start in a dry run"))
    result = run_cli("run", "--engines", "qwen3-tts", "--dry-run", "--config", _plan_config(tmp_path))
    assert result.code == 0
    assert "Benchmark plan: 15 sentences x 3 runs, +1 warm-up per engine (not counted), " \
           "30 s timeout per synthesis" in result.out
    gpu = r"GPU 0 \(NVIDIA GeForce RTX 3090 Ti, 24 GB\)"
    assert re.search(rf"^\s+qwen3-tts\s+built-in voice\s+{gpu}\s+46 syntheses\s*$", result.out, re.M)
    assert re.search(rf"^\s+qwen3-tts_clone\s+reference clip\s+{gpu}\s+46 syntheses\s*$", result.out, re.M)
    assert result.out.rstrip().endswith("Dry run: no engine was started.")


def test_the_gpu_option_moves_every_engine(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "list_gpus", lambda: GPUS)
    result = run_cli("run", "--engines", "chatterbox", "--modes", "default", "--gpu", "5060", "--dry-run",
                     "--config", _plan_config(tmp_path))
    assert "GPU 1 (NVIDIA GeForce RTX 5060 Ti, 16 GB)" in result.out and "3090" not in result.out


def test_linux_only_engines_say_why_they_cannot_run_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "list_gpus", lambda: GPUS)
    monkeypatch.setattr(runner, "os", OsAs("nt"))
    config = _plan_config(tmp_path)
    result = run_cli("run", "--engines", "voxtral", "--dry-run", "--config", config)
    assert result.code == 0 and "cannot run: its runtime has no Windows build" in result.out
    result = run_cli("run", "--engines", "voxtral", "--config", config)
    assert result.code == 1 and "Nothing can run" in result.out


@pytest.mark.parametrize("argv, message", [
    (["run", "--engines", "nope"], "error: unknown engine 'nope'"),
    (["run", "--modes", "both"], "error: unknown voice mode(s): both"),
    (["run", "--runs", "0"], "error: runs per sentence must be at least 1"),
    (["run", "--gpu", "4090"], "no GPU name contains it"),
    (["download"], "error: choose what to download"),
])
def test_bad_options_are_explained_and_exit_with_code_2(tmp_path, monkeypatch, argv, message):
    monkeypatch.setattr(runner, "list_gpus", lambda: GPUS)
    result = run_cli(*argv, "--dry-run" if argv[0] == "run" else "--config", *(
        [] if argv[0] == "run" else [_plan_config(tmp_path)]), *(["--config", _plan_config(tmp_path)]
                                                                 if argv[0] == "run" else []))
    assert result.code == 2 and message in result.err


# -- downloading ---------------------------------------------------------------

def _download_plan(**kw):
    item = DownloadItem(repo="BreezeBlue/Breeze-TTS-2", size_gb=7.68, engine="breeze", python=PYTHON,
                        cache_dir="C:/hf", commercial=False,
                        license="BreezeBlue Research and Non-Commercial License v1.1")
    return DownloadPlan(items=kw.get("items", [item]), present=[], problems=kw.get("problems", []))


@pytest.fixture()
def downloads(tmp_path, monkeypatch):
    """`download` with a canned plan; records what would have been fetched."""
    fetched = []
    state = {"plan": _download_plan()}
    monkeypatch.setattr(cli, "plan_downloads", lambda *a, **k: state["plan"])
    monkeypatch.setattr(cli, "_fetch", lambda python, repos: fetched.append((python, list(repos))) or 0)
    return fetched, state, _plan_config(tmp_path)


def test_download_shows_the_license_and_does_nothing_without_a_confirmation(downloads, monkeypatch):
    fetched, _, config = downloads
    monkeypatch.setattr(sys, "stdin", io.StringIO("y\n"))                            # piped input is no confirmation
    result = run_cli("download", "--engines", "breeze", "--config", config)
    assert result.code == 1 and fetched == []
    assert re.search(r"BreezeBlue/Breeze-TTS-2\s+7\.68 GB\s+breeze\s+NON-COMMERCIAL", result.out)
    assert "license of BreezeBlue/Breeze-TTS-2: BreezeBlue Research and Non-Commercial License v1.1" in result.out
    assert "Not confirmed: run in a terminal, or add --yes. Nothing was downloaded." in result.out


def test_download_with_yes_fetches_into_the_engines_own_environment(downloads):
    fetched, _, config = downloads
    result = run_cli("download", "--engines", "breeze", "--yes", "--config", config)
    assert result.code == 0 and fetched == [(PYTHON, ["BreezeBlue/Breeze-TTS-2"])]


def test_download_stops_when_an_environment_cannot_be_inspected_or_nothing_is_missing(downloads):
    fetched, state, config = downloads
    state["plan"] = _download_plan(problems=["python: exit code 1: boom"])
    assert run_cli("download", "--engines", "breeze", "--yes", "--config", config).code == 1
    state["plan"] = _download_plan(items=[])
    result = run_cli("download", "--engines", "breeze", "--yes", "--config", config)
    assert result.code == 0 and "Nothing to download." in result.out
    assert fetched == []


def test_download_leaves_the_engines_vcb_installs_itself_to_vcb(tmp_path, monkeypatch):
    fetched = []
    monkeypatch.setattr(cli, "_fetch", lambda python, repos: fetched.append((python, list(repos))) or 0)
    result = run_cli("download", "--engines", "kokoro,chatterbox", "--yes", "--config", _plan_config(tmp_path))
    assert result.code == 0 and fetched == []
    assert "Left to VCB, which installs its own voices: kokoro, chatterbox" in result.out
    assert result.out.rstrip().endswith("Nothing to download.")


# -- everything stays local ----------------------------------------------------

NETWORK_MODULES = {"requests", "httpx", "aiohttp", "urllib.request", "urllib3", "http.client", "http.server",
                   "socketserver", "socket", "ssl", "websocket", "websockets", "smtplib", "ftplib", "xmlrpc.client",
                   "grpc", "huggingface_hub", "openai", "anthropic", "mistralai", "boto3", "gradio", "wandb",
                   "sentry_sdk"}
ADAPTERS = ("breeze_tts.py", "qwen3_tts.py", "step_audio_editx_tts.py", "voxtral_tts.py")
ALLOWED_NETWORK_IMPORTS = {
    "fetch.py": {"huggingface_hub"},                                      # the explicit `download` command
    "serve.py": {"http.server", "socketserver"},                          # the page server: 127.0.0.1 only
    "netguard.py": {"socket"},                                            # it only replaces socket functions with refusals
    "fake_engine.py": {"socket"},                                         # the test engine looks up a name that cannot exist
    "breeze_tts.py": {"huggingface_hub"},                                 # a look in the local cache (offline)
    "qwen3_tts.py": {"huggingface_hub"},
    "step_audio_editx_tts.py": {"huggingface_hub"},
    "voxtral_tts.py": {"huggingface_hub", "http.client", "urllib.request", "socket"},   # its own server, on 127.0.0.1
}


def network_imports(path: Path) -> set[str]:
    """The network-capable modules a source file imports (anywhere in it, also inside functions)."""
    names = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)     # `from urllib import request`
    return {m for m in NETWORK_MODULES if any(n == m or n.startswith(m + ".") for n in names)}


def test_only_the_download_command_and_a_local_server_use_network_libraries():
    files = [*sorted((TOOLS_DIR / "tts_bench").glob("*.py")), TOOLS_DIR / "tts_benchmark.py",
             *(REPO_ROOT / "vcb" / "tts" / name for name in ADAPTERS)]
    offenders = {p.name: sorted(extra) for p in files
                 if (extra := network_imports(p) - ALLOWED_NETWORK_IMPORTS.get(p.name, set()))}
    assert offenders == {}
    assert network_imports(TOOLS_DIR / "tts_bench" / "fetch.py") == {"huggingface_hub"}


def test_voxtral_adapter_only_talks_to_its_own_loopback_server():
    source = (REPO_ROOT / "vcb" / "tts" / "voxtral_tts.py").read_text(encoding="utf-8")
    urls = re.findall(r"https?://[^\s\"'`)]+", source)
    assert urls and all(u.startswith("http://127.0.0.1") for u in urls)


HUB_PROBE = """
import socket, sys
attempts = []
def deny(*args, **kwargs):
    attempts.append(1)
    raise OSError("blocked by the test")
socket.socket.connect = deny
socket.create_connection = deny
socket.getaddrinfo = deny
import huggingface_hub
error = "none"
try:
    huggingface_hub.snapshot_download("nobody/never-downloaded", cache_dir=sys.argv[1])
except BaseException as exc:
    error = type(exc).__name__
print(error, len(attempts))
"""


def test_engines_cannot_reach_the_network_unless_downloading_is_allowed(tmp_path):
    pytest.importorskip("huggingface_hub")

    def hub_lookup(allow_download):
        env = runner.worker_env(os.environ, runner.Target("cpu", None), allow_download=allow_download)
        done = subprocess.run([sys.executable, "-c", HUB_PROBE, str(tmp_path)], env=env, capture_output=True,
                              text=True, timeout=180)
        error, attempts = done.stdout.split()
        return error, int(attempts)

    error, attempts = hub_lookup(allow_download=False)
    assert attempts == 0 and error in {"LocalEntryNotFoundError", "OfflineModeIsEnabled"}   # offline: no attempt
    assert hub_lookup(allow_download=True)[1] >= 1                  # the same code does try when allowed (blocked here)
