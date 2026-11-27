"""Tests for the TTS benchmark's reports (statistics, CSV, JSON, terminal tables) and the blind
listening page. They run on synthetic results, so no engine is started."""

from __future__ import annotations

import array
import base64
import contextlib
import csv
import hashlib
import io
import json
import math
import re
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import pytest

from tts_bench_support import GPUS, TOOLS_DIR, write_wav

from tts_bench import cli, report, review, runner  # noqa: E402
from tts_bench import sentences as sent  # noqa: E402
from tts_bench.registry import get_engine, variant_name  # noqa: E402

SENTENCES = sent.load_sentences(TOOLS_DIR / "tts_bench" / "sentences.txt")
VARIANTS = ("chatterbox", "breeze", "voxtral", "step-audio-editx", "qwen3-tts")
CONDITIONS = {"sentences_sha": "0123456789abcdef", "n_sentences": 15, "runs_per_sentence": 3,
              "warmup_runs": 1, "timeout_s": 30.0, "seed": 1234}


# -- synthetic results ---------------------------------------------------------

def make_row(sid, run=1, *, category="neutral", status="ok", total=1.0, audio=2.0, ttfa=None,
             error_type=None, counted=True, tags_in=0, tags_dropped=0, vram=None, wav=None):
    """One synthesis, as the runner records it (a successful one streams: TTFA is half the total)."""
    ok = status == "ok"
    return {
        "sentence_id": sid, "category": category, "intent": "plain", "run": run, "counted": counted,
        "status": status, "error_type": error_type, "error": None if ok else "boom",
        "ttfa_s": (total / 2 if ttfa is None else ttfa) if ok else None,
        "total_s": total, "audio_s": audio if ok else None, "rtf": total / audio if ok else None,
        "n_chunks": 2 if ok else None, "sample_rate": 24000 if ok else None, "text_sent": "text",
        "tags_in": tags_in, "tags_dropped": tags_dropped, "wav": wav, "vram_peak_mib": vram or {},
        "torch_peak_alloc_mib": None, "torch_peak_reserved_mib": None, "started_at": 1.0,
    }


def ok_rows(totals, audio=2.0, category="neutral"):
    return [make_row(f"s{i}", 1, total=t, audio=audio, category=category) for i, t in enumerate(totals)]


def make_variant(engine, rows, *, mode="default", gpu=0, **extra):
    """The contents of one output/<variant>/raw.json."""
    info, name = GPUS[gpu], variant_name(engine, mode)
    for row in rows:
        row.setdefault("engine", engine)
        row.setdefault("variant", name)
        row.setdefault("mode", mode)
    data = {
        "schema": runner.SCHEMA_VERSION, "engine": engine, "title": get_engine(engine).title,
        "variant": name, "mode": mode,
        "gpu": {"spec": str(gpu), "index": info.index, "name": info.name, "total_mib": info.total_mib},
        "conditions": dict(CONDITIONS), "complete": True, "restarts": 0, "aborted": None,
        "loads": [{"ok": True, "load_s": 4.2}], "vram": {}, "versions": {}, "voice": None, "rows": rows,
    }
    data.update(extra)
    return data


def vram_info(peak0=9000, peak1=5000):
    return {"baseline_mib": {"0": 1000, "1": 5000}, "peak_mib": {"0": peak0, "1": peak1},
            "delta_mib": {"0": peak0 - 1000, "1": peak1 - 5000}, "released": True, "residual_mib": 0}


def both_gpus():
    return [{"index": i, "name": g.name, "total_mib": g.total_mib} for i, g in enumerate(GPUS)]


def summary_of(*variants):
    return report.build_summary(list(variants))


# -- statistics per variant ----------------------------------------------------

def test_latency_statistics_use_only_counted_successful_runs():
    rows = ok_rows([1, 2, 3, 4, 5])
    rows.append(make_row("w", 0, counted=False, total=100, audio=1))          # the warm-up is not counted
    rows.append(make_row("t", 1, status="timeout", error_type="TimeoutError", total=30))
    s = report.summarise_variant(make_variant("qwen3-tts", rows))
    assert s["runs"] == {"expected": 6, "ok": 5, "failed": 1}
    assert (s["total_s"]["n"], s["total_s"]["mean"], s["total_s"]["median"]) == (5, 3, 3)
    assert s["total_s"]["p95"] == pytest.approx(4.8)
    assert (s["ttfa_s"]["mean"], s["ttfa_s"]["median"]) == (1.5, 1.5)
    assert s["failures"] == {"TimeoutError": 1}


def test_real_time_factor_is_reported_per_run_and_overall():
    rows = [make_row(f"s{i}", 1, total=3, audio=a) for i, a in enumerate([2, 2, 2, 2, 10])]
    s = report.summarise_variant(make_variant("breeze", rows))
    assert s["rtf"]["mean"] == pytest.approx((4 * 1.5 + 0.3) / 5)             # synthesis time / audio duration
    assert s["rtf"]["median"] == pytest.approx(1.5)
    assert s["rtf_overall"] == pytest.approx(15 / 18)                          # long clips weigh more here


def test_failures_are_counted_by_error_type():
    rows = ok_rows([1, 1]) + [
        make_row("a", status="error", error_type="RuntimeError"),
        make_row("b", status="error", error_type="RuntimeError"),
        make_row("c", status="error", error_type="SilentAudio"),
        make_row("d", status="timeout", error_type="TimeoutError", total=30)]
    s = report.summarise_variant(make_variant("breeze", rows, restarts=1))
    assert s["failures"] == {"RuntimeError": 2, "SilentAudio": 1, "TimeoutError": 1}
    assert s["runs"] == {"expected": 6, "ok": 2, "failed": 4}
    assert s["restarts"] == 1


def test_dropped_sound_tags_are_counted_once_per_sentence_not_per_run():
    rows = [make_row("h1", run, category="human", tags_in=2, tags_dropped=1) for run in (1, 2, 3)]
    rows += [make_row("h2", run, category="human", tags_in=1, tags_dropped=0) for run in (1, 2, 3)]
    assert report.summarise_variant(make_variant("voxtral", rows))["tags"] == {"in": 3, "dropped": 1}


def test_statistics_are_also_given_per_category():
    rows = [make_row("n1", total=1), make_row("e1", category="emotion", total=2)]
    by_category = report.summarise_variant(make_variant("breeze", rows))["by_category"]
    assert set(by_category) == {"neutral", "emotion"}                          # no human rows, no entry
    assert by_category["emotion"]["total_s"]["mean"] == 2


def test_peak_vram_of_the_synthesis_windows_is_kept_per_gpu():
    rows = [make_row("a", vram={"0": 9000, "1": 5000}), make_row("b", vram={"0": 9500, "1": 5000})]
    s = report.summarise_variant(make_variant("breeze", rows, vram=vram_info(9800, 5000)))
    assert s["vram"]["synth_peak_mib"] == {"0": 9500, "1": 5000}
    assert s["vram"]["peak_mib"] == {"0": 9800, "1": 5000}


# -- the summary and its comparability warnings --------------------------------

def test_runs_under_the_same_conditions_produce_no_warnings():
    a = make_variant("chatterbox", ok_rows([1, 2]), vram=vram_info())
    b = make_variant("qwen3-tts", ok_rows([1, 2]), vram=vram_info())
    assert summary_of(a, b)["warnings"] == []


def test_runs_under_different_conditions_are_flagged():
    a = make_variant("chatterbox", ok_rows([1]), vram=vram_info())
    b = make_variant("breeze", ok_rows([1]), vram=vram_info(), conditions={**CONDITIONS, "runs_per_sentence": 5})
    c = make_variant("qwen3-tts", ok_rows([1]), vram=vram_info(), gpu=1)
    d = make_variant("voxtral", ok_rows([1]), vram=vram_info(), complete=False)
    text = "\n".join(summary_of(a, b, c, d)["warnings"])
    assert "breeze ran with different settings than chatterbox (runs_per_sentence 5 vs 3)" in text
    assert "engines ran on different GPUs (GPU 0 RTX 3090 Ti; GPU 1 RTX 5060 Ti): " \
           "latency is not directly comparable" in text
    assert "voxtral: the run is incomplete" in text


def test_missing_vram_data_and_different_reference_clips_are_flagged():
    voice = {"clip": "a.wav", "clip_sha": "aa", "text": "Hi."}
    a = make_variant("chatterbox", ok_rows([1]), mode="clone", voice=voice)
    b = make_variant("qwen3-tts", ok_rows([1]), mode="clone", voice={**voice, "clip": "b.wav", "clip_sha": "bb"})
    warnings = summary_of(a, b)["warnings"]
    assert "clone variants used different reference clips or transcripts" in warnings
    assert "no VRAM data was recorded (nvidia-smi was not available)" in warnings


def test_summary_lists_every_watched_gpu_and_the_licenses_of_the_engines_present():
    a = make_variant("breeze", ok_rows([1]), vram=vram_info(), gpus_seen=both_gpus())
    b = make_variant("qwen3-tts", ok_rows([1]), vram=vram_info(), gpus_seen=both_gpus())
    summary = summary_of(a, b)
    assert [g["index"] for g in summary["gpus"]] == [0, 1]
    assert list(summary["variants"]) == ["breeze", "qwen3-tts"]
    assert set(summary["licenses"]) == {"breeze", "qwen3-tts"}
    assert summary["licenses"]["breeze"]["weights_commercial"] is False
    assert summary["licenses"]["qwen3-tts"]["weights_commercial"] is True
    assert [g["index"] for g in summary_of(make_variant("breeze", ok_rows([1]), gpu=1))["gpus"]] == [1]
    assert summary_of(make_variant("fake", ok_rows([1])))["licenses"] == {}        # the self-test engine


# -- the terminal summary ------------------------------------------------------

def test_terminal_summary_shows_latency_rtf_and_vram_for_each_gpu():
    a = make_variant("chatterbox", ok_rows([1, 2, 3]), vram=vram_info(9000, 5000), gpus_seen=both_gpus())
    b = make_variant("qwen3-tts", ok_rows([1, 2, 3]), gpus_seen=both_gpus()[:1],
                     vram={"peak_mib": {"0": 7000}, "delta_mib": {"0": 6000}, "released": True})
    text = report.format_summary(summary_of(a, b))
    assert "TTS benchmark: 15 sentences x 3 runs (+1 warm-up, not counted), 30 s timeout per synthesis" in text
    header = next(line for line in text.splitlines() if "rtf mean" in line)
    assert "peak GPU 0" in header and "peak GPU 1" in header and header.endswith("unloaded")
    assert re.search(r"^\s+chatterbox(\s+\S+){4}\s+8\.8 GB \(\+7\.8\)\s+4\.9 GB \(\+0\.0\)\s+yes$", text, re.M)
    assert re.search(r"^\s+qwen3-tts(\s+\S+){4}\s+6\.8 GB \(\+5\.9\)\s+-\s+yes$", text, re.M)
    assert re.search(r"^\s+chatterbox\s+3/3\s+GPU 0 RTX 3090 Ti\s", text, re.M)    # successes and the GPU used


def test_leaked_vram_is_called_out_in_the_summary():
    leak = make_variant("breeze", ok_rows([1]), vram={**vram_info(), "released": False, "residual_mib": 4000})
    assert "NO (4000 MiB left)" in report.format_summary(summary_of(leak))


def test_failures_and_restarts_are_listed_with_their_types():
    rows = ok_rows([1, 1, 1]) + [make_row("f", status="timeout", error_type="TimeoutError", total=30)]
    text = report.format_summary(summary_of(make_variant("breeze", rows, restarts=1, vram=vram_info())))
    assert "Failures:\n  breeze: 1 of 4 failed (1 x TimeoutError); engine restarted 1x" in text
    clean = report.format_summary(summary_of(make_variant("breeze", ok_rows([1]), vram=vram_info())))
    assert "Failures: none" in clean


def test_summary_warns_about_non_commercial_weights():
    variants = [make_variant(key, ok_rows([1]), vram=vram_info()) for key in VARIANTS]
    text = report.format_summary(summary_of(*variants))
    block = text.split("Licenses (check before any commercial use):")[1].split("\n\n")[0]
    assert "breeze: weights non-commercial" in block
    assert "voxtral: weights non-commercial" in block
    assert "step-audio-editx: weights not stated" in block
    assert "qwen3-tts" not in block and "chatterbox" not in block


def test_summary_notes_sound_tags_an_engine_cannot_perform():
    rows = [make_row("h1", category="human", tags_in=2, tags_dropped=2),
            make_row("h2", category="human", tags_in=1, tags_dropped=0)]
    text = report.format_summary(summary_of(make_variant("voxtral", rows, vram=vram_info())))
    assert "voxtral: 2 of 3 sound tags have no equivalent in this engine" in text


def test_summary_says_what_vcb_does_for_its_own_engines_that_cannot_perform_sound_tags():
    rows = [make_row("h1", category="human", tags_in=3, tags_dropped=3)]
    text = report.format_summary(summary_of(make_variant("kokoro", rows, vram=vram_info())))
    assert ('kokoro: performs no sound tags; as in VCB, <laugh> and <sigh> were spoken as the words '
            '"Haha!" and "Haaah." and the other tags were removed') in text
    assert "have no equivalent" not in text
    some = [make_row("h1", category="human", tags_in=3, tags_dropped=1)]            # Chatterbox performs two of them
    text = report.format_summary(summary_of(make_variant("chatterbox", some, vram=vram_info())))
    assert "chatterbox: 1 of 3 sound tags have no equivalent in this engine" in text


def test_summary_lists_the_connections_the_engines_were_kept_from():
    refused = ["look up huggingface.co", "connect 192.0.2.1:443"]
    xtts = make_variant("xtts", ok_rows([1]), mode="clone", vram=vram_info(), network_blocked=refused)
    quiet = make_variant("kokoro", ok_rows([1]), vram=vram_info())
    summary = summary_of(xtts, quiet)
    assert summary["variants"]["xtts_clone"]["network_blocked"] == refused
    assert summary["variants"]["kokoro"]["network_blocked"] == []
    text = report.format_summary(summary)
    assert ("Network: engines run offline; these connection attempts were refused (nothing was sent):\n"
            "  xtts_clone: look up huggingface.co; connect 192.0.2.1:443") in text
    assert "kokoro:" not in text.split("Network:")[1]
    assert "Network:" not in report.format_summary(summary_of(quiet))


def test_empty_results_give_a_clear_message():
    assert report.format_summary(summary_of()) == "No results found. Run the benchmark first."


# -- the files -----------------------------------------------------------------

def test_csv_has_one_row_per_synthesis_and_a_vram_column_per_gpu(tmp_path):
    rows = [make_row("w", 0, counted=False, total=2, vram={"0": 9000, "1": 5000}),
            make_row("a", 1, vram={"0": 9100, "1": 5000}),
            make_row("b", 1, status="error", error_type="RuntimeError", vram={"0": 9100})]
    path = tmp_path / "raw.csv"
    report.write_csv([make_variant("breeze", rows)], path)
    with open(path, newline="", encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
    assert len(table) == 3
    assert {"vram_peak_gpu0_mib", "vram_peak_gpu1_mib", "ttfa_s", "total_s", "rtf", "error_type"} <= set(table[0])
    assert [r["counted"] for r in table] == ["False", "True", "True"]          # the warm-up is flagged
    assert table[1]["vram_peak_gpu0_mib"] == "9100" and table[2]["vram_peak_gpu1_mib"] == ""
    assert (table[2]["error_type"], table[2]["ttfa_s"]) == ("RuntimeError", "")
    assert (table[0]["variant"], table[0]["gpu"]) == ("breeze", "GPU 0 RTX 3090 Ti")


def test_reports_are_written_from_the_per_variant_files_in_report_order(tmp_path):
    for data in (make_variant("qwen3-tts", ok_rows([1, 2])),
                 make_variant("chatterbox", ok_rows([1, 2]), mode="clone"),
                 make_variant("chatterbox", ok_rows([1, 2]))):
        (tmp_path / data["variant"]).mkdir()
        (tmp_path / data["variant"] / "raw.json").write_text(json.dumps(data), encoding="utf-8")
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "raw.json").write_text("{not json", encoding="utf-8")
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "raw.json").write_text(json.dumps({"schema": 0}), encoding="utf-8")

    summary, csv_path, json_path = report.write_reports(tmp_path)
    assert list(summary["variants"]) == ["chatterbox", "chatterbox_clone", "qwen3-tts"]
    assert (csv_path.name, json_path.name) == ("benchmark_raw.csv", "benchmark_summary.json")
    assert list(json.loads(json_path.read_text(encoding="utf-8"))["variants"]) == list(summary["variants"])
    assert any("broken: unreadable raw.json" in w for w in summary["warnings"])
    assert any("old: raw.json has schema 0" in w for w in summary["warnings"])
    with open(csv_path, newline="", encoding="utf-8") as fh:
        assert len(list(csv.DictReader(fh))) == 6


# -- the blind listening page --------------------------------------------------

def make_results(out: Path, keys, sentences, failed=(), amplitudes=None):
    """out/<variant>/raw.json plus one WAV per sentence, laid out like a finished run."""
    for key in keys:
        (out / key).mkdir(parents=True, exist_ok=True)
        rows = []
        for s in sentences:
            if (key, s.id) in failed:
                rows.append(make_row(s.id, 1, category=s.category, status="error", error_type="RuntimeError"))
                continue
            wav = f"{key}/{s.id}_1.wav"
            write_wav(out / wav, amplitude=(amplitudes or {}).get(key, 0.2))
            rows.append(make_row(s.id, 1, category=s.category, wav=wav))
        (out / key / "raw.json").write_text(json.dumps(make_variant(key, rows)), encoding="utf-8")


def levels(path: Path):
    """(RMS, peak) of a 16-bit mono WAV, as fractions of full scale."""
    with wave.open(str(path), "rb") as fh:
        samples = array.array("h")
        samples.frombytes(fh.readframes(fh.getnframes()))
    peak = max(max(samples), -min(samples)) / 32768
    return math.sqrt(sum(x * x for x in samples) / len(samples)) / 32768, peak


def page_meta(page: Path) -> dict:
    return json.loads(re.search(r'<script type="application/json" id="meta">(.*?)</script>',
                                page.read_text(encoding="utf-8"), re.S).group(1))


def page_mapping(page: Path) -> dict:
    return json.loads(base64.b64decode(page_meta(page)["mapping"]))


@pytest.fixture()
def results(tmp_path):
    """Three engines on four sentences (neutral, emotion, two with sounds); one clip failed."""
    out = tmp_path / "out"
    chosen = [s for s in SENTENCES if s.id in ("n01", "e01", "h01", "h03")]
    keys = ["chatterbox", "breeze", "qwen3-tts"]
    make_results(out, keys, chosen, failed={("breeze", "e01")}, amplitudes={"chatterbox": 0.05, "breeze": 0.6})
    return out, chosen, keys


def test_page_hides_the_engines_and_asks_for_three_ratings(results):
    out, chosen, keys = results
    page = review.build_review(out, chosen, seed=7)
    html = page.read_text(encoding="utf-8")
    assert page == out / "review.html"
    assert "http://" not in html and "https://" not in html                    # nothing is loaded from anywhere
    for key in keys:
        assert key not in html.lower() and get_engine(key).title.split()[0].lower() not in html.lower()

    def radios(field):
        return len(re.findall(rf'<input type="radio"[^>]*data-field="{field}"', html))

    assert radios("naturalness") == radios("emotion") == 11 * 5                # 12 clips minus the failed one
    assert radios("human_sounds") == 3 * 2 * 5                                 # only the two sentences with sounds
    assert html.count('<div class="row na">') == 5                             # n01 (3 clips) and e01 (2 clips)
    assert html.count("No audio: this engine failed on this sentence.") == 1
    assert 'id="reveal"' in html and 'id="meta"' in html
    assert [m for m in re.findall(r'id="s-(\w+)"', html)] == ["n01", "e01", "h01", "h03"]


def test_clips_on_the_page_have_names_that_do_not_give_the_engine_away(results):
    out, chosen, keys = results
    page = review.build_review(out, chosen, seed=7)
    mapping = page_mapping(page)
    clips = sorted((out / "review" / "clips").iterdir())
    assert len(clips) == 11 and all(re.fullmatch(r"[0-9a-f]{12}\.wav", c.name) for c in clips)
    assert len(mapping) == 12 and {m["variant"] for m in mapping.values()} == set(keys)
    assert {c.stem for c in clips} == {t for t, m in mapping.items() if m["wav"]}
    for clip in clips:
        assert f'src="review/clips/{clip.name}"' in page.read_text(encoding="utf-8")
        assert mapping[clip.stem]["wav"].startswith(mapping[clip.stem]["variant"] + "/")
    assert mapping[next(t for t, m in mapping.items() if (m["variant"], m["sentence"]) == ("breeze", "e01"))]["wav"] is None


def test_the_page_says_what_to_do_when_a_clip_cannot_be_loaded(results):
    out, chosen, _ = results
    html = review.build_review(out, chosen, seed=7).read_text(encoding="utf-8")
    notice = re.search(r'<div class="notice" id="audio-problem" role="alert" hidden>(.*?)</div>', html, re.S)
    assert notice                                                  # on the page from the start, shown when a clip fails
    advice = {name: re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", text))
              for name, text in re.findall(r'<span id="problem-(\w+)"[^>]*>(.*?)</span>', notice.group(1), re.S)}
    assert set(advice) == {"disk", "served"}                       # one text for a page opened from disk, one for a served page
    assert "tts_benchmark.py serve" in advice["disk"] and "tts_benchmark.py review" in advice["served"]
    for command in ("serve", "review"):                            # the commands it names exist
        with pytest.raises(SystemExit) as stop, contextlib.redirect_stdout(io.StringIO()):
            cli.build_parser().parse_args([command, "--help"])
        assert stop.value.code == 0
    assert 'location.protocol === "file:"' in html and 'document.addEventListener("error"' in html


def test_page_audio_is_loudness_matched_and_the_originals_are_untouched(results):
    out, chosen, _ = results
    originals = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*/*.wav")}
    mapping = page_mapping(review.build_review(out, chosen, seed=7))
    for token, entry in mapping.items():
        if entry["wav"]:
            rms, peak = levels(out / "review" / "clips" / f"{token}.wav")
            assert rms == pytest.approx(review.TARGET_RMS, rel=0.05) and peak <= review.PEAK_LIMIT
    assert originals == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*/*.wav")}

    mapping = page_mapping(review.build_review(out, chosen, seed=7, normalize=False))
    for token, entry in mapping.items():
        if entry["wav"]:
            assert (out / "review" / "clips" / f"{token}.wav").read_bytes() == (out / entry["wav"]).read_bytes()


def test_page_id_changes_only_when_the_audio_changes(results):
    out, chosen, _ = results
    first = page_meta(review.build_review(out, chosen, seed=7))["page_id"]
    assert page_meta(review.build_review(out, chosen, seed=7))["page_id"] == first
    write_wav(out / "chatterbox" / "n01_1.wav", seconds=0.5)                   # a different recording
    assert page_meta(review.build_review(out, chosen, seed=7))["page_id"] != first


def test_rebuilding_removes_old_clip_copies_but_not_other_files(results):
    out, chosen, _ = results
    clips = out / "review" / "clips"
    review.build_review(out, chosen, seed=1)
    stale, note = clips / "0123456789ab.wav", clips / "notes.txt"
    stale.write_bytes(b"old")
    note.write_text("keep me", encoding="utf-8")
    review.build_review(out, chosen, seed=2)
    assert not stale.exists() and note.read_text(encoding="utf-8") == "keep me"
    assert len(list(clips.glob("*.wav"))) == 11


def test_nothing_is_built_when_no_clip_succeeded(tmp_path):
    out = tmp_path / "out"
    few = SENTENCES[:2]
    make_results(out, ["breeze"], few, failed={("breeze", s.id) for s in few})
    assert review.build_review(out, few) is None
    assert not (out / "review.html").exists()


@pytest.mark.skipif(shutil.which("node") is None, reason="Node.js is not installed")
def test_page_script_is_valid_javascript(results, tmp_path):
    out, chosen, _ = results
    html = review.build_review(out, chosen).read_text(encoding="utf-8")
    (code,) = re.findall(r"<script>(.*?)</script>", html, re.S)
    script = tmp_path / "page.js"
    script.write_text(code, encoding="utf-8")
    done = subprocess.run([shutil.which("node"), "--check", str(script)], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr


# -- order and blinding --------------------------------------------------------

def _all_variants():
    return [make_variant(key, [make_row(s.id, 1, category=s.category, wav=f"{key}/{s.id}_1.wav") for s in SENTENCES])
            for key in VARIANTS]


def _orders(plan):
    return [tuple(c["variant"] for c in item["clips"]) for item in plan]


def test_clip_order_is_shuffled_for_each_sentence_and_reproducible():
    variants = _all_variants()
    plan = review.plan_review(variants, SENTENCES, seed=1)
    orders = _orders(plan)
    assert len(orders) == 15 and all(sorted(o) == sorted(VARIANTS) for o in orders)    # each engine once per sentence
    assert len(set(orders)) > 1 and len({o[0] for o in orders}) > 1                   # not the same order everywhere
    assert [c["letter"] for c in plan[0]["clips"]] == list("ABCDE")
    assert orders == _orders(review.plan_review(variants, SENTENCES, seed=1))
    assert orders != _orders(review.plan_review(variants, SENTENCES, seed=2))
    tokens = [c["token"] for item in plan for c in item["clips"]]
    assert len(tokens) == len(set(tokens)) == 75 and all(re.fullmatch(r"[0-9a-f]{12}", t) for t in tokens)


def test_a_failed_clip_keeps_its_slot_as_missing():
    good = make_variant("chatterbox", [make_row("n01", wav="chatterbox/n01_1.wav")])
    bad = make_variant("breeze", [make_row("n01", status="error", error_type="RuntimeError")])
    (item,) = review.plan_review([good, bad], [s for s in SENTENCES if s.id == "n01"], seed=3)
    by_variant = {c["variant"]: c for c in item["clips"]}
    assert (by_variant["breeze"]["wav"], by_variant["breeze"]["run"]) == (None, None)
    assert (by_variant["chatterbox"]["wav"], by_variant["chatterbox"]["run"]) == ("chatterbox/n01_1.wav", 1)


def test_the_reviewed_clip_is_the_first_successful_counted_run():
    rows = [make_row("n01", 0, counted=False, wav="x/n01_0.wav"),
            make_row("n01", 1, status="error", error_type="RuntimeError"),
            make_row("n01", 2, wav="x/n01_2.wav"), make_row("n01", 3, wav="x/n01_3.wav"),
            make_row("n02", 1, wav="x/n02_1.wav")]
    assert review.pick_clip(rows, "n01")["wav"] == "x/n01_2.wav"
    assert review.pick_clip(rows, "zz") is None


# -- loudness matching ---------------------------------------------------------

def test_loudness_matching_brings_quiet_and_loud_clips_to_the_same_level(tmp_path):
    for name, amplitude in (("quiet", 0.05), ("loud", 0.6)):
        assert review.match_loudness(write_wav(tmp_path / f"{name}.wav", amplitude=amplitude),
                                     tmp_path / f"{name}_matched.wav")
        rms, peak = levels(tmp_path / f"{name}_matched.wav")
        assert rms == pytest.approx(review.TARGET_RMS, rel=0.05) and peak <= review.PEAK_LIMIT


def test_loudness_matching_never_amplifies_into_clipping(tmp_path):
    samples = [int(32767 * 0.02 * math.sin(2 * math.pi * 220 * i / 24000)) for i in range(24000)]
    samples[12000] = int(32767 * 0.5)                                          # one loud spike in a quiet clip
    src = tmp_path / "spiky.wav"
    with wave.open(str(src), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(24000)
        fh.writeframes(struct.pack(f"<{len(samples)}h", *samples))
    assert review.match_loudness(src, tmp_path / "out.wav")
    rms, peak = levels(tmp_path / "out.wav")
    assert peak == pytest.approx(review.PEAK_LIMIT, abs=0.01) and rms < review.TARGET_RMS


@pytest.mark.parametrize("channels, width, frames", [(2, 2, 2400), (1, 1, 2400), (1, 2, 0)])
def test_loudness_matching_leaves_unusual_files_alone(tmp_path, channels, width, frames):
    src = tmp_path / "odd.wav"
    with wave.open(str(src), "wb") as fh:
        fh.setnchannels(channels)
        fh.setsampwidth(width)
        fh.setframerate(24000)
        fh.writeframes(b"\x40" * (frames * channels * width))
    assert review.match_loudness(src, tmp_path / "out.wav") is False
    assert not (tmp_path / "out.wav").exists()


def test_loudness_matching_leaves_silence_and_non_wav_files_alone(tmp_path):
    silent = tmp_path / "silent.wav"
    with wave.open(str(silent), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(24000)
        fh.writeframes(b"\x00" * 4800)
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"this is not a wav file")
    for src in (silent, junk, tmp_path / "missing.wav"):
        assert review.match_loudness(src, tmp_path / "out.wav") is False
    assert not (tmp_path / "out.wav").exists()
