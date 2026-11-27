"""Reports: reads every output/<variant>/raw.json and produces

  * output/benchmark_raw.csv       one row per synthesis (warm-ups included, flagged)
  * output/benchmark_summary.json  statistics per variant
  * the terminal summary tables

Because the inputs are the per-variant files, results from separate runs (for
instance engines that run inside WSL) are merged into one report.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime
from pathlib import Path

from .registry import BENCH_ENGINES, ENGINES
from .runner import SCHEMA_VERSION
from .sentences import CATEGORIES
from .stats import describe

CSV_NAME = "benchmark_raw.csv"
SUMMARY_NAME = "benchmark_summary.json"

_CSV_FIELDS = ["engine", "variant", "mode", "gpu", "sentence_id", "category", "intent", "run",
               "counted", "status", "error_type", "error", "ttfa_s", "total_s", "audio_s", "rtf",
               "n_chunks", "sample_rate", "text_sent", "tags_in", "tags_dropped", "wav"]
_CSV_TAIL = ["torch_peak_alloc_mib", "torch_peak_reserved_mib", "started_at"]
_METRICS = ("ttfa_s", "total_s", "rtf", "audio_s")


# -- reading the raw results ---------------------------------------------------

def load_variants(output_dir: Path) -> tuple[list[dict], list[str]]:
    """All variants found under output_dir, in report order, plus problems met."""
    variants: list[dict] = []
    problems: list[str] = []
    for path in sorted(Path(output_dir).glob("*/raw.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            problems.append(f"{path.parent.name}: unreadable raw.json ({exc})")
            continue
        if data.get("schema") != SCHEMA_VERSION:
            problems.append(f"{path.parent.name}: raw.json has schema {data.get('schema')}, "
                            f"expected {SCHEMA_VERSION}")
            continue
        variants.append(data)
    order = {key: i for i, key in enumerate(BENCH_ENGINES)}
    variants.sort(key=lambda v: (order.get(v["engine"], len(order)), v["mode"] != "default"))
    return variants, problems


def _gpu_label(gpu: dict) -> str:
    if "index" not in gpu:
        return "CPU"
    return f"GPU {gpu['index']} {_short_gpu_name(gpu['name'])}"


def _short_gpu_name(name: str) -> str:
    return name.replace("NVIDIA ", "").replace("GeForce ", "").strip()


def _int_keys(mapping: dict | None) -> dict[int, float]:
    return {int(k): v for k, v in (mapping or {}).items()}


# -- statistics ------------------------------------------------------------------

def summarise_variant(data: dict) -> dict:
    """Statistics of one variant. Latency figures use only the counted runs that
    succeeded; failed runs are reported separately as failures."""
    rows = data["rows"]
    counted = [r for r in rows if r["counted"]]
    ok = [r for r in counted if r["status"] == "ok"]
    failed = [r for r in counted if r["status"] != "ok"]

    by_category = {}
    for category in CATEGORIES:
        subset = [r for r in ok if r["category"] == category]
        if subset:
            by_category[category] = {m: describe(r[m] for r in subset)
                                     for m in ("ttfa_s", "total_s", "rtf")}

    # Tags the engine had to drop, counted once per sentence (not once per run).
    per_sentence: dict[str, tuple[int, int]] = {}
    for r in counted:
        if r["tags_dropped"] is not None and r["sentence_id"] not in per_sentence:
            per_sentence[r["sentence_id"]] = (r["tags_in"], r["tags_dropped"])

    # VRAM: the monitor's whole-session figures plus the peak of the synthesis windows alone.
    vram = dict(data.get("vram") or {})
    synth_peak: dict[int, float] = {}
    for r in rows:
        for gpu, mib in _int_keys(r.get("vram_peak_mib")).items():
            synth_peak[gpu] = max(synth_peak.get(gpu, 0), mib)
    if synth_peak:
        vram["synth_peak_mib"] = {str(g): v for g, v in synth_peak.items()}

    audio_total = sum(r["audio_s"] for r in ok)
    first_load = next((l for l in data.get("loads", []) if l.get("ok")), None)
    return {
        "engine": data["engine"], "title": data.get("title", data["engine"]),
        "mode": data["mode"], "gpu": _gpu_label(data["gpu"]),
        "complete": data.get("complete", False), "restarts": data.get("restarts", 0),
        "aborted": data.get("aborted"),
        "runs": {"expected": len(counted), "ok": len(ok), "failed": len(failed)},
        "failures": dict(Counter(r["error_type"] or r["status"] for r in failed)),
        "ttfa_s": describe(r["ttfa_s"] for r in ok),
        "total_s": describe(r["total_s"] for r in ok),
        "rtf": describe(r["rtf"] for r in ok),
        "rtf_overall": (sum(r["total_s"] for r in ok) / audio_total) if audio_total else None,
        "audio_s": describe(r["audio_s"] for r in ok),
        "by_category": by_category,
        "load_s": first_load["load_s"] if first_load else None,
        "vram": vram,
        "torch_peak_mib": {
            "allocated": max((r["torch_peak_alloc_mib"] for r in rows if r.get("torch_peak_alloc_mib")), default=None),
            "reserved": max((r["torch_peak_reserved_mib"] for r in rows if r.get("torch_peak_reserved_mib")), default=None),
        },
        "tags": {"in": sum(t for t, _ in per_sentence.values()),
                 "dropped": sum(d for _, d in per_sentence.values())},
        "network_blocked": list(data.get("network_blocked") or []),
        "versions": data.get("versions", {}),
        "voice": data.get("voice"),
    }


def _comparability_warnings(variants: list[dict]) -> list[str]:
    """Results are only comparable if every engine got the same test."""
    warnings: list[str] = []
    for v in variants:
        if not v.get("complete"):
            warnings.append(f"{v['variant']}: the run is incomplete (interrupted); its numbers cover fewer runs")
    if not variants:
        return warnings
    base = variants[0]
    for v in variants[1:]:
        diff = [k for k, val in base["conditions"].items() if v["conditions"].get(k) != val]
        if diff:
            detail = ", ".join(f"{k} {v['conditions'].get(k)} vs {base['conditions'][k]}" for k in diff)
            warnings.append(f"{v['variant']} ran with different settings than {base['variant']} ({detail})")
    clones = [v for v in variants if v["mode"] == "clone" and v.get("voice")]
    if len({(c["voice"]["clip_sha"], c["voice"]["text"]) for c in clones}) > 1:
        warnings.append("clone variants used different reference clips or transcripts")
    gpus = {_gpu_label(v["gpu"]) for v in variants}
    if len(gpus) > 1:
        warnings.append("engines ran on different GPUs (" + "; ".join(sorted(gpus)) +
                        "): latency is not directly comparable")
    if not any(v.get("vram") for v in variants):
        warnings.append("no VRAM data was recorded (nvidia-smi was not available)")
    return warnings


def build_summary(variants: list[dict], problems: list[str] | None = None) -> dict:
    # Every GPU the memory monitor watched (the other card shows what else is running),
    # plus the GPUs the engines were pinned to.
    gpus: dict[int, dict] = {}
    for v in variants:
        for g in v.get("gpus_seen", []):
            gpus[g["index"]] = g
        if "index" in v["gpu"]:
            gpus[v["gpu"]["index"]] = {"index": v["gpu"]["index"], "name": v["gpu"]["name"],
                                       "total_mib": v["gpu"]["total_mib"]}
    return {
        "schema": SCHEMA_VERSION,
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "conditions": variants[0]["conditions"] if variants else {},
        "gpus": [gpus[i] for i in sorted(gpus)],
        "warnings": (problems or []) + _comparability_warnings(variants),
        "licenses": {key: {"code": s.license_code, "weights": s.license_weights,
                           "weights_commercial": s.weights_commercial}
                     for key, s in ENGINES.items() if not s.hidden
                     and any(v["engine"] == key for v in variants)},
        "variants": {v["variant"]: summarise_variant(v) for v in variants},
    }


# -- files ---------------------------------------------------------------------

def write_csv(variants: list[dict], path: Path) -> None:
    gpu_ids = sorted({int(g) for v in variants for r in v["rows"] for g in r.get("vram_peak_mib", {})})
    vram_cols = [f"vram_peak_gpu{g}_mib" for g in gpu_ids]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS + vram_cols + _CSV_TAIL)
        writer.writeheader()
        for v in variants:
            for r in v["rows"]:
                line = {k: r.get(k) for k in _CSV_FIELDS + _CSV_TAIL}
                line["gpu"] = _gpu_label(v["gpu"])
                for g, col in zip(gpu_ids, vram_cols):
                    line[col] = r.get("vram_peak_mib", {}).get(str(g))
                for key in ("ttfa_s", "total_s", "audio_s", "rtf"):    # keep the file compact
                    if line[key] is not None:
                        line[key] = round(line[key], 4)
                writer.writerow({k: ("" if x is None else x) for k, x in line.items()})


def write_reports(output_dir: Path) -> tuple[dict, Path, Path]:
    """Write the CSV and the JSON summary next to the per-variant folders."""
    variants, problems = load_variants(output_dir)
    summary = build_summary(variants, problems)
    csv_path = Path(output_dir) / CSV_NAME
    json_path = Path(output_dir) / SUMMARY_NAME
    write_csv(variants, csv_path)
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary, csv_path, json_path


# -- terminal tables -----------------------------------------------------------

def _num(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def render_table(headers: list[str], rows: list[list[str]], left: int = 1) -> str:
    """Plain-text table; the first `left` columns are left-aligned, the rest right-aligned."""
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(headers)]

    def line(cells: list[str]) -> str:
        parts = [c.ljust(w) if i < left else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths))]
        return "  " + "  ".join(parts)

    rule = "  " + "  ".join("-" * w for w in widths)
    return "\n".join([line(headers), rule, *(line(r) for r in rows)])


def _vram_cell(vram: dict, gpu: int) -> str:
    peak = _int_keys(vram.get("peak_mib")).get(gpu)
    delta = _int_keys(vram.get("delta_mib")).get(gpu)
    if peak is None:
        return "-"
    cell = f"{peak / 1024:.1f} GB"
    return cell if delta is None else f"{cell} ({delta / 1024:+.1f})"


def _unloaded_cell(vram: dict) -> str:
    if vram.get("released") is None:
        return "-"
    return "yes" if vram["released"] else f"NO ({vram.get('residual_mib', 0):.0f} MiB left)"


def format_summary(summary: dict) -> str:
    variants = summary["variants"]
    if not variants:
        return "No results found. Run the benchmark first."
    out: list[str] = []
    cond = summary["conditions"]
    if cond:
        out.append(f"TTS benchmark: {cond['n_sentences']} sentences x {cond['runs_per_sentence']} runs "
                   f"(+{cond['warmup_runs']} warm-up, not counted), {cond['timeout_s']:g} s timeout per synthesis")
    else:
        out.append("TTS benchmark")
    out.append("")
    out.append("Latency in seconds (successful runs only). TTFA = time to first audio chunk;"
               " engines that do not stream deliver all audio at once, so TTFA = total.")
    out.append(render_table(
        ["variant", "ok/runs", "gpu", "ttfa mean", "median", "p95", "total mean", "median", "p95"],
        [[name, f"{s['runs']['ok']}/{s['runs']['expected']}", s["gpu"],
          _num(s["ttfa_s"]["mean"]), _num(s["ttfa_s"]["median"]), _num(s["ttfa_s"]["p95"]),
          _num(s["total_s"]["mean"]), _num(s["total_s"]["median"]), _num(s["total_s"]["p95"])]
         for name, s in variants.items()], left=3))

    gpu_ids = [g["index"] for g in summary["gpus"]]
    out.append("")
    out.append("Real-time factor = synthesis time / audio duration (below 1 is faster than real time)."
               " Peak VRAM is the whole GPU's use as seen by nvidia-smi, with the change over idle in brackets.")
    headers = ["variant", "rtf mean", "median", "p95", "load s"]
    headers += [f"peak GPU {g}" for g in gpu_ids] + ["unloaded"]
    out.append(render_table(headers, [
        [name, _num(s["rtf"]["mean"]), _num(s["rtf"]["median"]), _num(s["rtf"]["p95"]),
         _num(s["load_s"], 1), *(_vram_cell(s["vram"], g) for g in gpu_ids), _unloaded_cell(s["vram"])]
        for name, s in variants.items()]))

    failures = [(name, s) for name, s in variants.items() if s["runs"]["failed"] or s["restarts"]]
    out.append("")
    if failures:
        out.append("Failures:")
        for name, s in failures:
            kinds = ", ".join(f"{n} x {kind}" for kind, n in sorted(s["failures"].items(), key=lambda kv: -kv[1]))
            extra = f"; engine restarted {s['restarts']}x" if s["restarts"] else ""
            out.append(f"  {name}: {s['runs']['failed']} of {s['runs']['expected']} failed ({kinds or 'none'}){extra}")
    else:
        out.append("Failures: none")

    notes = []
    for name, s in variants.items():
        if s["tags"]["in"] and s["tags"]["dropped"]:
            spec = ENGINES.get(s["engine"])
            if spec and spec.vcb_native and not any(spec.tag_coverage.values()):
                # VCB's own text step speaks the vocal tags as words for an engine that cannot perform them.
                notes.append(f"{name}: performs no sound tags; as in VCB, <laugh> and <sigh> were spoken as "
                             "the words \"Haha!\" and \"Haaah.\" and the other tags were removed")
            else:
                notes.append(f"{name}: {s['tags']['dropped']} of {s['tags']['in']} sound tags have no equivalent "
                             "in this engine; they were removed and the sentences were spoken without them")
    if notes:
        out.append("")
        out.append("Sound tags:")
        out.extend(f"  {n}" for n in notes)

    blocked = {name: s["network_blocked"] for name, s in variants.items() if s.get("network_blocked")}
    if blocked:
        out.append("")
        out.append("Network: engines run offline; these connection attempts were refused (nothing was sent):")
        out.extend(f"  {name}: {'; '.join(attempts)}" for name, attempts in blocked.items())

    restricted = {k: v for k, v in summary["licenses"].items() if v["weights_commercial"] is not True}
    if restricted:
        out.append("")
        out.append("Licenses (check before any commercial use):")
        for key, lic in restricted.items():
            label = "non-commercial" if lic["weights_commercial"] is False else "not stated"
            out.append(f"  {key}: weights {label}: {lic['weights']}")
    if summary["warnings"]:
        out.append("")
        out.append("Warnings:")
        out.extend(f"  {w}" for w in summary["warnings"])
    return "\n".join(out)
