"""What the benchmark knows about the engines, printed for people: the `list`
command and the plan shown before `download`.

Both only read the registry (plain data) and the local Hugging Face cache, so
neither imports an engine or touches the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .check import run_probe
from .config import BenchConfig
from .registry import ENGINES, RETIRED_INSTALL_HINT, VCB_INSTALL_HINT, EngineSpec
from .report import render_table


def _licence_flag(spec: EngineSpec) -> str:
    if spec.weights_commercial is False:
        return "NON-COMMERCIAL"
    if spec.weights_commercial is None:
        return "NOT STATED"
    return "commercial use OK"


def _size_cell(spec: EngineSpec) -> str:
    """Hugging Face download size; XTTS and IndexTTS keep their files elsewhere, where only VCB put them."""
    if spec.weights:
        return f"{spec.download_gb():g} GB"
    if spec.vcb_native:
        return "retired from VCB" if spec.retired else "by VCB"
    return "none"


def format_engine_list(cfg: BenchConfig, specs: list[EngineSpec] | None = None) -> str:
    specs = specs or [s for s in ENGINES.values() if not s.hidden]
    out = ["TTS engines (choose with --engines a,b, --engines vcb for the voices VCB offers, or --engines all; "
           "the default is chatterbox)", ""]
    rows = []
    for s in specs:
        tags = ", ".join(t for t, performed in s.tag_coverage.items() if performed) or "none"
        rows.append([s.key, ", ".join(s.modes), tags, _size_cell(s), _licence_flag(s)])
    out.append(render_table(["engine", "voices", "sound tags performed", "weights", "weights license"],
                            rows, left=5))
    out += ["", "Voices: default = the engine's built-in voice, clone = the reference clip from [voice].",
            "Sound tags are as documented by each engine; the review page is where you hear whether they work.",
            "Sizes are the Hugging Face download sizes for the listed voices.",
            "VCB's own voices are installed by VCB, never by this tool's `download` command. Its retired "
            "voices (no longer offered or installed by VCB) run only where an earlier VCB installed them.", ""]
    for s in specs:
        out.append(f"{s.key}: {s.title}")
        for w in s.weights:
            out.append(f"  weights   {w.repo}  {w.size_gb:g} GB  ({', '.join(w.modes)})")
        if not s.weights:
            out.append("  weights   none")
        out.append(f"  license   code: {s.license_code}")
        out.append(f"            weights: {s.license_weights}")
        if s.license_url:
            out.append(f"            {s.license_url}")
        out.append(f"  platform  {s.platform}")
        if s.packages:
            out.append(f"  packages  {', '.join(s.packages)}")
        for note in s.setup_notes:
            out.append(f"  setup     {note}")
        own = cfg.settings_for(s.key)
        out.append(f"  config    gpu = {cfg.gpu_for(s.key)!r}, python = {cfg.python_for(s.key)}"
                   + (f", options = {own.options}" if own.options else ""))
        out.append("")
    return "\n".join(out).rstrip()


# -- download plan -------------------------------------------------------------

@dataclass(frozen=True)
class DownloadItem:
    repo: str
    size_gb: float
    engine: str
    python: Path                 # the interpreter whose cache receives it
    cache_dir: str
    commercial: bool | None
    license: str


@dataclass
class DownloadPlan:
    items: list[DownloadItem]
    present: list[str]           # already complete in the cache
    problems: list[str]          # environments that could not be inspected
    native: list[str] = field(default_factory=list)   # VCB's own engines: VCB installs them, not this tool

    @property
    def total_gb(self) -> float:
        return round(sum(i.size_gb for i in self.items), 2)


def plan_downloads(cfg: BenchConfig, specs: list[EngineSpec], modes: list[str] | None = None,
                   probe: Callable[..., dict] = run_probe) -> DownloadPlan:
    """Which weights the selected engines are still missing. The cache that
    counts is the one of the engine's own interpreter, so engines that run in
    WSL are checked there. VCB's own engines are left out: VCB installs them
    (its loaders fetch single files, which a whole-repo download would not match)."""
    plan = DownloadPlan([], [], [])
    wanted: dict[Path, list[tuple[EngineSpec, str, float]]] = {}
    for spec in specs:
        if spec.vcb_native:
            plan.native.append(spec.key)
            continue
        chosen = tuple(m for m in spec.modes if modes is None or m in modes)
        wanted.setdefault(cfg.python_for(spec.key), []).extend(
            (spec, w.repo, w.size_gb) for w in spec.weight_repos(chosen))
    for python, entries in wanted.items():
        if not entries:
            continue
        if not python.is_file():
            plan.problems.append(f"interpreter not found: {python}")
            continue
        answer = probe(python, [], sorted({repo for _, repo, _ in entries}))
        if "error" in answer:
            plan.problems.append(f"{python}: {answer['error']}")
            continue
        cache = answer["hf_cache"]
        for spec, repo, size_gb in entries:
            have = cache["repos"].get(repo, {})
            complete = (have.get("present") and not have.get("partial_files")
                        and have.get("size_bytes", 0) >= 0.9 * size_gb * 1e9)
            if complete:
                plan.present.append(repo)
            else:
                plan.items.append(DownloadItem(repo, size_gb, spec.key, python, cache["dir"],
                                               spec.weights_commercial, spec.license_weights))
    return plan


def format_download_plan(plan: DownloadPlan) -> str:
    out = []
    if plan.problems:
        out += ["Could not inspect:"] + [f"  {p}" for p in plan.problems] + [""]
    if plan.native:
        offered = [k for k in plan.native if not ENGINES[k].retired]
        retired = [k for k in plan.native if ENGINES[k].retired]
        if offered:
            out += [f"Left to VCB, which installs its own voices: {', '.join(offered)}",
                    f"  to install: {VCB_INSTALL_HINT}", ""]
        if retired:
            out += [f"Retired from VCB, which no longer installs them: {', '.join(retired)}",
                    f"  {RETIRED_INSTALL_HINT}", ""]
    if plan.present:
        out += [f"Already downloaded: {', '.join(plan.present)}", ""]
    if not plan.items:
        out.append("Nothing to download.")
        return "\n".join(out).rstrip()
    out.append("Weights to download from huggingface.co (nothing is uploaded):")
    rows = [[i.repo, f"{i.size_gb:g} GB", i.engine,
             "NON-COMMERCIAL" if i.commercial is False else
             "NOT STATED" if i.commercial is None else "ok to use commercially"] for i in plan.items]
    out.append(render_table(["repo", "size", "engine", "weights license"], rows, left=4))
    out.append(f"  total {plan.total_gb:g} GB")
    for cache_dir in sorted({i.cache_dir for i in plan.items}):
        out.append(f"  cache folder: {cache_dir}")
    for i in plan.items:
        if i.commercial is not True:
            out.append(f"  license of {i.repo}: {i.license}")
    return "\n".join(out)
