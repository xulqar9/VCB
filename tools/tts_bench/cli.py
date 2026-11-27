"""Command line of the TTS benchmark.

    python tools\\tts_benchmark.py list                       engines, weights, licenses, packages
    python tools\\tts_benchmark.py check                      is each engine ready? (read-only)
    python tools\\tts_benchmark.py run --engines chatterbox   run the benchmark
    python tools\\tts_benchmark.py run --engines vcb          ... with the voices VCB offers
    python tools\\tts_benchmark.py report                     rebuild the CSV, JSON and tables
    python tools\\tts_benchmark.py review                     rebuild the blind listening page
    python tools\\tts_benchmark.py serve                      listen to it in any browser, VS Code's included
    python tools\\tts_benchmark.py download --engines breeze  fetch model weights (uses the network)

Only `download` uses the network. `run` keeps every engine offline: the engine's process cannot
open a connection to anything but this machine (see netguard.py). `serve` listens on 127.0.0.1 only.
Exit codes: 0 success, 1 nothing ran or an engine failed completely, 2 bad usage or configuration.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import webbrowser
from pathlib import Path

from .catalog import format_download_plan, format_engine_list, plan_downloads
from .check import format_check, run_check
from .config import DEFAULT_CONFIG, REPO_ROOT, BenchConfig, load_config
from .registry import MODES, EngineSpec, select_engines
from .report import format_summary, render_table, write_reports
from .review import REVIEW_HTML, build_review
from .runner import Plan, Target, run_plan, worker_env, build_plan
from .sentences import load_sentences
from .serve import DEFAULT_PORT, make_server, page_url

EXIT_OK, EXIT_FAILED, EXIT_USAGE, EXIT_INTERRUPTED = 0, 1, 2, 130

# VS Code's own browser answers every request for a local audio file with 403, so the page is served to it.
SERVE_COMMAND = "python tools\\tts_benchmark.py serve"
SERVE_REASON = "VS Code's browser blocks audio files opened from disk"


# -- shared helpers ------------------------------------------------------------

def _split(value: str | None) -> list[str] | None:
    parts = [p.strip() for p in value.split(",") if p.strip()] if value else []
    return parts or None


def _modes(value: str | None) -> list[str] | None:
    modes = _split(value)
    bad = [m for m in modes or [] if m not in MODES]
    if bad:
        raise ValueError(f"unknown voice mode(s): {', '.join(bad)} (choose from: {', '.join(MODES)})")
    return modes


def _specs(args: argparse.Namespace, default: str | None) -> list[EngineSpec]:
    return select_engines(args.engines or default)


def _config(args: argparse.Namespace) -> BenchConfig:
    """The config file, with the command-line options laid over it."""
    cfg = load_config(args.config)
    if getattr(args, "runs", None) is not None:
        cfg.runs_per_sentence = args.runs
    if getattr(args, "warmup", None) is not None:
        cfg.warmup_runs = args.warmup
    if getattr(args, "timeout", None) is not None:
        cfg.timeout_s = args.timeout
    if getattr(args, "gpu", None):
        cfg.default_gpu = args.gpu
        for settings in cfg.engines.values():     # --gpu means every engine
            settings.gpu = None
    if getattr(args, "output", None):
        cfg.output_dir = Path(args.output).expanduser().resolve()
    cfg.validate()
    return cfg


def _tidy_console() -> None:
    """Engine errors may contain characters the Windows console cannot print."""
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


# -- list / check --------------------------------------------------------------

def cmd_list(args: argparse.Namespace) -> int:
    print(format_engine_list(_config(args)))
    return EXIT_OK


def cmd_check(args: argparse.Namespace) -> int:
    cfg = _config(args)
    report = run_check(cfg, _specs(args, "all"), _modes(args.modes))
    print(format_check(report, cfg))
    return EXIT_OK if report.ready else EXIT_FAILED


# -- run -----------------------------------------------------------------------

def format_plan(plan: Plan, cfg: BenchConfig, n_sentences: int) -> str:
    """What is about to run, and what cannot and why."""
    per_variant = cfg.warmup_runs + cfg.runs_per_sentence * n_sentences
    out = [f"Benchmark plan: {n_sentences} sentences x {cfg.runs_per_sentence} runs, "
           f"+{cfg.warmup_runs} warm-up per engine (not counted), {cfg.timeout_s:g} s timeout per synthesis"]
    rows = [[v.variant, "reference clip" if v.mode == "clone" else "built-in voice", v.target.label,
             f"cannot run: {v.skip}" if v.skip else f"{per_variant} syntheses"] for v in plan.variants]
    if rows:
        out.append(render_table(["variant", "voice", "gpu", "status"], rows, left=4))
    out.extend(f"  note: {n}" for n in plan.notes)
    return "\n".join(out)


def _variant_failed(raw_file: Path) -> bool:
    """True if the engine produced nothing usable: it never loaded, gave up
    after too many restarts, was cut short, or every counted run failed."""
    data = json.loads(raw_file.read_text(encoding="utf-8"))
    counted = [r for r in data["rows"] if r["counted"]]
    return bool(data.get("aborted")) or not data.get("complete") \
        or not any(r["status"] == "ok" for r in counted)


def _print_files(cfg: BenchConfig, csv_path: Path, json_path: Path) -> None:
    print("\nFiles:")
    print(f"  raw per-run data   {csv_path}")
    print(f"  summary            {json_path}")
    print(f"  audio clips        {cfg.output_dir / '<engine>' / '<sentence_id>_<run>.wav'}")


def cmd_run(args: argparse.Namespace) -> int:
    cfg = _config(args)
    sentences = load_sentences(cfg.sentences)
    plan = build_plan(cfg, select_engines(args.engines), _modes(args.modes))
    print(format_plan(plan, cfg, len(sentences)))
    if args.dry_run:
        print("\nDry run: no engine was started.")
        return EXIT_OK
    if not plan.runnable():
        print("\nNothing can run. `check` explains what each engine is missing.")
        return EXIT_FAILED

    try:
        raw_files = run_plan(plan, cfg, sentences, allow_download=args.allow_download)
    except KeyboardInterrupt:
        print(f"\nInterrupted. Partial results are in {cfg.output_dir}; "
              "`tts_benchmark.py report` summarises them.", file=sys.stderr)
        return EXIT_INTERRUPTED

    summary, csv_path, json_path = write_reports(cfg.output_dir)
    print("\n" + format_summary(summary))
    _print_files(cfg, csv_path, json_path)
    if not args.no_review:
        page = build_review(cfg.output_dir, sentences, seed=cfg.seed)
        if page:
            print(f"  blind listening    {page}")
            print(f"  listen             {SERVE_COMMAND}   ({SERVE_REASON})")
        else:
            print("  blind listening    not created: there are no successful clips to review")

    failed = [p for p in raw_files if _variant_failed(p)]
    if failed:
        print("\nIncomplete: " + ", ".join(p.parent.name for p in failed) +
              ". Engine logs are in each variant's worker.log; `check` shows missing packages or weights.")
    return EXIT_FAILED if failed else EXIT_OK


# -- report / review -----------------------------------------------------------

def cmd_report(args: argparse.Namespace) -> int:
    cfg = _config(args)
    if not cfg.output_dir.is_dir():
        print(f"No results folder at {cfg.output_dir}. Run the benchmark first.")
        return EXIT_FAILED
    summary, csv_path, json_path = write_reports(cfg.output_dir)
    print(format_summary(summary))
    if not summary["variants"]:
        return EXIT_FAILED
    _print_files(cfg, csv_path, json_path)
    return EXIT_OK


def cmd_review(args: argparse.Namespace) -> int:
    cfg = _config(args)
    page = build_review(cfg.output_dir, load_sentences(cfg.sentences),
                        seed=cfg.seed if args.seed is None else args.seed, normalize=not args.no_normalize)
    if page is None:
        print(f"No successful clips under {cfg.output_dir}: nothing to review.")
        return EXIT_FAILED
    print(f"Blind listening page: {page}")
    print(f"Open it in a browser, or run `{SERVE_COMMAND}` ({SERVE_REASON}).")
    return EXIT_OK


def cmd_serve(args: argparse.Namespace) -> int:
    cfg = _config(args)
    page = cfg.output_dir / REVIEW_HTML
    if not page.is_file():
        print(f"No listening page at {page}. Run the benchmark, or `tts_benchmark.py review` "
              "to build it from saved results.")
        return EXIT_FAILED
    try:
        server = make_server(cfg.output_dir, args.port)
    except OSError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_FAILED
    url = page_url(server)
    # flush: a terminal that is not interactive (a task runner, a pipe) would otherwise show nothing until the end
    print(f"Blind listening page: {url}", flush=True)
    print("  Open it in any browser on this machine, VS Code's included. It is served to this computer only.",
          flush=True)
    if args.port and server.server_port != args.port:
        print(f"  Port {args.port} is busy, so this is port {server.server_port}: ratings made on "
              "another port are not shown here.", flush=True)
    else:
        print("  The browser keeps your ratings per address: use the same port next time to see them again.",
              flush=True)
    print("  Press Ctrl+C to stop.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    print("Stopped.", flush=True)
    return EXIT_OK


# -- download ------------------------------------------------------------------

def _confirm(question: str) -> bool:
    if not sys.stdin.isatty():
        return False
    return input(f"{question} [y/N] ").strip().lower() in {"y", "yes"}


def _fetch(python: Path, repos: list[str]) -> int:
    """Download in the engine's own interpreter, so its Hugging Face cache (a
    WSL one, say) is the one that receives the files."""
    env = worker_env(os.environ, Target("all", None), allow_download=True)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    return subprocess.run([str(python), "-m", "tts_bench.fetch", *repos], cwd=REPO_ROOT, env=env).returncode


def cmd_download(args: argparse.Namespace) -> int:
    cfg = _config(args)
    if not args.engines:                          # never guess what to download
        raise ValueError("choose what to download: --engines breeze,qwen3-tts (or --engines all)")
    plan = plan_downloads(cfg, _specs(args, None), _modes(args.modes))
    print(format_download_plan(plan))
    if plan.problems:
        return EXIT_FAILED
    if not plan.items:
        return EXIT_OK
    print("\nThis downloads model weights only. Python packages and the Breeze / Step-Audio-EditX source "
          "folders are installed separately (see `list`).")
    if not args.yes and not _confirm(f"\nDownload {len(plan.items)} repo(s), {plan.total_gb:g} GB?"):
        print("Cancelled. Nothing was downloaded." if sys.stdin.isatty()
              else "Not confirmed: run in a terminal, or add --yes. Nothing was downloaded.")
        return EXIT_FAILED
    status = EXIT_OK
    for python in dict.fromkeys(i.python for i in plan.items):
        repos = [i.repo for i in plan.items if i.python == python]
        if _fetch(python, repos) != 0:
            status = EXIT_FAILED
    return status


# -- argument parser -----------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tts_benchmark.py", description="Compare text-to-speech engines under identical conditions.",
        epilog="Only `download` uses the network. Nothing is ever uploaded.")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    def command(name: str, func, help_text: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help_text, description=help_text)
        p.add_argument("--config", metavar="FILE", default=None,
                       help=f"settings file (default: {DEFAULT_CONFIG.relative_to(REPO_ROOT)})")
        p.set_defaults(func=func)
        return p

    def engines(p: argparse.ArgumentParser, default: str) -> None:
        p.add_argument("--engines", metavar="A,B",
                       help=f"engine names, 'vcb' (the voices VCB offers) or 'all' (default: {default})")
        p.add_argument("--modes", metavar="MODES", help="voice modes to use: default, clone (default: both)")

    command("list", cmd_list, "show the engines, their weights, licenses and packages")

    p = command("check", cmd_check, "check that each engine is ready (read-only: loads and downloads nothing)")
    engines(p, "all")

    p = command("run", cmd_run, "run the benchmark")
    engines(p, "chatterbox")
    p.add_argument("--runs", type=int, metavar="N", help="runs per sentence (default: from the config, 3)")
    p.add_argument("--warmup", type=int, metavar="N", help="warm-up runs per engine, not counted")
    p.add_argument("--timeout", type=float, metavar="SECONDS", help="timeout per synthesis (default 30)")
    p.add_argument("--gpu", metavar="GPU", help="GPU for every engine: an index or part of a name, e.g. '3090 Ti'")
    p.add_argument("--output", metavar="DIR", help="results folder (default: output)")
    p.add_argument("--dry-run", action="store_true", help="show the plan without starting any engine")
    p.add_argument("--no-review", action="store_true", help="skip the blind listening page")
    p.add_argument("--allow-download", action="store_true",
                   help="let engines reach the network and fetch missing weights themselves (otherwise they "
                        "run offline and every outgoing connection is blocked; prefer the `download` command)")

    p = command("report", cmd_report, "rebuild the CSV, the JSON summary and the tables from saved results")
    p.add_argument("--output", metavar="DIR", help="results folder (default: output)")

    p = command("review", cmd_review, "rebuild the blind listening page from saved results")
    p.add_argument("--output", metavar="DIR", help="results folder (default: output)")
    p.add_argument("--seed", type=int, help="seed of the randomised clip order")
    p.add_argument("--no-normalize", action="store_true", help="do not match the loudness of the clips")

    p = command("serve", cmd_serve, "serve the blind listening page to this machine only, for browsers that "
                                    "cannot play local audio files (VS Code's)")
    p.add_argument("--output", metavar="DIR", help="results folder (default: output)")
    p.add_argument("--port", type=int, default=DEFAULT_PORT, metavar="PORT",
                   help=f"port (default {DEFAULT_PORT}; the next free one if it is busy; 0 for any)")
    p.add_argument("--open", action="store_true", help="also open the page in the default browser")

    p = command("download", cmd_download, "download model weights (the only command that uses the network)")
    engines(p, "(required)")
    p.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    return parser


def main(argv: list[str] | None = None) -> int:
    _tidy_console()
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (ValueError, FileNotFoundError) as exc:      # bad config, unknown engine or GPU
        print(f"error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return EXIT_INTERRUPTED
