"""Downloads model weights into the local Hugging Face cache.

    python -m tts_bench.fetch org/name [org/name ...]

This is the only part of the benchmark that uses the network, and it only
talks to huggingface.co to download the repos named on the command line.
Nothing is uploaded. It is started by `tts_benchmark.py download`, which
shows the sizes and asks for confirmation first.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("repos", nargs="+", help="Hugging Face repo ids, e.g. Qwen/Qwen3-TTS-12Hz-1.7B-Base")
    args = parser.parse_args(argv)
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("huggingface_hub is not installed in this Python environment", file=sys.stderr)
        return 3
    failed = 0
    for repo in args.repos:
        try:
            path = snapshot_download(repo_id=repo)     # resumes an interrupted download
            print(f"downloaded {repo} -> {path}", flush=True)
        except Exception as exc:
            failed += 1
            print(f"FAILED {repo}: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
