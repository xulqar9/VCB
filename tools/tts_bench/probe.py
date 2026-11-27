"""Read-only look at one Python environment, run inside the engine's own interpreter:

    python -m tts_bench.probe --modules torch,qwen_tts --repos Qwen/Qwen3-TTS-12Hz-1.7B-Base

Prints one JSON object on the last line of stdout. It loads no model and never
touches the network. Only torch is imported (to read the GPU build and run a
tiny matrix multiplication on each GPU); the other modules are looked up with
find_spec, which does not execute them.

The checks stay stdlib-only so they run in any engine venv, whatever its Python.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
from pathlib import Path

_SM = re.compile(r"^sm_(\d+)([a-z]?)$")
_PTX = re.compile(r"^compute_(\d+)$")


def _split_capability(digits: str) -> tuple[int, int]:
    """'86' -> (8, 6), '120' -> (12, 0): the last digit is the minor version."""
    return int(digits[:-1]), int(digits[-1])


def arch_support(arch_list: list[str], major: int, minor: int) -> str:
    """Can a torch build with this architecture list run on a GPU of this compute
    capability? "native" = it ships kernels for the GPU (cubins run on any newer
    minor version of the same major version); "ptx" = only generic PTX that the
    driver compiles on first use (works, slow start); "no" = it will not run."""
    for arch in arch_list:
        m = _SM.match(arch)
        if m:
            a_major, a_minor = _split_capability(m.group(1))
            if a_major == major and a_minor <= minor:
                return "native"
    for arch in arch_list:
        m = _PTX.match(arch)
        if m and _split_capability(m.group(1)) <= (major, minor):
            return "ptx"
    return "no"


def _module_info(name: str) -> dict:
    try:
        found = importlib.util.find_spec(name) is not None
    except (ImportError, ValueError, AttributeError):
        found = False
    version = None
    if found:
        for dist in (name, name.replace("_", "-")):
            try:
                version = importlib.metadata.version(dist)
                break
            except importlib.metadata.PackageNotFoundError:
                continue
    return {"found": found, "version": version}


def _torch_info() -> dict | None:
    try:
        import torch
    except Exception as exc:                       # missing, or broken install
        return {"error": f"{type(exc).__name__}: {exc}"[:300]}
    info = {"version": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()), "arch_list": [], "devices": []}
    if not info["cuda_available"]:
        return info
    info["arch_list"] = list(torch.cuda.get_arch_list())
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        device = {"index": index, "name": props.name, "capability": [props.major, props.minor],
                  "total_mib": int(props.total_memory // 2**20),
                  "support": arch_support(info["arch_list"], props.major, props.minor)}
        try:                                       # the only real proof that kernels launch on this GPU
            a = torch.randn(128, 128, device=f"cuda:{index}")
            device["matmul_ok"] = bool(torch.isfinite((a @ a).sum()).item())
        except Exception as exc:
            device["matmul_ok"] = False
            device["matmul_error"] = f"{type(exc).__name__}: {exc}"[:300]
        info["devices"].append(device)
    return info


def hf_cache_dir() -> Path:
    for var in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        if os.environ.get(var):
            return Path(os.environ[var])
    home = os.environ.get("HF_HOME")
    if not home:
        base = os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")
        home = os.path.join(base, "huggingface")
    return Path(home) / "hub"


def _tree_size(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        try:
            if path.is_file() and not path.name.endswith(".incomplete"):
                total += path.stat().st_size     # follows the cache's symlinks
        except OSError:
            continue
    return total


def cache_status(repo: str, cache: Path) -> dict:
    """What the local Hugging Face cache holds for a repo (no network)."""
    folder = cache / ("models--" + repo.replace("/", "--"))
    if not folder.is_dir():
        return {"present": False, "size_bytes": 0, "partial_files": 0}
    blobs = folder / "blobs"
    blob_bytes = _tree_size(blobs) if blobs.is_dir() else 0
    snapshots = folder / "snapshots"
    snap_bytes = max((_tree_size(p) for p in snapshots.iterdir() if p.is_dir()), default=0) \
        if snapshots.is_dir() else 0
    partial = len(list(blobs.glob("*.incomplete"))) if blobs.is_dir() else 0
    return {"present": snap_bytes > 0 or blob_bytes > 0, "size_bytes": max(blob_bytes, snap_bytes),
            "partial_files": partial}


def probe(modules: list[str], repos: list[str]) -> dict:
    cache = hf_cache_dir()
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "platform": sys.platform,
        "torch": _torch_info(),
        "modules": {name: _module_info(name) for name in modules if name != "torch"},
        "hf_cache": {"dir": str(cache), "repos": {r: cache_status(r, cache) for r in repos}},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--modules", default="")
    parser.add_argument("--repos", default="")
    args = parser.parse_args(argv)
    result = probe([m for m in args.modules.split(",") if m], [r for r in args.repos.split(",") if r])
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
