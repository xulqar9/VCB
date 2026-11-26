"""Installs, checks and removes what online search needs -- only when the user
turns it on. Stdlib only (it runs before requests/PySocks exist).

  python -m vcb.search.install --ensure     # install anything missing (launcher)
  python -m vcb.search.install --setup      # install, then run the self-test (settings)
  python -m vcb.search.install --remove     # uninstall what setup added
  add --json-progress for one JSON object per line (the settings window).
  Send "cancel" on stdin to stop; whatever this run added is rolled back.

What it installs:
  * Python packages from requirements-search.txt (requests + PySocks + psutil),
    within requirements-lock.txt's pins.
  * For the managed Tor route: the official Tor Expert Bundle, checked against
    a pinned SHA-256, into .runtime/tor. (Downloading it goes directly to
    torproject.org: your network can see that you downloaded Tor, nothing more.)

A failed or cancelled install is rolled back: packages this run added are
uninstalled, packages it changed are put back, and a Tor it unpacked is
deleted. .env is never touched here; the settings window saves the switch
only after setup succeeded. A failing self-test keeps the components (so a
retry is quick) and is reported; "Remove components" deletes them.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

from .tor import TOR_DIR, find_tor_exe

ROOT = Path(__file__).resolve().parent.parent.parent
RUNTIME = ROOT / ".runtime"
STAMP = RUNTIME / "search_installed.json"
REQS = ROOT / "requirements-search.txt"
LOCK = ROOT / "requirements-lock.txt"
UV = RUNTIME / "uv" / ("uv.exe" if sys.platform == "win32" else "uv")
MODULES = ("requests", "socks", "psutil")

TOR_VERSION = "16.0.3"
TOR_FILE = f"tor-expert-bundle-windows-x86_64-{TOR_VERSION}.tar.gz"
TOR_SHA256 = "b70cabe2b93b5b1fd144466e42afaabe7a1a470228912657b3b5776e9cb7a9f1"
TOR_URLS = (f"https://dist.torproject.org/torbrowser/{TOR_VERSION}/{TOR_FILE}",
            f"https://archive.torproject.org/tor-package-archive/torbrowser/{TOR_VERSION}/{TOR_FILE}")
TOR_MAX_BYTES = 100_000_000
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


class SetupError(Exception):
    """Install step failed; the message is shown to the user."""


class Cancelled(Exception):
    pass


def needs_tor(cfg) -> bool:
    return cfg.route == "tor" and cfg.tor_mode == "managed"


def missing_modules() -> list[str]:
    return [m for m in MODULES if importlib.util.find_spec(m) is None]


def is_installed(cfg) -> bool:
    return not missing_modules() and (not needs_tor(cfg) or find_tor_exe(TOR_DIR) is not None)


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _snapshot() -> dict[str, str]:
    importlib.invalidate_caches()
    return {_norm(d.metadata["Name"]): d.version for d in md.distributions() if d.metadata["Name"]}


def _read_stamp() -> dict:
    try:
        data = json.loads(STAMP.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def _write_stamp(data: dict) -> None:
    STAMP.parent.mkdir(parents=True, exist_ok=True)
    tmp = STAMP.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, STAMP)


# ---------------------------------------------------------------------------
# Running uv / pip, cancellable
# ---------------------------------------------------------------------------
def _pip_cmd(*args: str) -> list[str]:
    if UV.exists():
        return [str(UV), "pip", *args[:1], "--python", sys.executable, *args[1:]]
    return [sys.executable, "-m", "pip", *args[:1], "--disable-pip-version-check", *args[1:]]


def _run(cmd: list[str], cancel: threading.Event, progress, step: str) -> None:
    env = dict(os.environ)
    env.setdefault("UV_CACHE_DIR", str(RUNTIME / "cache"))
    env.setdefault("UV_PYTHON_INSTALL_DIR", str(RUNTIME / "python"))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                            errors="replace", env=env, creationflags=CREATE_NO_WINDOW)
    tail: list[str] = []

    def read():
        for line in proc.stdout:
            line = line.strip()
            if line:
                tail[:] = (tail + [line])[-6:]
                progress(step, line[:160], None)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    while proc.poll() is None:
        if cancel.wait(0.2):
            proc.terminate()
            proc.wait(10)
            raise Cancelled()
    reader.join(5)
    if proc.returncode != 0:
        raise SetupError("Installing the search components failed:\n" + "\n".join(tail[-3:]))


def _install_packages(cancel, progress) -> None:
    progress("packages", "Installing search components (requests, PySocks)â€¦", None)
    args = ["install", "-r", str(REQS)]
    if LOCK.exists():
        args += ["-c", str(LOCK)]
    for attempt in range(3):              # antivirus can briefly lock new files
        try:
            _run(_pip_cmd(*args), cancel, progress, "packages")
            return
        except SetupError:
            if attempt == 2:
                raise
            progress("packages", "Install hiccup, retryingâ€¦", None)
            time.sleep(3)


def _restore_packages(added: list[str], changed: dict[str, str], progress) -> None:
    never = threading.Event()
    if added:
        progress("rollback", "Removing the components that were just installedâ€¦", None)
        try:
            _run(_pip_cmd("uninstall", *([] if UV.exists() else ["-y"]), *added),
                 never, progress, "rollback")
        except SetupError:
            pass
    if changed:
        try:
            _run(_pip_cmd("install", "--no-deps", *(f"{n}=={v}" for n, v in changed.items())),
                 never, progress, "rollback")
        except SetupError:
            pass


# ---------------------------------------------------------------------------
# Tor Expert Bundle
# ---------------------------------------------------------------------------
def _download(url: str, dest: Path, cancel, progress) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "VCB-setup"})
    with urllib.request.urlopen(req, timeout=30) as r, open(dest, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        got, last = 0, -1
        while True:
            if cancel.is_set():
                raise Cancelled()
            chunk = r.read(65536)
            if not chunk:
                break
            got += len(chunk)
            if got > TOR_MAX_BYTES:
                raise SetupError("The Tor download is unexpectedly large; stopped.")
            f.write(chunk)
            pct = int(got * 100 / total) if total else None
            if pct != last:
                last = pct
                progress("tor", f"Downloading Tor {TOR_VERSION}â€¦", pct)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _install_tor(cancel, progress) -> None:
    """Download, verify and unpack Tor into TOR_DIR/bundle."""
    RUNTIME.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="tor-setup-", dir=RUNTIME))
    try:
        archive = work / TOR_FILE
        for url in TOR_URLS:
            try:
                _download(url, archive, cancel, progress)
                break
            except (OSError, ValueError):
                continue
        else:
            raise SetupError("Couldn't download Tor from torproject.org. Check your internet "
                             "connection (some networks block it), or use VPN mode instead.")
        progress("tor", "Verifying the Tor downloadâ€¦", None)
        if _sha256(archive) != TOR_SHA256:
            raise SetupError("The Tor download failed its integrity check, so it wasn't used.")
        progress("tor", "Unpacking Torâ€¦", None)
        out = work / "bundle"
        with tarfile.open(archive, "r:gz") as tar:
            tar.extractall(out, filter="data")      # refuses absolute paths, .., links out
        if find_tor_exe(out) is None:
            raise SetupError("The Tor bundle didn't contain tor.exe.")
        TOR_DIR.mkdir(parents=True, exist_ok=True)
        target = TOR_DIR / "bundle"
        if target.exists():
            shutil.rmtree(target)
        os.replace(out, target)
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------
def ensure(cfg, cancel: threading.Event | None = None, progress=lambda *a: None) -> None:
    """Install whatever is missing for `cfg`. Raises SetupError/Cancelled,
    after rolling this run's changes back."""
    cancel = cancel or threading.Event()
    before = _snapshot()
    tor_existed = TOR_DIR.exists()
    tor_touched = False
    pkgs_touched = False
    try:
        if missing_modules():
            pkgs_touched = True
            _install_packages(cancel, progress)
            if missing_modules():
                raise SetupError("The search components installed but can't be loaded.")
        if needs_tor(cfg) and find_tor_exe(TOR_DIR) is None:
            tor_touched = True
            _install_tor(cancel, progress)
            if cancel.is_set():
                raise Cancelled()
    except BaseException:
        if pkgs_touched:
            after = _snapshot()
            added = sorted(set(after) - set(before))
            changed = {n: v for n, v in before.items() if n in after and after[n] != v}
            _restore_packages(added, changed, progress)
        if tor_touched:
            shutil.rmtree(TOR_DIR if not tor_existed else TOR_DIR / "bundle", ignore_errors=True)
        raise
    stamp = _read_stamp()
    added = sorted(set(_snapshot()) - set(before))
    stamp["packages_added"] = sorted(set(stamp.get("packages_added", [])) | set(added))
    if needs_tor(cfg):
        stamp["tor"] = TOR_VERSION
    _write_stamp(stamp)


def remove(progress=lambda *a: None) -> None:
    """Uninstall what setup added (never VCB's own dependencies) and delete
    VCB's Tor. Raises SetupError if Tor is still running."""
    if TOR_DIR.exists():
        progress("remove", "Deleting Torâ€¦", None)
        try:
            shutil.rmtree(TOR_DIR)
        except OSError:
            raise SetupError("Tor is still in use. Close VCB, then try again.") from None
    stamp = _read_stamp()
    pkgs = stamp.get("packages_added") or []
    if pkgs:
        progress("remove", "Removing search componentsâ€¦", None)
        _run(_pip_cmd("uninstall", *([] if UV.exists() else ["-y"]), *pkgs),
             threading.Event(), progress, "remove")
    STAMP.unlink(missing_ok=True)


def _selftest(progress, cancel) -> dict:
    """Run vcb.search.selftest in a fresh interpreter (the new packages are
    importable there for sure) and relay its progress."""
    proc = subprocess.Popen([sys.executable, "-m", "vcb.search.selftest", "--json-progress"],
                            cwd=str(ROOT), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                            stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                            errors="replace", creationflags=CREATE_NO_WINDOW,
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    result = {"ok": False, "error": "The self-test didn't report a result."}

    def watch():
        cancel.wait()
        if proc.poll() is None:
            proc.terminate()

    threading.Thread(target=watch, daemon=True).start()
    for line in proc.stdout:
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if ev.get("event") == "progress":
            progress("test", ev.get("text", ""), ev.get("pct"))
        elif ev.get("event") == "result":
            ev.pop("event")
            result = ev
    proc.wait()
    if cancel.is_set():
        raise Cancelled()
    return result


def _plain(text: str) -> str:
    """ASCII-only text for consoles and pipes that may not be UTF-8."""
    return (text.replace("\u2026", "...").replace("\u2713", "OK").replace("\u2717", "X")
            .encode("ascii", "replace").decode())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--ensure", action="store_true")
    mode.add_argument("--setup", action="store_true")
    mode.add_argument("--remove", action="store_true")
    ap.add_argument("--json-progress", action="store_true")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            # JSON consumers read UTF-8; a console (or the launcher's pipe) may not.
            stream.reconfigure(encoding="utf-8" if args.json_progress else None,
                               errors="replace")

    def emit(obj):
        print(json.dumps(obj, ensure_ascii=False), flush=True)

    def progress(step, text, pct=None):
        if args.json_progress:
            emit({"event": "progress", "step": step, "text": text, "pct": pct})
        elif pct is None or pct % 10 == 0:
            print(_plain(f"  {text}" + (f" {pct}%" if pct is not None else "")), flush=True)

    cancel = threading.Event()
    installing = False
    if args.json_progress:
        def listen():
            for line in sys.stdin:
                if line.strip() == "cancel":
                    cancel.set()
                    return
        threading.Thread(target=listen, daemon=True).start()

    from ..config import SearchConfig
    cfg = SearchConfig()
    result: dict = {"ok": True}
    try:
        if args.remove:
            remove(progress)
            result["summary"] = ["Online search components removed."]
        else:
            if not is_installed(cfg):
                installing = True
                ensure(cfg, cancel, progress)
                installing = False
            result["summary"] = ["Online search components are installed."]
            if args.setup:
                result["test"] = _selftest(progress, cancel)
                result["ok"] = bool(result["test"].get("ok"))
                result["summary"] = result["test"].get("summary") or [result["test"].get("error", "")]
    except Cancelled:
        result = {"ok": False, "cancelled": True,
                  "summary": ["Setup cancelled; nothing was changed." if installing
                              else "Test cancelled."]}
    except SetupError as exc:
        result = {"ok": False, "summary": [str(exc)]}
    except Exception as exc:
        result = {"ok": False, "summary": [f"Setup failed unexpectedly ({type(exc).__name__})."]}
    if args.json_progress:
        emit({"event": "result", **result})
    else:
        print(_plain("\n".join(result.get("summary", []))))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
