"""VCB-managed Tor: runs the Tor Expert Bundle's tor.exe (installed on demand
into .runtime/tor by vcb.search.install) as a private SOCKS proxy for online
search only.

Starting it doesn't block VCB: Tor bootstraps in the background while the
models load, and searches stay off until the route verifier has confirmed
(via check.torproject.org) that traffic really leaves through Tor.
Stdlib only, so the launcher and the settings window can use it too.
"""

from __future__ import annotations

import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from urllib.parse import urlsplit

log = logging.getLogger("vcb.search.tor")

ROOT = Path(__file__).resolve().parent.parent.parent
TOR_DIR = ROOT / ".runtime" / "tor"
_BOOT = re.compile(r"Bootstrapped (\d+)%")
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


def find_tor_exe(base: Path = TOR_DIR) -> Path | None:
    name = "tor.exe" if sys.platform == "win32" else "tor"
    return next((p for p in sorted(base.glob(f"**/{name}")) if p.is_file()), None)


def port_in_use(host: str, port: int) -> bool:
    try:
        with socket.create_connection((host, port), timeout=0.5):
            return True
    except OSError:
        return False


def proxy_addr(proxy: str) -> tuple[str, int]:
    p = urlsplit(proxy)
    return p.hostname or "127.0.0.1", p.port or 9050


class ManagedTor:
    """One tor process with its own torrc and data directory."""

    def __init__(self, port: int, *, base: Path = TOR_DIR,
                 cmd_prefix: list[str] | None = None):
        self.port = port
        self.base = base
        self.data_dir = base / "state"       # not "data": the bundle ships a data/ folder
        self.torrc = base / "torrc"
        self._prefix = cmd_prefix
        self.proc: subprocess.Popen | None = None
        self.progress = 0
        self.error = ""
        self.ready = threading.Event()
        self.exited = threading.Event()

    def command(self) -> list[str]:
        prefix = self._prefix
        if prefix is None:
            exe = find_tor_exe(self.base)
            if exe is None:
                raise FileNotFoundError("tor.exe isn't installed")
            prefix = [str(exe)]
        return [*prefix, "-f", str(self.torrc), "--defaults-torrc", str(self.base / "torrc-defaults"),
                "--ignore-missing-torrc"]

    def write_torrc(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Quoted torrc values are C-escaped, so use forward slashes (fine on Windows).
        q = lambda p: '"' + Path(p).as_posix().replace('"', '\\"') + '"'  # noqa: E731
        lines = [
            f"SocksPort 127.0.0.1:{self.port} IsolateSOCKSAuth",
            f"DataDirectory {q(self.data_dir)}",
            "ClientOnly 1",
            "SafeLogging 1",               # never put addresses in Tor's log
            "AvoidDiskWrites 1",
            "Log notice stdout",
            # Tor exits by itself if VCB dies, even without the job object.
            f"__OwningControllerProcess {os.getpid()}",
        ]
        for name, key in (("geoip", "GeoIPFile"), ("geoip6", "GeoIPv6File")):
            hit = next(iter(sorted(self.base.glob(f"**/data/{name}"))), None)
            if hit is not None:
                lines.append(f"{key} {q(hit)}")
        self.torrc.write_text("\n".join(lines) + "\n", encoding="utf-8")
        (self.base / "torrc-defaults").write_text("", encoding="utf-8")

    def start(self) -> None:
        """Launch tor without waiting for it. Raises OSError/FileNotFoundError."""
        cmd = self.command()
        self.write_torrc()
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
            cwd=str(self.base), creationflags=CREATE_NO_WINDOW,
            text=True, encoding="utf-8", errors="replace")
        threading.Thread(target=self._read, name="tor-log", daemon=True).start()

    def _read(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        shown = -1
        last_warn = ""
        for line in self.proc.stdout:
            m = _BOOT.search(line)
            if m:
                self.progress = int(m.group(1))
                if self.progress // 25 > shown:
                    shown = self.progress // 25
                    log.info("Tor connecting: %d%%", self.progress)
                if self.progress >= 100:
                    self.ready.set()
            elif "[warn]" in line:
                last_warn = line.split("[warn]", 1)[1].strip()[:200]
            elif "[err]" in line:
                err = line.split("[err]", 1)[1].strip()[:200]
                self.error = last_warn if "see warnings above" in err and last_warn else err
                log.warning("Tor: %s", self.error)
        self.exited.set()
        if not self.ready.is_set():
            log.warning("Tor stopped before it was connected%s.",
                        f" ({self.error})" if self.error else "")

    def wait_bootstrapped(self, timeout: float) -> bool:
        """True once Tor reports 100%; False on timeout or if it exited."""
        deadline = time.monotonic() + timeout
        while not self.ready.is_set() and not self.exited.is_set():
            left = deadline - time.monotonic()
            if left <= 0:
                break
            self.ready.wait(min(left, 0.2))
        return self.ready.is_set()

    def stop(self) -> None:
        proc, self.proc = self.proc, None
        if proc is None or proc.poll() is not None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def start_managed_tor(cfg, job_assign=None) -> ManagedTor | None:
    """Start VCB's own Tor for the search route when it's configured, or
    return None (with a log line saying why). Never raises."""
    host, port = proxy_addr(cfg.tor_proxy)
    if port_in_use(host, port):
        log.info("Something is already listening on %s:%d; online search will use it "
                 "only if it proves to be Tor.", host, port)
        return None
    tor = ManagedTor(port)
    try:
        tor.start()
    except FileNotFoundError:
        log.warning("Online search: Tor isn't installed yet. Open settings > Online "
                    "search > Set up & test.")
        return None
    except OSError as exc:
        log.warning("Online search: couldn't start Tor (%s).", type(exc).__name__)
        return None
    if job_assign is not None:
        job_assign(tor.proc)
    log.info("Starting Tor for online search (it connects in the background).")
    return tor
