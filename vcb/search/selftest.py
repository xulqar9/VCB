"""Online search self-test, run by the settings window and the installer:

  python -m vcb.search.selftest [--json-progress]

1. Route: proves traffic leaves through the selected route (Tor: the Tor
   Project's own check; VPN: an IP echo through the VPN adapter/proxy). For
   managed Tor it starts a temporary Tor if VCB isn't already running one.
2. Exit IP: shown masked (e.g. 185.220.x.x) with its country, so you can see
   it isn't your home connection. Never logged or saved.
3. DNS leak: every name lookup this process makes through the OS resolver is
   recorded. Any hostname there is a leak. A hostname request that succeeded
   through the route without one proves names were resolved by the route. A
   DNS leak-test service additionally reports which resolvers it saw.
4. Backend: one harmless canary search through the route.

Everything goes through the route; there is never a direct connection, so the
check can't reveal your real IP to anyone. The configuration comes from .env,
overridden by VCB_* environment variables (the settings window passes unsaved
choices that way).
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import socket
import sys
import time
from urllib.parse import urlsplit
CANARY_QUERY = "weather"
DNS_LEAK_SERVICE = "bash.ws"
DNS_PROBES = 3
TOR_BOOT_TIMEOUT_S = 180.0


def mask_ip(ip: str | None) -> str:
    """185.220.101.4 -> 185.220.x.x; 2a0b:f4c2:…:1 -> 2a0b:f4c2:…"""
    if not ip:
        return ""
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return "?"
    if addr.version == 4:
        a, b, _, _ = str(addr).split(".")
        return f"{a}.{b}.x.x"
    return ":".join(addr.exploded.split(":")[:2]) + ":…"


def _is_ip_literal(host: str) -> bool:
    try:
        ipaddress.ip_address(host.strip("[]").split("%")[0])
        return True
    except ValueError:
        return False


class DnsMonitor:
    """Records every name handed to the OS resolver while active. With a
    socks5h proxy or the VPN-adapter resolver, only IP literals should appear."""

    HOOKED = ("getaddrinfo", "gethostbyname", "gethostbyname_ex")

    def __init__(self) -> None:
        self.lookups: list[str] = []
        self._orig: dict = {}

    def __enter__(self) -> "DnsMonitor":
        for name in self.HOOKED:
            orig = getattr(socket, name)
            self._orig[name] = orig
            setattr(socket, name, self._wrap(orig))
        return self

    def __exit__(self, *exc) -> None:
        for name, orig in self._orig.items():
            setattr(socket, name, orig)

    def _wrap(self, orig):
        def hooked(host, *args, **kwargs):
            if host is not None:
                self.lookups.append(host.decode() if isinstance(host, bytes) else str(host))
            return orig(host, *args, **kwargs)
        return hooked

    def hook_works(self) -> bool:
        before = len(self.lookups)
        try:
            socket.getaddrinfo("127.0.0.1", None)
        except OSError:
            pass
        ok = len(self.lookups) == before + 1
        del self.lookups[before:]
        return ok

    def leaked(self, allowed: set[str] = frozenset()) -> list[str]:
        """Hostnames sent to the OS resolver, except `allowed` (the proxy's
        own hostname, which has to be looked up to reach it)."""
        return sorted({h.lower() for h in self.lookups
                       if not _is_ip_literal(h) and h.lower() not in allowed})


def dns_verdict(hook_ok: bool, leaked: list[str], remote_proof: bool) -> tuple[str, str]:
    """("leak" | "no_leak" | "inconclusive", explanation)."""
    if leaked:
        return "leak", "Your normal DNS resolver was asked for: " + ", ".join(leaked[:5])
    if not hook_ok:
        return "inconclusive", "Couldn't watch DNS lookups on this system."
    if not remote_proof:
        return "inconclusive", "Nothing got through the route, so DNS couldn't be checked."
    return "no_leak", "No lookups went to your normal resolver; names were resolved through the route."


def _resolver_probe(session, net, timeout: float) -> list[dict]:
    """Which resolvers a DNS leak-test service saw for unique names that were
    only ever given to the route. Only ASN/country are kept."""
    base = f"https://{DNS_LEAK_SERVICE}"
    test_id = net.fetch(session, f"{base}/id", timeout=timeout).body.decode("ascii", "replace").strip()
    if not test_id.isalnum() or len(test_id) > 64:
        raise net.FetchError("bad_response")
    for n in range(1, DNS_PROBES + 1):
        try:
            net.fetch(session, f"https://{n}.{test_id}.{DNS_LEAK_SERVICE}/", timeout=timeout)
        except net.FetchError:
            pass                                  # only the lookup matters
    time.sleep(1.0)
    resp = net.fetch(session, f"{base}/dnsleak/test/{test_id}?json", timeout=timeout)
    if resp.status != 200:
        raise net.FetchError(f"http_{resp.status}")
    entries = json.loads(resp.body.decode("utf-8"))
    if not isinstance(entries, list):
        raise net.FetchError("bad_response")
    out = []
    for e in entries:
        if isinstance(e, dict) and e.get("type") == "dns":
            out.append({"asn": str(e.get("asn") or "")[:80], "country": str(e.get("country_name")
                        or e.get("country") or "")[:40]})
    return out


def _local_public_ips() -> set[str]:
    """Public addresses on this PC's own adapters (rare; usually all private)."""
    try:
        import psutil
    except ImportError:
        return set()
    ips = set()
    for addrs in psutil.net_if_addrs().values():
        for a in addrs:
            try:
                ip = ipaddress.ip_address(a.address.split("%")[0])
            except ValueError:
                continue
            if ip.is_global:
                ips.add(str(ip))
    return ips


def _check_searxng_token(route, backend, timeout: float) -> str | None:
    """Return a safe failure kind if the token fails to reveal a private engine."""
    from . import net

    try:
        without = route.fetch(backend.config_request(include_token=False), timeout)
        with_token = route.fetch(backend.config_request(), timeout)
    except (net.FetchError, net.RouteUnavailable) as exc:
        return getattr(exc, "kind", "unavailable")

    before_names = backend.parse_config(without.body) if without.status == 200 else None
    after_names = backend.parse_config(with_token.body) if with_token.status == 200 else None
    if before_names is None or after_names is None:
        if without.status in (401, 403) or with_token.status in (401, 403):
            return "auth"
        return "bad_response"
    return None if after_names - before_names else "auth"


def run(cfg, progress=lambda step, text, pct=None: None) -> dict:
    """Run every check and return a JSON-safe report (no unmasked IPs)."""
    report: dict = {"ok": False, "route": cfg.route,
                    "backend": getattr(cfg, "backend", ""), "steps": {}}
    problem = cfg.problem()
    if problem:
        report["error"] = problem
        return report

    tor = None
    if cfg.route == "tor" and cfg.tor_mode == "managed":
        from .tor import ManagedTor, find_tor_exe, port_in_use, proxy_addr
        host, port = proxy_addr(cfg.tor_proxy)
        if not port_in_use(host, port):
            if find_tor_exe() is None:
                report["error"] = "Tor isn't installed. Run setup again."
                return report
            progress("tor", "Connecting to the Tor network…", 0)
            tor = ManagedTor(port)
            tor.start()
            deadline = time.monotonic() + TOR_BOOT_TIMEOUT_S
            last = 0
            while not tor.wait_bootstrapped(0.5):
                if tor.exited.is_set() or time.monotonic() > deadline:
                    tor.stop()
                    report["error"] = ("Tor couldn't connect" + (f": {tor.error}" if tor.error else
                                       ". Your network may block Tor."))
                    return report
                if tor.progress != last:
                    last = tor.progress
                    progress("tor", "Connecting to the Tor network…", last)
            progress("tor", "Connected to the Tor network.", 100)

    try:
        from . import net
        from .backends import classify, make_backend
        from .service import install_log_guards
        install_log_guards()
        with DnsMonitor() as mon:
            hook_ok = mon.hook_works()
            report.update(_checks(cfg, net, classify, make_backend, progress))
            leaked = mon.leaked({(urlsplit(cfg.proxy).hostname or "").lower()} - {""})
        status, why = dns_verdict(hook_ok, leaked, report["steps"]["route"]["ok"])
        report["steps"]["dns"].update({"status": status, "detail": why})
        r, d, b = (report["steps"][k] for k in ("route", "dns", "backend"))
        report["ok"] = bool(r["ok"] and d["status"] != "leak" and b.get("kind") in ("ok", "empty")
                            and not report["steps"]["exit"].get("is_local"))
    finally:
        if tor is not None:
            tor.stop()
    return report


def _checks(cfg, net, classify, make_backend, progress) -> dict:
    steps: dict = {"route": {"ok": False}, "exit": {}, "dns": {"resolvers": []},
                   "backend": {}}
    route = net.PrivateRoute(cfg)
    timeout = max(cfg.effective_timeout_s * 2, 10.0)

    progress("route", f"Checking the {cfg.route.upper()} route…")
    if not route.verify():
        steps["route"] = {"ok": False, "reason": route.reason, "detail": net.describe(route.reason)}
        steps["backend"] = {"kind": "skipped"}
        return {"steps": steps}
    steps["route"] = {"ok": True, "detail": ("Traffic leaves through the Tor network."
                                             if cfg.route == "tor" else
                                             "Traffic leaves through your VPN.")}
    try:
        progress("exit", "Looking up the address websites will see…")
        ip, country = net.ip_echo(route.session, timeout)
        local = bool(ip) and ip in _local_public_ips()
        steps["exit"] = {"ip_masked": mask_ip(ip), "country": country or "unknown", "is_local": local}
        if local:
            steps["exit"]["detail"] = "This is your own connection's address, not a private route!"
        del ip

        progress("dns", "Checking for DNS leaks…")
        try:
            steps["dns"]["resolvers"] = _resolver_probe(route.session, net, timeout)
        except (net.FetchError, ValueError, UnicodeDecodeError) as exc:
            steps["dns"]["probe_error"] = getattr(exc, "kind", type(exc).__name__)

        progress("backend", f"Trying a test search on {cfg.backend}…")
        backend = make_backend(cfg)
        if (cfg.backend == "searxng" and backend.api_key
                and backend.auth_mode == "cookie"):
            progress("backend", "Checking the SearXNG server token…")
            token_problem = _check_searxng_token(route, backend, timeout)
            if token_problem:
                steps["backend"] = {"kind": token_problem}
                return {"steps": steps}
        try:
            resp = route.fetch(backend.request(CANARY_QUERY, cfg.max_results), timeout)
            outcome = classify(resp.status, resp.content_type, resp.body, backend, cfg.max_results)
            steps["backend"] = {"kind": outcome.kind, "results": len(outcome.results)}
        except (net.FetchError, net.RouteUnavailable) as exc:
            steps["backend"] = {"kind": getattr(exc, "kind", "unavailable")}
    finally:
        route.close()
    return {"steps": steps}


def summarize(report: dict) -> list[str]:
    """Human-readable lines for the terminal and the settings window."""
    if report.get("error"):
        return [f"✗ {report['error']}"]
    s = report["steps"]
    lines = [("✓ " if s["route"]["ok"] else "✗ ") + s["route"].get("detail", "")]
    if s["exit"]:
        e = s["exit"]
        lines.append(f"{'✗' if e.get('is_local') else '✓'} Websites see {e['ip_masked'] or '?'} "
                     f"({e['country']})" + (f" - {e['detail']}" if e.get("detail") else ""))
    d = s["dns"]
    if "status" in d:
        mark = {"leak": "✗", "no_leak": "✓"}.get(d["status"], "?")
        extra = ""
        if d.get("resolvers"):
            where = sorted({r["asn"] or r["country"] for r in d["resolvers"]} - {""})
            extra = f" Resolvers seen: {', '.join(where[:3])}." if where else ""
        lines.append(f"{mark} DNS: {d['detail']}{extra}")
    b = s["backend"]
    kind = b.get("kind")
    if kind and kind != "skipped":
        text = {"ok": f"Test search worked ({b.get('results', 0)} results).",
                "empty": "Test search worked (no results).",
                "auth": ("The SearXNG server or its gateway didn't accept this token. Check it "
                         "with whoever runs the server and confirm the token has the right access."
                         if report.get("backend") == "searxng" else
                         "The search service didn't accept the API key. Check that you copied "
                         "the whole key and that your plan is active."),
                "captcha": ("The search service wants a CAPTCHA solved on this route, so VCB "
                            "can't use it. Try Brave Search or another SearXNG server."),
                "blocked": ("The search service blocks this route. Try Brave Search or another "
                            "SearXNG server."),
                "rate_limited": "The search service is busy (too many requests). Try again in a few minutes.",
                "bad_response": ("The search service didn't answer in a form VCB can read. For "
                                 "SearXNG, the server must have JSON output turned on."),
                "timeout": "The test search took too long. Try again; if it keeps happening, try the other route."}.get(
                    kind, f"The test search failed ({kind}).")
        lines.append(("✓ " if kind in ("ok", "empty") else "✗ ") + text)
    return lines


def _plain(text: str) -> str:
    """ASCII-only text for consoles and pipes that may not be UTF-8."""
    return (text.replace("\u2026", "...").replace("\u2713", "OK").replace("\u2717", "X")
            .encode("ascii", "replace").decode())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json-progress", action="store_true",
                    help="emit one JSON object per line (for the settings window)")
    args = ap.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            # JSON consumers read UTF-8; a console (or the launcher's pipe) may not.
            stream.reconfigure(encoding="utf-8" if args.json_progress else None,
                               errors="replace")
    from ..config import SearchConfig
    cfg = SearchConfig()

    def emit(obj: dict) -> None:
        print(json.dumps(obj, ensure_ascii=False), flush=True)

    def progress(step, text, pct=None):
        if args.json_progress:
            emit({"event": "progress", "step": step, "text": text, "pct": pct})
        else:
            print(_plain(f"  {text}" + (f" {pct}%" if pct is not None else "")), flush=True)

    try:
        report = run(cfg, progress)
    except ImportError:
        report = {"ok": False, "error": "Online search components aren't installed. Run setup again."}
    except Exception as exc:                  # report, don't traceback (could include URLs)
        report = {"ok": False, "error": f"The self-test failed unexpectedly ({type(exc).__name__})."}
    report["summary"] = summarize(report)
    if args.json_progress:
        emit({"event": "result", **report})
    else:
        print(_plain("\n".join(report["summary"])))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
