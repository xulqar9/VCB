#!/usr/bin/env python3
"""Benchmark web-search performance over two network routes: VPN and Tor.

Each route is run separately (the VPN may need to be toggled between runs),
then a side-by-side report is built from the saved results:

    python tools/search_route_benchmark.py vpn      # system VPN already connected
    python tools/search_route_benchmark.py tor      # via socks5h://127.0.0.1:9050
    python tools/search_route_benchmark.py report   # merge + print + save

Requirements:  pip install requests pysocks

Configuration (environment variables, defaults in the CONFIG section below):
    SEARCH_BACKEND_URL    e.g. https://searx.example.org/search, or the keyless
                          https://wiby.me/json/ (set SEARCH_EXTRA_PARAMS to {})
    SEARCH_QUERY_PARAM    query-string parameter carrying the query (default "q")
    SEARCH_EXTRA_PARAMS   JSON object of extra params (default {"format": "json"})
    SEARCH_EXTRA_HEADERS  JSON object of extra headers, e.g. an API key
    TOR_PROXY             default socks5h://127.0.0.1:9050
    IP_ECHO_URL           comma-separated IP echo services, tried in order until one
                          answers (default ipinfo.io, ifconfig.co, geojs.io)
    DNS_LEAK_TEST_URL     default https://bash.ws (set to "" to skip the external test)
    REAL_IP               your real public IP (or pass --real-ip)

Notes:
  * SearXNG only answers format=json if "json" is listed under search.formats in
    its settings.yml; otherwise it returns 403, which is counted as a block.
  * The measured hop is client -> search backend. A backend on localhost or a
    private address cannot be reached through Tor, so the Tor run refuses to start.
  * A request is a success only if it returns HTTP 200 and at least one parseable
    result. CAPTCHA and block pages are detected and counted, never bypassed.
  * Latency (mean/median/p95) is computed over successful requests only; raw
    per-request timings for every request are kept in the CSV.
  * The DNS leak check (Tor route) combines two pieces of evidence:
      1. In-process: every hostname this process resolves through the OS
         resolver is recorded; anything other than the proxy address is a leak.
      2. External: a DNS leak-test service reports which resolvers looked up
         unique hostnames - once via your OS resolver (your normal resolver)
         and once via Tor only - so the two can be compared.
    A clean verdict additionally requires proof that names really were resolved
    through the proxy; if every Tor request failed the verdict is "inconclusive".
    Measuring the normal resolver contacts the leak-test service directly.
"""

from __future__ import annotations

import argparse
import csv
import ipaddress
import json
import math
import os
import random
import socket
import statistics
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

try:
    import requests
    from urllib3.exceptions import TimeoutError as Urllib3Timeout
except ImportError:
    sys.exit("Missing dependency. Install with: pip install requests pysocks")

# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
SEARCH_BACKEND_URL = os.environ.get("SEARCH_BACKEND_URL", "http://127.0.0.1:8080/search")
SEARCH_QUERY_PARAM = os.environ.get("SEARCH_QUERY_PARAM", "q")
SEARCH_EXTRA_PARAMS = json.loads(os.environ.get("SEARCH_EXTRA_PARAMS", '{"format": "json"}'))
SEARCH_EXTRA_HEADERS = json.loads(os.environ.get("SEARCH_EXTRA_HEADERS", "{}"))

# socks5h (not socks5) makes the proxy resolve hostnames, so DNS goes through Tor.
TOR_PROXY = os.environ.get("TOR_PROXY", "socks5h://127.0.0.1:9050")
IP_ECHO_URLS = [u.strip() for u in os.environ.get(
    "IP_ECHO_URL",
    # Tried in order until one answers: some services (e.g. ipinfo.io) reject Tor exits.
    "https://ipinfo.io/json,https://ifconfig.co/json,https://get.geojs.io/v1/ip/geo.json",
).split(",") if u.strip()]
TOR_CHECK_URL = os.environ.get("TOR_CHECK_URL", "https://check.torproject.org/api/ip")
# bash.ws-compatible API: GET /id, look up <n>.<id>.<domain>, GET /dnsleak/test/<id>?json
DNS_LEAK_TEST_URL = os.environ.get("DNS_LEAK_TEST_URL", "https://bash.ws")
DNS_PROBES = 3                  # unique hostnames looked up per resolver probe

REQUEST_TIMEOUT = 30            # seconds; connect, per-read and total-body budget
REPEATS = 3                     # runs per query per route
DELAY_RANGE = (1.0, 3.0)        # random pause between requests, seconds
TOP_N = 5                       # results compared across routes
MAX_BODY_BYTES = 5_000_000
# One identical, honest User-Agent on both routes so only the route differs.
USER_AGENT = "Mozilla/5.0 (compatible; route-benchmark/1.0)"
DEFAULT_OUT_DIR = Path("benchmark_results")

# 20 queries: 7 factual, 6 news, 7 technical.
QUERIES = [
    # factual
    "capital city of Australia",
    "how tall is Mount Everest",
    "who wrote One Hundred Years of Solitude",
    "boiling point of water at high altitude",
    "population of the Tokyo metropolitan area",
    "when did the Berlin Wall fall",
    "speed of light in meters per second",
    # news
    "latest news global climate summit",
    "stock market today",
    "SpaceX launch schedule this week",
    "world cup qualifiers results",
    "central bank interest rate decision",
    "European Union AI regulation news",
    # technical
    "python asyncio vs threading performance",
    "configure nginx reverse proxy with TLS",
    "docker container exit code 137 meaning",
    "postgresql index scan vs sequential scan explain analyze",
    "git rebase vs merge difference",
    "rust borrow checker error E0499",
    "kubernetes CrashLoopBackOff troubleshooting",
]

CAPTCHA_MARKERS = (
    "captcha", "recaptcha", "hcaptcha", "cf-challenge", "challenge-platform",
    "are you a robot", "unusual traffic", "verify you are human",
)

# Every outcome a request can end in; "success" is the only good one.
OUTCOMES = (
    "success", "timeout", "connection_error", "blocked", "captcha",
    "http_error", "parse_error", "no_results", "other_error",
)
# Failure breakdown shown in the report (label, outcomes that feed it).
FAILURE_GROUPS = (
    ("Timeouts", ("timeout",)),
    ("Connection errors", ("connection_error",)),
    ("Blocked (403/429)", ("blocked",)),
    ("CAPTCHA pages", ("captcha",)),
    ("Empty result sets (HTTP 200)", ("no_results",)),
    ("Other failures", ("http_error", "parse_error", "other_error")),
)
DNS_STATUS_LABELS = {"leak": "LEAK", "no_leak": "none (verified)", "inconclusive": "inconclusive"}

CSV_FIELDS = [
    "route", "run", "query_index", "query", "timestamp", "http_status", "outcome",
    "ttfb_s", "total_s", "result_count", "top_urls", "bytes", "error",
]


# ----------------------------------------------------------------------------
# HELPERS: sessions, URL handling, stats
# ----------------------------------------------------------------------------
def make_session(proxy: str | None) -> requests.Session:
    """Build a session for one route. proxy=None means a direct (VPN) connection."""
    session = requests.Session()
    # Ignore HTTP(S)_PROXY / NO_PROXY from the environment so the route is
    # exactly what you configure here and nothing else.
    session.trust_env = False
    session.headers.update({"User-Agent": USER_AGENT, "Accept": "application/json"})
    if proxy:
        session.proxies = {"http": proxy, "https": proxy}
    return session


def sanitize_url(url: str) -> str:
    """Drop query string and credentials so the URL is safe to write to reports."""
    p = urlsplit(url)
    host = p.hostname or ""
    netloc = f"{host}:{p.port}" if p.port else host
    return urlunsplit((p.scheme, netloc, p.path, "", ""))


def is_local_host(host: str | None) -> bool:
    """True for localhost / loopback / private literals a Tor exit cannot reach."""
    if not host:
        return False
    if host.lower() == "localhost":
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def is_hostname(host: str | None) -> bool:
    """True if host is a DNS name (needs resolving) rather than an IP literal."""
    if not host:
        return False
    try:
        ipaddress.ip_address(host)
        return False
    except ValueError:
        return True


def normalize_url(url: str) -> str:
    """Canonical form for comparing results: no scheme, no www, no fragment/trailing slash."""
    p = urlsplit(url.strip().lower())
    host = p.netloc[4:] if p.netloc.startswith("www.") else p.netloc
    key = host + p.path.rstrip("/")
    return f"{key}?{p.query}" if p.query else key


def percentile(values: list[float], pct: float) -> float:
    """Linear-interpolated percentile (same method as numpy's default)."""
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100
    low, high = math.floor(rank), math.ceil(rank)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def latency_stats(values: list[float]) -> dict:
    """Mean / median / p95 in seconds, or None values when there is no data."""
    if not values:
        return {"mean": None, "median": None, "p95": None}
    return {
        "mean": round(statistics.fmean(values), 3),
        "median": round(statistics.median(values), 3),
        "p95": round(percentile(values, 95), 3),
    }


# ----------------------------------------------------------------------------
# SEARCH REQUEST: timing, parsing and failure classification
# ----------------------------------------------------------------------------
def extract_result_urls(body: bytes) -> list[str] | None:
    """Parse a JSON search response into an ordered list of result URLs.

    Returns None if the body is not a recognizable results payload, and []
    if it is valid but empty. Handles SearXNG ("results"), and common shapes of
    other APIs ("organic_results", "items", "web.results").
    """
    try:
        payload = json.loads(body.decode("utf-8", errors="replace"))
    except ValueError:
        return None

    items = None
    if isinstance(payload, list):
        items = payload
    elif isinstance(payload, dict):
        for key in ("results", "organic_results", "items"):
            if isinstance(payload.get(key), list):
                items = payload[key]
                break
        else:
            web = payload.get("web")
            if isinstance(web, dict) and isinstance(web.get("results"), list):
                items = web["results"]
    if items is None:
        return None

    urls = []
    for item in items:
        if isinstance(item, dict):
            fields = {str(k).lower(): v for k, v in item.items()}  # e.g. Wiby uses "URL"
            url = fields.get("url") or fields.get("link") or fields.get("href")
            if isinstance(url, str) and url:
                urls.append(url)
    return urls


def looks_like_captcha(text: str) -> bool:
    lowered = text[:200_000].lower()
    return any(marker in lowered for marker in CAPTCHA_MARKERS)


def classify(status: int | None, text: str, urls: list[str] | None) -> str:
    """Map an HTTP response to exactly one outcome."""
    if status == 200 and urls:
        return "success"
    # A CAPTCHA page is often served with 200, 403, 429 or 503, so check it
    # before the status code. SearXNG also reports upstream-engine CAPTCHAs in
    # an otherwise empty 200 response, which lands here as well.
    if looks_like_captcha(text):
        return "captcha"
    if status in (403, 429):
        return "blocked"
    if status == 200:
        return "parse_error" if urls is None else "no_results"
    return "http_error"


def is_timeout(exc: Exception) -> bool:
    """True for connect/read timeouts, including read timeouts that requests
    re-raises as a ConnectionError while streaming the body."""
    return isinstance(exc, requests.exceptions.Timeout) or any(
        isinstance(arg, Urllib3Timeout) for arg in exc.args
    )


def timed_search(route: str, proxy: str | None, run: int, query_index: int, query: str) -> dict:
    """Send one search and return a raw-data row.

    ttfb_s  = time from starting the request until the response headers arrive
              (includes DNS, TCP, TLS and any proxy handshake, like curl's
              time_starttransfer).
    total_s = time until the full body has been downloaded.
    """
    row = {
        "route": route, "run": run, "query_index": query_index, "query": query,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "http_status": None, "outcome": None, "ttfb_s": None, "total_s": None,
        "result_count": None, "top_urls": "[]", "bytes": None, "error": "",
    }
    params = {**SEARCH_EXTRA_PARAMS, SEARCH_QUERY_PARAM: query}
    start = time.perf_counter()
    try:
        with make_session(proxy) as session:
            # stream=True returns as soon as the headers arrive, which is what
            # lets us time first byte separately from the full download.
            response = session.get(
                SEARCH_BACKEND_URL, params=params, headers=SEARCH_EXTRA_HEADERS,
                timeout=REQUEST_TIMEOUT, stream=True,
            )
            row["ttfb_s"] = time.perf_counter() - start
            row["http_status"] = response.status_code

            body = bytearray()
            for chunk in response.iter_content(chunk_size=16384):
                body.extend(chunk)
                # requests' timeout is per socket operation, so also enforce an
                # overall budget while the body streams in.
                if time.perf_counter() - start > REQUEST_TIMEOUT:
                    raise requests.exceptions.ReadTimeout("total time budget exceeded")
                if len(body) > MAX_BODY_BYTES:
                    break
            row["total_s"] = time.perf_counter() - start
            row["bytes"] = len(body)

        urls = extract_result_urls(bytes(body))
        text = body.decode("utf-8", errors="replace")
        row["outcome"] = classify(row["http_status"], text, urls)
        if urls is not None:
            row["result_count"] = len(urls)
            row["top_urls"] = json.dumps(urls[:TOP_N])
        if row["outcome"] != "success":
            row["error"] = f"HTTP {row['http_status']}"
    except requests.exceptions.RequestException as exc:  # timeouts, connection, proxy, SSL, ...
        row["outcome"] = "timeout" if is_timeout(exc) else "connection_error"
        row["error"] = str(exc)
    except Exception as exc:  # never let one bad request abort the whole run
        row["outcome"], row["error"] = "other_error", f"{type(exc).__name__}: {exc}"

    row["error"] = " ".join(row["error"].split())[:200]
    for key in ("ttfb_s", "total_s"):
        if row[key] is not None:
            row[key] = round(row[key], 4)
    return row


# ----------------------------------------------------------------------------
# IP CHECK: exit IP and country per route
# ----------------------------------------------------------------------------
def check_exit_ip(proxy: str | None, real_ip: str | None) -> dict:
    """Ask an IP echo service which address and country the route exits from.

    Services are tried in order and the first usable answer wins, because some
    of them block Tor exits. "error" is only set if every service failed.
    """
    result = {"service": None, "ip": None, "country": None, "hides_real_ip": None,
              "error": None, "skipped": []}
    with make_session(proxy) as session:
        for url in IP_ECHO_URLS:
            try:
                response = session.get(url, timeout=REQUEST_TIMEOUT)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict) or not data.get("ip"):
                    raise ValueError("response has no 'ip' field")
            except Exception as exc:
                result["skipped"].append(f"{urlsplit(url).hostname}: {type(exc).__name__}: {exc}"[:160])
                continue
            result["service"], result["ip"] = url, data["ip"]
            # Prefer ISO codes; field names differ between services.
            result["country"] = (data.get("country_code") or data.get("country_iso")
                                 or data.get("country") or data.get("country_name"))
            break
    if result["ip"] is None:
        result["error"] = "; ".join(result["skipped"]) or "no IP echo service configured"
    if real_ip and result["ip"]:
        result["hides_real_ip"] = result["ip"] != real_ip
    return result


def check_tor_exit() -> dict:
    """Ask the Tor Project whether our requests really arrive from a Tor exit node."""
    result = {"service": TOR_CHECK_URL, "is_tor": None, "ip": None, "error": None}
    try:
        with make_session(TOR_PROXY) as session:
            response = session.get(TOR_CHECK_URL, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            data = response.json()
        result["is_tor"], result["ip"] = data.get("IsTor"), data.get("IP")
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return result


# ----------------------------------------------------------------------------
# DNS LEAK CHECK: local resolver hook + resolver comparison via a leak-test service
# ----------------------------------------------------------------------------
def resolver_probe(proxy: str | None) -> dict:
    """Ask a DNS leak-test service which resolvers looked up a few unique names.

    proxy=None:  the names go to the OS resolver, so the service sees your
                 normal resolver (the baseline).
    proxy=TOR:   the names are only ever handed to the SOCKS proxy (socks5h),
                 so the service should see the Tor exit's resolver instead.
    Each probe uses a fresh test id, so the two result sets cannot mix.
    """
    base = DNS_LEAK_TEST_URL.rstrip("/")
    domain = urlsplit(base).hostname
    result = {"service": base, "resolvers": [], "error": None}
    try:
        with make_session(proxy) as session:
            response = session.get(f"{base}/id", timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            test_id = response.text.strip()
            if not test_id.isalnum():
                raise ValueError(f"unexpected test id {test_id[:40]!r}")
            for n in range(1, DNS_PROBES + 1):
                name = f"{n}.{test_id}.{domain}"
                try:
                    if proxy:
                        session.get(f"http://{name}/", timeout=15)  # exit node resolves name
                    else:
                        socket.getaddrinfo(name, 80)                # OS resolver resolves name
                except (requests.exceptions.RequestException, OSError):
                    pass  # only the lookup matters, not whether a connection follows
            time.sleep(1)  # let the service record the lookups
            response = session.get(f"{base}/dnsleak/test/{test_id}?json", timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            entries = response.json()
        result["resolvers"] = [
            {"ip": e.get("ip"), "asn": e.get("asn"), "country": e.get("country")}
            for e in entries if isinstance(e, dict) and e.get("type") == "dns"
        ]
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"[:200]
    return result


class DnsMonitor:
    """Context manager that records every name resolved through the OS resolver.

    It hooks socket.getaddrinfo (used by urllib3 and for the proxy address) and
    socket.gethostbyname[_ex] (used by PySocks for client-side DNS with plain
    socks5://). With socks5h the only lookup expected is the proxy address
    itself; any other name means the OS resolver was used, i.e. a DNS leak.
    """

    HOOKED = ("getaddrinfo", "gethostbyname", "gethostbyname_ex")

    def __init__(self) -> None:
        self.lookups: list[str] = []
        self._originals: dict = {}

    def __enter__(self) -> "DnsMonitor":
        for name in self.HOOKED:
            original = getattr(socket, name)
            self._originals[name] = original
            setattr(socket, name, self._recorder(original))
        return self

    def __exit__(self, *exc_info) -> None:
        for name, original in self._originals.items():
            setattr(socket, name, original)

    def _recorder(self, original):
        def wrapper(host, *args, **kwargs):
            self.lookups.append(host.decode() if isinstance(host, bytes) else str(host))
            return original(host, *args, **kwargs)
        return wrapper

    def verdict(self, proxy_url: str, targets: set[str], proxied_name_responses: int,
                tor_probe: dict | None, normal_probe: dict | None) -> dict:
        """Combine the evidence into "leak", "no_leak" or "inconclusive".

        Call while the monitor is still active.
          leak:         the OS resolver was asked for a name other than the proxy.
          no_leak:      the hook provably records lookups, none leaked, AND names
                        were verifiably resolved through the proxy: a proxied
                        request to a hostname got an HTTP response, or the leak-test
                        service saw lookups of names that only the proxy was given.
          inconclusive: anything else, e.g. every Tor request failed, so silence
                        from the hook proves nothing.
        """
        before = len(self.lookups)
        socket.getaddrinfo("localhost", None)  # control: proves the hook records lookups
        hook_works = len(self.lookups) == before + 1
        del self.lookups[before:]

        proxy_host = (urlsplit(proxy_url).hostname or "").lower()
        seen = sorted({h.lower() for h in self.lookups})
        unexpected = [h for h in seen if h != proxy_host]
        tor_resolvers = (tor_probe or {}).get("resolvers") or []
        normal_resolvers = (normal_probe or {}).get("resolvers") or []
        remote_verified = proxied_name_responses > 0 or bool(tor_resolvers)

        if unexpected:
            status, reason = "leak", "the OS resolver was used for: " + ", ".join(unexpected)
        elif not hook_works:
            status, reason = "inconclusive", "the resolver hook did not record a control lookup"
        elif not remote_verified:
            status, reason = "inconclusive", ("no request through the proxy succeeded, so there "
                                              "is no proof that names were resolved remotely")
        else:
            status, reason = "no_leak", "no OS resolver lookups; names verifiably resolved via the proxy"
        return {
            "status": status,
            "leak_detected": status == "leak",
            "reason": reason,
            "proxy_scheme": urlsplit(proxy_url).scheme,
            "proxy_resolves_dns": urlsplit(proxy_url).scheme == "socks5h",
            "local_lookups": seen,
            "targets_resolved_locally": [
                h for h in unexpected if any(h == t or h.endswith("." + t) for t in targets)
            ],
            "hook_verified": hook_works,
            "proxied_name_responses": proxied_name_responses,
            "tor_route_resolvers": tor_resolvers,
            "normal_resolvers": normal_resolvers,
            # Same resolver IP on both sides normally just means the exit node uses
            # the same public DNS provider; the queries still came from the exit.
            "shared_resolver_ips": sorted({r["ip"] for r in tor_resolvers}
                                          & {r["ip"] for r in normal_resolvers}),
            "leak_test_errors": {k: p["error"] for k, p in (("tor", tor_probe), ("normal", normal_probe))
                                 if p and p.get("error")},
        }


def tor_proxy_reachable(proxy_url: str) -> bool:
    p = urlsplit(proxy_url)
    try:
        with socket.create_connection((p.hostname, p.port or 9050), timeout=5):
            return True
    except OSError:
        return False


# ----------------------------------------------------------------------------
# RUN ONE ROUTE: collect raw data and write raw CSV + meta JSON
# ----------------------------------------------------------------------------
def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def run_route(route: str, args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    proxy = TOR_PROXY if route == "tor" else None
    backend_host = urlsplit(SEARCH_BACKEND_URL).hostname

    # --- preflight: fail early with a clear message instead of 60 failed requests
    if route == "tor":
        try:
            import socks  # noqa: F401  (PySocks: requests needs it for socks5h:// proxies)
        except ImportError:
            sys.exit("PySocks is required for the Tor route. Install with: pip install pysocks")
        if not tor_proxy_reachable(TOR_PROXY):
            sys.exit(f"Tor SOCKS proxy not reachable at {TOR_PROXY}. Start Tor first.")
        if is_local_host(backend_host):
            sys.exit(f"Search backend host '{backend_host}' is local/private; a Tor exit node "
                     "cannot reach it. Point SEARCH_BACKEND_URL at a publicly reachable backend.")
        if urlsplit(TOR_PROXY).scheme != "socks5h":
            print("WARNING: TOR_PROXY is not socks5h; DNS will be resolved locally (leak).")
    elif is_local_host(backend_host):
        print(f"WARNING: backend '{backend_host}' is local; the VPN only affects the "
              "client->backend hop, not the backend's own upstream traffic.")

    queries = QUERIES[: args.limit] if args.limit else QUERIES
    total = len(queries) * args.repeats
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"Route: {route.upper()}  |  backend: {sanitize_url(SEARCH_BACKEND_URL)}  |  "
          f"{len(queries)} queries x {args.repeats} runs = {total} requests")

    rows: list[dict] = []
    monitor = DnsMonitor() if route == "tor" else None
    ip_check = tor_check = dns_check = normal_probe = tor_probe = None

    # --- DNS baseline (Tor run only): which resolver does the OS normally use?
    # Runs before the monitor starts because these lookups use the OS resolver on purpose.
    if monitor and DNS_LEAK_TEST_URL:
        print("DNS leak test: measuring your normal resolver ...")
        normal_probe = resolver_probe(None)

    with monitor if monitor else nullcontext():
        try:
            # --- IP check (once per route)
            ip_check = check_exit_ip(proxy, args.real_ip)
            tor_check = check_tor_exit() if route == "tor" else None
            print(f"Exit IP: {ip_check['ip']} ({ip_check['country']})"
                  + (f"  [error: {ip_check['error']}]" if ip_check["error"] else ""))
            if ip_check["hides_real_ip"] is False:
                print("WARNING: the exit IP equals your real IP; this route is NOT hiding it.")
            if tor_check:
                print(f"Tor check: IsTor={tor_check['is_tor']}"
                      + (f"  [error: {tor_check['error']}]" if tor_check["error"] else ""))

            # --- DNS leak test through Tor: names only the proxy is ever given
            if monitor and DNS_LEAK_TEST_URL:
                print("DNS leak test: resolving test names through Tor ...")
                tor_probe = resolver_probe(TOR_PROXY)

            # --- search loop: repeats outermost so identical queries are spread out
            n = 0
            for run in range(1, args.repeats + 1):
                for index, query in enumerate(queries, start=1):
                    n += 1
                    row = timed_search(route, proxy, run, index, query)
                    rows.append(row)
                    latency = f"{row['total_s']:.2f}s" if row["total_s"] is not None else "--"
                    print(f"[{route} {n:>3}/{total}] run {run} {row['outcome']:<16} {latency:>7}  {query[:50]}")
                    if n < total:
                        time.sleep(random.uniform(args.delay_min, args.delay_max))
        except KeyboardInterrupt:
            print("\nInterrupted; saving the requests collected so far.")

        # --- DNS verdict: must be taken while the resolver hook is still active
        if monitor:
            leak_host = urlsplit(DNS_LEAK_TEST_URL).hostname if DNS_LEAK_TEST_URL else None
            targets = {h.lower() for h in (backend_host, urlsplit(TOR_CHECK_URL).hostname, leak_host,
                                           *(urlsplit(u).hostname for u in IP_ECHO_URLS)) if h}
            # Responses to hostname URLs received over the proxy prove the proxy resolved them.
            proxied_name_responses = (
                (sum(1 for r in rows if r["http_status"] is not None) if is_hostname(backend_host) else 0)
                + (1 if ip_check and ip_check["service"] and is_hostname(urlsplit(ip_check["service"]).hostname) else 0)
                + (1 if tor_check and tor_check["is_tor"] is not None else 0)
            )
            dns_check = monitor.verdict(TOR_PROXY, targets, proxied_name_responses, tor_probe, normal_probe)

    # --- save raw rows and route metadata for the report step
    write_csv(out_dir / f"raw_{route}.csv", rows)
    meta = {
        "route": route,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "backend": sanitize_url(SEARCH_BACKEND_URL),
        "proxy": proxy,
        "repeats": args.repeats,
        "queries": len(queries),
        "ip_check": ip_check,
        "tor_check": tor_check,
        "dns_check": dns_check,
    }
    (out_dir / f"meta_{route}.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    summary = summarize_route(rows)
    print(f"\n{route.upper()} done: {summary['success_count']}/{summary['requests']} succeeded. "
          f"Saved {out_dir / f'raw_{route}.csv'}")
    if dns_check:
        print(f"DNS leak check: {DNS_STATUS_LABELS[dns_check['status']]} - {dns_check['reason']}")
    print("Run the other route, then: python tools/search_route_benchmark.py report")


# ----------------------------------------------------------------------------
# SUMMARIES: per-route statistics and cross-route result quality
# ----------------------------------------------------------------------------
def summarize_route(rows: list[dict]) -> dict:
    """Success rate, failure breakdown, latency and result counts for one route."""
    ok = [r for r in rows if r["outcome"] == "success"]
    counts = {outcome: sum(1 for r in rows if r["outcome"] == outcome)
              for outcome in OUTCOMES if outcome != "success"}
    returned = [r["result_count"] for r in rows if r["result_count"] is not None]
    return {
        "requests": len(rows),
        "success_count": len(ok),
        "success_rate_pct": round(100 * len(ok) / len(rows), 1) if rows else None,
        "failures": counts,
        "ttfb_s": latency_stats([r["ttfb_s"] for r in ok if r["ttfb_s"] is not None]),
        "total_s": latency_stats([r["total_s"] for r in ok if r["total_s"] is not None]),
        "avg_results_returned": round(statistics.fmean(returned), 1) if returned else None,
    }


def jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if a | b else 0.0


def compare_quality(rows_by_route: dict[str, list[dict]]) -> dict:
    """Per-query overlap of the top-N results between VPN and Tor.

    Cross-route overlap uses the first successful run of each route. The
    "self" columns show run-to-run overlap within one route, a noise floor:
    cross-route overlap near the self-overlap means the routes agree about as
    well as a route agrees with itself.
    """
    def per_query(rows):
        grouped: dict[int, dict] = {}
        for r in rows:
            entry = grouped.setdefault(r["query_index"], {"query": r["query"], "runs": [], "counts": []})
            if r["outcome"] == "success":
                entry["runs"].append({normalize_url(u) for u in json.loads(r["top_urls"])})
                entry["counts"].append(r["result_count"])
        return grouped

    vpn, tor = per_query(rows_by_route["vpn"]), per_query(rows_by_route["tor"])

    def self_jaccard(runs):
        pairs = [jaccard(runs[i], runs[j]) for i in range(len(runs)) for j in range(i + 1, len(runs))]
        return round(statistics.fmean(pairs), 3) if pairs else None

    def mean_count(entry):
        return round(statistics.fmean(entry["counts"]), 1) if entry and entry["counts"] else None

    per_query_rows = []
    for index in sorted(set(vpn) | set(tor)):
        v, t = vpn.get(index), tor.get(index)
        a = v["runs"][0] if v and v["runs"] else None
        b = t["runs"][0] if t and t["runs"] else None
        per_query_rows.append({
            "query_index": index,
            "query": (v or t)["query"],
            "vpn_results": mean_count(v),
            "tor_results": mean_count(t),
            "overlap_count": len(a & b) if a is not None and b is not None else None,
            "jaccard": round(jaccard(a, b), 3) if a is not None and b is not None else None,
            "vpn_self_jaccard": self_jaccard(v["runs"]) if v else None,
            "tor_self_jaccard": self_jaccard(t["runs"]) if t else None,
        })

    def mean_of(key):
        values = [q[key] for q in per_query_rows if q[key] is not None]
        return round(statistics.fmean(values), 3) if values else None

    return {
        "top_n": TOP_N,
        "queries_compared": sum(1 for q in per_query_rows if q["overlap_count"] is not None),
        "mean_overlap_count": mean_of("overlap_count"),
        "mean_jaccard": mean_of("jaccard"),
        "mean_vpn_self_jaccard": mean_of("vpn_self_jaccard"),
        "mean_tor_self_jaccard": mean_of("tor_self_jaccard"),
        "per_query": per_query_rows,
    }


# ----------------------------------------------------------------------------
# REPORT: load both routes, print side-by-side tables, write combined files
# ----------------------------------------------------------------------------
def load_rows(path: Path) -> list[dict]:
    """Read a raw CSV back, restoring numeric types."""
    def num(value, cast):
        return cast(value) if value not in ("", None) else None

    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        r["run"], r["query_index"] = int(r["run"]), int(r["query_index"])
        r["http_status"] = num(r["http_status"], int)
        r["bytes"] = num(r["bytes"], int)
        r["result_count"] = num(r["result_count"], int)
        r["ttfb_s"], r["total_s"] = num(r["ttfb_s"], float), num(r["total_s"], float)
    return rows


def fmt(value, suffix: str = "") -> str:
    return "n/a" if value is None else f"{value}{suffix}"


def print_table(headers: list[str], table: list[list], aligns: str | None = None) -> None:
    cells = [[str(c) for c in headers]] + [[str(c) for c in row] for row in table]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    aligns = aligns or "l" + "r" * (len(headers) - 1)

    def line(row):
        return "  ".join(c.ljust(w) if a == "l" else c.rjust(w) for c, w, a in zip(row, widths, aligns))

    print(line(cells[0]))
    print("  ".join("-" * w for w in widths))
    for row in cells[1:]:
        print(line(row))


def describe_resolvers(resolvers: list[dict]) -> str:
    return ", ".join(f"{r.get('ip')} ({r.get('asn') or '?'}, {(r.get('country') or '?').upper()})"
                     for r in resolvers) or "none seen"


def print_dns_details(dns: dict) -> None:
    """Explain the evidence behind the Tor-route DNS leak verdict."""
    print("\n=== DNS leak check (Tor route) ===")
    print(f"Verdict: {DNS_STATUS_LABELS.get(dns['status'], dns['status'])} - {dns['reason']}")
    print(f"  OS resolver lookups during the Tor run : {', '.join(dns['local_lookups']) or 'none'}"
          f"  (hook verified: {dns['hook_verified']}; the proxy's own address is expected)")
    print(f"  Proxied hostname requests answered     : {dns['proxied_name_responses']}")
    print(f"  Resolvers that saw Tor-route lookups   : {describe_resolvers(dns['tor_route_resolvers'])}")
    print(f"  Your normal resolver(s)                : {describe_resolvers(dns['normal_resolvers'])}")
    if dns["shared_resolver_ips"]:
        print(f"  Note: {', '.join(dns['shared_resolver_ips'])} appears on both sides. The exit node "
              "uses the same public DNS provider; the queries still came from the exit, not from you.")
    for side, error in dns["leak_test_errors"].items():
        print(f"  Leak-test service error ({side}): {error}")


def build_report(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    paths = {r: out_dir / f"raw_{r}.csv" for r in ("vpn", "tor")}
    missing = [r for r, p in paths.items() if not p.exists()]
    if missing:
        sys.exit(f"Missing results for: {', '.join(missing)}. Run: "
                 + " and ".join(f"python tools/search_route_benchmark.py {r}" for r in missing))

    rows = {r: load_rows(p) for r, p in paths.items()}
    meta = {r: json.loads((out_dir / f"meta_{r}.json").read_text(encoding="utf-8"))
            if (out_dir / f"meta_{r}.json").exists() else {} for r in rows}
    summaries = {r: summarize_route(rows[r]) for r in rows}
    quality = compare_quality(rows)

    # --- side-by-side summary table
    v, t = summaries["vpn"], summaries["tor"]
    ip = {r: meta[r].get("ip_check") or {} for r in rows}
    dns = meta["tor"].get("dns_check")
    table = [
        ["Requests", v["requests"], t["requests"]],
        ["Success rate (%)", fmt(v["success_rate_pct"]), fmt(t["success_rate_pct"])],
    ]
    for label, key in (("TTFB", "ttfb_s"), ("Total", "total_s")):
        for stat in ("mean", "median", "p95"):
            table.append([f"{label} {stat} (s)", fmt(v[key][stat]), fmt(t[key][stat])])
    for label, outcomes in FAILURE_GROUPS:
        table.append([label, sum(v["failures"][o] for o in outcomes), sum(t["failures"][o] for o in outcomes)])
    table += [
        ["Avg results returned", fmt(v["avg_results_returned"]), fmt(t["avg_results_returned"])],
        ["Exit IP", fmt(ip["vpn"].get("ip")), fmt(ip["tor"].get("ip"))],
        ["Exit country", fmt(ip["vpn"].get("country")), fmt(ip["tor"].get("country"))],
        ["Real IP hidden", fmt(ip["vpn"].get("hides_real_ip")), fmt(ip["tor"].get("hides_real_ip"))],
        ["Confirmed Tor exit", "-", fmt((meta["tor"].get("tor_check") or {}).get("is_tor"))],
        ["DNS leak", "-", DNS_STATUS_LABELS.get(dns.get("status"), "n/a") if dns else "n/a"],
    ]
    print("\n=== VPN vs Tor: side-by-side (latency over successful requests) ===")
    print_table(["Metric", "VPN", "Tor"], table)

    if ip["vpn"].get("ip") and ip["vpn"].get("ip") == ip["tor"].get("ip"):
        print("\nWARNING: both routes report the same exit IP; one route is not doing what you think.")
    if dns:
        print_dns_details(dns)

    # --- result quality table
    print(f"\n=== Result quality: top-{TOP_N} overlap per query ===")
    print_table(
        ["#", "Query", "VPN n", "Tor n", f"Overlap/{TOP_N}", "Jaccard", "VPN self", "Tor self"],
        [[q["query_index"], q["query"][:44], fmt(q["vpn_results"]), fmt(q["tor_results"]),
          fmt(q["overlap_count"]), fmt(q["jaccard"]), fmt(q["vpn_self_jaccard"]), fmt(q["tor_self_jaccard"])]
         for q in quality["per_query"]],
        aligns="rlrrrrrr",
    )
    print(f"\nQueries compared: {quality['queries_compared']}  |  mean overlap: {fmt(quality['mean_overlap_count'])}/{TOP_N}"
          f"  |  mean Jaccard: {fmt(quality['mean_jaccard'])}"
          f"  (self-overlap baseline: VPN {fmt(quality['mean_vpn_self_jaccard'])}, Tor {fmt(quality['mean_tor_self_jaccard'])})")

    # --- combined files: raw CSV (both routes) and summary JSON
    write_csv(out_dir / "benchmark_raw.csv", rows["vpn"] + rows["tor"])
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"timeout_s": REQUEST_TIMEOUT, "top_n": TOP_N,
                   "delay_range_s": list(DELAY_RANGE), "queries": len(QUERIES)},
        "routes": {r: {**summaries[r], "meta": meta[r]} for r in rows},
        "quality": quality,
    }
    (out_dir / "benchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nSaved {out_dir / 'benchmark_raw.csv'} and {out_dir / 'benchmark_summary.json'}")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark web search over a VPN vs Tor.")
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR), help="where results are written")

    sub = parser.add_subparsers(dest="command", required=True)
    for name, text in (("vpn", "run the VPN route (VPN must already be connected)"),
                       ("tor", "run the Tor route via the local SOCKS proxy")):
        p = sub.add_parser(name, parents=[common], help=text)
        p.add_argument("--repeats", type=int, default=REPEATS, help="runs per query")
        p.add_argument("--limit", type=int, help="only use the first N queries (smoke test)")
        p.add_argument("--delay-min", type=float, default=DELAY_RANGE[0])
        p.add_argument("--delay-max", type=float, default=DELAY_RANGE[1])
        p.add_argument("--real-ip", default=os.environ.get("REAL_IP"),
                       help="your real public IP (measured with VPN off) to confirm it is hidden")
    sub.add_parser("report", parents=[common], help="merge both routes into a side-by-side report")
    return parser.parse_args()


def main() -> None:
    # Never crash on printing odd characters from server error messages.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args()
    if args.command == "report":
        build_report(args)
    else:
        run_route(args.command, args)


if __name__ == "__main__":
    main()
