"""The only module that opens network connections for online search, and the
only one that imports requests/PySocks. It is imported only when the user
has turned search on.

Every connection goes through the private route, and nothing else is ever
tried:
  * Tor / VPN proxy: socks5h://, so even DNS names are resolved by the proxy.
    With Tor, every session uses its own random SOCKS login, which Tor uses to
    keep it on separate circuits (stream isolation).
  * VPN adapter: every socket is bound to the VPN adapter's address, and names
    are resolved by the VPN's own DNS server (asked from that same address, so
    the query travels inside the tunnel), falling back to DNS-over-HTTPS
    through the tunnel. The system resolver is never asked.
If the route isn't verified, nothing is sent (fail closed).

Never put a URL, query, result or IP address in a log message or exception
text here: requests' own exceptions include the full URL, so only their type
is reported.
"""

from __future__ import annotations

import ipaddress
import json
import logging
import secrets
import socket
import struct
import threading
import time
from dataclasses import dataclass
from urllib.parse import urlsplit

import requests
import socks  # noqa: F401  -- PySocks; fail at import (not mid-search) if it's missing
from requests.adapters import HTTPAdapter
from urllib3 import connection as u3conn
from urllib3 import connectionpool as u3pool
from urllib3.util import connection as u3util

from .adapters import adapter_dns_servers, list_adapters, looks_like_vpn  # noqa: F401  -- re-exported
from .backends import RequestSpec

log = logging.getLogger(__name__)

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:128.0) Gecko/20100101 Firefox/128.0"
MAX_BODY = 1_000_000
TOR_CHECK_URL = "https://check.torproject.org/api/ip"
IP_ECHO_URLS = ("https://ipinfo.io/json", "https://ifconfig.co/json",
                "https://get.geojs.io/v1/ip/geo.json")
# DNS-over-HTTPS by IP literal, so finding the resolver needs no DNS itself.
# Only a fallback: many VPNs block third-party DNS (including DoH) on purpose.
DOH_URLS = ("https://1.1.1.1/dns-query", "https://8.8.8.8/resolve")
DNS_PORT = 53


class RouteUnavailable(Exception):
    """The private route isn't verified; nothing was sent."""


class FetchError(Exception):
    """A request through the route failed. `kind` is safe to log."""

    def __init__(self, kind: str):
        super().__init__(kind)
        self.kind = kind


@dataclass(frozen=True)
class Response:
    status: int
    content_type: str
    body: bytes


# ---------------------------------------------------------------------------
# VPN adapter mode: bind sockets to the adapter, resolve names inside the VPN
# ---------------------------------------------------------------------------
def adapter_ipv4(name: str) -> str:
    """The adapter's IPv4 address. RouteUnavailable if it's missing or down."""
    import psutil

    st = psutil.net_if_stats().get(name)
    if st is None:
        raise RouteUnavailable("adapter_missing")
    if not st.isup:
        raise RouteUnavailable("adapter_down")
    for a in psutil.net_if_addrs().get(name, ()):
        if a.family == socket.AF_INET:
            ip = ipaddress.ip_address(a.address)
            if not (ip.is_loopback or ip.is_link_local):
                return a.address
    raise RouteUnavailable("adapter_no_ipv4")


def _dns_query_a(server: str, host: str, bind_ip: str, timeout: float) -> tuple[str, int]:
    """(IPv4, TTL) for `host` from a plain-DNS server, asked from a UDP socket
    bound to `bind_ip` so the query travels inside the VPN tunnel. A random
    query ID plus the connected socket's peer filter reject stray replies."""
    try:
        labels = host.rstrip(".").encode("idna").split(b".")
    except UnicodeError:
        raise FetchError("dns_failed") from None
    question = b"".join(bytes([len(p)]) + p for p in labels) + b"\0" + struct.pack("!HH", 1, 1)
    qid = secrets.randbits(16)
    edns = b"\0" + struct.pack("!HHIH", 41, 1232, 0, 0)   # OPT: accept 1232-byte UDP replies
    packet = struct.pack("!HHHHHH", qid, 0x0100, 1, 0, 0, 1) + question + edns
    deadline = time.monotonic() + timeout
    resend_at = time.monotonic() + min(1.0, timeout / 2)
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.bind((bind_ip, 0))
            s.connect((server, DNS_PORT))
            s.send(packet)
            while True:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise FetchError("dns_timeout")
                s.settimeout(min(left, max(resend_at - time.monotonic(), 0.05)))
                try:
                    data = s.recv(4096)
                except TimeoutError:
                    if resend_at < deadline and time.monotonic() >= resend_at:
                        s.send(packet)          # one retransmit, like any stub resolver
                        resend_at = deadline
                    continue
                if data[:2] == packet[:2]:
                    break
    except OSError:
        raise FetchError("dns_failed") from None
    try:
        return _parse_a(data, question)
    except (IndexError, struct.error):
        raise FetchError("dns_failed") from None


def _parse_a(data: bytes, question: bytes) -> tuple[str, int]:
    _, flags, qdcount, ancount = struct.unpack("!HHHH", data[:8])
    if not flags & 0x8000 or qdcount != 1:
        raise FetchError("dns_failed")
    if flags & 0x0200:
        raise FetchError("dns_truncated")
    if flags & 0xF == 3:
        raise FetchError("dns_no_answer")       # NXDOMAIN
    if flags & 0xF or data[12:12 + len(question)].lower() != question.lower():
        raise FetchError("dns_failed")
    pos = 12 + len(question)
    for _ in range(ancount):
        while data[pos] and data[pos] < 0xC0:   # skip the owner name
            pos += data[pos] + 1
        pos += 2 if data[pos] >= 0xC0 else 1
        rtype, rclass, ttl, rdlen = struct.unpack("!HHIH", data[pos:pos + 10])
        pos += 10
        if rtype == 1 and rclass == 1 and rdlen == 4:
            return socket.inet_ntoa(data[pos:pos + 4]), ttl
        pos += rdlen
    raise FetchError("dns_no_answer")


class VpnResolver:
    """Name lookups for the VPN-adapter route that never touch the system
    resolver: first the VPN's own DNS servers (from the adapter's address, so
    inside the tunnel), then DNS-over-HTTPS through the tunnel. Many VPNs
    block third-party DNS, so their own server comes first. Answers are cached
    for their TTL (10 s to 5 min)."""

    def __init__(self, bind_ip: str, timeout: float, dns_servers=()):
        self.bind_ip = bind_ip
        self.timeout = timeout
        self.dns_servers = tuple(dns_servers)
        self._cache: dict[str, tuple[float, str]] = {}
        self._lock = threading.Lock()
        self._session = _base_session()
        adapter = _BoundAdapter(bind_ip, resolver=None)
        self._session.mount("https://", adapter)
        self._session.mount("http://", _RefuseAdapter())

    def resolve(self, host: str) -> str:
        try:
            ipaddress.ip_address(host)
            return host
        except ValueError:
            pass
        now = time.monotonic()
        with self._lock:
            hit = self._cache.get(host)
            if hit and hit[0] > now:
                return hit[1]
        for server in self.dns_servers:
            try:
                ip, ttl = _dns_query_a(server, host, self.bind_ip, min(self.timeout, 3.0))
                return self._remember(host, ip, ttl, now)
            except FetchError as exc:
                if exc.kind == "dns_no_answer":
                    raise                       # a real answer: the name doesn't exist
        last = "dns_failed"
        for url in DOH_URLS:
            try:
                r = self._session.get(url, params={"name": host, "type": "A"},
                                      headers={"Accept": "application/dns-json"},
                                      timeout=self.timeout, allow_redirects=False)
                if r.status_code != 200:
                    continue
                data = r.json()
                for ans in data.get("Answer") or ():
                    if ans.get("type") == 1:
                        ip = str(ipaddress.IPv4Address(ans.get("data", "")))
                        return self._remember(host, ip, int(ans.get("TTL", 60)), now)
                last = "dns_no_answer"
            except (requests.RequestException, ValueError, TypeError):
                last = "dns_failed"
        raise FetchError(last)

    def _remember(self, host: str, ip: str, ttl: int, now: float) -> str:
        with self._lock:
            self._cache[host] = (now + min(max(ttl, 10), 300), ip)
        return ip

    def close(self) -> None:
        self._session.close()


def _make_conn_classes(bind_ip: str, resolver: VpnResolver | None):
    """urllib3 connection classes that never touch the system resolver and
    always leave from the VPN adapter's address."""

    def new_conn(self):
        host = self._dns_host
        try:
            ipaddress.ip_address(host)
            ip = host
        except ValueError:
            if resolver is None:
                raise FetchError("dns_refused")     # DoH itself must use IP literals
            ip = resolver.resolve(host)
        try:
            return u3util.create_connection(
                (ip, self.port), self.timeout, source_address=(bind_ip, 0),
                socket_options=self.socket_options)
        except OSError as exc:
            raise u3conn.NewConnectionError(self, "bound connection failed") from exc

    class BoundHTTPS(u3conn.HTTPSConnection):
        _new_conn = new_conn

    class BoundHTTPSPool(u3pool.HTTPSConnectionPool):
        ConnectionCls = BoundHTTPS

    return BoundHTTPSPool


class _BoundAdapter(HTTPAdapter):
    def __init__(self, bind_ip: str, resolver: VpnResolver | None):
        self._pool_cls = _make_conn_classes(bind_ip, resolver)
        super().__init__(max_retries=0)

    def init_poolmanager(self, *args, **kwargs):
        super().init_poolmanager(*args, **kwargs)
        self.poolmanager.pool_classes_by_scheme = {"https": self._pool_cls}

    def proxy_manager_for(self, *args, **kwargs):
        raise FetchError("proxy_refused")


class _RefuseAdapter(HTTPAdapter):
    """Mounted for any scheme the route must not carry."""

    def send(self, request, **kwargs):
        raise FetchError("scheme_refused")


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------
def _base_session() -> requests.Session:
    s = requests.Session()
    s.trust_env = False                 # ignore HTTP(S)_PROXY, NO_PROXY, .netrc
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "en-US,en;q=0.5"})
    return s


def tor_isolated_proxy(proxy: str) -> str:
    """socks5h://127.0.0.1:9050 -> same proxy with a random SOCKS login, which
    Tor uses to put this session on its own circuits."""
    p = urlsplit(proxy)
    user, pw = "vcb" + secrets.token_hex(6), secrets.token_hex(12)
    return f"socks5h://{user}:{pw}@{p.hostname}:{p.port}"


def make_session(cfg, *, allow_http_onion: bool = True):
    """(session, resolver-or-None) for the configured route."""
    s = _base_session()
    if cfg.route == "tor" or cfg.vpn_mode == "proxy":
        proxy = tor_isolated_proxy(cfg.proxy) if cfg.route == "tor" else cfg.proxy
        if not proxy.startswith("socks5h://"):
            raise RouteUnavailable("proxy_not_socks5h")
        s.proxies = {"https": proxy, "http": proxy}
        a = HTTPAdapter(max_retries=0)
        s.mount("https://", a)
        s.mount("http://", a if (cfg.route == "tor" and allow_http_onion) else _RefuseAdapter())
        return s, None
    bind_ip = adapter_ipv4(cfg.vpn_adapter)
    resolver = VpnResolver(bind_ip, cfg.effective_timeout_s, adapter_dns_servers(cfg.vpn_adapter))
    s.mount("https://", _BoundAdapter(bind_ip, resolver))
    s.mount("http://", _RefuseAdapter())
    return s, resolver


def fetch(session, url: str, *, params=None, headers=None, timeout: float) -> Response:
    """GET through the route. No redirects (they could leave the backend), at
    most MAX_BODY bytes, and the whole call bounded by `timeout`."""
    parts = urlsplit(url)
    host = parts.hostname or ""
    if parts.scheme != "https" and not (parts.scheme == "http" and host.endswith(".onion")):
        raise FetchError("scheme_refused")
    deadline = time.monotonic() + timeout
    try:
        r = session.get(url, params=params, headers=headers, timeout=(timeout, timeout),
                        allow_redirects=False, stream=True)
        with r:
            body = bytearray()
            for chunk in r.iter_content(16384):
                body += chunk
                if len(body) > MAX_BODY:
                    raise FetchError("too_large")
                if time.monotonic() > deadline:
                    raise FetchError("timeout")
            return Response(r.status_code, r.headers.get("Content-Type", "").lower(), bytes(body))
    except FetchError:
        raise
    except requests.exceptions.Timeout:
        raise FetchError("timeout") from None
    except requests.exceptions.ProxyError:
        raise FetchError("proxy") from None
    except requests.exceptions.SSLError:
        raise FetchError("tls") from None
    except requests.exceptions.ConnectionError:
        raise FetchError("connection") from None
    except requests.RequestException:
        raise FetchError("request") from None


def _json(resp: Response) -> dict:
    if resp.status != 200:
        raise FetchError(f"http_{resp.status}")
    try:
        data = json.loads(resp.body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise FetchError("bad_json") from None
    if not isinstance(data, dict):
        raise FetchError("bad_json")
    return data


def ip_echo(session, timeout: float, *, strict: bool = False) -> tuple[str | None, str | None]:
    """(exit IP, country) via an IP echo service, or (None, None). With
    `strict`, raise the last FetchError instead (it says *why* nothing got
    out). Callers must never log the IP; the route itself only keeps the
    country."""
    error = None
    for url in IP_ECHO_URLS:
        try:
            data = _json(fetch(session, url, timeout=timeout))
        except FetchError as exc:
            error = exc
            continue
        if data.get("ip"):
            c = data.get("country_code") or data.get("country_iso") or data.get("country")
            return str(data["ip"])[:64], (str(c)[:40] if c else "unknown")
    if strict and error is not None:
        raise error
    return None, None


# ---------------------------------------------------------------------------
# The route: verify, then fetch
# ---------------------------------------------------------------------------
class PrivateRoute:
    """Holds the verified session. `state` is "starting", "verified" or
    "failed"; `reason` is a short code safe to log and show."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.state = "starting"
        self.reason = ""
        self.country: str | None = None
        self.verified_at = 0.0
        self._session = None
        self._resolver: VpnResolver | None = None
        self._lock = threading.Lock()

    def verify(self) -> bool:
        """Build a fresh session and prove it goes through the route. Blocking."""
        timeout = max(self.cfg.effective_timeout_s * 2, 10.0)
        session = resolver = None
        try:
            session, resolver = make_session(self.cfg)
            if self.cfg.route == "tor":
                data = _json(fetch(session, TOR_CHECK_URL, timeout=timeout))
                if data.get("IsTor") is not True:
                    raise RouteUnavailable("not_tor")
                country = None
            else:
                _, country = ip_echo(session, timeout, strict=True)
                if country is None:
                    raise RouteUnavailable("no_egress")
        except (RouteUnavailable, FetchError) as exc:
            if session is not None:
                session.close()
            if resolver is not None:
                resolver.close()
            self._swap(None, None)
            self.state, self.reason = "failed", str(exc) or type(exc).__name__
            return False
        self._swap(session, resolver)
        self.state, self.reason, self.country = "verified", "", country
        self.verified_at = time.monotonic()
        return True

    def _swap(self, session, resolver) -> None:
        with self._lock:
            old, old_res = self._session, self._resolver
            self._session, self._resolver = session, resolver
        for x in (old, old_res):
            if x is not None:
                x.close()

    def fetch(self, spec: RequestSpec, timeout: float) -> Response:
        with self._lock:
            session = self._session
        if self.state != "verified" or session is None:
            raise RouteUnavailable(self.reason or "not_verified")
        if self.cfg.route == "vpn" and self.cfg.vpn_mode == "adapter":
            try:
                ip = adapter_ipv4(self.cfg.vpn_adapter)  # VPN dropped since verify? stop now.
                if self._resolver is not None and ip != self._resolver.bind_ip:
                    raise RouteUnavailable("adapter_changed")   # reconnected: verify again first
            except RouteUnavailable as exc:
                self.state, self.reason = "failed", str(exc)
                raise
        try:
            return fetch(session, spec.url, params=spec.params, headers=spec.headers, timeout=timeout)
        except FetchError as exc:
            if exc.kind in ("proxy", "connection", "dns_failed"):
                self.state, self.reason = "failed", exc.kind   # re-verify before the next search
            raise

    def close(self) -> None:
        self._swap(None, None)
        self.state = "failed"
        self.reason = "closed"

    @property
    def session(self):
        return self._session

    @property
    def resolver(self) -> VpnResolver | None:
        return self._resolver


REASONS = {
    "not_tor": "Traffic through the proxy isn't coming out of the Tor network.",
    "proxy": "Couldn't reach the proxy. Is Tor (or your VPN's proxy) running?",
    "connection": "Couldn't connect through the route.",
    "timeout": "The route is too slow or not connected yet (timed out).",
    "tls": "A secure connection couldn't be verified; something may be intercepting traffic.",
    "adapter_missing": ("The VPN adapter isn't there right now. Connect your VPN (or wait "
                        "for it to finish reconnecting) and try again."),
    "adapter_changed": "The VPN reconnected; the new connection is checked before searching again.",
    "adapter_down": "The VPN adapter is disconnected. Connect your VPN first.",
    "adapter_no_ipv4": "The VPN adapter has no IPv4 address. Connect your VPN first.",
    "no_egress": "Nothing gets out through the VPN adapter. Is the VPN connected?",
    "dns_failed": ("The VPN is connected, but names couldn't be looked up through it "
                   "(neither its own DNS nor DNS over HTTPS answered)."),
    "dns_no_answer": "The check service's name couldn't be found through the VPN's DNS.",
    "proxy_not_socks5h": "The proxy must be socks5h:// so DNS goes through it.",
}


def describe(reason: str) -> str:
    """User-facing text for a route failure code."""
    if reason in REASONS:
        return REASONS[reason]
    if reason.startswith("http_") or reason == "bad_json":
        return "The check service gave an unexpected answer. Try again in a minute."
    return "The private route isn't working."
