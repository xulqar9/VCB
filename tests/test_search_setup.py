"""Online search setup: rollback on failure/cancel, Tor integrity checks, and
the self-test's DNS-leak verdict and IP masking. No real network or pip."""

import hashlib
import io
import json
import socket
import tarfile
import threading
from types import SimpleNamespace

import pytest

from vcb.search import install, selftest


def _cfg(**kw):
    base = dict(route="tor", tor_mode="managed")
    base.update(kw)
    return SimpleNamespace(**base)


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setattr(install, "RUNTIME", tmp_path)
    monkeypatch.setattr(install, "TOR_DIR", tmp_path / "tor")
    monkeypatch.setattr(install, "STAMP", tmp_path / "search_installed.json")
    return tmp_path


def _tarball(members: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for name, data in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def _serve(monkeypatch, blob: bytes, cancel_after: bool = False):
    def fake_download(url, dest, cancel, progress):
        dest.write_bytes(blob)
        if cancel_after:
            cancel.set()
    monkeypatch.setattr(install, "_download", fake_download)
    monkeypatch.setattr(install, "TOR_SHA256", hashlib.sha256(blob).hexdigest())


def test_failed_package_install_rolls_back(sandbox, monkeypatch):
    snaps = iter([{"urllib3": "2.7.0"}, {"urllib3": "2.6.0", "pysocks": "1.7.1"}])
    monkeypatch.setattr(install, "_snapshot", lambda: next(snaps))
    monkeypatch.setattr(install, "missing_modules", lambda: ["socks"])
    monkeypatch.setattr(install, "_install_packages",
                        lambda c, p: (_ for _ in ()).throw(install.SetupError("boom")))
    restored = {}
    monkeypatch.setattr(install, "_restore_packages",
                        lambda added, changed, p: restored.update(added=added, changed=changed))
    with pytest.raises(install.SetupError):
        install.ensure(_cfg(route="vpn"))
    assert restored == {"added": ["pysocks"], "changed": {"urllib3": "2.7.0"}}
    assert not install.STAMP.exists()


def test_tor_installs_and_verifies(sandbox, monkeypatch):
    monkeypatch.setattr(install, "missing_modules", lambda: [])
    _serve(monkeypatch, _tarball({"tor/tor.exe": b"MZ", "data/geoip": b""}))
    install.ensure(_cfg())
    assert (install.TOR_DIR / "bundle" / "tor" / "tor.exe").exists()
    assert install.STAMP.exists()
    assert not list(sandbox.glob("tor-setup-*"))          # temp dir cleaned up


def test_tampered_tor_is_rejected(sandbox, monkeypatch):
    monkeypatch.setattr(install, "missing_modules", lambda: [])
    _serve(monkeypatch, _tarball({"tor/tor.exe": b"MZ"}))
    monkeypatch.setattr(install, "TOR_SHA256", "0" * 64)
    with pytest.raises(install.SetupError, match="integrity"):
        install.ensure(_cfg())
    assert not install.TOR_DIR.exists()


def test_tor_archive_cannot_escape(sandbox, monkeypatch):
    monkeypatch.setattr(install, "missing_modules", lambda: [])
    _serve(monkeypatch, _tarball({"../evil.txt": b"x", "tor/tor.exe": b"MZ"}))
    with pytest.raises(tarfile.FilterError):
        install.ensure(_cfg())
    assert not (sandbox / "evil.txt").exists()
    assert not install.TOR_DIR.exists()


def test_cancel_rolls_back_tor(sandbox, monkeypatch):
    monkeypatch.setattr(install, "missing_modules", lambda: [])
    _serve(monkeypatch, _tarball({"tor/tor.exe": b"MZ"}), cancel_after=True)
    with pytest.raises(install.Cancelled):
        install.ensure(_cfg(), threading.Event())
    assert not install.TOR_DIR.exists()


def test_remove_only_uninstalls_what_setup_added(sandbox, monkeypatch):
    (install.TOR_DIR / "bundle").mkdir(parents=True)
    install._write_stamp({"packages_added": ["pysocks"]})
    ran = []
    monkeypatch.setattr(install, "_run", lambda cmd, *a: ran.append(cmd))
    install.remove()
    assert len(ran) == 1 and ran[0][-1] == "pysocks"
    assert not install.TOR_DIR.exists() and not install.STAMP.exists()


def test_mask_ip_hides_host_part():
    assert selftest.mask_ip("185.220.101.4") == "185.220.x.x"
    assert selftest.mask_ip("2a0b:f4c2::1") == "2a0b:f4c2:…"
    assert "101" not in selftest.mask_ip("185.220.101.4")
    assert selftest.mask_ip(None) == ""


def test_dns_monitor_flags_hostnames_not_literals(monkeypatch):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [])
    with selftest.DnsMonitor() as mon:
        assert mon.hook_works()
        socket.getaddrinfo("127.0.0.1", 9050)
        socket.getaddrinfo("proxy.example", 1080)
        socket.getaddrinfo("check.torproject.org", 443)
    assert mon.leaked({"proxy.example"}) == ["check.torproject.org"]


@pytest.mark.parametrize("hook,leaked,proof,want", [
    (True, ["x.com"], True, "leak"),
    (True, [], True, "no_leak"),
    (True, [], False, "inconclusive"),
    (False, [], True, "inconclusive"),
])
def test_dns_verdict(hook, leaked, proof, want):
    assert selftest.dns_verdict(hook, leaked, proof)[0] == want


def test_searxng_token_check_requires_newly_visible_engine():
    from vcb.search.backends import SearxngBackend

    class Route:
        def __init__(self, names_without, names_with):
            self.names = iter((names_without, names_with))
            self.specs = []

        def fetch(self, spec, timeout):
            from types import SimpleNamespace
            self.specs.append(spec)
            names = next(self.names)
            body = json.dumps({"engines": [{"name": name} for name in names]}).encode()
            return SimpleNamespace(status=200, body=body)

    backend = SearxngBackend("https://search.example.org", api_key="private-token")
    accepted = Route({"public"}, {"public", "private"})
    assert selftest._check_searxng_token(accepted, backend, 5) is None
    assert "Cookie" not in accepted.specs[0].headers
    assert accepted.specs[1].headers["Cookie"] == "tokens=private-token"

    rejected = Route({"public"}, {"public"})
    assert selftest._check_searxng_token(rejected, backend, 5) == "auth"


def test_summarize_explains_rejected_searxng_token():
    report = {
        "backend": "searxng",
        "steps": {
            "route": {"ok": True, "detail": "Traffic leaves through Tor."},
            "exit": {},
            "dns": {},
            "backend": {"kind": "auth"},
        },
    }
    summary = selftest.summarize(report)
    assert any("SearXNG server or its gateway didn't accept this token" in line
               for line in summary)


def test_selftest_refuses_bad_config():
    cfg = SimpleNamespace(route="tor", problem=lambda: "The proxy must be socks5h://")
    report = selftest.run(cfg)
    assert report["ok"] is False and "socks5h" in report["error"]
    assert selftest.summarize(report)[0].startswith("✗")


def test_selftest_reports_leak(monkeypatch):
    """A lookup of a real hostname during the checks fails the whole test."""
    def leaky_checks(cfg, *a):
        socket.getaddrinfo("search.example", 443)
        return {"steps": {"route": {"ok": True, "detail": "ok"},
                          "exit": {"ip_masked": "1.2.x.x", "country": "DE", "is_local": False},
                          "dns": {"resolvers": []}, "backend": {"kind": "ok", "results": 5}}}
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [])
    monkeypatch.setattr(selftest, "_checks", leaky_checks)
    cfg = SimpleNamespace(route="vpn", tor_mode="managed", proxy="", problem=lambda: None)
    report = selftest.run(cfg)
    assert report["steps"]["dns"]["status"] == "leak"
    assert report["ok"] is False
    assert any("search.example" in line for line in selftest.summarize(report))
