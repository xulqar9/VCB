"""VCB-managed Tor: the torrc it writes, bootstrap tracking, and that a busy
port or missing tor.exe never crashes VCB. A fake tor script stands in."""

import sys
import textwrap
from types import SimpleNamespace

from vcb.search import tor as tormod
from vcb.search.tor import ManagedTor, find_tor_exe, start_managed_tor


def _fake(tmp_path, body):
    script = tmp_path / "fake_tor.py"
    script.write_text(textwrap.dedent(body), encoding="utf-8")
    return [sys.executable, str(script)]


def test_torrc_is_private_and_owned(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "geoip").write_text("")
    t = ManagedTor(9150, base=tmp_path, cmd_prefix=["tor"])
    t.write_torrc()
    rc = (tmp_path / "torrc").read_text()
    assert "SocksPort 127.0.0.1:9150 IsolateSOCKSAuth" in rc
    assert "SafeLogging 1" in rc and "ClientOnly 1" in rc
    assert "__OwningControllerProcess" in rc
    assert "GeoIPFile" in rc
    assert "ControlPort" not in rc
    cmd = t.command()
    assert cmd[cmd.index("-f") + 1] == str(tmp_path / "torrc")
    assert "--defaults-torrc" in cmd


def test_bootstrap_progress(tmp_path):
    cmd = _fake(tmp_path, """
        import time
        for p in (5, 50, 100):
            print(f"Jan 01 [notice] Bootstrapped {p}% (x): y", flush=True)
        time.sleep(30)
    """)
    t = ManagedTor(9150, base=tmp_path, cmd_prefix=cmd)
    t.start()
    try:
        assert t.wait_bootstrapped(20) is True
        assert t.progress == 100
    finally:
        t.stop()
    assert t.proc is None


def test_early_exit_reports_error(tmp_path):
    cmd = _fake(tmp_path, """
        print("Jan 01 [err] Could not bind to 127.0.0.1:9150", flush=True)
    """)
    t = ManagedTor(9150, base=tmp_path, cmd_prefix=cmd)
    t.start()
    assert t.wait_bootstrapped(20) is False
    assert "Could not bind" in t.error


def test_missing_or_busy_never_raises(tmp_path, monkeypatch):
    cfg = SimpleNamespace(tor_proxy="socks5h://127.0.0.1:9")
    monkeypatch.setattr(tormod, "TOR_DIR", tmp_path)
    monkeypatch.setattr(tormod.ManagedTor.__init__, "__kwdefaults__",
                        {"base": tmp_path, "cmd_prefix": None})
    assert find_tor_exe(tmp_path) is None
    assert start_managed_tor(cfg) is None                   # not installed
    monkeypatch.setattr(tormod, "port_in_use", lambda h, p: True)
    assert start_managed_tor(cfg) is None                   # someone else's Tor (verified later)
