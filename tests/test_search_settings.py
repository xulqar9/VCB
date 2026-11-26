"""Settings window: online search can only be switched on after setup and
the privacy check passed, and the window works without requests installed."""

import tkinter as tk

import pytest

from vcb.search import settings_ui as su


@pytest.fixture(scope="module")
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def owner(root):
    frame = tk.Frame(root)
    yield type("Owner", (), {"root": root, "frame": frame, "row": 0,
                             "backend": tk.StringVar(value="ollama")})()
    frame.destroy()


def test_ticking_opens_setup_instead_of_enabling(owner, monkeypatch):
    opened = []
    monkeypatch.setattr(su, "SetupDialog", lambda section: opened.append(section))
    s = su.SearchSection(owner, {})
    s.enabled.set(True)
    s._toggled()
    assert opened and s.enabled.get() is False
    assert s.updates() == {"VCB_ONLINE_SEARCH": "false"}


def test_accepted_setup_saves_route_and_key(owner):
    s = su.SearchSection(owner, {})
    values = dict(su.FIELDS, VCB_SEARCH_ROUTE="vpn", VCB_SEARCH_VPN_ADAPTER="WG0")
    s.accept(values, "secret")
    out = s.updates()
    assert out["VCB_ONLINE_SEARCH"] == "true"
    assert out["VCB_SEARCH_ROUTE"] == "vpn" and out["VCB_SEARCH_VPN_ADAPTER"] == "WG0"
    assert out["VCB_BRAVE_API_KEY"] == "secret"
    s.enabled.set(False)
    s._toggled()
    assert s.updates()["VCB_ONLINE_SEARCH"] == "false"


def test_invalid_settings_never_start_setup(owner, monkeypatch):
    monkeypatch.setattr(su.subprocess, "Popen", lambda *a, **k: pytest.fail("started"))
    d = su.SetupDialog(su.SearchSection(owner, {}))
    d.backend.set("searxng")
    d.url.set("http://192.168.1.5")
    d.refresh()
    d.run_test()
    assert "https" in d.result.get("1.0", "end")
    d.url.set("https://192.168.1.5")              # the old error goes once the input changes
    assert d.result.get("1.0", "end").strip() == ""
    d.cancel()


def test_each_service_keeps_its_own_key(owner, monkeypatch):
    started = []
    monkeypatch.setattr(su.SetupDialog, "_start",
                        lambda self, args, env: started.append(env))
    s = su.SearchSection(owner, {"VCB_BRAVE_API_KEY": "brave-secret"})
    d = su.SetupDialog(s)
    assert d.key.get() == "brave-secret"
    d.backend.set("searxng")
    d.refresh()
    assert d.key.get() == ""                      # the Brave key isn't shown for SearXNG
    d.url.set("https://search.example.org")
    d.key.set("searx-token")
    d.run_test()
    assert started[0]["VCB_SEARXNG_API_KEY"] == "searx-token"
    assert started[0]["VCB_SEARXNG_AUTH_MODE"] == "cookie"
    assert "VCB_BRAVE_API_KEY" not in started[0]
    d.advanced.set(True)
    d.refresh()
    d.auth_mode.set("bearer")
    assert d.collect()["VCB_SEARXNG_AUTH_MODE"] == "bearer"
    d.backend.set("brave")
    d.refresh()
    assert d.key.get() == "brave-secret"
    d.cancel()

    s.accept(dict(su.FIELDS, VCB_SEARCH_BACKEND="searxng"), "searx-token")
    out = s.updates()
    assert out["VCB_SEARXNG_API_KEY"] == "searx-token"
    assert out["VCB_SEARXNG_AUTH_MODE"] == "cookie"
    assert "VCB_BRAVE_API_KEY" not in out         # left as it is in .env
    s.accept(dict(su.FIELDS, VCB_SEARCH_BACKEND="searxng",
                  VCB_SEARXNG_AUTH_MODE="bearer"), "gateway-token")
    assert s.updates()["VCB_SEARXNG_AUTH_MODE"] == "bearer"


def test_vpn_status_updates_while_the_window_is_open(owner, monkeypatch):
    """Connecting the VPN after opening the window is noticed without
    reopening it, and refreshing the list doesn't undo a passed check."""
    tap = ("HotspotShield Network Adapter", False, True)     # installed, never used
    monkeypatch.setattr(su, "list_adapters", lambda: [tap, ("Wi-Fi", True, False)])
    d = su.SetupDialog(su.SearchSection(owner, {"VCB_SEARCH_ROUTE": "vpn",
                                                "VCB_SEARCH_VPN_ADAPTER": tap[0]}))
    assert d.watch_id is not None
    assert "Not connected" in d.vpn_status.cget("text")
    assert "Wi-Fi" not in str(d.adapter_combo.cget("values"))  # non-VPNs only in advanced
    monkeypatch.setattr(su, "list_adapters",
                        lambda: [("HotspotShield WinTun", True, True), tap, ("Wi-Fi", True, False)])
    d._watch()
    assert d.collect()["VCB_SEARCH_VPN_ADAPTER"] == "HotspotShield WinTun"
    assert "Connected" in d.vpn_status.cget("text")
    d.passed, d.tested = True, (d.collect(), d._key())
    d._changed()
    monkeypatch.setattr(su, "list_adapters",
                        lambda: [("HotspotShield WinTun", False, True), tap, ("Wi-Fi", True, False)])
    d._watch()
    assert "Not connected" in d.vpn_status.cget("text")
    assert str(d.ok_btn.cget("state")) == "normal"
    d.cancel()
    assert d.watch_id is None


def test_adapter_list_refreshes_and_old_labels_keep_the_name(owner, monkeypatch):
    """VPNs re-create their adapter when they reconnect: the dropdown re-reads
    it, and a label from an older listing still means the same adapter."""
    monkeypatch.setattr(su, "list_adapters", lambda: [("WG0", False, True)])
    d = su.SetupDialog(su.SearchSection(owner, {}))
    d.adapter.set(d._adapter_label(("WG0", False, True)))
    monkeypatch.setattr(su, "list_adapters", lambda: [("WG0", True, True)])
    d.adapters = su.list_adapters()
    assert d.collect()["VCB_SEARCH_VPN_ADAPTER"] == "WG0"
    d._reload_adapters()
    assert "(connected" in d.adapter.get()
    d.cancel()
