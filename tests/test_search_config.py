"""Online search configuration: off by default, and every unsafe setup is
refused before anything touches the network."""

import pytest

from vcb.config import AppConfig, SearchConfig


def _cfg(**kw) -> SearchConfig:
    base = dict(enabled=True, route="tor", tor_mode="managed",
                tor_proxy="socks5h://127.0.0.1:9050", vpn_mode="adapter",
                vpn_adapter="", vpn_proxy="", backend="searxng",
                url="https://search.example.org", params="", api_key_env="", timeout_s=0.0)
    base.update(kw)
    return SearchConfig(**base)


def test_off_by_default():
    cfg = AppConfig()
    assert cfg.search.enabled is False
    assert cfg.search.route == "tor"
    assert cfg.search.backend == "brave"


def test_valid_tor_searxng():
    assert _cfg().problem() is None
    assert _cfg().effective_timeout_s == 6.0
    assert _cfg(route="vpn", vpn_adapter="WinTun").effective_timeout_s == 4.0


@pytest.mark.parametrize("proxy", [
    "socks5://127.0.0.1:9050",      # DNS would leak to the local resolver
    "http://127.0.0.1:9050",
    "socks5h://10.0.0.5:9050",      # Tor proxy must be on this PC
    "socks5h://127.0.0.1",          # no port
    "",
])
def test_tor_proxy_must_be_local_socks5h(proxy):
    assert _cfg(tor_proxy=proxy).problem()


def test_vpn_modes():
    assert "adapter" in _cfg(route="vpn").problem()
    assert _cfg(route="vpn", vpn_adapter="WinTun").problem() is None
    assert _cfg(route="vpn", vpn_mode="proxy", vpn_proxy="socks5://1.2.3.4:1080").problem()
    assert _cfg(route="vpn", vpn_mode="proxy", vpn_proxy="socks5h://vpn.example.net:1080").problem() is None
    assert _cfg(route="vpn", vpn_mode="proxy").proxy == ""
    assert _cfg(route="vpn", vpn_adapter="x").proxy == ""


@pytest.mark.parametrize("url", [
    "http://search.example.org",         # plaintext
    "https://localhost:8888",
    "https://127.0.0.1",
    "https://192.168.1.20",
    "https://searx.local",
    "https://intranet",
    "https://user:pw@search.example.org",
    "ftp://search.example.org",
])
def test_backend_url_rejected(url):
    assert _cfg(url=url).problem()


def test_onion_allows_http_but_only_over_tor():
    onion = "http://abcdefghijklmnop.onion"
    assert _cfg(url=onion).problem() is None
    assert _cfg(url=onion, route="vpn", vpn_adapter="WinTun").problem()


def test_searxng_needs_url():
    assert _cfg(url="").problem()


def test_searxng_auth_mode_defaults_to_cookie_and_validates():
    assert _cfg().searxng_auth_mode == "cookie"
    assert _cfg(searxng_auth_mode="bearer").problem() is None
    assert "token type" in _cfg(searxng_auth_mode="invalid").problem()
    assert "control characters" in _cfg(params='{"tokens": "bad\\nvalue"}').problem()


def test_brave_needs_key_from_named_env(monkeypatch):
    cfg = _cfg(backend="brave", url="", api_key_env="VCB_TEST_BRAVE_KEY")
    assert cfg.backend_url.startswith("https://api.search.brave.com/")
    assert "VCB_TEST_BRAVE_KEY" in cfg.problem()
    monkeypatch.setenv("VCB_TEST_BRAVE_KEY", "k")
    assert cfg.problem() is None
    assert cfg.api_key == "k"


def test_each_backend_reads_only_its_own_key(monkeypatch):
    """A Brave key must never be sent to a SearXNG server (it once was)."""
    from vcb.search.backends import make_backend
    monkeypatch.setenv("VCB_BRAVE_API_KEY", "brave-secret")
    searx = _cfg()
    assert searx.key_env == "VCB_SEARXNG_API_KEY" and searx.api_key == ""
    assert "Authorization" not in make_backend(searx).request("q", 5).headers
    monkeypatch.setenv("VCB_SEARXNG_API_KEY", "searx-token")
    assert make_backend(searx).request("q", 5).headers["Cookie"] == "tokens=searx-token"
    brave = _cfg(backend="brave", url="")
    assert make_backend(brave).request("q", 5).headers["X-Subscription-Token"] == "brave-secret"


def test_params_must_be_json_object():
    assert _cfg(params='{"language": "en"}').extra_params() == {"language": "en"}
    assert _cfg(params="[1, 2]").problem()
    assert _cfg(params="language=en").problem()


def test_env_overrides(monkeypatch):
    monkeypatch.setenv("VCB_ONLINE_SEARCH", "true")
    monkeypatch.setenv("VCB_SEARCH_ROUTE", "VPN")
    monkeypatch.setenv("VCB_SEARCH_TIMEOUT_S", "nonsense")
    cfg = SearchConfig()
    assert cfg.enabled and cfg.route == "vpn" and cfg.timeout_s == 0.0
