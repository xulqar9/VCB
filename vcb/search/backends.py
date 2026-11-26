"""Search backends as pure data: build a request spec, parse a response.
No networking here (that's vcb.search.net), so this is trivially testable."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from http.cookies import SimpleCookie

from .sanitize import clean_text, domain_of

_TITLE_CHARS = 120
_SNIPPET_CHARS = 300
_CAPTCHA_MARKERS = ("captcha", "are you a robot", "unusual traffic", "cf-chl",
                    "challenge-platform", "verify you are human", "hcaptcha")


@dataclass(frozen=True)
class SearchResult:
    title: str
    domain: str
    snippet: str


@dataclass(frozen=True)
class RequestSpec:
    url: str
    params: dict[str, str]
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Outcome:
    """What a response meant. `kind` is one of: ok, empty, captcha, blocked,
    rate_limited, auth, bad_response, http_error."""
    kind: str
    results: tuple[SearchResult, ...] = ()


def _clean(items, title_key: str, url_key: str, text_key: str, limit: int) -> tuple[SearchResult, ...]:
    out: list[SearchResult] = []
    for item in items if isinstance(items, list) else ():
        if not isinstance(item, dict):
            continue
        title = clean_text(item.get(title_key), _TITLE_CHARS)
        snippet = clean_text(item.get(text_key), _SNIPPET_CHARS)
        if not (title or snippet):
            continue
        out.append(SearchResult(title=title, domain=domain_of(item.get(url_key)), snippet=snippet))
        if len(out) >= limit:
            break
    return tuple(out)


class SearxngBackend:
    name = "searxng"

    def __init__(self, url: str, params: dict[str, str] | None = None, api_key: str = "",
                 auth_mode: str = "cookie"):
        base = url.rstrip("/")
        self.url = base if base.endswith("/search") else base + "/search"
        self.params = dict(params or {})
        # Older setup instructions put private-engine tokens in extra params.
        # Keep accepting those settings, but never put the token in the URL.
        param_token = self.params.pop("tokens", "")
        self.api_key = api_key or param_token
        self.auth_mode = auth_mode

    def _auth_headers(self, include_token: bool = True) -> dict[str, str]:
        headers: dict[str, str] = {}
        if include_token and self.api_key:
            if self.auth_mode == "bearer":
                headers["Authorization"] = f"Bearer {self.api_key}"
            else:
                cookie = SimpleCookie()
                cookie["tokens"] = self.api_key
                headers["Cookie"] = cookie.output(header="", sep=";").strip()
        return headers

    def request(self, query: str, limit: int) -> RequestSpec:
        params = {**self.params, "q": query, "format": "json"}
        headers = {"Accept": "application/json", **self._auth_headers()}
        return RequestSpec(self.url, params, headers)

    def config_request(self, include_token: bool = True) -> RequestSpec:
        """Request SearXNG's engine list, optionally with its private-engine token."""
        config_url = self.url[:-len("/search")] + "/config"
        headers = {"Accept": "application/json", **self._auth_headers(include_token)}
        return RequestSpec(config_url, {}, headers)

    @staticmethod
    def parse_config(body: bytes) -> frozenset[str] | None:
        """Parse visible engine names from SearXNG's /config response."""
        try:
            data = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            return None
        engines = data.get("engines") if isinstance(data, dict) else None
        if not isinstance(engines, list):
            return None
        names = {engine["name"] for engine in engines
                 if isinstance(engine, dict) and isinstance(engine.get("name"), str)}
        return frozenset(names)

    def parse(self, data: dict, limit: int) -> tuple[SearchResult, ...]:
        return _clean(data.get("results"), "title", "url", "content", limit)


class BraveBackend:
    name = "brave"

    def __init__(self, url: str, params: dict[str, str] | None = None, api_key: str = ""):
        self.url = url
        self.params = dict(params or {})
        self.api_key = api_key

    def request(self, query: str, limit: int) -> RequestSpec:
        params = {"safesearch": "moderate", **self.params, "q": query, "count": str(limit)}
        headers = {"Accept": "application/json", "X-Subscription-Token": self.api_key}
        return RequestSpec(self.url, params, headers)

    def parse(self, data: dict, limit: int) -> tuple[SearchResult, ...]:
        web = data.get("web") if isinstance(data, dict) else None
        items = web.get("results") if isinstance(web, dict) else None
        return _clean(items, "title", "url", "description", limit)


def make_backend(cfg):
    """SearchConfig -> backend instance."""
    cls = BraveBackend if cfg.backend == "brave" else SearxngBackend
    if cls is SearxngBackend:
        return cls(cfg.backend_url, cfg.extra_params(), cfg.api_key, cfg.searxng_auth_mode)
    return cls(cfg.backend_url, cfg.extra_params(), cfg.api_key)


def classify(status: int, content_type: str, body: bytes, backend, limit: int) -> Outcome:
    """Map an HTTP response to an Outcome. CAPTCHA / bot-wall pages are only
    detected and reported -- never solved or worked around."""
    head = body[:4096].decode("utf-8", "replace").lower()
    if status == 200 and any(m in head for m in _CAPTCHA_MARKERS) and "json" not in content_type:
        return Outcome("captcha")
    if status in (401, 422):
        return Outcome("auth")
    if status == 403:
        return Outcome("captcha" if any(m in head for m in _CAPTCHA_MARKERS) else "blocked")
    if status == 429:
        return Outcome("rate_limited")
    if status != 200:
        return Outcome("http_error")
    try:
        data = json.loads(body.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        return Outcome("bad_response")
    if not isinstance(data, dict):
        return Outcome("bad_response")
    results = backend.parse(data, limit)
    return Outcome("ok", results) if results else Outcome("empty")
