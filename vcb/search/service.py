"""Async face of online search for the orchestrator: keeps the private route
verified in the background and runs time-boxed searches.

Only outcomes are logged (kind, route, duration, result count), never the
query, the results or any IP address.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

from .backends import classify, make_backend
from .net import FetchError, PrivateRoute, RouteUnavailable
from .sanitize import clean_query, frame_results

log = logging.getLogger(__name__)

# What the LLM is told when no results are available. Fixed strings: nothing
# from the query or the network ends up in them.
MSG_UNAVAILABLE = ("Online search is unavailable right now (the private connection isn't "
                   "ready, so nothing was sent). Answer from what you already know and "
                   "briefly say you couldn't check online.")
MSG_TIMEOUT = ("The web search took too long and was cancelled. Answer from what you "
               "already know and briefly say you couldn't check online.")
MSG_REFUSED = ("The search service refused the request. Answer from what you already "
               "know and briefly say online search isn't working right now.")
MSG_EMPTY = "The web search found nothing useful. Answer from what you know, or say you don't know."
MSG_BAD_QUERY = "No search was run because the query was empty. Answer without searching."

_REFUSED = {"captcha", "blocked", "rate_limited", "auth", "http_error", "bad_response"}

_RETRY_MIN_S, _RETRY_MAX_S = 5.0, 60.0


class SearchService:
    def __init__(self, cfg, route: PrivateRoute | None = None, backend=None):
        self.cfg = cfg
        self.route = route or PrivateRoute(cfg)
        self.backend = backend or make_backend(cfg)
        self._wake = asyncio.Event()
        self._reported: str | None = None

    @property
    def ready(self) -> bool:
        return self.route.state == "verified"

    async def run_verifier(self) -> None:
        """Verify the route now, then re-check: quickly while it's down
        (backing off), every reverify_s while it's healthy, and right after a
        search found it broken."""
        delay = _RETRY_MIN_S
        while True:
            ok = await asyncio.to_thread(self.route.verify)
            self._report(ok)
            if ok:
                delay = _RETRY_MIN_S
                wait = self.cfg.reverify_s
            else:
                wait, delay = delay, min(delay * 2, _RETRY_MAX_S)
            self._wake.clear()
            try:
                await asyncio.wait_for(self._wake.wait(), wait)
            except asyncio.TimeoutError:
                pass

    def _report(self, ok: bool) -> None:
        key = "ok" if ok else self.route.reason
        if key == self._reported:
            return
        self._reported = key
        if ok:
            extra = f", exit country {self.route.country}" if self.route.country else ""
            log.info("Online search ready via %s%s.", self.cfg.route, extra)
        else:
            log.warning("Online search paused: the %s route isn't available (%s). "
                        "Nothing will be sent until it is.", self.cfg.route, self.route.reason)

    async def search(self, query: str) -> str:
        """Search and return text for the LLM: framed results or a fixed notice."""
        q = clean_query(query)
        if not q:
            return MSG_BAD_QUERY
        t0 = time.perf_counter()
        if not self.ready:
            self._log("unavailable", t0, 0)
            self._wake.set()
            return MSG_UNAVAILABLE
        timeout = self.cfg.effective_timeout_s
        try:
            outcome = await asyncio.wait_for(asyncio.to_thread(self._search_sync, q, timeout),
                                             timeout + 0.5)
        except asyncio.TimeoutError:
            self._log("timeout", t0, 0)
            return MSG_TIMEOUT
        except RouteUnavailable:
            self._log("unavailable", t0, 0)
            self._wake.set()
            return MSG_UNAVAILABLE
        except FetchError as exc:
            self._log(exc.kind, t0, 0)
            if not self.ready:
                self._wake.set()
            return MSG_TIMEOUT if exc.kind == "timeout" else MSG_UNAVAILABLE
        except Exception as exc:                        # never let search break a turn
            self._log(type(exc).__name__, t0, 0)
            return MSG_UNAVAILABLE
        self._log(outcome.kind, t0, len(outcome.results))
        if outcome.kind == "ok":
            return frame_results(outcome.results, max_chars=self.cfg.max_chars)
        if outcome.kind in _REFUSED:
            return MSG_REFUSED
        return MSG_EMPTY

    def _search_sync(self, q: str, timeout: float):
        spec = self.backend.request(q, self.cfg.max_results)
        resp = self.route.fetch(spec, timeout)
        return classify(resp.status, resp.content_type, resp.body, self.backend, self.cfg.max_results)

    def _log(self, kind: str, t0: float, n: int) -> None:
        log.info("web search: %s via %s in %.1fs (%d results)",
                 kind, self.cfg.route, time.perf_counter() - t0, n)

    def close(self) -> None:
        self.route.close()


# ---------------------------------------------------------------------------
# Logging guards
# ---------------------------------------------------------------------------
_QUERY_STRING = re.compile(r"\?[^\s\"']*")
_NOISY = ("urllib3", "requests", "socks", "charset_normalizer")
_URLLIB3_CHILDREN = ("urllib3.connectionpool", "urllib3.connection", "urllib3.util.retry",
                     "urllib3.response", "urllib3.poolmanager", "urllib3.contrib.socks")


class _RedactQueryStrings(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:
            return False
        record.msg, record.args = _QUERY_STRING.sub("?…", msg), ()
        return True


def install_log_guards() -> None:
    """Keep HTTP libraries from logging URLs (which hold the query), even at
    VCB_LOG_LEVEL=DEBUG: cap them at WARNING and strip query strings from
    whatever they still log."""
    for name in _NOISY:
        lg = logging.getLogger(name)
        if lg.level == logging.NOTSET or lg.level < logging.WARNING:
            lg.setLevel(logging.WARNING)
    redact = _RedactQueryStrings()
    for name in _NOISY + _URLLIB3_CHILDREN:
        lg = logging.getLogger(name)
        if not any(isinstance(f, _RedactQueryStrings) for f in lg.filters):
            lg.addFilter(redact)
