"""Serves the blind listening page to a browser on this machine, and to nobody else.

Why it exists: VS Code's built-in browser refuses to play audio that a file:// page
points at (it answers those requests with 403), so output/review.html, which plays
fine when an ordinary browser opens it as a file, stays silent there. Served from
http://127.0.0.1 the same page and clips play in every browser.

Only the page and its clip copies are served:
    /  and  /review.html          the page
    /review/clips/<12 hex>.wav    its loudness-matched clip copies (random names)
There are no directory listings and no way to reach the engine folders (their file
names would give the engines away) or anything else on disk: a request path is
matched against these two patterns and is never decoded or joined into a path.
The server listens on the loopback interface only, answers only requests addressed
to 127.0.0.1 or localhost (so a web page on the internet cannot reach it through
DNS rebinding), takes no uploads, and the page it sends forbids itself every
outgoing connection (Content-Security-Policy). Nothing leaves this machine.
"""

from __future__ import annotations

import os
import re
import socketserver
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .review import CLIPS_DIR, REVIEW_HTML

HOST = "127.0.0.1"
DEFAULT_PORT = 8780        # a fixed default: the browser keeps the ratings per address, port included
PORT_TRIES = 20            # a busy port moves the page to the next free one

_CLIP_URL = re.compile(r"/review/clips/([0-9a-f]{12})\.wav")
_RANGE = re.compile(r"bytes=(\d*)-(\d*)")
_LOCAL_NAMES = ("127.0.0.1", "localhost")

# The page may run its own script and style and play its own clips. It may connect nowhere.
_PAGE_POLICY = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; media-src 'self'; "
                "connect-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


def route(path: str) -> tuple[Path, str] | None:
    """What a request path stands for: (file below the results folder, content type), or None."""
    path = path.split("?", 1)[0].split("#", 1)[0]
    if path in ("/", f"/{REVIEW_HTML}"):
        return Path(REVIEW_HTML), "text/html; charset=utf-8"
    clip = _CLIP_URL.fullmatch(path)
    if clip:
        return CLIPS_DIR / f"{clip.group(1)}.wav", "audio/wav"
    return None


def byte_range(header: str | None, size: int) -> tuple[int, int] | None:
    """The first and last byte that a `Range: bytes=a-b` header asks for. None: send the whole file
    (no header, or one this server does not honour). ValueError: the range lies outside the file."""
    match = _RANGE.fullmatch((header or "").strip())
    if not match or match.group(0) == "bytes=-":
        return None
    first, last = match.groups()
    if size == 0:                                   # nothing to take a range from: send the (empty) file
        return None
    if not first:                                   # bytes=-N: the last N bytes
        count = int(last)
        if count == 0:
            raise ValueError("empty suffix range")
        return max(size - count, 0), size - 1
    start = int(first)
    if last and int(last) < start:                  # malformed, so ignored (RFC 9110)
        return None
    if start >= size:
        raise ValueError("range starts after the end of the file")
    return start, min(int(last), size - 1) if last else size - 1


class _Handler(BaseHTTPRequestHandler):
    server_version = "VCBReview"

    def version_string(self) -> str:
        return self.server_version                      # not "VCBReview <python version>": nobody needs to know it

    def do_GET(self) -> None:
        self._answer(with_body=True)

    def do_HEAD(self) -> None:
        self._answer(with_body=False)

    def _not_allowed(self) -> None:
        self._text(405, "Only GET and HEAD are served.", Allow="GET, HEAD")

    do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _not_allowed

    def log_request(self, code="-", size="-") -> None:
        if isinstance(code, int) and code >= 400:       # only what went wrong: a page of clips is 100+ requests
            super().log_request(code, size)

    def _answer(self, with_body: bool) -> None:
        if self.headers.get("Host", "") not in self.server.allowed_hosts:
            return self._text(403, f"Open this page as http://{HOST}:{self.server.server_port}/", with_body)
        if self.path.split("?", 1)[0] == "/favicon.ico":
            return self._send(204, "text/plain", 0)     # the browser asks for it; there is none
        target = route(self.path)
        if target is None:
            return self._text(404, "Not found.", with_body)
        try:
            handle = (self.server.output_dir / target[0]).open("rb")
        except FileNotFoundError:                       # not built yet, or being rebuilt this very moment
            return self._text(404, "Not found.", with_body)
        with handle:
            size = os.fstat(handle.fileno()).st_size    # of the file that is read, whatever happens to the folder
            try:
                span = byte_range(self.headers.get("Range"), size)
            except ValueError:
                return self._text(416, "That range is outside the file.", with_body,
                                  **{"Content-Range": f"bytes */{size}"})
            start, end = span or (0, size - 1)
            handle.seek(start)
            body = handle.read(end - start + 1) if with_body else None
        extra = {"Accept-Ranges": "bytes"}
        if span:
            extra["Content-Range"] = f"bytes {start}-{end}/{size}"
        self._send(206 if span else 200, target[1], end - start + 1, body, **extra)

    def _text(self, code: int, message: str, with_body: bool = True, **headers: str) -> None:
        data = (message + "\n").encode("utf-8")
        self._send(code, "text/plain; charset=utf-8", len(data), data if with_body else None, **headers)

    def _send(self, code: int, content_type: str, length: int, body: bytes | None = None, **headers: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        if code != 204:                                     # a 204 has no body and so no length
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")       # a rebuilt page must never be shown from a cache
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cross-Origin-Resource-Policy", "same-origin")
        if content_type.startswith("text/html"):
            self.send_header("Content-Security-Policy", _PAGE_POLICY)
        for name, value in headers.items():
            self.send_header(name, value)
        self.end_headers()
        if body:
            self.wfile.write(body)


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    request_queue_size = 128                    # a page that preloads its clips opens dozens of connections at once
    allow_reuse_address = os.name != "nt"       # on Windows this option would let a second server share the port
    output_dir: Path
    allowed_hosts: frozenset[str]

    def server_bind(self) -> None:
        socketserver.TCPServer.server_bind(self)    # not HTTPServer's: it also asks the resolver for this machine's name
        self.server_name, self.server_port = HOST, self.server_address[1]

    def handle_error(self, request, client_address) -> None:
        if not isinstance(sys.exc_info()[1], (ConnectionError, TimeoutError)):    # a closed tab is not an error
            super().handle_error(request, client_address)


def make_server(output_dir: Path, port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """A server on the first free port from `port` on (port 0: any free one). ValueError for a port that
    cannot exist; OSError if none is free."""
    if not 0 <= port <= 65535:
        raise ValueError(f"the port must be between 0 and 65535, not {port}")
    ports = [0] if port == 0 else range(port, min(port + PORT_TRIES, 65536))
    error: OSError | None = None
    for number in ports:
        try:
            server = _Server((HOST, number), _Handler)
        except OSError as exc:                  # in use: try the next one
            error = exc
            continue
        server.output_dir = Path(output_dir)
        server.allowed_hosts = frozenset(f"{name}:{server.server_port}" for name in _LOCAL_NAMES)
        return server
    raise OSError(f"no free port from {ports[0]} to {ports[-1]} on {HOST}") from error


def page_url(server: ThreadingHTTPServer) -> str:
    return f"http://{HOST}:{server.server_port}/{REVIEW_HTML}"
