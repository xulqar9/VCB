"""Tests of the page server (tools/tts_bench/serve.py) and of the `serve` command.

A real server runs on a free loopback port in a thread of the test process and is asked with http.client, so no
browser, GPU, model or outside network is involved.
"""

from __future__ import annotations

import contextlib
import http.client
import io
import os
import queue
import random
import re
import socket
import socketserver
import subprocess
import sys
import threading
from collections import namedtuple

import pytest

from tts_bench_support import GPUS, TOOLS_DIR, NoMonitor, write_wav

from tts_bench import cli, runner, serve  # noqa: E402
from tts_bench.sentences import load_sentences  # noqa: E402

CLIP = "0123456789ab"
CLIP_PATH = f"/review/clips/{CLIP}.wav"
Result = namedtuple("Result", "code out err")


# -- helpers -------------------------------------------------------------------

def run_cli(*argv) -> Result:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = cli.main([str(a) for a in argv])
    return Result(code, out.getvalue(), err.getvalue())


@contextlib.contextmanager
def serving(folder, port=0):
    """A running page server for `folder`."""
    server = serve.make_server(folder, port)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def fetch(server, path, method="GET", headers=None):
    """(status, lower-case headers, body) of one request to the server."""
    conn = http.client.HTTPConnection(serve.HOST, server.server_port, timeout=10)
    try:
        conn.request(method, path, headers=headers or {})
        reply = conn.getresponse()
        return reply.status, {name.lower(): value for name, value in reply.getheaders()}, reply.read()
    finally:
        conn.close()


@contextlib.contextmanager
def port_in_use():
    """A port that another program is listening on. It lies below the range that Windows gives to programs asking
    for any port (they count up from wherever the last one stopped), so the ports above it are free."""
    start = random.randrange(20000, 40000)
    for port in range(start, start + 200):
        other = socket.socket()
        try:
            other.bind((serve.HOST, port))
        except OSError:                                                     # taken or reserved: try the next
            other.close()
            continue
        break
    else:
        pytest.skip("no port to occupy")
    with other:
        other.listen()
        yield port


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    """A results folder: the page and one clip, and the files that must never be served."""
    out = tmp_path_factory.mktemp("site")
    (out / "review" / "clips").mkdir(parents=True)
    (out / "review.html").write_text("<!doctype html><title>the page</title>\n", encoding="utf-8")
    write_wav(out / "review" / "clips" / f"{CLIP}.wav", seconds=0.1)
    write_wav(out / "review" / "clips" / "abcdef.wav", seconds=0.1)         # hex, but not a name the page ever uses
    (out / "review" / "clips" / "notes.txt").write_text("not a clip", encoding="utf-8")
    (out / "kokoro").mkdir()
    write_wav(out / "kokoro" / "n01_1.wav")                 # an engine's own clip: its folder name names the engine
    (out / "benchmark_raw.csv").write_text("variant,total_s\nkokoro,1.0\n", encoding="utf-8")
    return out


@pytest.fixture(scope="module")
def server(site):
    with serving(site) as running:
        yield running


@pytest.fixture(scope="module")
def real_site(tmp_path_factory):
    """The results folder of a real run of the test engine, with the page that run built."""
    work = tmp_path_factory.mktemp("real_site")
    sentences, out, config = work / "sentences.txt", work / "out", work / "bench.toml"
    sentences.write_text("".join(f"{s.id} | {s.category} | {s.intent} | {s.text}\n"
                                 for s in load_sentences(TOOLS_DIR / "tts_bench" / "sentences.txt")
                                 if s.id in ("n01", "e01", "h01")), encoding="utf-8")
    config.write_text(f"[run]\nruns_per_sentence = 1\nwarmup_runs = 0\ntimeout_s = 10\nsentences = '{sentences}'\n"
                      f"output_dir = '{out}'\n[gpu]\ndefault = \"3090 Ti\"\n[python]\ndefault = '{sys.executable}'\n"
                      "[engines.fake]\noptions = { latency_s = 0.01 }\n", encoding="utf-8")
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(runner, "list_gpus", lambda: GPUS)
        mp.setattr(runner, "VramMonitor", NoMonitor)
        result = run_cli("run", "--engines", "fake", "--modes", "default", "--config", config)
    assert result.code == 0 and (out / "review.html").is_file(), result.out + result.err
    return out


# -- what is served ------------------------------------------------------------

@pytest.mark.parametrize("path", ["/", "/review.html", "/review.html?seed=3"])
def test_the_page_is_served_at_its_own_address_and_at_the_root(server, site, path):
    status, headers, body = fetch(server, path)
    assert status == 200 and headers["content-type"] == "text/html; charset=utf-8"
    assert body == (site / "review.html").read_bytes()


def test_a_clip_is_served_as_audio(server, site):
    status, headers, body = fetch(server, CLIP_PATH)
    assert status == 200 and headers["content-type"] == "audio/wav" and headers["accept-ranges"] == "bytes"
    assert body == (site / "review" / "clips" / f"{CLIP}.wav").read_bytes() and int(headers["content-length"]) == len(body)


@pytest.mark.parametrize("path", [
    "/benchmark_raw.csv", "/kokoro/n01_1.wav", "/review/clips/", "/review/clips", "/review/clips/notes.txt",
    "/review/clips/abcdef.wav",                                         # a file in the clips folder with another name
    "/review/clips/ffffffffffff.wav",                                   # no such clip
    f"/review/clips/{CLIP.upper()}.wav",                                # Windows would open the file under this name
    f"/review/clips/{CLIP}.WAV", f"/review/clips/{CLIP}.wav/", f"/review/clips/{CLIP}.wav.",
    "/review/clips/../../benchmark_raw.csv", "/%2e%2e/benchmark_raw.csv", "/review/clips/%30123456789ab.wav",
    "/review/clips/..%5c..%5cbenchmark_raw.csv", "/review.html/", "/Review.html",
])
def test_nothing_else_is_served(server, path):
    status, _, body = fetch(server, path)
    assert status == 404 and b"kokoro" not in body and b"the page" not in body


def test_the_favicon_request_is_answered_without_a_body(server):
    status, headers, body = fetch(server, "/favicon.ico")
    assert status == 204 and body == b"" and "content-length" not in headers      # a 204 has none (RFC 9110)


def test_responses_are_never_cached_and_the_page_may_connect_nowhere(server):
    _, page, _ = fetch(server, "/review.html")
    _, clip, _ = fetch(server, CLIP_PATH)
    for headers in (page, clip):
        assert headers["cache-control"] == "no-store" and headers["x-content-type-options"] == "nosniff"
        assert headers["referrer-policy"] == "no-referrer" and headers["cross-origin-resource-policy"] == "same-origin"
    policy = dict(directive.split(None, 1) for directive in page["content-security-policy"].split("; "))
    for name in ("default-src", "connect-src", "form-action", "frame-ancestors", "base-uri"):
        assert policy[name] == "'none'"
    sources = {token for value in policy.values() for token in value.split()}
    assert sources <= {"'none'", "'self'", "'unsafe-inline'"}               # no host, no scheme, no wildcard


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def test_the_server_takes_nothing_that_is_sent_to_it(server, method):
    status, headers, _ = fetch(server, "/review.html", method)
    assert status == 405 and headers["allow"] == "GET, HEAD"


def test_head_gives_the_headers_of_a_get_and_no_body(server):
    got = fetch(server, CLIP_PATH)
    # a raw socket: http.client never reads a body after HEAD, so it could not tell whether one was sent
    with socket.create_connection((serve.HOST, server.server_port), timeout=10) as conn:
        conn.sendall(f"HEAD {CLIP_PATH} HTTP/1.0\r\nHost: {serve.HOST}:{server.server_port}\r\n\r\n".encode())
        reply = b""
        while chunk := conn.recv(65536):
            reply += chunk
    head, _, body = reply.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.0 200") and body == b""
    assert f"Content-Length: {got[1]['content-length']}".encode() in head


def test_the_server_does_not_say_which_python_runs_it(server):
    assert fetch(server, "/review.html")[1]["server"] == "VCBReview"


def test_only_requests_that_went_wrong_are_logged(server, capsys):
    # a page of 90 clips is 90 requests: the log is for the ones that failed
    fetch(server, "/review.html")
    fetch(server, CLIP_PATH)
    assert capsys.readouterr().err == ""
    fetch(server, "/nothing")
    assert '"GET /nothing HTTP/1.1" 404' in capsys.readouterr().err         # as the client (http.client) wrote it


def test_a_closed_tab_is_not_reported_as_an_error_but_a_mistake_is(server, capsys):
    for closed in (ConnectionResetError(), BrokenPipeError(), ConnectionAbortedError(), TimeoutError()):
        try:
            raise closed
        except OSError:
            server.handle_error(None, (serve.HOST, 1))
    assert capsys.readouterr().err == ""
    try:
        raise ValueError("a real mistake")
    except ValueError:
        server.handle_error(None, (serve.HOST, 1))
    assert "a real mistake" in capsys.readouterr().err


# -- ranges: a browser seeks in a clip by asking for part of it ----------------

@pytest.mark.parametrize("header, size, expected", [
    (None, 100, None), ("", 100, None),
    ("bytes=10-19", 100, (10, 19)),
    ("bytes=10-", 100, (10, 99)),
    ("bytes=-4", 100, (96, 99)),
    ("bytes=-500", 100, (0, 99)),                       # a tail longer than the file is the whole file
    ("bytes=90-500", 100, (90, 99)),                    # an end beyond the file stops at its end
    ("bytes=0-0", 100, (0, 0)),
    ("bytes=99-", 100, (99, 99)),
    ("bytes=0-1,5-6", 100, None),                       # several ranges: not honoured, the whole file is sent
    ("bytes=5-3", 100, None), ("bytes=-", 100, None), ("bytes=abc", 100, None), ("items=0-5", 100, None),
    ("bytes=0-5", 0, None),                             # an empty file has no parts
    ("bytes=-5", 0, None),
])
def test_a_range_header_names_the_part_of_the_file_to_send(header, size, expected):
    assert serve.byte_range(header, size) == expected


@pytest.mark.parametrize("header", ["bytes=100-", "bytes=250-300", "bytes=-0"])
def test_a_range_outside_the_file_is_refused(header):
    with pytest.raises(ValueError):
        serve.byte_range(header, 100)


def test_the_server_answers_range_requests_the_way_a_browser_needs(server, site):
    data = (site / "review" / "clips" / f"{CLIP}.wav").read_bytes()
    status, headers, body = fetch(server, CLIP_PATH, headers={"Range": "bytes=10-19"})
    assert status == 206 and body == data[10:20] and headers["content-length"] == "10"
    assert headers["content-range"] == f"bytes 10-19/{len(data)}"
    status, headers, _ = fetch(server, CLIP_PATH, headers={"Range": f"bytes={len(data)}-"})
    assert status == 416 and headers["content-range"] == f"bytes */{len(data)}"
    status, _, body = fetch(server, CLIP_PATH, headers={"Range": "bytes=0-1,5-6"})
    assert status == 200 and body == data


# -- who may ask, and which port -----------------------------------------------

def test_the_server_listens_on_the_loopback_interface_only(server):
    assert server.server_address[0] == "127.0.0.1" == server.socket.getsockname()[0]


def test_only_requests_addressed_to_this_machine_are_answered(server):
    port = server.server_port
    for host in (f"127.0.0.1:{port}", f"localhost:{port}"):
        assert fetch(server, "/review.html", headers={"Host": host})[0] == 200
    # a web page on the internet whose name points at 127.0.0.1 (DNS rebinding) arrives under its own name
    for host in ("evil.example", f"evil.example:{port}", "127.0.0.1", f"127.0.0.1.evil.example:{port}", ""):
        status, _, body = fetch(server, "/review.html", headers={"Host": host})
        assert status == 403 and b"the page" not in body


def test_a_busy_port_moves_the_page_to_the_next_free_one(site):
    with port_in_use() as busy, serving(site, busy) as moved:
        assert busy < moved.server_port < busy + serve.PORT_TRIES
        assert fetch(moved, "/review.html")[0] == 200


def test_a_second_server_never_shares_the_port_of_a_running_one(server, site):
    # the usual address-reuse option would, on Windows, let both listen on one port and a browser reach either
    with serving(site, server.server_port) as second:
        assert second.server_port != server.server_port


def test_no_free_port_is_reported(site, monkeypatch):
    monkeypatch.setattr(serve, "PORT_TRIES", 1)
    with port_in_use() as busy, pytest.raises(OSError, match="no free port"):
        serve.make_server(site, busy)


def test_the_search_for_a_free_port_stops_at_the_last_port_there_is(site):
    other = socket.socket()
    try:
        other.bind((serve.HOST, 65535))
    except OSError:
        other.close()
        pytest.skip("port 65535 is not free here")
    with other, pytest.raises(OSError, match="no free port from 65535 to 65535"):       # not an OverflowError at 65536
        other.listen()
        serve.make_server(site, 65535)


@pytest.mark.parametrize("port", [-1, 65536, 70000])
def test_a_port_that_cannot_exist_is_refused(site, port):
    with pytest.raises(ValueError, match="between 0 and 65535"):
        serve.make_server(site, port)


# -- the page a real run builds ------------------------------------------------

def test_every_clip_the_real_page_links_to_is_served(real_site):
    with serving(real_site) as server:
        status, _, body = fetch(server, "/review.html")
        assert status == 200
        urls = re.findall(r'src="(review/clips/[^"]+)"', body.decode("utf-8"))
        assert len(urls) == len(list((real_site / "review" / "clips").glob("*.wav"))) == 3
        for url in urls:
            status, headers, body = fetch(server, "/" + url)
            assert status == 200 and headers["content-type"] == "audio/wav" and body[:4] == b"RIFF"


def test_the_page_uses_nothing_that_its_security_policy_blocks(real_site):
    """Under `serve` the page may only run its own script and play its own clips: anything else would fail silently."""
    page = (real_site / "review.html").read_text(encoding="utf-8")
    for needle in ("url(", "data:", "<img", "<link", "<iframe", "<embed", "<object", "@import", "fetch(",
                   "XMLHttpRequest", "WebSocket", "EventSource", "sendBeacon", "new Worker"):
        assert needle not in page


# -- the command ---------------------------------------------------------------

@pytest.fixture()
def stop_at_once(monkeypatch):
    """Runs `serve` in this process, which returns the moment the server starts, as when Ctrl+C is pressed.
    Gives the servers it made and the addresses it opened in a browser."""
    made, opened, real = [], [], cli.make_server

    def make(*args):
        made.append(real(*args))
        return made[-1]

    def ctrl_c(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(cli, "make_server", make)
    monkeypatch.setattr(socketserver.BaseServer, "serve_forever", ctrl_c)
    monkeypatch.setattr(cli.webbrowser, "open", opened.append)
    return made, opened


def test_serve_needs_a_page_to_serve(tmp_path):
    result = run_cli("serve", "--output", tmp_path, "--port", 0)
    assert result.code == 1 and f"No listening page at {tmp_path / 'review.html'}" in result.out
    assert "`tts_benchmark.py review`" in result.out                         # and how to get one


def test_serve_prints_the_address_and_stops_cleanly_on_ctrl_c(site, stop_at_once):
    made, opened = stop_at_once
    result = run_cli("serve", "--output", site, "--port", 0, "--open")
    url = f"http://127.0.0.1:{made[0].server_port}/review.html"
    assert result.code == 0 and f"Blind listening page: {url}" in result.out and "Ctrl+C" in result.out
    assert opened == [url]
    assert made[0].socket.fileno() == -1                                    # the port was given back


def test_serve_does_not_open_a_browser_unless_asked(site, stop_at_once):
    result = run_cli("serve", "--output", site, "--port", 0)
    assert result.code == 0 and stop_at_once[1] == []


def test_serve_says_when_it_had_to_take_another_port(site, stop_at_once):
    with port_in_use() as busy:
        result = run_cli("serve", "--output", site, "--port", busy)
    assert f"Port {busy} is busy, so this is port {stop_at_once[0][0].server_port}" in result.out


def test_serve_refuses_a_port_that_cannot_exist(site):
    result = run_cli("serve", "--output", site, "--port", 70000)
    assert result.code == 2 and "between 0 and 65535" in result.err


def test_serve_reports_that_no_port_is_free(site, monkeypatch):
    monkeypatch.setattr(serve, "PORT_TRIES", 1)
    with port_in_use() as busy:
        result = run_cli("serve", "--output", site, "--port", busy)
    assert result.code == 1 and "error: no free port" in result.err and "Blind listening page" not in result.out


def test_the_command_as_a_user_starts_it_prints_an_address_that_answers(site):
    # without PYTHONUNBUFFERED the first line only arrives if the command flushes it, as it must when output is piped
    env = {name: value for name, value in os.environ.items() if name != "PYTHONUNBUFFERED"}
    proc = subprocess.Popen([sys.executable, str(TOOLS_DIR / "tts_benchmark.py"), "serve", "--output", str(site),
                             "--port", "0"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env,
                            encoding="utf-8", errors="replace")
    lines: queue.Queue = queue.Queue()
    threading.Thread(target=lambda: [lines.put(line) for line in proc.stdout], daemon=True).start()
    try:
        first = lines.get(timeout=30)
        found = re.search(r"http://127\.0\.0\.1:(\d+)/review\.html", first)
        assert found, first
        conn = http.client.HTTPConnection(serve.HOST, int(found.group(1)), timeout=10)
        conn.request("GET", "/review.html")
        reply = conn.getresponse()
        assert reply.status == 200 and reply.read() == (site / "review.html").read_bytes()
        conn.close()
    finally:
        proc.kill()
        proc.wait(10)
        proc.stdout.close()
