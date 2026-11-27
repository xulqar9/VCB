"""Tests of the network guard that keeps an engine's process on this machine (tools/tts_bench/netguard.py).

The guard replaces functions of the socket module for the whole process, so everything that installs it runs in a
fresh Python process and never in pytest's. No test sends a packet: the refused addresses are from the documentation
range 192.0.2.0/24, the refused names end in .invalid, and the guard stops both before anything leaves the process.
What is allowed stays on the loopback interface."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import namedtuple

import pytest

from tts_bench_support import TOOLS_DIR

from tts_bench import netguard  # noqa: E402


# -- what counts as this machine -----------------------------------------------

@pytest.mark.parametrize("host", [None, "", "localhost", "LocalHost.", "127.0.0.1", "127.8.9.10", "::1", "[::1]",
                                  "::1%lo", "::ffff:127.0.0.1", "0.0.0.0", "::", b"localhost"])
def test_this_machine_is_local(host):
    assert netguard.is_local(host) is True


@pytest.mark.parametrize("host", ["huggingface.co", "bench.invalid", "192.0.2.1", "10.0.0.5", "192.168.1.20",
                                  "2001:db8::1", "::ffff:8.8.8.8", "foo.localhost", "localhost.example.com"])
def test_nothing_else_is_local(host):
    assert netguard.is_local(host) is False          # another computer, even one on the same network


# -- a process with the guard on -----------------------------------------------

PROBE = r'''
import asyncio, json, socket, urllib.request
from tts_bench import netguard

original = (socket.getaddrinfo, socket.socket.connect)
restore = netguard.block_outbound()
guarded = (socket.getaddrinfo, socket.socket.connect)
netguard.block_outbound()()                   # a second installation, and "undoing" it, change nothing
once = (socket.getaddrinfo, socket.socket.connect) == guarded and socket.getaddrinfo.__wrapped__ is original[0]

outcome = {}

def attempt(name, call):
    """"refused" when the guard stopped it, "went through" when nothing did, else what went wrong."""
    try:
        call()
    except Exception as exc:
        stopped = isinstance(exc, netguard.NetworkBlocked) or "blocked by the benchmark" in str(exc)
        outcome[name] = "refused" if stopped else f"{type(exc).__name__}: {exc}"
    else:
        outcome[name] = "went through"

def tcp(address, ex=False):
    sock = socket.socket()
    sock.settimeout(1)
    try:
        if ex:
            sock.connect_ex(address)
        else:
            sock.connect(address)
    finally:
        sock.close()

def udp(address):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.sendto(b"x", address)
    finally:
        sock.close()

async def open_and_close(host, port):
    reader, writer = await asyncio.open_connection(host, port)
    writer.close()

# Something on this machine to talk to: a listening TCP socket (the handshake completes without an accept)
# and a UDP socket.
server = socket.socket()
server.bind(("127.0.0.1", 0))
server.listen(5)
datagrams = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
datagrams.bind(("127.0.0.1", 0))
tcp_address, udp_address = server.getsockname(), datagrams.getsockname()
no_proxy = urllib.request.build_opener(urllib.request.ProxyHandler({}))      # ignore the system's proxy settings

attempt("getaddrinfo", lambda: socket.getaddrinfo("bench-guard.invalid", 443))
attempt("getaddrinfo again", lambda: socket.getaddrinfo("bench-guard.invalid", 443))
attempt("gethostbyname", lambda: socket.gethostbyname("bench-name.invalid"))
attempt("gethostbyname_ex", lambda: socket.gethostbyname_ex("bench-ex.invalid"))
attempt("gethostbyaddr", lambda: socket.gethostbyaddr("192.0.2.1"))
attempt("getnameinfo", lambda: socket.getnameinfo(("192.0.2.2", 80), 0))
attempt("create_connection", lambda: socket.create_connection(("192.0.2.3", 443), timeout=1))
attempt("connect", lambda: tcp(("192.0.2.4", 443)))
attempt("connect_ex", lambda: tcp(("192.0.2.5", 443), ex=True))
attempt("sendto", lambda: udp(("192.0.2.6", 9)))
attempt("asyncio", lambda: asyncio.run(open_and_close("192.0.2.7", 443)))
attempt("urlopen", lambda: no_proxy.open("http://bench-url.invalid/", timeout=2))

attempt("loopback tcp", lambda: tcp(tcp_address))
attempt("loopback udp", lambda: udp(udp_address))
attempt("loopback asyncio", lambda: asyncio.run(open_and_close(*tcp_address)))
attempt("localhost", lambda: socket.getaddrinfo("localhost", tcp_address[1]))

blocked = netguard.blocked_attempts()
restore()
restored = (socket.getaddrinfo, socket.socket.connect) == original
print(json.dumps({"outcome": outcome, "blocked": blocked, "once": once, "restored": restored}))
'''

Guarded = namedtuple("Guarded", "outcome blocked once restored stderr")

REFUSED = ["getaddrinfo", "getaddrinfo again", "gethostbyname", "gethostbyname_ex", "gethostbyaddr", "getnameinfo",
           "create_connection", "connect", "connect_ex", "sendto", "asyncio", "urlopen"]
LOCAL = ["loopback tcp", "loopback udp", "loopback asyncio", "localhost"]
# What was refused, each attempt once and in the order it happened.
BLOCKED = ["look up bench-guard.invalid", "look up bench-name.invalid", "look up bench-ex.invalid",
           "look up 192.0.2.1", "look up 192.0.2.2", "look up 192.0.2.3", "connect to 192.0.2.4:443",
           "connect to 192.0.2.5:443", "send to 192.0.2.6:9", "connect to 192.0.2.7:443",
           "look up bench-url.invalid"]


@pytest.fixture(scope="module")
def guarded():
    env = {**os.environ, "PYTHONPATH": str(TOOLS_DIR)}
    done = subprocess.run([sys.executable, "-W", "ignore", "-c", PROBE], env=env, capture_output=True, text=True,
                          timeout=120)
    assert done.returncode == 0, done.stderr
    return Guarded(**json.loads(done.stdout.strip().splitlines()[-1]), stderr=done.stderr)


def test_a_guarded_process_cannot_look_up_or_reach_anything_away_from_this_machine(guarded):
    assert {name: guarded.outcome[name] for name in REFUSED} == dict.fromkeys(REFUSED, "refused")


def test_a_guarded_process_can_still_talk_to_itself(guarded):
    assert {name: guarded.outcome[name] for name in LOCAL} == dict.fromkeys(LOCAL, "went through")


def test_every_refused_attempt_is_listed_and_logged_once(guarded):
    assert guarded.blocked == BLOCKED                  # the second lookup of bench-guard.invalid is not repeated
    logged = [line.removeprefix("[netguard] blocked: ").split(" (")[0]
              for line in guarded.stderr.splitlines() if line.startswith("[netguard] blocked: ")]
    assert logged == BLOCKED


def test_the_guard_is_installed_once_and_can_be_taken_off(guarded):
    assert guarded.once is True and guarded.restored is True
