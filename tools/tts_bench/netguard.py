"""Keeps an engine's process from reaching anything outside this machine.

The benchmark promises that no audio, text or metrics leave the computer. Its own code never opens a
connection, but VCB's engines are third-party code that can: Coqui's model manager, for one, asks
huggingface.co whether the XTTS config changed every time XTTS loads, and downloads the whole model
again when it did. The offline switches (HF_HUB_OFFLINE and friends) do not reach that code, so the
worker installs this guard before it imports an engine. From then on a name lookup, a connection or
a datagram to anything but this machine raises NetworkBlocked. Loopback keeps working, because an
engine may run its own helper server there.

It sits in Python, so it is a safety net and not a firewall: a child process (IndexTTS runs its model
in one) or a C library that opens its own sockets is not covered and relies on the offline variables
the runner puts in that process's environment.
"""

from __future__ import annotations

import functools
import ipaddress
import socket
import sys

__all__ = ["NetworkBlocked", "block_outbound", "blocked_attempts", "is_local"]


class NetworkBlocked(ConnectionRefusedError):
    """Raised instead of reaching a host outside this machine."""


_blocked: dict[str, int] = {}    # what was refused -> how many times
_MISSING = object()
_active = False


def is_local(host) -> bool:
    """True for this machine: no host at all, localhost, or a loopback / unspecified address."""
    if host is None:
        return True
    if isinstance(host, (bytes, bytearray)):
        host = bytes(host).decode("ascii", "replace")
    text = str(host).strip().strip("[]").split("%", 1)[0].rstrip(".").lower()   # "[::1%lo]" -> "::1"
    if text in ("", "localhost"):
        return True
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False                       # a name that is not localhost needs a real lookup
    if address.version == 6 and address.ipv4_mapped is not None:
        address = address.ipv4_mapped
    return address.is_loopback or address.is_unspecified


def blocked_attempts() -> list[str]:
    """What the guard refused so far, each distinct attempt once, in the order it happened."""
    return list(_blocked)


def _refuse(action: str, host, port=None) -> NetworkBlocked:
    """Record an attempt (the first time it is seen it also goes to the engine's log) and build the error."""
    what = f"{action} {host}" + (f":{port}" if port is not None else "")
    if what not in _blocked:
        print(f"[netguard] blocked: {what} (the benchmark keeps engines on this machine; "
              f"--allow-download lifts this)", file=sys.stderr, flush=True)
    _blocked[what] = _blocked.get(what, 0) + 1
    return NetworkBlocked(f"blocked by the benchmark, which keeps engines offline: {what}")


def _check_address(sock, address, action: str) -> None:
    """Refuse an (host, port) address of an internet socket that points away from this machine.
    Unix sockets and the other families carry no host and pass."""
    if (isinstance(address, tuple) and address and sock.family in (socket.AF_INET, socket.AF_INET6)
            and not is_local(address[0])):
        raise _refuse(action, address[0], address[1] if len(address) > 1 else None)


def _guard_lookup(original, parameter: str):
    @functools.wraps(original)
    def guarded(*args, **kwargs):
        host = args[0] if args else kwargs.get(parameter)
        if not is_local(host):
            raise _refuse("look up", host)
        return original(*args, **kwargs)
    return guarded


def _guard_getnameinfo(original):
    @functools.wraps(original)
    def guarded(sockaddr, *args, **kwargs):
        host = sockaddr[0] if isinstance(sockaddr, tuple) and sockaddr else None
        if not is_local(host):
            raise _refuse("look up", host)
        return original(sockaddr, *args, **kwargs)
    return guarded


def _guard_connect(original, action: str):
    @functools.wraps(original)
    def guarded(self, address):
        _check_address(self, address, action)
        return original(self, address)
    return guarded


def _guard_sendto(original):
    @functools.wraps(original)
    def guarded(self, data, *args):
        if args:                                   # sendto(data, address) or sendto(data, flags, address)
            _check_address(self, args[-1], "send to")
        return original(self, data, *args)
    return guarded


def _guard_proactor_connect(original):
    @functools.wraps(original)
    def guarded(self, conn, address):
        _check_address(conn, address, "connect to")
        return original(self, conn, address)
    return guarded


def block_outbound():
    """Install the guard in this process and return `restore()`, which removes it again
    (the worker never restores it; the tests do)."""
    global _active
    if _active:
        return lambda: None                       # already on: leave the first installation alone
    undo: list[tuple] = []

    def patch(owner, name: str, replacement) -> None:
        undo.append((owner, name, vars(owner).get(name, _MISSING)))
        setattr(owner, name, replacement)

    # Name lookups (an `.invalid` host never resolves anyway, but a real name would ask a DNS server).
    for name, parameter in (("getaddrinfo", "host"), ("gethostbyname", "hostname"),
                            ("gethostbyname_ex", "hostname"), ("gethostbyaddr", "ip_address")):
        patch(socket, name, _guard_lookup(getattr(socket, name), parameter))
    patch(socket, "getnameinfo", _guard_getnameinfo(socket.getnameinfo))

    # Connections and datagrams, whatever library opens them (requests, urllib3, aiohttp, asyncio, ssl...).
    patch(socket.socket, "connect", _guard_connect(socket.socket.connect, "connect to"))
    patch(socket.socket, "connect_ex", _guard_connect(socket.socket.connect_ex, "connect to"))
    patch(socket.socket, "sendto", _guard_sendto(socket.socket.sendto))
    if sys.platform == "win32":
        # asyncio's default loop on Windows connects with ConnectEx and never calls socket.connect.
        from asyncio import windows_events
        patch(windows_events.IocpProactor, "connect",
              _guard_proactor_connect(windows_events.IocpProactor.connect))
    _active = True

    def restore() -> None:
        global _active
        for owner, name, original in reversed(undo):
            if original is _MISSING:
                delattr(owner, name)              # it was inherited: removing ours uncovers it again
            else:
                setattr(owner, name, original)
        _active = False

    return restore
