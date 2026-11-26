"""Network adapter listing for the VPN-adapter route. No requests/PySocks, so
the settings window can use it before online search is installed."""

from __future__ import annotations

import ipaddress
import re
import socket
import sys

_VPN_HINTS = re.compile(r"vpn|wintun|wireguard|tap|tun|nord|proton|mullvad|express|surfshark|"
                        r"hotspot|openvpn|cyberghost|pia|private internet|windscribe|ipvanish", re.I)


def looks_like_vpn(name: str) -> bool:
    return bool(_VPN_HINTS.search(name or ""))


def list_adapters() -> list[tuple[str, bool, bool]]:
    """(name, is_up, looks_like_vpn) for every adapter with an IPv4 address,
    likely VPNs first. Windows reads this straight from the OS, so it works
    before online search is installed; elsewhere it needs psutil (else [])."""
    if sys.platform == "win32":
        try:
            out = [(a["name"], a["up"], looks_like_vpn(a["name"]) or looks_like_vpn(a["description"]))
                   for a in _win_adapters() if a["ipv4"] and not a["loopback"]]
        except (OSError, ValueError, AttributeError):
            out = _psutil_adapters()
    else:
        out = _psutil_adapters()
    return sorted(out, key=lambda t: (not t[2], not t[1], t[0].lower()))


def _psutil_adapters() -> list[tuple[str, bool, bool]]:
    try:
        import psutil
    except ImportError:
        return []
    stats = psutil.net_if_stats()
    out = []
    for name, addrs in psutil.net_if_addrs().items():
        if not any(a.family == socket.AF_INET for a in addrs):
            continue
        st = stats.get(name)
        out.append((name, bool(st and st.isup), looks_like_vpn(name)))
    return out


def adapter_dns_servers(name: str) -> list[str]:
    """IPv4 DNS servers the VPN configured on this adapter (Windows only;
    [] elsewhere or on any error). Loopback/unspecified entries are skipped:
    they can't be reached from the adapter's address."""
    if sys.platform != "win32":
        return []
    try:
        dns = next((a["dns"] for a in _win_adapters() if a["name"] == name), [])
        return [ip for ip in dns
                if not (ipaddress.ip_address(ip).is_loopback or ipaddress.ip_address(ip).is_unspecified)]
    except (OSError, ValueError, AttributeError):
        return []


def _win_adapters() -> list[dict]:
    """Every adapter from GetAdaptersAddresses: name (FriendlyName, the same
    name psutil uses), description, up, loopback, ipv4 and dns address lists."""
    import ctypes
    from ctypes import wintypes

    class SocketAddress(ctypes.Structure):
        _fields_ = [("lpSockaddr", ctypes.c_void_p), ("iSockaddrLength", ctypes.c_int)]

    class AddrEntry(ctypes.Structure):  # head shared by the unicast and DNS server lists
        pass

    AddrEntry._fields_ = [("Length", wintypes.ULONG), ("Flags", wintypes.DWORD),
                          ("Next", ctypes.POINTER(AddrEntry)), ("Address", SocketAddress)]

    class Adapter(ctypes.Structure):    # leading fields of IP_ADAPTER_ADDRESSES
        pass

    Adapter._fields_ = [("Length", wintypes.ULONG), ("IfIndex", wintypes.DWORD),
                        ("Next", ctypes.POINTER(Adapter)), ("AdapterName", ctypes.c_char_p),
                        ("FirstUnicastAddress", ctypes.POINTER(AddrEntry)),
                        ("FirstAnycastAddress", ctypes.c_void_p),
                        ("FirstMulticastAddress", ctypes.c_void_p),
                        ("FirstDnsServerAddress", ctypes.POINTER(AddrEntry)),
                        ("DnsSuffix", ctypes.c_wchar_p), ("Description", ctypes.c_wchar_p),
                        ("FriendlyName", ctypes.c_wchar_p),
                        ("PhysicalAddress", ctypes.c_ubyte * 8),
                        ("PhysicalAddressLength", wintypes.ULONG), ("Flags", wintypes.ULONG),
                        ("Mtu", wintypes.ULONG), ("IfType", wintypes.DWORD),
                        ("OperStatus", ctypes.c_int)]

    def ipv4s(entry) -> list[str]:
        out = []
        while entry:
            sa = entry.contents.Address
            if sa.lpSockaddr and sa.iSockaddrLength >= 8:
                raw = ctypes.string_at(sa.lpSockaddr, sa.iSockaddrLength)
                if int.from_bytes(raw[:2], "little") == socket.AF_INET:
                    out.append(socket.inet_ntoa(raw[4:8]))
            entry = entry.contents.Next
        return out

    gaa = ctypes.WinDLL("iphlpapi").GetAdaptersAddresses
    gaa.argtypes = [wintypes.ULONG, wintypes.ULONG, ctypes.c_void_p, ctypes.c_void_p,
                    ctypes.POINTER(wintypes.ULONG)]
    gaa.restype = wintypes.ULONG
    skip_anycast_multicast = 0x2 | 0x4
    size = wintypes.ULONG(32 * 1024)
    for _ in range(3):
        buf = ctypes.create_string_buffer(size.value)
        rc = gaa(socket.AF_UNSPEC, skip_anycast_multicast, None, buf, ctypes.byref(size))
        if rc != 111:                   # ERROR_BUFFER_OVERFLOW: retry with the size it asked for
            break
    if rc != 0:
        raise OSError(rc, "GetAdaptersAddresses failed")
    out = []
    p = ctypes.cast(buf, ctypes.POINTER(Adapter))
    while p:
        a = p.contents
        out.append({"name": a.FriendlyName or "", "description": a.Description or "",
                    "up": a.OperStatus == 1, "loopback": a.IfType == 24,
                    "ipv4": ipv4s(a.FirstUnicastAddress), "dns": ipv4s(a.FirstDnsServerAddress)})
        p = a.Next
    return out
