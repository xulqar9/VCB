"""A minimal SOCKS5 server for tests. It records what clients ask for (auth
and CONNECT target, including whether the target arrived as a hostname or
as an IP address), then refuses the connection. Nothing leaves this PC."""

from __future__ import annotations

import socket
import struct
import threading


class FakeSocks5:
    def __init__(self):
        self.requests: list[dict] = []
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self.port = self._sock.getsockname()[1]
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    @property
    def url(self) -> str:
        return f"socks5h://127.0.0.1:{self.port}"

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            threading.Thread(target=self._handle, args=(conn,), daemon=True).start()

    @staticmethod
    def _recv(conn, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = conn.recv(n - len(buf))
            if not chunk:
                raise ConnectionError
            buf += chunk
        return buf

    def _handle(self, conn) -> None:
        rec: dict = {}
        try:
            with conn:
                conn.settimeout(5)
                _, n = self._recv(conn, 2)
                methods = self._recv(conn, n)
                if 2 in methods:                       # username/password
                    conn.sendall(b"\x05\x02")
                    _, ulen = self._recv(conn, 2)
                    user = self._recv(conn, ulen).decode()
                    (plen,) = self._recv(conn, 1)
                    rec["auth"] = (user, self._recv(conn, plen).decode())
                    conn.sendall(b"\x01\x00")
                else:
                    conn.sendall(b"\x05\x00")
                _, _, _, atyp = self._recv(conn, 4)
                if atyp == 3:
                    (ln,) = self._recv(conn, 1)
                    rec["host"], rec["atyp"] = self._recv(conn, ln).decode(), "domain"
                elif atyp == 1:
                    rec["host"], rec["atyp"] = socket.inet_ntoa(self._recv(conn, 4)), "ipv4"
                else:
                    rec["host"] = socket.inet_ntop(socket.AF_INET6, self._recv(conn, 16))
                    rec["atyp"] = "ipv6"
                (rec["port"],) = struct.unpack("!H", self._recv(conn, 2))
                self.requests.append(rec)
                conn.sendall(b"\x05\x05\x00\x01" + b"\x00" * 6)   # 0x05 = refused
        except (OSError, ConnectionError, ValueError):
            if rec and rec not in self.requests:
                self.requests.append(rec)

    def close(self) -> None:
        self._sock.close()
