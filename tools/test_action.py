"""Standalone avatar-bridge test server: fire body actions into Unreal without
running the full VCB pipeline or an LLM.

It mimics VcbAvatar's bridge: listens on 127.0.0.1:8765, sends the audio_format
handshake when Unreal connects, then sends an "action" message on a timer (or
whatever names you pass on the command line).

Usage (from d:\\VCB):
    python tools/test_action.py                 # fires "wave" every 4s
    python tools/test_action.py highfive         # fires "highfive" every 4s
    python tools/test_action.py wave crouch bow   # cycles through the list
    python tools/test_action.py --interval 2 wave

Start this, then press Play in Unreal. Watch the character perform the action.
"""

import argparse
import asyncio
import json
import struct

HOST = "127.0.0.1"
PORT = 8765
_JSON = 0
_HEADER = struct.Struct(">BI")


def _frame(obj: dict) -> bytes:
    payload = json.dumps(obj).encode("utf-8")
    return _HEADER.pack(_JSON, len(payload)) + payload


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("actions", nargs="*", default=["wave"],
                    help="action name(s) to fire; cycles if more than one")
    ap.add_argument("--interval", type=float, default=4.0,
                    help="seconds between actions")
    args = ap.parse_args()
    actions = args.actions or ["wave"]

    async def on_client(reader: asyncio.StreamReader,
                        writer: asyncio.StreamWriter) -> None:
        peer = writer.get_extra_info("peername")
        print(f"Unreal connected from {peer}")
        # Handshake so the client is happy (it expects audio_format on connect).
        writer.write(_frame({"type": "audio_format", "sample_rate": 24000,
                             "channels": 1, "dtype": "float32"}))
        writer.write(_frame({"type": "state", "state": "idle"}))
        await writer.drain()
        i = 0
        try:
            while not writer.is_closing():
                name = actions[i % len(actions)]
                i += 1
                print(f"-> action: {name}")
                writer.write(_frame({"type": "action", "name": name}))
                await writer.drain()
                await asyncio.sleep(args.interval)
        except (ConnectionResetError, BrokenPipeError):
            print("Unreal disconnected")

    server = await asyncio.start_server(on_client, HOST, PORT)
    print(f"test action server listening on {HOST}:{PORT}")
    print(f"actions={actions} interval={args.interval}s - press Play in Unreal")
    async with server:
        await server.serve_forever()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
