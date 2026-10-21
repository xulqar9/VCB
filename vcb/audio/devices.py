"""Audio device lookup by name. PortAudio indices shift when devices are
plugged in or out, so settings store the device *name*; it's resolved to the
current index at stream-open time."""

import sounddevice as sd


def _default_hostapi() -> int:
    try:
        return sd.query_devices(kind="input")["hostapi"]
    except Exception:
        return 0


def list_devices(kind: str) -> list[str]:
    """Device names of `kind` ("input"/"output") on the default host API (the
    same device shows up once per Windows audio API; list each just once)."""
    channels = "max_input_channels" if kind == "input" else "max_output_channels"
    api = _default_hostapi()
    names: list[str] = []
    for dev in sd.query_devices():
        if dev[channels] > 0 and dev["hostapi"] == api and dev["name"] not in names:
            names.append(dev["name"])
    return names


def resolve_device(device: int | str | None, kind: str) -> int | None:
    """Index for a configured device (index, exact name, or name fragment).
    None or an unknown name means the system default."""
    if device is None or isinstance(device, int):
        return device
    channels = "max_input_channels" if kind == "input" else "max_output_channels"
    devices = [(i, d) for i, d in enumerate(sd.query_devices()) if d[channels] > 0]
    api = _default_hostapi()
    want = device.lower()
    for match in (
        lambda d: d["name"].lower() == want and d["hostapi"] == api,
        lambda d: d["name"].lower() == want,
        lambda d: want in d["name"].lower() and d["hostapi"] == api,
        lambda d: want in d["name"].lower(),
    ):
        for i, d in devices:
            if match(d):
                return i
    return None
