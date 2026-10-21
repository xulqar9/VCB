"""Read and update a `.env` file in place, keeping comments, ordering, and any
keys we don't manage (e.g. a user's own HF_TOKEN) untouched."""

import re
from pathlib import Path

_LINE = re.compile(r"^\s*([A-Za-z_][A-Za-z0-9_]*)\s*=(.*)$")


def _parse_value(raw: str) -> str:
    value = raw.strip()
    if value[:1] in ('"', "'"):
        quote = value[0]
        end = value.find(quote, 1)
        return value[1:end] if end > 0 else value[1:]
    return value.split(" #", 1)[0].rstrip()


def _format_value(value: str) -> str:
    if value == "" or re.search(r"[\s#'\"]", value):
        return '"' + value.replace('"', "'") + '"'
    return value


def read_env(path: Path) -> dict[str, str]:
    """Active (uncommented) KEY=VALUE pairs."""
    if not path.is_file():
        return {}
    out = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        m = _LINE.match(line)
        if m:
            out[m.group(1)] = _parse_value(m.group(2))
    return out


def update_env(path: Path, updates: dict[str, str | None]) -> None:
    """Set each key to its value (None removes it). Existing lines are edited
    where they are; new keys are appended at the end."""
    lines = path.read_text(encoding="utf-8-sig").splitlines() if path.is_file() else []
    pending = dict(updates)
    out: list[str] = []
    for line in lines:
        m = _LINE.match(line)
        if m and m.group(1) in pending:
            value = pending.pop(m.group(1))
            if value is not None:
                out.append(f"{m.group(1)}={_format_value(value)}")
            continue
        out.append(line)
    new = {k: v for k, v in pending.items() if v is not None}
    if new:
        if out and out[-1].strip():
            out.append("")
        out.extend(f"{k}={_format_value(v)}" for k, v in new.items())
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
