"""Rolling consolidation: fold old turns into a compact summary + durable profile
facts, using the same LLM that answers the user. Runs in the background so it
never gates the voice loop.
"""

import contextlib
import json
import logging
import re

from ..llm.base import LLMEngine

log = logging.getLogger("vcb.memory")

_SYS = (
    "You maintain long-term memory for a voice assistant. Given the previous "
    "running summary and the latest conversation turns, produce an updated, "
    "compact summary and a set of durable facts about the user.\n"
    "Rules:\n"
    "- The summary is cumulative: fold the new turns into the previous summary and "
    "keep it concise (a few short sentences). Drop small talk; keep decisions, "
    "preferences, ongoing topics, and anything worth remembering long-term.\n"
    "- Facts are stable, durable attributes only (name, location, job, "
    "relationships, pets, preferences, recurring goals). No transient chit-chat.\n"
    "- Use short snake_case keys and short values.\n"
    'Respond with ONLY a JSON object: {"summary": "...", "facts": {"key": "value"}}. '
    "No prose, no code fences."
)


def _format_turns(turns: list[tuple[int, str, str]]) -> str:
    return "\n".join(f"{role}: {text}" for _id, role, text in turns)


def _extract_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.DOTALL)  # tolerate fences/prose around it
    if not m:
        return None
    with contextlib.suppress(json.JSONDecodeError):
        return json.loads(m.group(0))
    return None


async def consolidate(
    llm: LLMEngine,
    previous_summary: str | None,
    turns: list[tuple[int, str, str]],
) -> tuple[str | None, dict[str, str]]:
    """Return (summary, facts). Best-effort; may return (None, {}) on parse failure."""
    if not turns:
        return None, {}
    user = (
        f"PREVIOUS SUMMARY:\n{previous_summary or '(none yet)'}\n\n"
        f"NEW TURNS:\n{_format_turns(turns)}"
    )
    messages = [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": user},
    ]
    acc = ""
    async for delta in llm.stream(messages):
        acc += delta
    data = _extract_json(acc)
    if not isinstance(data, dict):
        log.warning("consolidation: could not parse JSON from model output")
        return None, {}
    summary = data.get("summary")
    raw_facts = data.get("facts") or {}
    facts = {str(k): str(v) for k, v in raw_facts.items() if k and v} \
        if isinstance(raw_facts, dict) else {}
    summary = summary.strip() if isinstance(summary, str) and summary.strip() else None
    return summary, facts
