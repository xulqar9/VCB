"""Rolling consolidation: fold old turns into a compact summary + durable profile
facts, using the same LLM that answers the user. Runs in the background so it
never gates the voice loop.
"""

import contextlib
import json
import logging
import re

from ..clock import Clock
from ..llm.base import LLMEngine

log = logging.getLogger("vcb.memory")

_SYS = (
    "You maintain long-term memory for a voice assistant. Given the previous "
    "running summary, the facts already stored, and the latest conversation "
    "turns, produce an updated summary and changes to the stored facts.\n"
    "Rules:\n"
    "- Only record what the USER actually said or did. Lines starting with "
    "'assistant:' are the assistant's own guesses, suggestions and questions -- "
    "never treat them as facts about the user unless the user confirmed them.\n"
    "- The summary is cumulative: fold the new turns into the previous summary and "
    "keep it concise (a few short sentences). Drop small talk; keep decisions, "
    "preferences, ongoing topics, and anything worth remembering long-term.\n"
    "- Facts are stable, durable attributes only (name, location, job, "
    "relationships, pets, hobbies, preferences, recurring goals, skill levels). "
    "Never store things that are only true right now: weather, scores, the "
    "current match or task, today's mood, what they're doing at the moment.\n"
    "- Reuse an existing key when the fact is about the same thing (update its "
    "value) instead of adding a near-duplicate key.\n"
    "- In \"remove\", list existing keys that are duplicates of another key, "
    "temporary, or that the user said are wrong.\n"
    "- Use short snake_case keys and short values.\n"
    "- Dates: each turn is prefixed with when it was said. Turn relative dates "
    "('tomorrow', 'next Monday', 'in two weeks', 'the 20th') into absolute "
    "dates (YYYY-MM-DD), counted from when they were said, in both the summary "
    "and facts. Store upcoming events and deadlines as facts with their date "
    "(e.g. permit_card_expiry: 2026-11-20); remove ones that are over and no "
    "longer matter.\n"
    'Respond with ONLY a JSON object: {"summary": "...", "facts": {"key": "value"}, '
    '"remove": ["key"]}. No prose, no code fences.'
)

# Fact keys that describe a moment, not the person. Dropped even if the model
# proposes them.
_TRANSIENT_KEY = re.compile(
    r"weather|temperature|score|mood|feeling|today|tonight|right_now|"
    r"current_(?:activity|task|match|game_state|score|mood)|session",
    re.IGNORECASE,
)
# Words too generic to show a fact was really mentioned by the user.
_GENERIC = {
    "user", "enjoys", "enjoy", "likes", "like", "loves", "love", "prefers",
    "prefer", "playing", "plays", "play", "currently", "usually", "often",
    "really", "very", "with", "from", "that", "this", "their", "they", "about",
    "some", "into", "also", "being", "doing", "does", "have", "has", "using",
    "uses", "wants", "want", "would", "should", "things", "stuff",
}


def _content_words(text: str) -> list[str]:
    # Numbers are skipped: "2026-11-20" is a normalized form of "20th November"
    # and can't be matched word for word.
    return [w for w in re.findall(r"[a-z0-9]+", text.lower())
            if len(w) >= 4 and w not in _GENERIC and not w.isdigit()]


def grounded(value: str, user_text: str) -> bool:
    """True if at least half of the fact's distinctive words (matched on their
    first 4 letters, so "windy" ~ "wind", "Platinum" ~ "Plat") appear in what
    the user said. Rejects invented facts like "enjoys playing guitar" when
    the user never mentioned a guitar."""
    words = _content_words(value)
    if not words:
        return True                       # nothing distinctive to check (e.g. "yes")
    corpus = user_text.lower()
    hits = sum(1 for w in words if w[:4] in corpus)
    return hits * 2 >= len(words)


def _norm(value: str) -> str:
    return re.sub(r"[\s_]+", " ", value.strip().lower())


def _format_turns(turns, clock: Clock | None = None) -> str:
    """'[Sat 24 Oct 2026, 01:47] user: ...' -- turns are (id, role, text[, ts])."""
    lines = []
    for turn in turns:
        _id, role, text = turn[0], turn[1], turn[2]
        if clock is not None and len(turn) > 3 and turn[3]:
            when = clock.from_timestamp(turn[3])
            lines.append(f"[{when:%a %d %b %Y, %H:%M}] {role}: {text}")
        else:
            lines.append(f"{role}: {text}")
    return "\n".join(lines)


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
    turns: list[tuple],
    existing_facts: dict[str, str] | None = None,
    clock: Clock | None = None,
) -> tuple[str | None, dict[str, str], list[str]]:
    """Return (summary, facts to add/update, fact keys to remove). Best-effort;
    returns (None, {}, []) on parse failure. Proposed facts are filtered in code
    too: momentary ones, ones the user never actually mentioned, and values
    that duplicate another key are dropped. Turns are (id, role, text[, ts]);
    with a clock, they're dated so relative dates can be resolved."""
    if not turns:
        return None, {}, []
    existing = existing_facts or {}
    facts_text = "\n".join(f"- {k}: {v}" for k, v in existing.items()) or "(none yet)"
    today = f"TODAY: {clock.now():%A, %Y-%m-%d}\n\n" if clock is not None else ""
    user = (
        f"{today}PREVIOUS SUMMARY:\n{previous_summary or '(none yet)'}\n\n"
        f"STORED FACTS:\n{facts_text}\n\n"
        f"NEW TURNS:\n{_format_turns(turns, clock)}"
    )
    messages = [
        {"role": "system", "content": _SYS},
        {"role": "user", "content": user},
    ]
    acc = ""
    # A summary plus a JSON fact list is longer than a spoken reply.
    async for delta in llm.stream(messages, temperature=0.2, max_tokens=800):
        acc += delta
    data = _extract_json(acc)
    if not isinstance(data, dict):
        log.warning("consolidation: could not parse JSON from model output")
        return None, {}, []
    summary = data.get("summary")
    summary = summary.strip() if isinstance(summary, str) and summary.strip() else None

    raw_facts = data.get("facts") or {}
    raw_facts = raw_facts if isinstance(raw_facts, dict) else {}
    user_text = " ".join(t[2] for t in turns if t[1] == "user")
    known = user_text + " " + " ".join(existing.values())
    taken = {_norm(v): k for k, v in existing.items()}
    facts: dict[str, str] = {}
    for k, v in raw_facts.items():
        key, value = str(k).strip(), str(v).strip()
        if not key or not value:
            continue
        if _TRANSIENT_KEY.search(key):
            log.info("memory: skipping momentary fact %s=%r", key, value)
            continue
        if not grounded(value, known):
            log.info("memory: skipping fact the user never mentioned %s=%r", key, value)
            continue
        dup_of = taken.get(_norm(value))
        if dup_of and dup_of != key:
            continue                      # same value already stored under another key
        facts[key] = value
        taken[_norm(value)] = key

    raw_remove = data.get("remove") or []
    remove = [str(k) for k in raw_remove if isinstance(k, str) and str(k) in existing] \
        if isinstance(raw_remove, list) else []
    # Momentary facts stored by older versions are cleaned up too.
    remove += [k for k in existing if _TRANSIENT_KEY.search(k) and k not in remove]
    remove = [k for k in remove if k not in facts]
    return summary, facts, remove
