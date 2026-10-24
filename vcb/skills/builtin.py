"""Built-in offline skills: date/time, calculator, timers, remember/recall,
notes. All run locally — no network. Online skills (web/weather) come later,
gated on the privacy-proxy decision."""

from __future__ import annotations

import ast
import asyncio
import contextlib
import operator
import time
from datetime import datetime, timedelta

from ..clock import Clock
from .base import Skill, SkillContext

# -- safe arithmetic ---------------------------------------------------------

_BIN_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod, ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("only numbers are allowed")
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("unsupported expression")


# -- skills ------------------------------------------------------------------

class DateTimeSkill(Skill):
    name = "datetime"
    description = "Get the current date and/or time."
    parameters = {"what": '"time", "date", or "both" (default both)'}

    async def run(self, args: dict, ctx: SkillContext) -> str:
        what = str(args.get("what", "both")).lower()
        now = (ctx.clock or Clock()).now()
        date = f"{now:%A}, {now.day} {now:%B %Y}"
        clock = now.strftime("%I:%M %p").lstrip("0")
        if what == "time":
            return f"The time is {clock}."
        if what == "date":
            return f"Today is {date}."
        return f"It is {clock} on {date}."


_AT_FORMATS = ("%Y-%m-%d %H:%M", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M:%S",
               "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")


def _parse_local(text: str, clock: Clock) -> datetime | None:
    """'2026-10-26 08:00' (or just '08:00' = next time it's 8) in the user's zone."""
    text = text.strip()
    for fmt in _AT_FORMATS:
        try:
            naive = datetime.strptime(text, fmt)
            if fmt == "%Y-%m-%d":
                naive = naive.replace(hour=9)          # a date alone = 9 AM
            return clock.localize(naive)
        except ValueError:
            continue
    try:
        hm = datetime.strptime(text, "%H:%M")
    except ValueError:
        return None
    now = clock.now()
    when = clock.localize(datetime.combine(now.date(), hm.time()))
    if when <= now:
        when = clock.localize(datetime.combine(now.date() + timedelta(days=1), hm.time()))
    return when


def _spoken_when(when: datetime, clock: Clock) -> str:
    now = clock.now()
    t = when.strftime("%I:%M %p").lstrip("0")
    days = (when.date() - now.date()).days
    if days == 0:
        return f"today at {t}"
    if days == 1:
        return f"tomorrow at {t}"
    return f"{when:%A}, {when.day} {when:%B} at {t}"


class SetReminderSkill(Skill):
    name = "set_reminder"
    description = ("Remind the user at a specific date and time (kept even if VCB "
                   "restarts). Work out the exact local time from the current date "
                   "and time you were given.")
    parameters = {
        "at": 'local date and time "YYYY-MM-DD HH:MM" (or use in_minutes)',
        "in_minutes": "alternatively: minutes from now",
        "text": "what to remind them about",
    }
    param_types = {"in_minutes": "number"}

    async def run(self, args: dict, ctx: SkillContext) -> str:
        clock = ctx.clock or Clock()
        text = str(args.get("text", "")).strip() or "your reminder"
        when = None
        if args.get("at"):
            when = _parse_local(str(args["at"]), clock)
            if when is None:
                return 'Error: "at" must look like "2026-10-26 08:00".'
        elif args.get("in_minutes") is not None:
            try:
                minutes = float(args["in_minutes"])
            except (TypeError, ValueError):
                return "Error: in_minutes must be a number."
            when = clock.now() + timedelta(minutes=minutes)
        if when is None:
            return 'Error: give either "at" or "in_minutes".'
        if when <= clock.now() - timedelta(minutes=1):
            return f"Error: {_spoken_when(when, clock)} has already passed."
        spoken = _spoken_when(when, clock)
        if ctx.memory is None:
            delay = max(0.0, when.timestamp() - time.time())

            async def _fire() -> None:
                await asyncio.sleep(delay)
                await ctx.speak(f"Hey, here's your reminder: {text}.")

            ctx.loop.create_task(_fire())
            return (f"Reminder set for {spoken}: {text}. (Memory is off, so it only "
                    "works while VCB stays open.)")
        rid = await asyncio.to_thread(ctx.memory.add_reminder, when.timestamp(), text)
        return f"Reminder {rid} set for {spoken}: {text}."


class ListRemindersSkill(Skill):
    name = "list_reminders"
    description = "List the user's upcoming reminders."
    parameters = {}

    async def run(self, args: dict, ctx: SkillContext) -> str:
        if ctx.memory is None:
            return "Memory is off, so there are no saved reminders."
        clock = ctx.clock or Clock()
        pending = await asyncio.to_thread(ctx.memory.pending_reminders)
        if not pending:
            return "No upcoming reminders."
        return "Upcoming reminders:\n" + "\n".join(
            f"- (id {rid}) {_spoken_when(clock.from_timestamp(due), clock)}: {text}"
            for rid, due, text in pending)


class CancelReminderSkill(Skill):
    name = "cancel_reminder"
    description = "Cancel an upcoming reminder."
    parameters = {"id": "reminder id (from list_reminders)",
                  "text": "or: words from the reminder"}

    async def run(self, args: dict, ctx: SkillContext) -> str:
        if ctx.memory is None:
            return "Memory is off, so there are no saved reminders."
        pending = await asyncio.to_thread(ctx.memory.pending_reminders)
        target = None
        if args.get("id") is not None:
            with contextlib.suppress(TypeError, ValueError):
                rid = int(args["id"])
                target = next((p for p in pending if p[0] == rid), None)
        elif args.get("text"):
            needle = str(args["text"]).lower()
            matches = [p for p in pending if needle in p[2].lower()]
            if len(matches) > 1:
                return "Several reminders match; which one? " + "; ".join(
                    f"id {p[0]}: {p[2]}" for p in matches)
            target = matches[0] if matches else None
        if target is None:
            return "I couldn't find that reminder."
        await asyncio.to_thread(ctx.memory.cancel_reminder, target[0])
        return f"Cancelled the reminder: {target[2]}."


class CalculatorSkill(Skill):
    name = "calculator"
    description = "Evaluate an arithmetic expression."
    parameters = {"expression": 'e.g. "12.5 * (3 + 4)"'}

    async def run(self, args: dict, ctx: SkillContext) -> str:
        expr = str(args.get("expression", "")).strip()
        if not expr:
            return "Error: no expression given."
        try:
            tree = ast.parse(expr, mode="eval")
            value = _safe_eval(tree)
        except Exception:
            return f"Error: could not evaluate {expr!r}."
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        return f"{expr} = {value}"


class SetTimerSkill(Skill):
    name = "set_timer"
    description = "Set a timer that speaks a reminder when it fires."
    parameters = {"seconds": "how many seconds from now", "label": "optional reminder text"}

    async def run(self, args: dict, ctx: SkillContext) -> str:
        try:
            seconds = float(args.get("seconds", 0))
        except (TypeError, ValueError):
            return "Error: seconds must be a number."
        if seconds <= 0:
            return "Error: seconds must be greater than zero."
        label = str(args.get("label", "")).strip()

        async def _fire() -> None:
            await asyncio.sleep(seconds)
            msg = f"Hey, your timer is up: {label}." if label else "Your timer is up."
            await ctx.speak(msg)

        ctx.loop.create_task(_fire())
        mins = seconds / 60
        pretty = f"{seconds:.0f} seconds" if seconds < 90 else f"{mins:.0f} minutes"
        return f"Timer set for {pretty}" + (f" ({label})." if label else ".")


class RememberSkill(Skill):
    name = "remember"
    description = "Save a durable fact about the user."
    parameters = {"key": 'short label, e.g. "birthday"', "value": "the fact to store"}

    async def run(self, args: dict, ctx: SkillContext) -> str:
        if ctx.memory is None:
            return "Memory is turned off, so I can't save that right now."
        key = str(args.get("key", "")).strip()
        value = str(args.get("value", "")).strip()
        if not key or not value:
            return "Error: need both a key and a value to remember."
        await asyncio.to_thread(ctx.memory.upsert_fact, key, value, "user")
        return f"Saved: {key} = {value}."


class RecallSkill(Skill):
    name = "recall"
    description = "Look up a saved fact, or list all saved facts."
    parameters = {"key": "the fact label to fetch (omit to list everything)"}

    async def run(self, args: dict, ctx: SkillContext) -> str:
        if ctx.memory is None:
            return "Memory is turned off, so I have nothing saved."
        key = str(args.get("key", "")).strip()
        if not key:
            block = await asyncio.to_thread(ctx.memory.facts_block)
            return f"Saved facts:\n{block}" if block else "Nothing saved yet."
        value = await asyncio.to_thread(ctx.memory.get_fact, key)
        return f"{key} = {value}" if value is not None else f"I have nothing saved for {key!r}."


class AddNoteSkill(Skill):
    name = "add_note"
    description = "Append an item to the user's notes list."
    parameters = {"text": "the note to add"}

    async def run(self, args: dict, ctx: SkillContext) -> str:
        if ctx.memory is None:
            return "Memory is turned off, so I can't save notes right now."
        text = str(args.get("text", "")).strip()
        if not text:
            return "Error: the note is empty."
        await asyncio.to_thread(ctx.memory.add_note, text)
        return f"Added a note: {text}"


class ListNotesSkill(Skill):
    name = "list_notes"
    description = "Read back the user's notes list."
    parameters = {}

    async def run(self, args: dict, ctx: SkillContext) -> str:
        if ctx.memory is None:
            return "Memory is turned off, so I have no notes."
        notes = await asyncio.to_thread(ctx.memory.list_notes)
        if not notes:
            return "The notes list is empty."
        return "Notes:\n" + "\n".join(f"{i}. {n}" for i, n in enumerate(notes, 1))


_ALL: dict[str, type[Skill]] = {
    "datetime": DateTimeSkill,
    "calculator": CalculatorSkill,
    "set_timer": SetTimerSkill,
    "set_reminder": SetReminderSkill,
    "list_reminders": ListRemindersSkill,
    "cancel_reminder": CancelReminderSkill,
    "remember": RememberSkill,
    "recall": RecallSkill,
    "add_note": AddNoteSkill,
    "list_notes": ListNotesSkill,
}


def build_skills(enabled: tuple[str, ...]) -> list[Skill]:
    """Instantiate the enabled built-in skills, preserving config order."""
    return [_ALL[name]() for name in enabled if name in _ALL]
