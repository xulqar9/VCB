"""Built-in offline skills: date/time, calculator, timers, remember/recall,
notes. All run locally — no network. Online skills (web/weather) come later,
gated on the privacy-proxy decision."""

from __future__ import annotations

import ast
import asyncio
import operator
import time

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
        now = time.localtime()
        date = time.strftime("%A, %B %d, %Y", now)
        clock = time.strftime("%I:%M %p", now).lstrip("0")
        if what == "time":
            return f"The time is {clock}."
        if what == "date":
            return f"Today is {date}."
        return f"It is {clock} on {date}."


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
            msg = f"Reminder: {label}." if label else "Your timer is up."
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
    "remember": RememberSkill,
    "recall": RecallSkill,
    "add_note": AddNoteSkill,
    "list_notes": ListNotesSkill,
}


def build_skills(enabled: tuple[str, ...]) -> list[Skill]:
    """Instantiate the enabled built-in skills, preserving config order."""
    return [_ALL[name]() for name in enabled if name in _ALL]
