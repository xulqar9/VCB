"""Skill registry: holds the enabled skills, renders their spec for the system
prompt, and dispatches a parsed tool call to the right skill."""

from __future__ import annotations

import logging

from .base import Skill, SkillContext

log = logging.getLogger("vcb.skills")

_HONESTY = ("Only a tool call actually does something: never tell the user you "
            "set a reminder or timer, or saved a fact or note, unless you called "
            "the tool for it and got its result. If they ask for one, call the "
            "tool first.")


class SkillRegistry:
    def __init__(self, skills: list[Skill]) -> None:
        self._skills: dict[str, Skill] = {s.name: s for s in skills}

    def __bool__(self) -> bool:
        return bool(self._skills)

    def names(self) -> list[str]:
        return list(self._skills)

    def prompt_block(self) -> str:
        """Instruction + tool catalogue appended to the system prompt (for
        models used with VCB's JSON tool protocol)."""
        if not self._skills:
            return ""
        lines = [
            "TOOLS:",
            "You can call a tool when it genuinely helps (math, timers, "
            "reminders, saving or recalling facts and notes). You already know "
            "the current date and time, so you don't need a tool for that. To "
            "call one, reply with ONLY this JSON and nothing else, no code "
            "fences, no words before or after:",
            '{"tool": "<name>", "args": {<arguments>}}',
            _HONESTY,
            "After you receive the tool result, reply to the user in one or two "
            "natural spoken sentences. Never say the tool's name or read raw "
            "JSON aloud. If no tool is needed, just answer normally.",
            "",
            "Available tools:",
        ]
        for skill in self._skills.values():
            args = ", ".join(
                f"{k} ({v})" for k, v in skill.parameters.items()
            ) or "no arguments"
            lines.append(f"- {skill.name}: {skill.description} Args: {args}.")
        return "\n".join(lines)

    def native_prompt_block(self) -> str:
        """Short guidance when tools are passed natively (function calling):
        the API carries the catalogue, the model just needs the ground rules."""
        if not self._skills:
            return ""
        return ("TOOLS: You have tools for math, timers, reminders, and saving or "
                "recalling facts and notes. You already know the current date and "
                "time. " + _HONESTY + " After a tool result, reply in one or two "
                "natural spoken sentences and never mention the tool by name.")

    def openai_tools(self) -> list[dict]:
        """The skills as OpenAI function-calling specs."""
        specs = []
        for skill in self._skills.values():
            props = {
                name: {"type": skill.param_types.get(name, "string"), "description": desc}
                for name, desc in skill.parameters.items()
            }
            specs.append({"type": "function", "function": {
                "name": skill.name,
                "description": skill.description,
                "parameters": {"type": "object", "properties": props},
            }})
        return specs

    async def dispatch(self, name: str, args: dict, ctx: SkillContext) -> str:
        skill = self._skills.get(name)
        if skill is None:
            return f"Error: no tool named {name!r}."
        try:
            return await skill.run(args, ctx)
        except Exception as exc:                     # skills must never crash a turn
            log.exception("skill %s failed", name)
            return f"Error running {name}: {exc}"
