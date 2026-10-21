"""Skill registry: holds the enabled skills, renders their spec for the system
prompt, and dispatches a parsed tool call to the right skill."""

from __future__ import annotations

import logging

from .base import Skill, SkillContext

log = logging.getLogger("vcb.skills")


class SkillRegistry:
    def __init__(self, skills: list[Skill]) -> None:
        self._skills: dict[str, Skill] = {s.name: s for s in skills}

    def __bool__(self) -> bool:
        return bool(self._skills)

    def names(self) -> list[str]:
        return list(self._skills)

    def prompt_block(self) -> str:
        """Instruction + tool catalogue appended to the system prompt."""
        if not self._skills:
            return ""
        lines = [
            "TOOLS:",
            "You can call a tool when it genuinely helps (current time, math, "
            "timers, saving or recalling facts and notes). To call one, reply "
            "with ONLY this JSON and nothing else, no code fences, no words "
            "before or after:",
            '{"tool": "<name>", "args": {<arguments>}}',
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

    async def dispatch(self, name: str, args: dict, ctx: SkillContext) -> str:
        skill = self._skills.get(name)
        if skill is None:
            return f"Error: no tool named {name!r}."
        try:
            return await skill.run(args, ctx)
        except Exception as exc:                     # skills must never crash a turn
            log.exception("skill %s failed", name)
            return f"Error running {name}: {exc}"
