"""Skill framework: a Skill is a named async action the model can invoke.

Many local GGUF chat templates (and llama.cpp run with --no-jinja) don't support
native OpenAI tool-calling, so we don't rely on it. Instead the model emits a JSON action
block, which parse_tool_call() extracts. The orchestrator runs the skill and
feeds the result back so the model can phrase a natural spoken reply.
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import asyncio

    from ..memory import MemoryStore


@dataclass
class SkillContext:
    """Everything a skill might need, injected at dispatch time."""
    memory: "MemoryStore | None"
    speak: Callable[[str], Awaitable[None]]   # enqueue text for immediate TTS
    loop: "asyncio.AbstractEventLoop"


@dataclass
class ToolCall:
    name: str
    args: dict


class Skill(ABC):
    name: str = ""
    description: str = ""
    # arg name -> short human description, rendered into the system prompt.
    parameters: dict[str, str] = {}

    @abstractmethod
    async def run(self, args: dict, ctx: SkillContext) -> str:
        """Do the work and return a short result string fed back to the model."""


_JSON_OBJ = re.compile(r"\{.*\}", re.DOTALL)


def looks_like_tool_call(head: str) -> bool:
    """Cheap check on the start of a streamed reply: a JSON object (optionally
    fenced) means the model is trying to call a tool rather than speak."""
    s = head.lstrip()
    if not s:
        return False
    if s[0] == "{":
        return True
    return s.startswith("```") and "{" in s


def parse_tool_call(text: str) -> ToolCall | None:
    """Extract {"tool": name, "args": {...}} from a model reply, tolerating code
    fences and surrounding prose. Returns None if it isn't a valid tool call."""
    match = _JSON_OBJ.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(data, dict):
        return None
    name = data.get("tool") or data.get("name")
    if not isinstance(name, str) or not name:
        return None
    args = data.get("args") or data.get("arguments") or {}
    if not isinstance(args, dict):
        args = {}
    return ToolCall(name=name, args=args)
