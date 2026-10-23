"""Memory consolidation: only grounded, durable, non-duplicate facts survive."""

import asyncio
import json

from vcb.memory.consolidate import consolidate, grounded


class _ScriptedLLM:
    def __init__(self, payload):
        self.payload = payload
        self.kwargs = None

    async def stream(self, messages, temperature=None, max_tokens=None):
        self.kwargs = {"temperature": temperature, "max_tokens": max_tokens}
        self.messages = messages
        yield json.dumps(self.payload)


def test_grounded():
    user = "I'm playing Rocket League on keyboard and mouse. I was Diamond, now Platz 3."
    assert grounded("keyboard and mouse", user)
    assert grounded("Rocket League", user)
    assert grounded("Platinum 3", user)               # "plat" matches "Platz"
    assert not grounded("Enjoys playing guitar", user)
    assert grounded("yes", user)                      # nothing distinctive to check


def test_consolidate_filters_invented_momentary_and_duplicate_facts():
    turns = [
        (1, "assistant", "You could grab your guitar and jam!"),
        (2, "user", "Nah. I play Rocket League with keyboard and mouse."),
        (3, "user", "It's clear and a bit cold outside."),
    ]
    existing = {"control_method": "keyboard and mouse", "weather": "windy"}
    llm = _ScriptedLLM({
        "summary": "Plays Rocket League on keyboard and mouse.",
        "facts": {
            "hobby": "plays guitar",                     # assistant's idea, never said
            "game": "Rocket League",                     # real
            "controls": "keyboard_and_mouse",            # duplicate of control_method
            "weather": "clear and cold",                 # momentary
        },
        "remove": ["control_method_old", "weather"],
    })
    summary, facts, remove = asyncio.run(consolidate(llm, None, turns, existing))
    assert summary.startswith("Plays Rocket League")
    assert facts == {"game": "Rocket League"}
    assert remove == ["weather"]                      # unknown keys ignored
    assert llm.kwargs["max_tokens"] >= 600            # room for summary + JSON
    assert "STORED FACTS" in llm.messages[1]["content"]


def test_consolidate_handles_bad_output():
    class _Junk:
        async def stream(self, messages, temperature=None, max_tokens=None):
            yield "sorry, I can't do that"

    assert asyncio.run(consolidate(_Junk(), None, [(1, "user", "hi")])) == (None, {}, [])
