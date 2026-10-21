"""Skills: tool-call parsing, calculator safety, dispatch, and stateful skills."""

import asyncio

from vcb.memory.store import MemoryStore
from vcb.skills import (
    SkillContext,
    SkillRegistry,
    build_skills,
    looks_like_tool_call,
    parse_tool_call,
)
from vcb.skills.builtin import CalculatorSkill, RecallSkill, RememberSkill, SetTimerSkill


async def _noop(_text):
    pass


# A loop object is only needed by timer skills; these ctx uses never run it.
_IDLE_LOOP = asyncio.new_event_loop()


def _ctx(memory=None):
    return SkillContext(memory=memory, speak=_noop, loop=_IDLE_LOOP)


# -- parsing -----------------------------------------------------------------

def test_looks_like_tool_call():
    assert looks_like_tool_call('{"tool": "x"}')
    assert looks_like_tool_call("```json\n{\"tool\": \"x\"}")
    assert not looks_like_tool_call("Hello there")
    assert not looks_like_tool_call("   ")


def test_parse_tool_call_standard():
    call = parse_tool_call('{"tool": "calculator", "args": {"expression": "2+2"}}')
    assert call is not None
    assert call.name == "calculator"
    assert call.args == {"expression": "2+2"}


def test_parse_tool_call_aliases_and_prose():
    call = parse_tool_call('sure: {"name": "datetime", "arguments": {"what": "time"}} ok')
    assert call is not None
    assert call.name == "datetime"
    assert call.args == {"what": "time"}


def test_parse_tool_call_rejects_non_calls():
    assert parse_tool_call("just talking") is None
    assert parse_tool_call('{"tool": 123}') is None


# -- calculator safety -------------------------------------------------------

def test_calculator_evaluates():
    res = asyncio.run(CalculatorSkill().run({"expression": "2 + 3 * 4"}, _ctx()))
    assert res == "2 + 3 * 4 = 14"


def test_calculator_rejects_names_and_calls():
    res = asyncio.run(CalculatorSkill().run({"expression": "__import__('os')"}, _ctx()))
    assert res.startswith("Error")


def test_calculator_rejects_statements():
    res = asyncio.run(CalculatorSkill().run({"expression": "import os"}, _ctx()))
    assert res.startswith("Error")


# -- registry ----------------------------------------------------------------

def test_build_skills_filters_unknown():
    skills = build_skills(("calculator", "datetime", "does_not_exist"))
    names = {s.name for s in skills}
    assert names == {"calculator", "datetime"}


def test_dispatch_unknown_tool():
    reg = SkillRegistry(build_skills(("calculator",)))
    res = asyncio.run(reg.dispatch("nope", {}, _ctx()))
    assert res.startswith("Error: no tool named")


def test_dispatch_runs_skill():
    reg = SkillRegistry(build_skills(("calculator",)))
    res = asyncio.run(reg.dispatch("calculator", {"expression": "10/4"}, _ctx()))
    assert res == "10/4 = 2.5"


# -- stateful skills against a real in-memory store --------------------------

def test_remember_then_recall():
    async def _run():
        store = MemoryStore("unused", ephemeral=True)
        ctx = SkillContext(memory=store, speak=_noop, loop=asyncio.get_running_loop())
        saved = await RememberSkill().run({"key": "name", "value": "Alex"}, ctx)
        assert "Alex" in saved
        got = await RecallSkill().run({"key": "name"}, ctx)
        store.close()
        return got

    assert asyncio.run(_run()) == "name = Alex"


def test_recall_without_memory_degrades_gracefully():
    res = asyncio.run(RecallSkill().run({"key": "name"}, _ctx(memory=None)))
    assert "Memory is turned off" in res


def test_set_timer_fires_and_speaks():
    captured = []

    async def _run():
        fired = asyncio.Event()

        async def speak(msg):
            captured.append(msg)
            fired.set()

        ctx = SkillContext(memory=None, speak=speak, loop=asyncio.get_running_loop())
        res = await SetTimerSkill().run({"seconds": 0.01, "label": "tea"}, ctx)
        assert "Timer set" in res
        await asyncio.wait_for(fired.wait(), timeout=1.0)

    asyncio.run(_run())
    assert captured and "tea" in captured[0]


def test_set_timer_rejects_non_positive():
    res = asyncio.run(SetTimerSkill().run({"seconds": 0}, _ctx()))
    assert res.startswith("Error")
