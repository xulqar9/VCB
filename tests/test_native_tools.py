"""Native (function-calling) tools: stream parsing, Gemini's thought signature
round trip, and the orchestrator's tool loop."""

import asyncio
from types import SimpleNamespace

from vcb.config import LLMConfig
from vcb.llm.base import ToolCallRequest
from vcb.llm.openai_compat import OpenAICompatLLM
from vcb.skills import SkillRegistry, build_skills


def _tc(index, id=None, name=None, args=None, extra=None):
    fn = SimpleNamespace(name=name, arguments=args)
    return SimpleNamespace(index=index, id=id, function=fn,
                           model_extra={"extra_content": extra} if extra else {})


def _ev(content=None, tool_calls=None):
    delta = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(delta=delta)])


class _Stream:
    def __init__(self, events):
        self.events = events

    def __aiter__(self):
        async def gen():
            for e in self.events:
                yield e
        return gen()


def test_stream_assembles_fragmented_tool_calls_with_signature():
    sig = {"google": {"thought_signature": "abc"}}
    events = [
        _ev(content="Sure. "),
        _ev(tool_calls=[_tc(0, id="call_1", name="set_reminder", args='{"text": "wat', extra=sig)]),
        _ev(tool_calls=[_tc(0, args='er", "in_minutes": 20}')]),
    ]
    seen = {}

    class _Completions:
        async def create(self, **kwargs):
            seen.update(kwargs)
            return _Stream(events)

    llm = OpenAICompatLLM(LLMConfig())
    llm._client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))

    async def _run():
        return [d async for d in llm.stream([{"role": "user", "content": "x"}],
                                            tools=[{"type": "function"}])]

    out = asyncio.run(_run())
    assert out[0] == "Sure. "
    assert out[1] == ToolCallRequest([("call_1", "set_reminder",
                                       {"text": "water", "in_minutes": 20}, sig)])
    assert seen["tools"] == [{"type": "function"}]


def test_openai_tool_specs():
    specs = SkillRegistry(build_skills(("set_reminder", "set_timer"))).openai_tools()
    rem = next(s["function"] for s in specs if s["function"]["name"] == "set_reminder")
    assert rem["parameters"]["properties"]["in_minutes"]["type"] == "number"
    assert rem["parameters"]["properties"]["at"]["type"] == "string"


def test_orchestrator_native_tool_loop():
    from tests.test_orchestrator import _drain, _orch
    from vcb.memory.store import MemoryStore

    sig = {"google": {"thought_signature": "abc"}}

    class _ToolLLM:
        def __init__(self):
            self.calls = []

        async def stream(self, messages, temperature=None, max_tokens=None, tools=None):
            self.calls.append((messages, tools))
            if len(self.calls) == 1:
                yield ToolCallRequest([("c1", "set_reminder",
                                        {"in_minutes": 30, "text": "tea"}, sig)])
            else:
                yield "Done, I'll remind you about tea."

    async def _run():
        llm = _ToolLLM()
        orch = _orch(llm)
        orch.cfg.skills.enabled = True
        from vcb.skills import SkillRegistry, build_skills
        orch._skills = SkillRegistry(build_skills(("set_reminder",)))
        orch._tool_specs = orch._skills.openai_tools()
        orch._memory = MemoryStore("unused", ephemeral=True)
        task = asyncio.create_task(orch._speak_reply([{"role": "user", "content": "x"}], None))
        await asyncio.sleep(0.05)
        orch._reply_drained.set()
        await task
        return llm, orch

    llm, orch = asyncio.run(_run())
    assert orch._memory.pending_reminders()[0][2] == "tea"
    followup, tools = llm.calls[1]
    assert tools                                         # tools offered again
    assistant = followup[-2]
    assert assistant["tool_calls"][0]["extra_content"] == sig     # signature echoed
    assert followup[-1] == {"role": "tool", "tool_call_id": "c1",
                            "content": followup[-1]["content"]}
    assert "tea" in followup[-1]["content"]
    assert orch.convo.last_assistant() == "Done, I'll remind you about tea."


def test_tool_mode_auto_uses_native_only_for_cloud():
    from tests.test_orchestrator import _FakeLLM, _FakeSTT, _FakeTTS
    from vcb.config import AppConfig
    from vcb.orchestrator import Orchestrator

    def make(backend):
        cfg = AppConfig()
        cfg.backend = backend
        cfg.memory.enabled = False
        return Orchestrator(cfg, _FakeSTT(), _FakeLLM([]), _FakeTTS())

    cloud, local = make("cloud"), make("ollama")
    assert cloud._tool_specs and '{"tool"' not in cloud._tools_block
    assert local._tool_specs is None and '{"tool"' in local._tools_block
