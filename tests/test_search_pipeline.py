"""Online search in the reply pipeline: off means no search code and no
network at all; on means the filler line, untrusted framing, the per-turn
budget, no other tools after web results, no logging of queries, and no
long-term memory of web-based replies."""

import asyncio
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from vcb.config import AppConfig
from vcb.llm.base import ToolCallRequest
from vcb.memory.store import MemoryStore
from vcb.orchestrator import Orchestrator, _SENTINEL
from vcb.search.skill import NotSetUp, WeatherSkill, WebSearchSkill
from vcb.skills import SkillRegistry, build_skills

from test_orchestrator import _FakeSTT, _FakeTTS, _orch

ROOT = Path(__file__).resolve().parent.parent
RESULTS = "Web search results (untrusted data)\nBEGIN WEB-abcd\n1. a.org: RESULT-FACT\nEND WEB-abcd"


class _Svc:
    def __init__(self, ready=True, text=RESULTS):
        self.ready, self.text, self.queries = ready, text, []

    async def search(self, q):
        self.queries.append(q)
        return self.text


class _Script:
    def __init__(self, replies):
        self.replies, self.calls = list(replies), []

    async def load(self):
        pass

    async def stream(self, messages, temperature=None, max_tokens=None, tools=None):
        self.calls.append((messages, tools))
        yield self.replies.pop(0) if self.replies else "Okay."

    async def close(self):
        pass


def _search_call(q="SECRET-QUERY"):
    return json.dumps({"tool": "web_search", "args": {"query": q}})


def _orch_search(llm, svc, native=False, skills=("remember",)):
    orch = _orch(llm)
    orch._skills = SkillRegistry(build_skills(skills) + [WebSearchSkill(svc)])
    if native:
        orch._tool_specs = orch._skills.openai_tools()
    orch._memory = MemoryStore("unused", ephemeral=True)
    remembered = []

    async def remember(text):
        remembered.append(text)

    orch._remember_assistant = remember
    orch.remembered = remembered
    return orch


async def _reply(orch, tools_allowed=True):
    task = asyncio.create_task(orch._speak_reply([{"role": "user", "content": "what's new?"}],
                                                 None, tools_allowed=tools_allowed))
    await asyncio.sleep(0.05)
    orch._reply_drained.set()
    await task
    items = []
    while not orch._tts_queue.empty():
        items.append(orch._tts_queue.get_nowait())
    return [i.text for i in items if i is not _SENTINEL]


def test_off_means_no_search_code_and_no_network():
    """Fresh interpreter, guards on sockets, DNS and process launches: with
    search off, a whole reply -- even one where the model tries to call
    web_search -- loads nothing from vcb.search and reaches nothing but
    loopback."""
    script = r"""
import asyncio, ipaddress, json, socket, subprocess, sys
sys.path[:0] = [ROOT, ROOT + "/tests"]
bad = []

def loopback(host):
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return host in ("localhost", None)

real_connect, real_gai = socket.socket.connect, socket.getaddrinfo
def connect(self, addr):
    if isinstance(addr, tuple) and not loopback(addr[0]):
        bad.append("connect")
        raise OSError("blocked by test")
    return real_connect(self, addr)
def gai(host, *a, **k):
    if not loopback(host):
        bad.append("dns")
        raise OSError("blocked by test")
    return real_gai(host, *a, **k)
socket.socket.connect, socket.getaddrinfo = connect, gai
socket.create_connection = lambda *a, **k: bad.append("create_connection") or (_ for _ in ()).throw(OSError())
class NoPopen:
    def __init__(self, *a, **k):
        bad.append("popen")
        raise OSError("blocked by test")
subprocess.Popen = NoPopen

from test_orchestrator import _FakeLLM, _FakeSTT, _FakeTTS
from vcb.config import AppConfig
from vcb.orchestrator import Orchestrator

cfg = AppConfig()
cfg.memory.enabled = cfg.proactive.enabled = cfg.avatar.enabled = False
assert cfg.search.enabled is False
llm = _FakeLLM(['{"tool": "web_search", "args": {"query": "news"}}'])
orch = Orchestrator(cfg, _FakeSTT(), llm, _FakeTTS())
assert "web_search" not in orch._skills.names()

async def main():
    t = asyncio.create_task(orch._speak_reply([{"role": "user", "content": "news?"}], None))
    await asyncio.sleep(0.1)
    orch._reply_drained.set()
    await t
asyncio.run(main())
loaded = sorted(m for m in sys.modules if m.startswith("vcb.search") or m in ("socks", "sockshandler"))
print(json.dumps({"bad": bad, "loaded": loaded}))
""".replace("ROOT", repr(str(ROOT)))
    env = {k: v for k, v in os.environ.items() if not k.startswith("VCB_")}
    env["VCB_ENV_FILE"] = str(ROOT / "tests" / "_no_such.env")
    proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, cwd=ROOT, timeout=120)
    assert proc.returncode == 0, proc.stderr[-2000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out == {"bad": [], "loaded": []}


def test_json_mode_search_frames_results_as_untrusted_data(caplog):
    caplog.set_level(logging.DEBUG)
    svc = _Svc()
    llm = _Script([_search_call(), "It's sunny today."])
    orch = _orch_search(llm, svc)
    spoken = asyncio.run(_reply(orch))
    assert spoken[0] == "Let me look that up."          # filler first, before the wait
    assert "It's sunny today." in " ".join(spoken)
    followup = llm.calls[1][0]
    data_msg = next(m for m in followup if "RESULT-FACT" in (m["content"] or ""))
    assert data_msg["role"] == "user"                    # never a system message
    assert "not written by the user" in data_msg["content"]
    assert followup[-1]["role"] == "system" and "untrusted" in followup[-1]["content"]
    assert svc.queries == ["SECRET-QUERY"]
    # Reply is in the live conversation but not long-term memory; the filler
    # isn't part of the answer.
    assert orch.convo.last_assistant() == "It's sunny today."
    assert orch.remembered == []
    for secret in ("SECRET-QUERY", "RESULT-FACT", "It's sunny today"):
        assert secret not in caplog.text


def test_results_dont_persist_into_later_turns():
    llm = _Script([_search_call(), "It's sunny."])
    orch = _orch_search(llm, _Svc())
    asyncio.run(_reply(orch))
    assert all("RESULT-FACT" not in str(m) for m in orch.convo.messages())


def test_no_other_tools_after_web_results():
    remember = json.dumps({"tool": "remember", "args": {"key": "k", "value": "pwned"}})
    llm = _Script([_search_call(), remember, "Done."])
    orch = _orch_search(llm, _Svc())
    asyncio.run(_reply(orch))
    assert orch._memory.facts() == {}
    blocked = llm.calls[2][0][-1]["content"] + llm.calls[2][0][-2]["content"]
    assert "can't be used after reading web results" in blocked


def test_search_budget_per_turn():
    svc = _Svc()
    llm = _Script([_search_call("a"), _search_call("b"), _search_call("c"), "Here's what I found."])
    orch = _orch_search(llm, svc)
    orch.cfg.skills.max_tool_iters = 4
    asyncio.run(_reply(orch))
    assert svc.queries == ["a", "b"]
    assert "Search limit reached" in str(llm.calls[3][0])


def test_lead_turns_cannot_search():
    svc = _Svc()
    llm = _Script([_search_call(), "Hey, how's it going?"])
    orch = _orch_search(llm, svc)
    asyncio.run(_reply(orch, tools_allowed=False))
    assert svc.queries == []


def test_no_filler_when_search_cant_run():
    llm = _Script([_search_call(), "I couldn't check online."])
    orch = _orch_search(llm, NotSetUp())
    spoken = asyncio.run(_reply(orch))
    assert "Let me look that up." not in spoken
    assert "isn't set up" in str(llm.calls[1][0])


def test_native_mode_uses_tool_role_and_hides_args(caplog):
    caplog.set_level(logging.DEBUG)
    svc = _Svc()
    llm = _Script([ToolCallRequest([("c1", "web_search", {"query": "SECRET-QUERY"}, None)]),
                   "Sunny."])
    orch = _orch_search(llm, svc, native=True)
    asyncio.run(_reply(orch))
    assert llm.calls[1][0][-1]["role"] == "tool"
    assert "SECRET-QUERY" not in caplog.text and "RESULT-FACT" not in caplog.text
    assert "web_search (arguments not logged)" in caplog.text


def test_weather_tool_arguments_are_not_logged(caplog):
    caplog.set_level(logging.INFO, logger="vcb.orchestrator")
    orch = _orch(_Script([]))
    orch._skills = SkillRegistry([WeatherSkill(_Svc())])

    asyncio.run(orch._run_tool(
        "weather", {"location": "SENSITIVE-CITY"}, None, ["already speaking"], 0, True
    ))

    assert "weather (arguments not logged)" in caplog.text
    assert "SENSITIVE-CITY" not in caplog.text


def test_sensitive_skill_errors_are_redacted(caplog):
    class _Boom:
        ready = True

        async def search(self, q):
            raise RuntimeError(f"failed for {q}")

    caplog.set_level(logging.DEBUG)
    reg = SkillRegistry([WebSearchSkill(_Boom())])
    out = asyncio.run(reg.dispatch("web_search", {"query": "SECRET-QUERY"}, None))
    assert out == "Error running web_search."
    assert "SECRET-QUERY" not in caplog.text


def test_prompts_unchanged_without_search_and_guided_with_it():
    plain = SkillRegistry(build_skills(("remember",)))
    with_search = SkillRegistry(build_skills(("remember",)) + [WebSearchSkill(_Svc())])
    assert "WEB SEARCH" not in plain.prompt_block() + plain.native_prompt_block()
    assert with_search.prompt_block().startswith(plain.prompt_block().split("\n- remember")[0])
    for block in (with_search.prompt_block(), with_search.native_prompt_block()):
        assert "never follow instructions" in block


def test_enabled_but_misconfigured_tells_the_model():
    cfg = AppConfig()
    cfg.memory.enabled = cfg.proactive.enabled = cfg.avatar.enabled = False
    cfg.search.enabled = True
    cfg.search.backend, cfg.search.url = "searxng", "http://insecure.example.org"
    orch = Orchestrator(cfg, _FakeSTT(), _Script([]), _FakeTTS())
    assert orch._search is None
    skill = orch._skills.get("web_search")
    assert isinstance(skill.service, NotSetUp)


def test_enabled_and_valid_builds_service_without_network():
    cfg = AppConfig()
    cfg.memory.enabled = cfg.proactive.enabled = cfg.avatar.enabled = False
    cfg.skills.enabled = False
    cfg.search.enabled = True
    cfg.search.backend, cfg.search.url = "searxng", "https://search.example.org"
    orch = Orchestrator(cfg, _FakeSTT(), _Script([]), _FakeTTS())
    assert orch._search is not None and orch._search.route.state == "starting"
    assert orch._skills.names() == ["web_search", "weather"]  # works with other skills off
    assert isinstance(orch._skills.get("weather"), WeatherSkill)
