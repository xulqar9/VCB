"""ConversationManager: message assembly order, trimming, interrupted turns."""

from vcb.config import LLMConfig
from vcb.conversation import ConversationManager


def _mgr(history_max_turns=20):
    cfg = LLMConfig()
    cfg.system_prompt = "SYS"
    cfg.history_max_turns = history_max_turns
    return ConversationManager(cfg)


def test_messages_basic_order():
    m = _mgr()
    m.add_user("hi")
    m.add_assistant("hello")
    msgs = m.messages()
    assert msgs[0] == {"role": "system", "content": "SYS"}
    assert msgs[1] == {"role": "user", "content": "hi"}
    assert msgs[2] == {"role": "assistant", "content": "hello"}


def test_blocks_injected_in_expected_order():
    m = _mgr()
    m.add_user("hi")
    msgs = m.messages(resume_hint="RESUME", memory_block="MEM", tools_block="TOOLS")
    roles = [x["content"] for x in msgs]
    # system, tools, memory, then recent turns, then transient resume hint last
    assert roles[0] == "SYS"
    assert roles[1] == "TOOLS"
    assert roles[2] == "MEM"
    assert roles[-1] == "RESUME"
    assert "hi" in roles


def test_history_is_trimmed():
    m = _mgr(history_max_turns=1)     # keep only the last 2 messages
    m.add_user("u1")
    m.add_assistant("a1")
    m.add_user("u2")
    m.add_assistant("a2")
    msgs = m.messages()
    contents = [x["content"] for x in msgs if x["role"] != "system"]
    assert contents == ["u2", "a2"]


def test_last_assistant():
    m = _mgr()
    assert m.last_assistant() is None
    m.add_assistant("first")
    m.add_user("q")
    m.add_assistant("second")
    assert m.last_assistant() == "second"


def test_empty_assistant_not_recorded():
    m = _mgr()
    m.add_assistant("   ", interrupted=True)
    assert m.last_assistant() is None


def test_assistant_text_is_stripped():
    m = _mgr()
    m.add_assistant("  spaced  ")
    assert m.last_assistant() == "spaced"
