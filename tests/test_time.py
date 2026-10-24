"""Date/time awareness: the user's clock, reminders, dated memories."""

import asyncio
import json
import time
from datetime import datetime, timedelta

import pytest

from vcb.clock import Clock, part_of_day, spoken_datetime, valid_timezone
from vcb.memory.store import MemoryStore
from vcb.skills.base import SkillContext
from vcb.skills.builtin import (
    CancelReminderSkill, DateTimeSkill, ListRemindersSkill, SetReminderSkill,
)


# -- clock ---------------------------------------------------------------------

def test_explicit_timezone_and_offsets():
    berlin = Clock("Europe/Berlin")
    assert berlin.explicit and berlin.zone_name == "Europe/Berlin"
    # Daylight saving is handled per date: summer UTC+2, winter UTC+1.
    summer = berlin.localize(datetime(2026, 7, 1, 12, 0))
    winter = berlin.localize(datetime(2026, 12, 1, 12, 0))
    assert berlin.offset_label(summer) == "UTC+02:00"
    assert berlin.offset_label(winter) == "UTC+01:00"
    ny = Clock("America/New_York")
    assert ny.offset_label(ny.localize(datetime(2026, 12, 1, 12, 0))) == "UTC-05:00"


def test_unknown_timezone_falls_back_to_pc():
    c = Clock("Mars/Olympus_Mons")
    assert not c.explicit and c.zone_name == ""
    assert c.now().tzinfo is not None
    assert not valid_timezone("Mars/Olympus_Mons") and valid_timezone("Asia/Tokyo")


def test_now_line_is_unambiguous():
    line = Clock("Europe/Berlin").now_line()
    now = Clock("Europe/Berlin").now()
    assert now.strftime("%A") in line and now.strftime("%B") in line
    assert now.strftime("%Y-%m-%d") in line and "Europe/Berlin" in line


def test_spoken_datetime_and_part_of_day():
    dt = datetime(2026, 10, 24, 1, 53)
    assert spoken_datetime(dt) == "Saturday, 24 October 2026, 1:53 AM"
    assert part_of_day(dt) == "night"
    assert part_of_day(dt.replace(hour=3)) == "late at night"
    assert part_of_day(dt.replace(hour=9)) == "morning"
    assert part_of_day(dt.replace(hour=19)) == "evening"


def test_ago_phrasing():
    c = Clock("Europe/Berlin")
    now = c.localize(datetime(2026, 10, 24, 18, 0))
    at = lambda **kw: (now - timedelta(**kw)).timestamp()  # noqa: E731
    assert c.ago(at(seconds=30), now) == "just now"
    assert c.ago(at(minutes=10), now) == "10 minutes ago"
    assert c.ago(at(hours=3), now) == "3 hours ago"
    assert c.ago(at(days=1), now) == "yesterday evening"
    late = c.localize(datetime(2026, 10, 24, 1, 50))
    assert c.ago((late - timedelta(hours=3)).timestamp(), late) == "last night"
    late = c.localize(datetime(2026, 10, 24, 1, 50))
    assert c.ago((late - timedelta(hours=3)).timestamp(), late) == "last night"
    assert c.ago(at(days=3), now) == "3 days ago (Wednesday)"
    assert c.ago(at(days=14), now) == "2 weeks ago"
    assert c.ago(at(days=200), now).startswith("on ")


# -- reminders (skills + store) --------------------------------------------------

def _ctx(store, clock):
    async def _speak(_text):
        pass
    return SkillContext(memory=store, speak=_speak, loop=None, clock=clock)


def test_set_reminder_at_local_time_is_stored_in_utc():
    clock = Clock("America/New_York")
    store = MemoryStore("unused", ephemeral=True)
    tomorrow = (clock.now() + timedelta(days=1)).strftime("%Y-%m-%d")
    reply = asyncio.run(SetReminderSkill().run(
        {"at": f"{tomorrow} 08:00", "text": "go to the office"}, _ctx(store, clock)))
    assert "tomorrow at 8:00 AM" in reply and "go to the office" in reply
    (rid, due, text), = store.pending_reminders()
    expected = clock.localize(datetime.fromisoformat(f"{tomorrow} 08:00")).timestamp()
    assert due == pytest.approx(expected) and text == "go to the office"


def test_set_reminder_in_minutes_and_rejects_the_past():
    clock = Clock("Europe/Berlin")
    store = MemoryStore("unused", ephemeral=True)
    reply = asyncio.run(SetReminderSkill().run(
        {"in_minutes": 30, "text": "tea"}, _ctx(store, clock)))
    assert "tea" in reply
    assert store.pending_reminders()[0][1] == pytest.approx(time.time() + 1800, abs=5)
    past = asyncio.run(SetReminderSkill().run(
        {"at": "2020-01-01 08:00", "text": "x"}, _ctx(store, clock)))
    assert "already passed" in past
    bad = asyncio.run(SetReminderSkill().run({"at": "next monday"}, _ctx(store, clock)))
    assert bad.startswith("Error")


def test_list_and_cancel_reminders():
    clock = Clock("Europe/Berlin")
    store = MemoryStore("unused", ephemeral=True)
    store.add_reminder(time.time() + 3600, "call mom")
    store.add_reminder(time.time() + 7200, "immigration office")
    listing = asyncio.run(ListRemindersSkill().run({}, _ctx(store, clock)))
    assert "call mom" in listing and "immigration office" in listing
    out = asyncio.run(CancelReminderSkill().run({"text": "office"}, _ctx(store, clock)))
    assert "immigration office" in out
    assert [r[2] for r in store.pending_reminders()] == ["call mom"]


def test_datetime_skill_uses_users_zone():
    clock = Clock("Asia/Tokyo")
    out = asyncio.run(DateTimeSkill().run({"what": "date"}, _ctx(None, clock)))
    assert clock.now().strftime("%A") in out


def test_store_timestamps():
    import numpy as np

    store = MemoryStore("unused", ephemeral=True)
    assert store.last_turn_ts() is None
    store.add_turn("s", "user", "my card ends on the 20th", np.array([1, 0], np.float32))
    (ts, role, text), = store.recall_entries(np.array([1, 0], np.float32), 5, 0.5)
    assert role == "user" and abs(ts - time.time()) < 5
    assert store.last_turn_ts() == pytest.approx(ts)
    assert len(store.unconsolidated_turns()[0]) == 4      # (id, role, text, ts)


# -- consolidation gets dates ----------------------------------------------------

def test_consolidation_sees_dates_and_keeps_date_facts():
    from vcb.memory.consolidate import consolidate

    clock = Clock("Europe/Berlin")
    said = clock.localize(datetime(2026, 10, 24, 1, 47)).timestamp()

    class _LLM:
        async def stream(self, messages, temperature=None, max_tokens=None):
            self.prompt = messages[1]["content"]
            yield json.dumps({"summary": "Visa card ends 2026-11-20.",
                              "facts": {"permit_card_expiry": "2026-11-20"}})

    llm = _LLM()
    turns = [(1, "user", "My physical card ends in 20th November.", said)]
    _s, facts, _r = asyncio.run(consolidate(llm, None, turns, {}, clock=clock))
    assert facts == {"permit_card_expiry": "2026-11-20"}    # numbers don't fail grounding
    assert "[Sat 24 Oct 2026, 01:47] user:" in llm.prompt
    assert "TODAY:" in llm.prompt


# -- orchestrator wiring -----------------------------------------------------------

from tests.test_orchestrator import _FakeLLM, _drain, _drive_turn, _orch  # noqa: E402


class _CapturingLLM(_FakeLLM):
    def __init__(self, deltas):
        super().__init__(deltas)
        self.seen = []

    async def stream(self, messages, temperature=None, max_tokens=None):
        self.seen.append(messages)
        async for d in super().stream(messages, temperature, max_tokens):
            yield d


def _system_text(messages):
    return "\n".join(m["content"] for m in messages if m["role"] == "system")


def test_every_reply_gets_the_current_date():
    async def _run():
        llm = _CapturingLLM(["It's Saturday."])
        orch = _orch(llm)
        orch.clock = Clock("Europe/Berlin")
        await _drive_turn(orch, "What day is it?")
        return llm.seen[-1], orch

    messages, orch = asyncio.run(_run())
    system = _system_text(messages)
    assert "for your own reference only" in system
    assert orch.clock.now().strftime("%A") in system
    # Recomputed each turn, never stored in the conversation history.
    assert all("for your own reference only" not in m["content"] for m in orch.convo._history)


def test_now_block_mentions_previous_conversation():
    orch = _orch(_FakeLLM([]))
    orch.clock = Clock("Europe/Berlin")
    orch._last_seen_ts = time.time() - 3 * 86400
    assert "3 days ago" in orch._now_block()
    orch._last_seen_ts = None
    assert "previous conversation" not in orch._now_block()


def test_greeting_gets_clock_and_what_she_knows():
    async def _run():
        llm = _CapturingLLM(["Morning!"])
        orch = _orch(llm)
        orch._memory = MemoryStore("unused", ephemeral=True)
        orch._memory.upsert_fact("permit_card_expiry", "2026-11-20")
        orch._start_lead_turn(greeting=True)
        await asyncio.sleep(0.05)
        orch._reply_drained.set()
        await orch._turn_task
        return llm.seen[-1]

    system = _system_text(asyncio.run(_run()))
    assert "for your own reference only" in system and "permit_card_expiry: 2026-11-20" in system


def test_reminder_loop_announces_due_and_missed_reminders():
    async def _run():
        orch = _orch(_FakeLLM([]))
        orch._memory = MemoryStore("unused", ephemeral=True)
        orch._memory.add_reminder(time.time() - 1, "stretch")
        orch._memory.add_reminder(time.time() - 3 * 3600, "go to the office")
        orch._memory.add_reminder(time.time() + 3600, "later")
        loop = asyncio.create_task(orch._reminder_loop())
        await asyncio.sleep(2.3)
        loop.cancel()
        return orch, [i.text for i in _drain(orch) if i is not None], orch.clock

    orch, spoken, _clock = asyncio.run(_run())
    joined = " ".join(spoken)
    assert "here's your reminder: stretch." in joined
    assert "you had a reminder" in joined and "go to the office" in joined
    assert "later" not in joined
    assert [r[2] for r in orch._memory.pending_reminders()] == ["later"]
