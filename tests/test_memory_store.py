"""MemoryStore: recall precision, recent-exclusion, facts, notes, consolidation
bookkeeping. Uses an ephemeral in-memory DB and hand-built unit vectors."""

import numpy as np

from vcb.memory.store import MemoryStore


def _store():
    return MemoryStore("unused", ephemeral=True)


def _vec(*xs):
    return np.array(xs, dtype=np.float32)


def test_recall_returns_most_similar():
    s = _store()
    s.add_turn("sess", "user", "I have a dog named Rex", _vec(1, 0, 0))
    s.add_turn("sess", "user", "I love rust programming", _vec(0, 1, 0))
    hits = s.recall(_vec(1, 0, 0), top_k=5, min_similarity=0.5)
    assert hits == ["user: I have a dog named Rex"]
    s.close()


def test_recall_respects_min_similarity():
    s = _store()
    s.add_turn("sess", "user", "unrelated", _vec(0, 1, 0))
    assert s.recall(_vec(1, 0, 0), top_k=5, min_similarity=0.5) == []
    s.close()


def test_recall_excludes_recent():
    s = _store()
    s.add_turn("sess", "user", "old dog fact", _vec(1, 0, 0))       # id 1
    s.add_turn("sess", "user", "recent rust fact", _vec(0, 1, 0))   # id 2 (newest)
    # Query matches the newest row, but exclude_recent=1 skips it.
    assert s.recall(_vec(0, 1, 0), top_k=5, min_similarity=0.5, exclude_recent=1) == []
    # The older row is still recallable.
    assert s.recall(_vec(1, 0, 0), top_k=5, min_similarity=0.5, exclude_recent=1) == [
        "user: old dog fact"
    ]
    s.close()


def test_facts_upsert_and_block():
    s = _store()
    s.upsert_fact("name", "Alex")
    s.upsert_fact("city", "Oslo")
    s.upsert_fact("name", "Alexandra")   # update in place
    block = s.facts_block()
    assert "- city: Oslo" in block
    assert "- name: Alexandra" in block
    assert "Alex\n" not in block         # old value gone
    assert s.get_fact("name") == "Alexandra"
    s.close()


def test_notes_roundtrip():
    s = _store()
    s.add_note("buy milk")
    s.add_note("call mom")
    assert s.list_notes() == ["buy milk", "call mom"]
    s.close()


def test_consolidation_bookkeeping():
    s = _store()
    for i in range(3):
        s.add_turn("sess", "user", f"turn {i}", _vec(1, 0, 0))
    assert s.count_unconsolidated() == 3
    assert s.last_covered_id() == 0
    s.add_summary("a summary", 1, 2)
    assert s.last_covered_id() == 2
    assert s.count_unconsolidated() == 1        # only id 3 remains
    remaining = s.unconsolidated_turns()
    assert [r[0] for r in remaining] == [3]
    s.close()


def test_consolidation_update_can_be_undone_without_losing_previous_facts():
    s = _store()
    s.add_summary("Earlier summary", 1, 2)
    s.upsert_fact("city", "Oslo", source="user")
    s.upsert_fact("weather", "windy")

    run_id = s.apply_consolidation(
        "Updated summary", 3, 5, {"city": "Berlin", "hobby": "chess"}, ["weather"],
        kind="quiet-time",
    )
    assert run_id is not None
    assert s.latest_summary() == "Updated summary"
    assert s.facts() == {"city": "Berlin", "hobby": "chess"}
    entry, = s.consolidation_history()
    assert entry["kind"] == "quiet-time"
    assert entry["summary"] == "Updated summary"

    s.undo_consolidation(run_id)

    assert s.latest_summary() == "Earlier summary"
    assert s.facts() == {"city": "Oslo", "weather": "windy"}
    assert s._db.execute("SELECT source FROM facts WHERE key='city'").fetchone()[0] == "user"
    undone, = s.consolidation_history()
    assert undone["undone"] is True
    assert undone["summary"] == "Updated summary"
    s.close()


def test_undo_refuses_to_overwrite_a_fact_changed_since_consolidation():
    s = _store()
    s.upsert_fact("city", "Oslo")
    run_id = s.apply_consolidation("Moved", 1, 2, {"city": "Berlin"}, [])
    s.upsert_fact("city", "Paris", source="user")

    try:
        s.undo_consolidation(run_id)
    except ValueError as exc:
        assert "changed since" in str(exc)
    else:
        raise AssertionError("undo should refuse to overwrite a newer fact")
    assert s.get_fact("city") == "Paris"
    s.close()


def test_undo_only_allows_the_newest_applied_consolidation():
    s = _store()
    first = s.apply_consolidation("First", 1, 2, {}, [])
    second = s.apply_consolidation("Second", 3, 4, {}, [])
    assert first is not None and second is not None

    try:
        s.undo_consolidation(first)
    except ValueError as exc:
        assert "newest applied" in str(exc)
    else:
        raise AssertionError("undo should only allow the newest applied update")
    s.undo_consolidation(second)
    assert s.latest_summary() == "First"
    s.close()


def test_history_survives_restart_and_can_be_undone_from_settings(tmp_path):
    from vcb.memory.store import read_consolidation_history, undo_latest_consolidation

    db_path = tmp_path / "memory.db"
    s = MemoryStore(str(db_path))
    s.add_turn("sess", "user", "I moved to Berlin.", _vec(1, 0, 0))
    s.add_turn("sess", "assistant", "I will remember.", _vec(0, 1, 0))
    s.add_turn("sess", "user", "My old city was Oslo.", _vec(1, 0, 0))
    s.upsert_fact("city", "Oslo")
    run_id = s.apply_consolidation("Moved to Berlin", 1, 3, {"city": "Berlin"}, [])
    s.close()

    history, = read_consolidation_history(db_path)
    assert history["id"] == run_id
    assert [turn["text"] for turn in history["evidence"]] == [
        "I moved to Berlin.", "My old city was Oslo."
    ]
    undo_latest_consolidation(db_path, run_id)
    s = MemoryStore(str(db_path))
    assert s.get_fact("city") == "Oslo"
    assert s.consolidation_history()[0]["undone"] is True
    s.close()


def test_existing_database_without_history_table_is_migrated(tmp_path):
    from vcb.memory.store import read_consolidation_history

    db_path = tmp_path / "memory.db"
    s = MemoryStore(str(db_path))
    s._db.execute("DROP TABLE consolidation_runs")
    s._db.commit()
    s.close()

    assert read_consolidation_history(db_path) == []
    s = MemoryStore(str(db_path))
    assert s.consolidation_history() == []
    s.close()


def test_clear_wipes_everything():
    s = _store()
    s.add_turn("sess", "user", "x", _vec(1, 0, 0))
    s.upsert_fact("k", "v")
    s.add_note("n")
    s.clear()
    assert s.recall(_vec(1, 0, 0), top_k=5, min_similarity=0.0) == []
    assert s.facts_block() == ""
    assert s.list_notes() == []
    assert s.consolidation_history() == []
    s.close()


def test_recall_orders_hits_and_limits_top_k():
    s = _store()
    s.add_turn("sess", "user", "close", _vec(0.9, 0.1, 0))
    s.add_turn("sess", "user", "exact", _vec(1, 0, 0))
    s.add_turn("sess", "user", "far", _vec(0.6, 0.8, 0))
    # Top-2 by similarity, returned oldest-first.
    assert s.recall(_vec(1, 0, 0), top_k=2, min_similarity=0.0) == [
        "user: close", "user: exact"
    ]
    s.close()


def test_index_grows_past_initial_capacity():
    from vcb.memory import store as store_mod

    s = _store()
    n = store_mod._INITIAL_CAPACITY * 2 + 3
    for i in range(n):
        s.add_turn("sess", "user", f"t{i}", _vec(0, 1, 0))
    s.add_turn("sess", "user", "needle", _vec(1, 0, 0))
    assert s._n == n + 1
    assert s.recall(_vec(1, 0, 0), top_k=1, min_similarity=0.5) == ["user: needle"]
    s.close()


def test_index_cap_evicts_oldest_but_keeps_db():
    s = MemoryStore("unused", ephemeral=True, max_index_rows=20)
    s.add_turn("sess", "user", "oldest", _vec(1, 0, 0))
    for i in range(30):
        s.add_turn("sess", "user", f"t{i}", _vec(0, 1, 0))
    assert s._n <= 20
    assert len(s._ids) == s._n
    # Evicted from the RAM index, so no longer recalled...
    assert s.recall(_vec(1, 0, 0), top_k=5, min_similarity=0.5) == []
    # ...but still in SQLite for consolidation.
    assert s.count_unconsolidated() == 31
    s.close()


def test_index_cap_applies_on_reload(tmp_path):
    db = str(tmp_path / "mem.db")
    s = MemoryStore(db)
    for i in range(10):
        s.add_turn("sess", "user", f"t{i}", _vec(1, 0, 0))
    s.close()
    s = MemoryStore(db, max_index_rows=4)
    assert s._n == 4
    assert s.recall(_vec(1, 0, 0), top_k=10, min_similarity=0.5) == [
        "user: t6", "user: t7", "user: t8", "user: t9"
    ]
    s.close()
