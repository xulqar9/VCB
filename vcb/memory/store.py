"""SQLite-backed long-term memory.

Every turn is stored with its embedding; recall is an in-RAM cosine search over
those embeddings. The in-RAM index is a preallocated buffer grown by doubling
(no full copy per insert) and optionally capped to the newest `max_index_rows`
turns; older turns remain in SQLite. Ephemeral mode uses an in-memory DB so a
session behaves normally but leaves nothing on disk.
"""

import contextlib
import json
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS turns (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts         REAL NOT NULL,
    session_id TEXT NOT NULL,
    role       TEXT NOT NULL,
    text       TEXT NOT NULL,
    embedding  BLOB
);
CREATE TABLE IF NOT EXISTS facts (
    key    TEXT PRIMARY KEY,
    value  TEXT NOT NULL,
    ts     REAL NOT NULL,
    source TEXT
);
CREATE TABLE IF NOT EXISTS summaries (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    text         TEXT NOT NULL,
    covers_range TEXT
);
CREATE TABLE IF NOT EXISTS consolidation_runs (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    ts           REAL NOT NULL,
    kind         TEXT NOT NULL,
    summary_id   INTEGER,
    summary_text TEXT,
    covers_range TEXT NOT NULL,
    facts_delta  TEXT NOT NULL,
    undone       INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS notes (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts   REAL NOT NULL,
    text TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS reminders (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    due_ts     REAL NOT NULL,
    text       TEXT NOT NULL,
    created_ts REAL NOT NULL,
    fired      INTEGER NOT NULL DEFAULT 0
);
"""


_INITIAL_CAPACITY = 256
_CONSOLIDATION_HISTORY_LIMIT = 100


def _read_consolidation_history(db: sqlite3.Connection, limit: int) -> list[dict]:
    exists = db.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='consolidation_runs'"
    ).fetchone()
    if not exists:
        return []
    rows = db.execute(
        "SELECT r.id, r.ts, r.kind, r.summary_id, "
        "COALESCE(s.text, r.summary_text), r.covers_range, "
        "r.facts_delta, r.undone FROM consolidation_runs r "
        "LEFT JOIN summaries s ON s.id = r.summary_id "
        "ORDER BY r.id DESC LIMIT ?", (max(1, min(limit, _CONSOLIDATION_HISTORY_LIMIT)),)
    ).fetchall()
    history = []
    for row in rows:
        evidence = []
        if row[5] and "-" in row[5]:
            try:
                start_id, end_id = (int(value) for value in row[5].split("-", 1))
            except ValueError:
                start_id = end_id = 0
            if start_id and end_id >= start_id:
                evidence = [
                    {"id": turn[0], "text": turn[1]}
                    for turn in db.execute(
                        "SELECT id, text FROM turns WHERE id BETWEEN ? AND ? "
                        "AND role = 'user' ORDER BY id",
                        (start_id, end_id),
                    ).fetchall()
                ]
        history.append({
            "id": row[0],
            "ts": row[1],
            "kind": row[2],
            "summary": row[4],
            "covers_range": row[5],
            "facts_delta": json.loads(row[6]),
            "undone": bool(row[7]),
            "evidence": evidence,
        })
    return history


def _undo_consolidation(db: sqlite3.Connection, run_id: int) -> None:
    db.execute("BEGIN IMMEDIATE")
    with db:
        row = db.execute(
            "SELECT summary_id, facts_delta, undone FROM consolidation_runs WHERE id = ?",
            (run_id,),
        ).fetchone()
        if row is None:
            raise ValueError("That memory update is no longer available.")
        summary_id, encoded_delta, undone = row
        if undone:
            raise ValueError("That memory update has already been undone.")
        latest = db.execute(
            "SELECT id FROM consolidation_runs WHERE undone = 0 ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if latest is None or latest[0] != run_id:
            raise ValueError("Only the newest applied memory update can be undone.")

        delta = json.loads(encoded_delta)
        for key, change in delta.items():
            current = db.execute(
                "SELECT value, source FROM facts WHERE key = ?", (key,)
            ).fetchone()
            after = change["after"]
            if after is None:
                matches = current is None
            else:
                matches = current is not None and current == (after["value"], after["source"])
            if not matches:
                raise ValueError(
                    f"Memory fact {key!r} changed since this update; refusing to overwrite it."
                )

        if summary_id is not None:
            latest_summary = db.execute(
                "SELECT id FROM summaries ORDER BY id DESC LIMIT 1"
            ).fetchone()
            if latest_summary is None or latest_summary[0] != summary_id:
                raise ValueError(
                    "A newer summary exists; refusing to undo over a later memory update."
                )

        for key, change in delta.items():
            before = change["before"]
            if before is None:
                db.execute("DELETE FROM facts WHERE key = ?", (key,))
            else:
                db.execute(
                    "INSERT INTO facts (key, value, ts, source) VALUES (?,?,?,?) "
                    "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                    "ts=excluded.ts, source=excluded.source",
                    (key, before["value"], before["ts"], before["source"]),
                )
        if summary_id is not None:
            db.execute("DELETE FROM summaries WHERE id = ?", (summary_id,))
        db.execute("UPDATE consolidation_runs SET undone = 1 WHERE id = ?", (run_id,))


def read_consolidation_history(db_path: str | Path, limit: int = 20) -> list[dict]:
    """Read recent memory changes for the local settings UI."""
    path = Path(db_path)
    if not path.is_file():
        return []
    with sqlite3.connect(path, timeout=1) as db:
        return _read_consolidation_history(db, limit)


def undo_latest_consolidation(db_path: str | Path, run_id: int) -> None:
    """Undo the newest applied memory update without overwriting later edits."""
    path = Path(db_path)
    if not path.is_file():
        raise ValueError("The memory database is unavailable.")
    with sqlite3.connect(path, timeout=1) as db:
        _undo_consolidation(db, run_id)


class MemoryStore:
    def __init__(self, db_path: str, ephemeral: bool = False,
                 max_index_rows: int = 0) -> None:
        if ephemeral:
            target = ":memory:"
        else:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            target = db_path
        # Accessed from asyncio worker threads (to_thread); allow cross-thread
        # use and serialize every access with a lock.
        self._db = sqlite3.connect(target, check_same_thread=False)
        self._lock = threading.Lock()
        self._db.executescript(_SCHEMA)
        self._db.commit()
        self._max_rows = max(0, max_index_rows)   # 0 = unlimited
        self._ids: list[int] = []          # row ids, chronological (insertion order)
        self._buf: np.ndarray | None = None  # embedding rows; only [:_n] are valid
        self._n = 0
        self._load_index()

    def _load_index(self) -> None:
        if self._max_rows:
            rows = self._db.execute(
                "SELECT id, embedding FROM turns WHERE embedding IS NOT NULL "
                "ORDER BY id DESC LIMIT ?", (self._max_rows,)
            ).fetchall()[::-1]
        else:
            rows = self._db.execute(
                "SELECT id, embedding FROM turns WHERE embedding IS NOT NULL ORDER BY id"
            ).fetchall()
        self._ids, self._buf, self._n = [], None, 0
        for rid, blob in rows:
            self._append(rid, np.frombuffer(blob, dtype=np.float32))

    def _append(self, rid: int, vec: np.ndarray) -> None:
        """Add one embedding to the index. Caller holds the lock (or is init)."""
        if self._buf is None:
            cap = _INITIAL_CAPACITY
            if self._max_rows:
                cap = min(cap, self._max_rows)
            self._buf = np.empty((cap, vec.shape[0]), dtype=np.float32)
        if self._max_rows and self._n >= self._max_rows:
            # Evict the oldest ~10% at once so the shift is amortized, not per insert.
            drop = max(1, self._max_rows // 10)
            keep = self._n - drop
            self._buf[:keep] = self._buf[drop:self._n]
            del self._ids[:drop]
            self._n = keep
        if self._n == len(self._buf):
            new_cap = len(self._buf) * 2
            if self._max_rows:
                new_cap = min(new_cap, self._max_rows)
            grown = np.empty((new_cap, self._buf.shape[1]), dtype=np.float32)
            grown[: self._n] = self._buf[: self._n]
            self._buf = grown
        self._buf[self._n] = vec
        self._ids.append(rid)
        self._n += 1

    def add_turn(self, session_id: str, role: str, text: str,
                 embedding: np.ndarray | None) -> None:
        blob = embedding.astype(np.float32).tobytes() if embedding is not None else None
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO turns (ts, session_id, role, text, embedding) VALUES (?,?,?,?,?)",
                (time.time(), session_id, role, text, blob),
            )
            self._db.commit()
            if embedding is not None:
                self._append(cur.lastrowid, embedding.astype(np.float32).ravel())

    def recall(self, query_vec: np.ndarray, top_k: int, min_similarity: float,
               exclude_recent: int = 0) -> list[str]:
        """Most similar past turns as "role: text", oldest-first, skipping the
        newest `exclude_recent` rows (those are already in the live conversation)."""
        return [f"{role}: {text}" for _ts, role, text in
                self.recall_entries(query_vec, top_k, min_similarity, exclude_recent)]

    def recall_entries(self, query_vec: np.ndarray, top_k: int, min_similarity: float,
                       exclude_recent: int = 0) -> list[tuple[float, str, str]]:
        """Like recall(), but as (timestamp, role, text) so callers can say
        when something was said."""
        with self._lock:
            cutoff = self._n - max(0, exclude_recent)
            k = min(top_k, cutoff)
            if self._buf is None or k <= 0:
                return []
            sims = self._buf[:cutoff] @ query_vec.astype(np.float32)
            # Partial selection: O(n) instead of sorting every stored turn.
            top = np.argpartition(-sims, k - 1)[:k]
            top = top[np.argsort(-sims[top])]
            hits = [self._ids[i] for i in top if sims[i] >= min_similarity]
            if not hits:
                return []
            placeholders = ",".join("?" * len(hits))
            return self._db.execute(
                f"SELECT ts, role, text FROM turns WHERE id IN ({placeholders}) ORDER BY id",
                hits,
            ).fetchall()

    def last_turn_ts(self) -> float | None:
        """When the most recent stored turn happened (None if never)."""
        with self._lock:
            row = self._db.execute("SELECT MAX(ts) FROM turns").fetchone()
        return row[0] if row and row[0] is not None else None

    # -- consolidation: rolling summary + durable profile facts -------------

    def last_covered_id(self) -> int:
        """Highest turn id already folded into a summary (0 if none)."""
        with self._lock:
            row = self._db.execute(
                "SELECT covers_range FROM summaries ORDER BY id DESC LIMIT 1"
            ).fetchone()
        if not row or not row[0] or "-" not in row[0]:
            return 0
        with contextlib.suppress(ValueError):
            return int(row[0].split("-")[1])
        return 0

    def unconsolidated_turns(
        self, limit: int | None = None
    ) -> list[tuple[int, str, str, float]]:
        """Turns newer than the latest summary as (id, role, text, ts), oldest-first."""
        last = self.last_covered_id()
        with self._lock:
            query = "SELECT id, role, text, ts FROM turns WHERE id > ? ORDER BY id"
            params: tuple = (last,)
            if limit is not None:
                query += " LIMIT ?"
                params = (last, max(1, limit))
            return self._db.execute(query, params).fetchall()

    def count_unconsolidated(self) -> int:
        last = self.last_covered_id()
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) FROM turns WHERE id > ?", (last,)
            ).fetchone()
        return row[0] if row else 0

    def add_summary(self, text: str, start_id: int, end_id: int) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO summaries (ts, text, covers_range) VALUES (?,?,?)",
                (time.time(), text, f"{start_id}-{end_id}"),
            )
            self._db.commit()

    def apply_consolidation(self, summary: str | None, start_id: int, end_id: int,
                            facts: dict[str, str], remove: list[str],
                            kind: str = "automatic") -> int | None:
        """Apply and record one reversible summary/fact update atomically."""
        with self._lock, self._db:
            self._db.execute("BEGIN IMMEDIATE")
            delta: dict[str, dict] = {}
            desired = {key: value for key, value in facts.items()}
            desired.update({key: None for key in remove if key not in desired})
            for key, value in desired.items():
                row = self._db.execute(
                    "SELECT value, source, ts FROM facts WHERE key = ?", (key,)
                ).fetchone()
                before = ({"value": row[0], "source": row[1], "ts": row[2]}
                          if row is not None else None)
                after = ({"value": value, "source": "consolidation"}
                         if value is not None else None)
                if before is None and after is None:
                    continue
                if (before is not None and after is not None
                        and before["value"] == after["value"]
                        and before["source"] == after["source"]):
                    continue
                delta[key] = {"before": before, "after": after}

            summary_id = None
            if summary:
                cursor = self._db.execute(
                    "INSERT INTO summaries (ts, text, covers_range) VALUES (?,?,?)",
                    (time.time(), summary, f"{start_id}-{end_id}"),
                )
                summary_id = cursor.lastrowid

            for key, change in delta.items():
                after = change["after"]
                if after is None:
                    self._db.execute("DELETE FROM facts WHERE key = ?", (key,))
                else:
                    self._db.execute(
                        "INSERT INTO facts (key, value, ts, source) VALUES (?,?,?,?) "
                        "ON CONFLICT(key) DO UPDATE SET value=excluded.value, "
                        "ts=excluded.ts, source=excluded.source",
                        (key, after["value"], time.time(), after["source"]),
                    )

            if summary_id is None and not delta:
                return None
            cursor = self._db.execute(
                "INSERT INTO consolidation_runs "
                "(ts, kind, summary_id, summary_text, covers_range, facts_delta) "
                "VALUES (?,?,?,?,?,?)",
                (time.time(), kind, summary_id, summary, f"{start_id}-{end_id}",
                 json.dumps(delta, ensure_ascii=True)),
            )
            run_id = cursor.lastrowid
            self._db.execute(
                "DELETE FROM consolidation_runs WHERE id NOT IN "
                "(SELECT id FROM consolidation_runs ORDER BY id DESC LIMIT ?)",
                (_CONSOLIDATION_HISTORY_LIMIT,),
            )
            return run_id

    def consolidation_history(self, limit: int = 20) -> list[dict]:
        with self._lock:
            return _read_consolidation_history(self._db, limit)

    def undo_consolidation(self, run_id: int) -> None:
        with self._lock:
            _undo_consolidation(self._db, run_id)

    def latest_summary(self) -> str | None:
        entry = self.latest_summary_entry()
        return entry[0] if entry else None

    def latest_summary_entry(self) -> tuple[str, float] | None:
        """(text, when it was written) of the newest summary."""
        with self._lock:
            row = self._db.execute(
                "SELECT text, ts FROM summaries ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return (row[0], row[1]) if row else None

    # -- reminders (survive restarts; due times are UTC epoch seconds) -----

    def add_reminder(self, due_ts: float, text: str) -> int:
        with self._lock:
            cur = self._db.execute(
                "INSERT INTO reminders (due_ts, text, created_ts) VALUES (?,?,?)",
                (due_ts, text, time.time()),
            )
            self._db.commit()
            return cur.lastrowid

    def pending_reminders(self) -> list[tuple[int, float, str]]:
        """Not-yet-delivered reminders as (id, due_ts, text), soonest first."""
        with self._lock:
            return self._db.execute(
                "SELECT id, due_ts, text FROM reminders WHERE fired = 0 ORDER BY due_ts"
            ).fetchall()

    def mark_reminder_fired(self, reminder_id: int) -> None:
        with self._lock:
            self._db.execute("UPDATE reminders SET fired = 1 WHERE id = ?", (reminder_id,))
            self._db.commit()

    def cancel_reminder(self, reminder_id: int) -> bool:
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM reminders WHERE id = ? AND fired = 0", (reminder_id,))
            self._db.commit()
            return cur.rowcount > 0

    def upsert_fact(self, key: str, value: str, source: str = "consolidation") -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO facts (key, value, ts, source) VALUES (?,?,?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value, ts=excluded.ts, "
                "source=excluded.source",
                (key, value, time.time(), source),
            )
            self._db.commit()

    def facts_block(self) -> str:
        """Durable profile facts as a bullet list, or empty string."""
        with self._lock:
            rows = self._db.execute(
                "SELECT key, value FROM facts ORDER BY key"
            ).fetchall()
        return "\n".join(f"- {k}: {v}" for k, v in rows)

    def facts(self) -> dict[str, str]:
        with self._lock:
            rows = self._db.execute("SELECT key, value FROM facts ORDER BY key").fetchall()
        return dict(rows)

    def delete_fact(self, key: str) -> None:
        with self._lock:
            self._db.execute("DELETE FROM facts WHERE key = ?", (key,))
            self._db.commit()

    def get_fact(self, key: str) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT value FROM facts WHERE key = ?", (key,)
            ).fetchone()
        return row[0] if row else None

    # -- notes / lists ------------------------------------------------------

    def add_note(self, text: str) -> None:
        with self._lock:
            self._db.execute(
                "INSERT INTO notes (ts, text) VALUES (?,?)", (time.time(), text)
            )
            self._db.commit()

    def list_notes(self) -> list[str]:
        with self._lock:
            rows = self._db.execute("SELECT text FROM notes ORDER BY id").fetchall()
        return [r[0] for r in rows]

    def clear_notes(self) -> None:
        with self._lock:
            self._db.execute("DELETE FROM notes")
            self._db.commit()

    def clear(self) -> None:
        with self._lock:
            self._db.executescript(
                "DELETE FROM turns; DELETE FROM facts; DELETE FROM summaries; "
                "DELETE FROM consolidation_runs; DELETE FROM notes; DELETE FROM reminders;"
            )
            self._db.commit()
            self._ids = []
            self._buf = None
            self._n = 0

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._db.close()
