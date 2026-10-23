"""SQLite-backed long-term memory.

Every turn is stored with its embedding; recall is an in-RAM cosine search over
those embeddings. The in-RAM index is a preallocated buffer grown by doubling
(no full copy per insert) and optionally capped to the newest `max_index_rows`
turns; older turns remain in SQLite. Ephemeral mode uses an in-memory DB so a
session behaves normally but leaves nothing on disk.
"""

import contextlib
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
CREATE TABLE IF NOT EXISTS notes (
    id   INTEGER PRIMARY KEY AUTOINCREMENT,
    ts   REAL NOT NULL,
    text TEXT NOT NULL
);
"""


_INITIAL_CAPACITY = 256


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
        """Most similar past turns, oldest-first, skipping the newest
        `exclude_recent` rows (those are already in the live conversation)."""
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
            rows = self._db.execute(
                f"SELECT role, text FROM turns WHERE id IN ({placeholders}) ORDER BY id",
                hits,
            ).fetchall()
        return [f"{role}: {text}" for role, text in rows]

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

    def unconsolidated_turns(self) -> list[tuple[int, str, str]]:
        """Turns newer than the latest summary, oldest-first."""
        last = self.last_covered_id()
        with self._lock:
            return self._db.execute(
                "SELECT id, role, text FROM turns WHERE id > ? ORDER BY id",
                (last,),
            ).fetchall()

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

    def latest_summary(self) -> str | None:
        with self._lock:
            row = self._db.execute(
                "SELECT text FROM summaries ORDER BY id DESC LIMIT 1"
            ).fetchone()
        return row[0] if row else None

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
                "DELETE FROM notes;"
            )
            self._db.commit()
            self._ids = []
            self._buf = None
            self._n = 0

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self._db.close()
