"""Local settings window for inspecting and undoing memory consolidation."""

from __future__ import annotations

import sqlite3
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import messagebox, ttk

from .store import read_consolidation_history, undo_latest_consolidation


def _short(text: str, limit: int = 100) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[:limit - 3] + "..."


def _full_fact_changes(delta: dict) -> str:
    changes = []
    for key, change in delta.items():
        before = change["before"]
        after = change["after"]
        old = before["value"] if before else "(new)"
        new = after["value"] if after else "(removed)"
        changes.append(f"{key}: {old} -> {new}")
    return "; ".join(changes)


def _fact_changes(delta: dict) -> str:
    return _short(_full_fact_changes(delta))


class MemoryHistoryWindow:
    def __init__(self, parent: tk.Misc, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.window = tk.Toplevel(parent)
        self.window.title("VCB memory updates")
        self.window.geometry("800x500")
        self.window.transient(parent)
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(1, weight=1)

        ttk.Label(
            self.window,
            text=("Recent summary and profile changes. They stay on this PC. "
                  "Only the newest applied update can be undone, to protect newer changes."),
            wraplength=720,
            justify="left",
        ).grid(row=0, column=0, sticky="w", padx=12, pady=10)

        columns = ("when", "kind", "summary", "facts", "status")
        self.table = ttk.Treeview(self.window, columns=columns, show="headings", height=7)
        for column, title, width in (
            ("when", "When", 120), ("kind", "Update", 85), ("summary", "Summary", 190),
            ("facts", "Profile changes", 250), ("status", "Status", 75),
        ):
            self.table.heading(column, text=title)
            self.table.column(column, width=width, anchor="w")
        self.table.grid(row=1, column=0, sticky="nsew", padx=12)
        scrollbar = ttk.Scrollbar(self.window, orient="vertical", command=self.table.yview)
        scrollbar.grid(row=1, column=1, sticky="ns", pady=0)
        self.table.configure(yscrollcommand=scrollbar.set)
        self.table.bind("<<TreeviewSelect>>", self._selection_changed)

        self.details = tk.Text(self.window, height=7, wrap="word", state="disabled")
        self.details.grid(row=2, column=0, sticky="nsew", padx=12, pady=(8, 0))
        details_scrollbar = ttk.Scrollbar(
            self.window, orient="vertical", command=self.details.yview)
        details_scrollbar.grid(row=2, column=1, sticky="ns", pady=(8, 0))
        self.details.configure(yscrollcommand=details_scrollbar.set)

        buttons = ttk.Frame(self.window)
        buttons.grid(row=3, column=0, sticky="e", padx=12, pady=10)
        self.undo_button = ttk.Button(buttons, text="Undo selected update", command=self._undo)
        self.undo_button.pack(side="left", padx=(0, 8))
        ttk.Button(buttons, text="Close", command=self.window.destroy).pack(side="left")
        self._refresh()

    def _refresh(self) -> None:
        for item in self.table.get_children():
            self.table.delete(item)
        try:
            self.rows = read_consolidation_history(self.db_path)
        except (OSError, sqlite3.Error, ValueError) as exc:
            messagebox.showerror(
                "Could not read memory updates", str(exc), parent=self.window)
            self.rows = []
        latest = next((row["id"] for row in self.rows if not row["undone"]), None)
        self.latest_id = latest
        for row in self.rows:
            stamp = datetime.fromtimestamp(row["ts"]).strftime("%Y-%m-%d %H:%M")
            summary = _short(row["summary"] or "(no summary)")
            self.table.insert(
                "", "end", iid=str(row["id"]),
                values=(stamp, row["kind"], summary, _fact_changes(row["facts_delta"]),
                        "Undone" if row["undone"] else "Applied"),
            )
        if self.rows:
            self.table.selection_set(str(self.rows[0]["id"]))
            self.table.focus(str(self.rows[0]["id"]))
        self._selection_changed()

    def _selection_changed(self, _event=None) -> None:
        selected = self.table.selection()
        enabled = bool(selected and int(selected[0]) == self.latest_id)
        self.undo_button.configure(state="normal" if enabled else "disabled")
        row = next((item for item in self.rows
                    if selected and item["id"] == int(selected[0])), None)
        details = ""
        if row is not None:
            details = row["summary"] or "(no summary)"
            changes = _full_fact_changes(row["facts_delta"])
            if changes:
                details += f"\n\nProfile changes: {changes}"
            if row["evidence"]:
                details += "\n\nUser source turns:"
                for turn in row["evidence"]:
                    details += f"\n[{turn['id']}] {turn['text']}"
            else:
                details += "\n\nNo user source turns were retained for this update."
        elif not self.rows:
            details = "No recorded memory updates yet."
        self.details.configure(state="normal")
        self.details.delete("1.0", "end")
        self.details.insert("1.0", details)
        self.details.configure(state="disabled")

    def _undo(self) -> None:
        selected = self.table.selection()
        if not selected or int(selected[0]) != self.latest_id:
            return
        if not messagebox.askyesno(
            "Undo memory update",
            "Undo this update and restore the previous summary and profile facts?",
            parent=self.window,
        ):
            return
        try:
            undo_latest_consolidation(self.db_path, int(selected[0]))
        except (OSError, sqlite3.Error, ValueError) as exc:
            messagebox.showerror("Could not undo memory update", str(exc), parent=self.window)
            self._refresh()
            return
        self._refresh()
