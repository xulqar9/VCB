"""Settings window part for optional online search: the "Enable online
search" switch and its guided setup dialog.

Stdlib + tkinter only: it works before anything search-related is installed.
Installing and testing run in a separate process (vcb.search.install), so a
crash or hang there can't take the settings window down, and Cancel can stop
it cleanly (the installer rolls back what it added).
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import tkinter as tk
import webbrowser
from tkinter import messagebox, ttk

from ..config import SEARCH_KEY_ENVS, TOR_DEFAULT_PROXY, SearchConfig
from .adapters import list_adapters, looks_like_vpn

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BRAVE_KEY_PAGE = "https://api-dashboard.search.brave.com/"
CREATE_NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
ADAPTER_POLL_MS = 2000          # how often the VPN connection status is re-read
WRAP = 560
GRAY, GREEN, ORANGE, RED, LINK = "gray", "#107010", "#b06000", "#c00000", "#0a5fb4"

# .env key -> default
FIELDS = {
    "VCB_SEARCH_ROUTE": "tor",
    "VCB_SEARCH_TOR_MODE": "managed",
    "VCB_SEARCH_TOR_PROXY": TOR_DEFAULT_PROXY,
    "VCB_SEARCH_VPN_MODE": "adapter",
    "VCB_SEARCH_VPN_ADAPTER": "",
    "VCB_SEARCH_VPN_PROXY": "",
    "VCB_SEARCH_BACKEND": "brave",
    "VCB_SEARCH_URL": "",
    "VCB_SEARCH_PARAMS": "",
    "VCB_SEARXNG_AUTH_MODE": "cookie",
}

PRIVACY_TEXT = (
    "\u2022 Searches go only through the route you choose (Tor or your VPN). If it isn't "
    "working, VCB simply doesn't search; it never uses your normal connection instead.\n\n"
    "\u2022 Website names are looked up through the same route, so your internet provider "
    "can't see what is searched (no \"DNS leak\"). Setup checks this for you.\n\n"
    "\u2022 Your questions, the search results and your IP address are never written to "
    "VCB's logs.\n\n"
    "\u2022 Web pages can contain tricks. VCB treats results as untrusted text and tells the "
    "assistant never to follow instructions found in them.\n\n"
    "\u2022 The assistant decides to search only when a question needs current information. "
    "You can switch online search off at any time.\n\n"
    "\u2022 If the assistant's brain is a Cloud API, that provider sees your questions and "
    "the results, like the rest of the conversation.")

BRAVE_STEPS = (
    "Brave needs an API key. Getting one takes about 3 minutes:\n"
    "  1.  Click \"Get a key\" and create an account. Confirm the email Brave sends you.\n"
    "  2.  Choose a Search plan. Brave asks for a payment card; its plans page shows "
    "what's free and what costs money.\n"
    "  3.  Open \"API Keys\", click \"Add API key\", copy the key, and paste it below.")

SEARXNG_HELP = (
    "Optional: ask whoever runs the server for a private-engine token. If you run the server, "
    "its owner must first add that token to the engine's tokens setting. VCB checks that it "
    "unlocks an engine. JSON output must be on; most public servers block apps.")


def _truthy(v: str) -> bool:
    return v.strip().lower() in ("1", "true", "yes", "on")


def _installed(values: dict[str, str]) -> bool:
    from . import install
    cfg = SearchConfig(route=values["VCB_SEARCH_ROUTE"], tor_mode=values["VCB_SEARCH_TOR_MODE"])
    return install.is_installed(cfg)


def _anything_installed() -> bool:
    from . import install
    return install.STAMP.exists() or install.TOR_DIR.exists()


def _show(widget, on: bool) -> None:
    widget.grid() if on else widget.grid_remove()


def describe_choice(values: dict[str, str]) -> str:
    if values["VCB_SEARCH_ROUTE"] == "tor":
        route = ("Tor (run by VCB)" if values["VCB_SEARCH_TOR_MODE"] == "managed"
                 else "your own Tor")
    else:
        route = ("your VPN's proxy" if values["VCB_SEARCH_VPN_MODE"] == "proxy"
                 else f"your VPN ({values['VCB_SEARCH_VPN_ADAPTER'] or 'no adapter picked'})")
    backend = "Brave Search" if values["VCB_SEARCH_BACKEND"] == "brave" else "SearXNG"
    return f"{backend} through {route}"


class SearchSection:
    """The checkbox + status line in the main settings window. `updates()`
    gives the .env changes to save."""

    def __init__(self, owner, env: dict[str, str]) -> None:
        self.owner = owner                       # vcb.settings.SettingsWindow
        self.env = env
        self.values = {k: env.get(k, d) for k, d in FIELDS.items()}
        # Optional .env override of the key variable's NAME (advanced users).
        self.key_override = env.get("VCB_SEARCH_API_KEY_ENV", "").strip()
        # One key per search service, so a key is only ever sent to its own service.
        self.keys = {b: env.get(self.key_env_for(b), "") for b in SEARCH_KEY_ENVS}
        self.was_on = _truthy(env.get("VCB_ONLINE_SEARCH", "false"))
        self.configured = False                  # dialog OK'd this session
        self.enabled = tk.BooleanVar(value=self.was_on)

        frame = owner.frame
        ttk.Checkbutton(frame, text="Enable online search", variable=self.enabled,
                        command=self._toggled).grid(row=owner.row, column=0, columnspan=2,
                                                    sticky="w")
        self.setup_btn = ttk.Button(frame, text="Set up & test...", command=self.open_dialog)
        self.setup_btn.grid(row=owner.row, column=2, sticky="e")
        owner.row += 1
        self.status = ttk.Label(frame, foreground="gray", wraplength=600, justify="left")
        self.status.grid(row=owner.row, column=0, columnspan=3, sticky="w", pady=(2, 0))
        owner.row += 1
        self.refresh()

    def refresh(self) -> None:
        cloud = self.owner.backend.get() == "cloud"
        if not self.enabled.get():
            text = ("Off. Lets the assistant look up current information on the web "
                    "(news, weather, opening hours...), privately through Tor or your VPN. "
                    "Nothing is downloaded until you turn it on.")
            color = "gray"
        else:
            text = (f"On: {describe_choice(self.values)}. If that route is down, VCB "
                    "doesn't search (never a direct connection). Questions and results "
                    "aren't logged.")
            color = "gray"
            if cloud:
                text += ("\nNote: with a Cloud API brain, your cloud provider sees what "
                         "you ask and what the search finds.")
                color = "#b06000"
        self.status.config(text=text, foreground=color)

    def _toggled(self) -> None:
        if self.enabled.get() and not (self.configured or (self.was_on and _installed(self.values))):
            self.enabled.set(False)              # only on once setup has passed
            self.open_dialog()
        self.refresh()

    def open_dialog(self) -> None:
        SetupDialog(self)

    def key_env_for(self, backend: str) -> str:
        return self.key_override or SEARCH_KEY_ENVS.get(backend, "")

    def accept(self, values: dict[str, str], key: str) -> None:
        self.values = values
        self.keys[values["VCB_SEARCH_BACKEND"]] = key
        self.configured = True
        self.enabled.set(True)
        self.refresh()

    def removed(self) -> None:
        self.configured = False
        self.was_on = False
        self.enabled.set(False)
        self.refresh()

    def updates(self) -> dict[str, str | None]:
        out: dict[str, str | None] = {"VCB_ONLINE_SEARCH": "true" if self.enabled.get() else "false"}
        if self.configured:
            out.update({k: (v or None) for k, v in self.values.items()})
            backend = self.values["VCB_SEARCH_BACKEND"]  # only the tested service's key
            out[self.key_env_for(backend)] = self.keys.get(backend) or None
        return out


class SetupDialog:
    """Guided setup: 1 pick the private route, 2 pick the search service,
    3 install + privacy check. Rarely needed options sit behind "Show
    advanced options"; while hidden, their defaults apply."""

    def __init__(self, section: SearchSection) -> None:
        self.section = section
        self.mode = "setup"
        self.tested: tuple[dict[str, str], str] | None = None
        self.proc: subprocess.Popen | None = None
        self.events: queue.Queue = queue.Queue()
        self.passed = False
        self.closing = False
        self.precheck: tuple[dict[str, str], str] | None = None  # inputs behind a form error
        self.watch_id: str | None = None
        self.keys = dict(section.keys)
        v = section.values

        top = self.top = tk.Toplevel(section.owner.root)
        top.title("Set up online search")
        top.resizable(False, False)
        top.transient(section.owner.root)
        top.grab_set()
        top.protocol("WM_DELETE_WINDOW", self.cancel)
        f = ttk.Frame(top, padding=(18, 14))
        f.grid(sticky="nsew")
        f.columnconfigure(1, weight=1)
        r = 0

        def heading(text):
            nonlocal r
            ttk.Label(f, text=text, font=("Segoe UI", 10, "bold")).grid(
                row=r, column=0, columnspan=3, sticky="w", pady=(14, 2))
            r += 1

        def note(parent, text, row, padx=(22, 0), color=GRAY, span=3):
            lbl = ttk.Label(parent, text=text, foreground=color, wraplength=WRAP - padx[0],
                            justify="left")
            lbl.grid(row=row, column=0, columnspan=span, sticky="w", padx=padx)
            return lbl

        def option(var, value, title, badge="", desc=""):
            nonlocal r
            line = ttk.Frame(f)
            line.grid(row=r, column=0, columnspan=3, sticky="w", pady=(4, 0))
            ttk.Radiobutton(line, text=title, value=value, variable=var,
                            command=self.refresh).pack(side="left")
            if badge:
                ttk.Label(line, text=badge, foreground=GREEN if badge == "Recommended" else GRAY,
                          font=("Segoe UI", 8, "bold")).pack(side="left", padx=(8, 0))
            r += 1
            if desc:
                note(f, desc, r)
                r += 1

        # -- intro ---------------------------------------------------------------
        ttk.Label(f, wraplength=WRAP, justify="left", text=(
            "Online search lets the assistant look things up on the web when you ask about "
            "something current: news, weather, opening hours, prices. Searches are sent "
            "privately, so the search service can't see who you are.")).grid(
            row=r, column=0, columnspan=3, sticky="w")
        r += 1
        link = ttk.Label(f, text="How is my privacy protected?", foreground=LINK,
                         cursor="hand2", font=("Segoe UI", 9, "underline"))
        link.grid(row=r, column=0, columnspan=3, sticky="w", pady=(2, 0))
        link.bind("<Button-1>", lambda _e: self.show_privacy())
        r += 1

        # -- step 1: route ---------------------------------------------------------
        heading("Step 1 of 3:  How should your searches be sent?")
        self.route = tk.StringVar(value=v["VCB_SEARCH_ROUTE"])
        option(self.route, "tor", "Through the Tor network", "Recommended",
               "Free, and needs no other app. VCB downloads Tor once (about 15 MB). "
               "Searches take a few seconds longer.")
        self.tor_box = ttk.Frame(f, padding=(22, 2, 0, 0))
        self.tor_box.grid(row=r, column=0, columnspan=3, sticky="ew")
        r += 1
        self.tor_mode = tk.StringVar(value=v["VCB_SEARCH_TOR_MODE"])
        ttk.Radiobutton(self.tor_box, text="Let VCB download and run Tor", value="managed",
                        variable=self.tor_mode, command=self.refresh).grid(
            row=0, column=0, columnspan=2, sticky="w")
        ttk.Radiobutton(self.tor_box, text="Use a Tor I already run, at", value="external",
                        variable=self.tor_mode, command=self.refresh).grid(row=1, column=0,
                                                                           sticky="w")
        self.tor_proxy = tk.StringVar(value=v["VCB_SEARCH_TOR_PROXY"])
        self.tor_proxy_entry = ttk.Entry(self.tor_box, textvariable=self.tor_proxy, width=32)
        self.tor_proxy_entry.grid(row=1, column=1, sticky="w", padx=(4, 0))

        option(self.route, "vpn", "Through my VPN app", "",
               "For VPN apps such as Proton VPN, NordVPN or Hotspot Shield. Faster, but the "
               "VPN must be connected whenever you search, and your VPN company can see "
               "your searches.")
        vb = self.vpn_box = ttk.Frame(f, padding=(22, 4, 0, 0))
        vb.grid(row=r, column=0, columnspan=3, sticky="ew")
        vb.columnconfigure(0, weight=1)
        r += 1
        self.vpn_mode = tk.StringVar(value=v["VCB_SEARCH_VPN_MODE"])
        self.vpn_adapter_radio = ttk.Radiobutton(
            vb, text="Use my VPN app's connection", value="adapter", variable=self.vpn_mode,
            command=self.refresh)
        self.vpn_adapter_radio.grid(row=0, column=0, sticky="w")
        self.pick_row = ttk.Frame(vb)
        self.pick_row.grid(row=1, column=0, sticky="ew")
        self.pick_row.columnconfigure(1, weight=1)
        ttk.Label(self.pick_row, text="VPN connection:").grid(row=0, column=0, sticky="w", pady=2)
        self.adapters: list[tuple[str, bool, bool]] = []
        self.adapter = tk.StringVar(value=v["VCB_SEARCH_VPN_ADAPTER"])
        self.adapter_combo = ttk.Combobox(self.pick_row, textvariable=self.adapter, width=56,
                                          postcommand=self._reload_adapters)
        self.adapter_combo.grid(row=0, column=1, sticky="ew", padx=(6, 0), pady=2)
        self.adapter_combo.bind("<<ComboboxSelected>>", lambda _e: self.refresh())
        self.vpn_status = ttk.Label(vb, wraplength=WRAP - 44, justify="left")
        self.vpn_status.grid(row=2, column=0, sticky="w", pady=(2, 0))
        self.proxy_row = ttk.Frame(vb)
        self.proxy_row.grid(row=3, column=0, sticky="ew", pady=(6, 0))
        self.proxy_row.columnconfigure(1, weight=1)
        ttk.Radiobutton(self.proxy_row, text="Use my VPN provider's SOCKS5 proxy:", value="proxy",
                        variable=self.vpn_mode, command=self.refresh).grid(row=0, column=0,
                                                                           sticky="w")
        self.vpn_proxy = tk.StringVar(value=v["VCB_SEARCH_VPN_PROXY"] or "socks5h://")
        self.vpn_proxy_entry = ttk.Entry(self.proxy_row, textvariable=self.vpn_proxy)
        self.vpn_proxy_entry.grid(row=0, column=1, sticky="ew", padx=(6, 0))
        self.vpn_proxy_hint = note(vb, "Must start with socks5h:// so website names are looked "
                                   "up through the proxy too.", 4, padx=(22, 0), span=1)

        # -- step 2: search service --------------------------------------------------
        heading("Step 2 of 3:  Which search service should answer?")
        self.backend = tk.StringVar(value=v["VCB_SEARCH_BACKEND"])
        self.key_backend = self.backend.get()
        # The key/URL fields move under whichever service is selected.
        self.fields_row: dict[str, int] = {}
        option(self.backend, "brave", "Brave Search", "Recommended")
        self.brave_help = note(f, BRAVE_STEPS, r)
        self.fields_row["brave"] = r + 1
        r += 2
        option(self.backend, "searxng", "A SearXNG server", "Advanced")
        self.searx_help = note(f, SEARXNG_HELP, r)
        self.fields_row["searxng"] = r + 1
        r += 2

        fields = self.fields = ttk.Frame(f, padding=(22, 6, 0, 2))
        fields.grid(row=self.fields_row["brave"], column=0, columnspan=3, sticky="ew")
        fields.columnconfigure(1, weight=1)
        self.url_label = ttk.Label(fields, width=16)
        self.url_label.grid(row=0, column=0, sticky="w", pady=2)
        self.url = tk.StringVar(value=v["VCB_SEARCH_URL"])
        self.url_entry = ttk.Entry(fields, textvariable=self.url)
        self.url_entry.grid(row=0, column=1, sticky="ew", pady=2)
        self.url_hint = ttk.Label(fields, foreground=GRAY)
        self.url_hint.grid(row=0, column=2, sticky="w", padx=(6, 0))
        self.key_label = ttk.Label(fields, width=16)
        self.key_label.grid(row=1, column=0, sticky="w", pady=2)
        self.key = tk.StringVar(value=self.keys.get(self.key_backend, ""))
        self.key_entry = ttk.Entry(fields, textvariable=self.key, show="\u2022")
        self.key_entry.grid(row=1, column=1, sticky="ew", pady=2)
        key_tools = ttk.Frame(fields)
        key_tools.grid(row=1, column=2, sticky="w", padx=(6, 0))
        ttk.Button(key_tools, text="Paste", width=6, command=self._paste_key).pack(side="left")
        self.show_key = tk.BooleanVar(value=False)
        ttk.Checkbutton(key_tools, text="Show", variable=self.show_key, command=lambda:
                        self.key_entry.config(show="" if self.show_key.get() else "\u2022")
                        ).pack(side="left", padx=(6, 0))
        self.key_link = ttk.Button(key_tools, text="Get a key",
                                   command=lambda: webbrowser.open(BRAVE_KEY_PAGE))
        self.key_link.pack(side="left", padx=(6, 0))
        self.params_label = ttk.Label(fields, text="Extra parameters", width=16)
        self.params_label.grid(row=2, column=0, sticky="w", pady=2)
        self.params = tk.StringVar(value=v["VCB_SEARCH_PARAMS"])
        self.params_entry = ttk.Entry(fields, textvariable=self.params)
        self.params_entry.grid(row=2, column=1, sticky="ew", pady=2)
        self.params_hint = ttk.Label(fields, text='e.g. {"language": "en"}', foreground=GRAY)
        self.params_hint.grid(row=2, column=2, sticky="w", padx=(6, 0))
        self.auth_mode_label = ttk.Label(fields, text="Token type", width=16)
        self.auth_mode_label.grid(row=3, column=0, sticky="w", pady=2)
        self.auth_mode = tk.StringVar(value=v["VCB_SEARXNG_AUTH_MODE"])
        self.auth_mode_combo = ttk.Combobox(
            fields, textvariable=self.auth_mode, values=("cookie", "bearer"),
            state="readonly", width=16)
        self.auth_mode_combo.grid(row=3, column=1, sticky="w", pady=2)
        self.auth_mode_hint = ttk.Label(
            fields, text="Bearer is only for a gateway that specifically requires it.",
            foreground=GRAY)
        self.auth_mode_hint.grid(row=3, column=2, sticky="w", padx=(6, 0))

        # -- step 3: install + privacy check ----------------------------------------
        heading("Step 3 of 3:  Check that everything works")
        self.step = note(f, (
            "Click \"Set up & check\". VCB installs what's needed (first time only), then makes "
            "sure searches really go through the private route: it looks at the address "
            "websites see, tests for DNS leaks and runs a test search. This can take a minute."),
            r, padx=(0, 0))
        r += 1
        run_row = ttk.Frame(f)
        run_row.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(8, 0))
        run_row.columnconfigure(1, weight=1)
        r += 1
        self.test_btn = ttk.Button(run_row, text="Set up & check", command=self.run_test)
        self.test_btn.grid(row=0, column=0, sticky="w")
        self.bar = ttk.Progressbar(run_row, mode="determinate", maximum=100)
        self.bar.grid(row=0, column=1, sticky="ew", padx=(10, 0))
        self.result = tk.Text(f, height=1, width=74, wrap="word", relief="flat",
                              background=top.cget("background"), font=("Segoe UI", 9))
        self.result.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        self.result.tag_config("bad", foreground=RED)
        self.result.tag_config("good", foreground=GREEN)
        self.result.config(state="disabled")
        self.result.grid_remove()
        r += 1

        buttons = ttk.Frame(f)
        buttons.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(14, 0))
        self.advanced = tk.BooleanVar(value=(
            v["VCB_SEARCH_TOR_MODE"] != "managed" or v["VCB_SEARCH_VPN_MODE"] != "adapter"
            or bool(v["VCB_SEARCH_PARAMS"])
            or v["VCB_SEARXNG_AUTH_MODE"] != "cookie"
            or (v["VCB_SEARCH_BACKEND"] == "brave" and bool(v["VCB_SEARCH_URL"]))))
        ttk.Checkbutton(buttons, text="Show advanced options", variable=self.advanced,
                        command=self.refresh).pack(side="left")
        self.remove_btn = ttk.Button(buttons, text="Remove components", command=self.remove)
        self.remove_btn.pack(side="left", padx=(10, 0))
        self.cancel_btn = ttk.Button(buttons, text="Cancel", command=self.cancel)
        self.cancel_btn.pack(side="right")
        self.ok_btn = ttk.Button(buttons, text="Turn on online search", command=self.ok,
                                 state="disabled")
        self.ok_btn.pack(side="right", padx=6)

        for entry in (self.tor_proxy_entry, self.vpn_proxy_entry, self.url_entry,
                      self.key_entry, self.params_entry):
            self._edit_menu(entry)
        top.bind("<Escape>", lambda _e: self.cancel())
        for var in (self.tor_proxy, self.vpn_proxy, self.adapter, self.url, self.params):
            var.trace_add("write", lambda *_: self._changed())
        self.auth_mode.trace_add("write", lambda *_: self._changed())
        self.key.trace_add("write", lambda *_: self._key_typed())
        self._reload_adapters()
        self.refresh()
        self._place()
        self.test_btn.focus_set()

    # -- form ------------------------------------------------------------------

    def _place(self) -> None:
        """Open over the settings window, with room below to grow."""
        top, root = self.top, self.section.owner.root
        top.update_idletasks()
        x = root.winfo_rootx() + 40
        y = min(root.winfo_rooty() + 20, top.winfo_screenheight() - top.winfo_reqheight() - 160)
        top.geometry(f"+{max(x, 0)}+{max(y, 0)}")

    def _edit_menu(self, entry: ttk.Entry) -> None:
        """Right-click Cut/Copy/Paste: many people never use Ctrl+V."""
        menu = tk.Menu(entry, tearoff=0)
        for label, event in (("Cut", "<<Cut>>"), ("Copy", "<<Copy>>"), ("Paste", "<<Paste>>")):
            menu.add_command(label=label, command=lambda e=event: entry.event_generate(e))
        menu.add_separator()
        menu.add_command(label="Select all", command=lambda: entry.select_range(0, "end"))

        def popup(event):
            entry.focus_set()
            try:
                menu.tk_popup(event.x_root, event.y_root)
            finally:
                menu.grab_release()

        entry.bind("<Button-3>", popup)

    def show_privacy(self) -> None:
        messagebox.showinfo("How online search protects your privacy", PRIVACY_TEXT,
                            parent=self.top)

    def _paste_key(self) -> None:
        try:
            text = self.top.clipboard_get()
        except tk.TclError:
            text = ""
        text = "".join(text.split())             # keys have no spaces; drop stray ones
        if text:
            self.key.set(text)
        else:
            self.step.config(text="Nothing to paste yet. Copy the key first (select it and "
                             "press Ctrl+C), then click Paste.", foreground=ORANGE)

    def _key_typed(self) -> None:
        self.keys[self.key_backend] = self.key.get()
        self._changed()

    def _key(self) -> str:
        return self.keys.get(self.backend.get(), "").strip()

    def _advanced(self, name: str, default: str) -> str:
        """An advanced option's value: its default while advanced options are hidden."""
        return getattr(self, name).get() if self.advanced.get() else default

    @staticmethod
    def _adapter_label(a: tuple[str, bool, bool]) -> str:
        name, up, vpn = a
        tags = ["connected" if up else "not connected"] + (["VPN"] if vpn else [])
        return f"{name}   ({', '.join(tags)})"

    def _adapter_name(self) -> str:
        text = self.adapter.get().strip()
        for a in self.adapters:
            if text == self._adapter_label(a):
                return a[0]
        head, sep, _ = text.partition("   (")     # a label from an older listing
        return head.strip() if sep and text.endswith(")") else text

    def _reload_adapters(self) -> None:
        """Re-read the network adapters (on open, every few seconds while the
        VPN route is shown, and when the list opens): VPN apps create or
        re-create their adapter when they connect. If the chosen one isn't
        connected, a connected VPN is picked by itself (e.g. Hotspot Shield
        keeps an unused adapter next to the one it really uses)."""
        name = self._adapter_name()
        self.adapters = list_adapters()
        cur = next((a for a in self.adapters if a[0] == name), None)
        if cur is None or not cur[1]:
            best = next((a for a in self.adapters if a[1] and a[2]), None)
            if best is not None:
                name, cur = best[0], best
        everything = self.advanced.get()
        shown = [a for a in self.adapters if everything or a[2] or a[0] == name]
        self.adapter_combo.config(values=[self._adapter_label(a) for a in shown])
        if cur is not None:
            label = self._adapter_label(cur)
        else:
            label = self._adapter_label((name, False, looks_like_vpn(name))) if name else ""
        if label != self.adapter.get():
            self.adapter.set(label)

    def _vpn_status_text(self) -> tuple[str, str]:
        if not self.adapters:
            return ("Couldn't list this computer's network connections. Type the name of "
                    "your VPN's network adapter."), ORANGE
        name = self._adapter_name()
        cur = next((a for a in self.adapters if a[0] == name), None)
        auto = "Open your VPN app and connect; this window notices by itself."
        if not name:
            if any(a[2] for a in self.adapters):
                return f"\u25cb Your VPN isn't connected. {auto}", ORANGE
            text = f"\u25cb No VPN found. {auto}"
            if not self.advanced.get():
                text += (" If your VPN still doesn't appear, tick \"Show advanced options\" "
                         "to choose from all network connections.")
            return text, ORANGE
        if cur is None or not cur[1]:
            return f"\u25cb Not connected right now. {auto}", ORANGE
        if not cur[2]:
            return ("\u26a0 This doesn't look like a VPN. Searches would go out through it, "
                    "so only pick it if it really is your VPN."), ORANGE
        return "\u25cf Connected. Searches go through this VPN, and stop if it disconnects.", GREEN

    def _watch(self) -> None:
        self.watch_id = None
        if not self.top.winfo_exists():
            return
        if self.route.get() == "vpn" and self._advanced("vpn_mode", "adapter") == "adapter":
            self._reload_adapters()
            text, color = self._vpn_status_text()
            self.vpn_status.config(text=text, foreground=color)
            self.watch_id = self.top.after(ADAPTER_POLL_MS, self._watch)

    def refresh(self) -> None:
        adv = self.advanced.get()
        tor = self.route.get() == "tor"
        _show(self.tor_box, tor and adv)
        _show(self.vpn_box, not tor)
        self.tor_proxy_entry.config(
            state="normal" if self._advanced("tor_mode", "managed") == "external" else "disabled")

        adapter_mode = self._advanced("vpn_mode", "adapter") == "adapter"
        for w in (self.vpn_adapter_radio, self.proxy_row, self.vpn_proxy_hint):
            _show(w, adv)
        indent = (22 if adv else 0, 0)      # under "Use my VPN app's connection"
        self.pick_row.grid_configure(padx=indent)
        self.vpn_status.grid_configure(padx=indent)
        if not tor and adapter_mode:
            self._reload_adapters()
        self.adapter_combo.config(state=("readonly" if self.adapters else "normal")
                                  if adapter_mode else "disabled")
        self.vpn_proxy_entry.config(state="disabled" if adapter_mode else "normal")
        text, color = self._vpn_status_text() if adapter_mode else ("", GRAY)
        self.vpn_status.config(text=text, foreground=color)
        _show(self.vpn_status, adapter_mode)
        if not tor and adapter_mode and self.watch_id is None:
            self.watch_id = self.top.after(ADAPTER_POLL_MS, self._watch)

        brave = self.backend.get() == "brave"
        if self.backend.get() != self.key_backend:   # each service shows its own key
            self.key_backend = self.backend.get()
            self.key.set(self.keys.get(self.key_backend, ""))
        _show(self.brave_help, brave)
        _show(self.searx_help, not brave)
        self.fields.grid(row=self.fields_row["brave" if brave else "searxng"])
        for w in (self.url_label, self.url_entry, self.url_hint):
            _show(w, adv or not brave)
        self.url_label.config(text="Custom API URL" if brave else "Server address")
        self.url_hint.config(text="leave empty normally" if brave
                             else "e.g. https://search.example.org")
        self.key_label.config(text="Brave API key" if brave else "SearXNG token")
        self.key_link.pack(side="left", padx=(6, 0)) if brave else self.key_link.pack_forget()
        for w in (self.params_label, self.params_entry, self.params_hint):
            _show(w, adv)
        for w in (self.auth_mode_label, self.auth_mode_combo, self.auth_mode_hint):
            _show(w, adv and not brave)
        self.remove_btn.pack(side="left", padx=(10, 0)) if adv else self.remove_btn.pack_forget()
        self._changed()

    def _changed(self) -> None:
        """"Turn on" is only allowed for exactly what was last checked."""
        now = (self.collect(), self._key())
        if self.precheck is not None and self.precheck != now:
            self.precheck = None                  # the form error was about older input
            self._write([])
        ok = self.passed and self.tested == now
        self.ok_btn.config(state="normal" if ok else "disabled")

    def collect(self) -> dict[str, str]:
        proxy = self.vpn_proxy.get().strip()
        brave = self.backend.get() == "brave"
        return {
            "VCB_SEARCH_ROUTE": self.route.get(),
            "VCB_SEARCH_TOR_MODE": self._advanced("tor_mode", "managed"),
            "VCB_SEARCH_TOR_PROXY": self.tor_proxy.get().strip() or TOR_DEFAULT_PROXY,
            "VCB_SEARCH_VPN_MODE": self._advanced("vpn_mode", "adapter"),
            "VCB_SEARCH_VPN_ADAPTER": self._adapter_name(),
            "VCB_SEARCH_VPN_PROXY": "" if proxy == "socks5h://" else proxy,
            "VCB_SEARCH_BACKEND": self.backend.get(),
            "VCB_SEARCH_URL": self._advanced("url", "").strip() if brave else self.url.get().strip(),
            "VCB_SEARCH_PARAMS": self._advanced("params", "").strip(),
            "VCB_SEARXNG_AUTH_MODE": self._advanced("auth_mode", "cookie"),
        }

    def _problem(self, values: dict[str, str], key: str) -> str | None:
        if values["VCB_SEARCH_BACKEND"] == "brave" and not key:
            return ("Paste your Brave API key first. Step 2 shows how to get one "
                    "(\"Get a key\" opens Brave's page).")
        if (values["VCB_SEARCH_ROUTE"] == "vpn" and values["VCB_SEARCH_VPN_MODE"] == "adapter"
                and not values["VCB_SEARCH_VPN_ADAPTER"]):
            return "Connect your VPN app first, then pick its connection in Step 1."
        cfg = SearchConfig(
            enabled=True, route=values["VCB_SEARCH_ROUTE"],
            tor_mode=values["VCB_SEARCH_TOR_MODE"], tor_proxy=values["VCB_SEARCH_TOR_PROXY"],
            vpn_mode=values["VCB_SEARCH_VPN_MODE"], vpn_adapter=values["VCB_SEARCH_VPN_ADAPTER"],
            vpn_proxy=values["VCB_SEARCH_VPN_PROXY"], backend=values["VCB_SEARCH_BACKEND"],
            url=values["VCB_SEARCH_URL"], params=values["VCB_SEARCH_PARAMS"],
            api_key_env=self.section.key_override,
            searxng_auth_mode=values["VCB_SEARXNG_AUTH_MODE"])
        key_env = cfg.key_env
        old = os.environ.get(key_env)
        os.environ[key_env] = key
        try:
            return cfg.problem()
        finally:
            if old is None:
                os.environ.pop(key_env, None)
            else:
                os.environ[key_env] = old

    # -- running the installer -------------------------------------------------

    def _write(self, lines: list[str]) -> None:
        self.result.config(state="normal")
        self.result.delete("1.0", "end")
        for line in lines:
            tag = "bad" if line.startswith("✗") else "good" if line.startswith("✓") else ""
            self.result.insert("end", line + "\n", tag)
        self.result.config(state="disabled",
                           height=min(10, sum(1 + len(line) // 90 for line in lines)) or 1)
        _show(self.result, bool(lines))

    def _busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        self.test_btn.config(state=state)
        self.remove_btn.config(state=state)
        self.bar.config(mode="indeterminate" if busy else "determinate")
        if busy:
            self.bar.start(12)
        else:
            self.bar.stop()

    def _start(self, args: list[str], env_extra: dict[str, str]) -> None:
        env = {**os.environ, **env_extra, "PYTHONIOENCODING": "utf-8"}
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "vcb.search.install", *args, "--json-progress"],
            cwd=ROOT_DIR, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
            creationflags=CREATE_NO_WINDOW)
        threading.Thread(target=self._pump, args=(self.proc,), daemon=True).start()
        self._busy(True)
        self.top.after(100, self._poll)

    def _pump(self, proc: subprocess.Popen) -> None:
        result = None
        for line in proc.stdout:
            try:
                ev = json.loads(line)
            except ValueError:
                continue
            if ev.get("event") == "result":
                result = ev
            else:
                self.events.put(ev)
        proc.wait()
        self.events.put(result or {"event": "result", "ok": False,
                                   "summary": ["✗ Setup stopped unexpectedly."]})

    def _poll(self) -> None:
        try:
            while True:
                ev = self.events.get_nowait()
                if ev.get("event") == "result":
                    self._finished(ev)
                    return
                self.step.config(text=ev.get("text", ""), foreground="gray")
                pct = ev.get("pct")
                if isinstance(pct, int):
                    if str(self.bar.cget("mode")) != "determinate":
                        self.bar.stop()
                        self.bar.config(mode="determinate")
                    self.bar.config(value=pct)
                elif str(self.bar.cget("mode")) != "indeterminate":
                    self.bar.config(mode="indeterminate")
                    self.bar.start(12)
        except queue.Empty:
            pass
        self.top.after(100, self._poll)

    def _finished(self, ev: dict) -> None:
        self.proc = None
        self._busy(False)
        self.bar.config(value=100 if ev.get("ok") else 0)
        if self.closing:
            self._close()
            return
        summary = ev.get("summary") or []
        if self.mode == "remove":
            self.step.config(text=summary[0] if summary else "", foreground="gray")
            if ev.get("ok"):
                self.section.removed()
            return
        self._write(summary)
        if ev.get("ok"):
            self.passed = True
            self._changed()
            self.step.config(text="\u2713 All checks passed. Click \"Turn on online search\" "
                             "to finish.", foreground=GREEN)
        elif ev.get("cancelled"):
            self.step.config(text=summary[0] if summary else "Cancelled.", foreground=GRAY)
        else:
            self.step.config(text="Online search can't be turned on yet. Fix the problem marked "
                             "\u2717 below, then click \"Set up & check\" again.", foreground=RED)

    def run_test(self) -> None:
        values, key = self.collect(), self._key()
        problem = self._problem(values, key)
        if problem:
            self._write(["✗ " + problem])
            self.precheck = (values, key)
            return
        self.mode = "setup"
        self.precheck = None
        self._write([])
        self.passed = False
        self.tested = (values, key)
        self._changed()
        self.step.config(text="Starting...", foreground=GRAY)
        key_env = self.section.key_env_for(values["VCB_SEARCH_BACKEND"])
        self._start(["--setup"], {**values, "VCB_ONLINE_SEARCH": "true",
                                  "VCB_SEARCH_API_KEY_ENV": self.section.key_override,
                                  key_env: key})

    def remove(self) -> None:
        if not _anything_installed():
            self.step.config(text="Nothing to remove: no search components are installed.",
                             foreground="gray")
            return
        if not messagebox.askyesno(
                "Online search", "Remove Tor and the online search components?\n"
                "Online search will be turned off.", parent=self.top):
            return
        self.mode = "remove"
        self._start(["--remove"], {})

    def ok(self) -> None:
        if not self.passed or self.tested is None:
            return
        values, key = self.tested
        self.section.accept(values, key)
        self._close()

    def cancel(self) -> None:
        if self.proc is not None:
            if self.closing:
                return
            self.closing = True
            self.step.config(text="Cancelling and undoing changes...", foreground="gray")
            self.cancel_btn.config(state="disabled")
            try:
                self.proc.stdin.write("cancel\n")
                self.proc.stdin.flush()
            except OSError:
                pass
            return
        self._close()

    def _close(self) -> None:
        if self.watch_id is not None:
            self.top.after_cancel(self.watch_id)
            self.watch_id = None
        self.top.destroy()
