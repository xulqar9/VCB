"""Settings window: which voices it offers, and what it saves for a voice
that VCB has retired."""

import tkinter as tk
from tkinter import ttk

import pytest

from vcb import settings
from vcb.envfile import read_env


@pytest.fixture(scope="module")
def root():
    try:
        r = tk.Tk()
    except tk.TclError:
        pytest.skip("no display")
    r.withdraw()
    yield r
    r.destroy()


@pytest.fixture
def open_window(root, tmp_path, monkeypatch):
    """Opens the window on a .env with the given text. No GPU probe, Ollama,
    audio device or dialog is involved: only the window's own logic runs."""
    monkeypatch.setattr(settings, "detect_gpus", lambda: [])
    monkeypatch.setattr(settings, "detect_amd_gpus", lambda: [])
    monkeypatch.setattr(settings, "_ollama_installed_models", lambda url: [])
    monkeypatch.setattr(settings, "_ollama_running", lambda url: False)
    monkeypatch.setattr(settings, "_ollama_installed", lambda: True)
    monkeypatch.setattr(settings, "_audio_devices", lambda kind: [])
    monkeypatch.setattr(settings.messagebox, "showerror",
                        lambda *args: pytest.fail(f"error dialog: {args}"))
    env = tmp_path / ".env"
    tops = []

    def open_(text: str) -> settings.SettingsWindow:
        env.write_text(text, encoding="utf-8")
        top = tk.Toplevel(root)
        top.withdraw()
        tops.append(top)
        return settings.SettingsWindow(top, env)

    yield open_, env
    for top in tops:
        if top.winfo_exists():
            top.destroy()


def _offered_voices(window) -> list[str]:
    def walk(widget):
        for child in widget.winfo_children():
            yield child
            yield from walk(child)
    return [str(w.cget("value")) for w in walk(window.frame)
            if isinstance(w, ttk.Radiobutton) and str(w.cget("variable")) == str(window.engine)]


def test_a_retired_voice_is_not_offered_and_saving_switches_to_chatterbox(open_window):
    open_, env = open_window
    window = open_("VCB_TTS_ENGINE=xtts\nVCB_INDEXTTS_GPUS=1\n")
    assert _offered_voices(window) == ["chatterbox-turbo", "kokoro", "orpheus"]
    assert window.engine.get() == "chatterbox-turbo"
    assert "Your previous voice, XTTS v2, is no longer part of VCB" in window.engine_info.cget("text")
    window._save()
    saved = read_env(env)
    assert saved["VCB_TTS_ENGINE"] == "chatterbox-turbo"
    assert saved["VCB_INDEXTTS_GPUS"] == "1"            # no longer managed: left as it was


def test_an_offered_voice_opens_as_saved_without_the_retired_note(open_window):
    open_, env = open_window
    window = open_("VCB_TTS_ENGINE=orpheus\n")
    assert window.engine.get() == "orpheus"
    assert "no longer part of VCB" not in window.engine_info.cget("text")
    window._save()
    saved = read_env(env)
    assert saved["VCB_TTS_ENGINE"] == "orpheus" and "VCB_INDEXTTS_GPUS" not in saved
