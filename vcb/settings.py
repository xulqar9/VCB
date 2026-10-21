"""VCB settings window (tkinter, ships with Python -- no extra dependency).

Run by the launcher on first start and by "VCB Settings.bat". Writes the
choices to `.env`, leaving any other keys/comments in that file untouched.
Exit code 0 = saved, 1 = cancelled.

    python -m vcb.settings
"""

import shutil
import sys
import tkinter as tk
import urllib.request
import webbrowser
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from .config import ENV_FILE, USER_DIR, available_prompts, resolve_path
from .envfile import read_env, update_env

DEFAULT_LABEL = "System default"

# name -> (API base URL, default model, default thinking level)
CLOUD_PROVIDERS = {
    "Google Gemini (free tier)": (
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-3.5-flash-lite", "minimal"),
    "OpenAI": ("https://api.openai.com/v1", "gpt-4o-mini", ""),
    "OpenRouter": ("https://openrouter.ai/api/v1", "openai/gpt-4o-mini", ""),
    "Groq": ("https://api.groq.com/openai/v1", "llama-3.3-70b-versatile", ""),
    "Custom": ("", "", ""),
}
# "" = don't send it (provider default). Fast, light models answer within a
# second at minimal/low; big "thinking" models can take 30+ s to start talking.
THINKING_LEVELS = ("", "none", "minimal", "low", "medium", "high")
PROVIDER_KEY_PAGES = {
    "Google Gemini (free tier)": "https://aistudio.google.com/api-keys",
    "OpenAI": "https://platform.openai.com/api-keys",
    "OpenRouter": "https://openrouter.ai/keys",
    "Groq": "https://console.groq.com/keys",
}
OLLAMA_MODELS = (
    "llama3.1:8b-instruct-q4_K_M",
    "qwen2.5:7b-instruct",
    "gemma3:4b",
    "gemma3:12b",
    "mistral-nemo:12b",
)
KOKORO_VOICES = (
    "af_heart", "af_bella", "af_nicole", "af_sarah", "af_sky",
    "am_adam", "am_michael", "bf_emma", "bf_isabella", "bm_george", "bm_lewis",
)


def _ollama_running(base_url: str) -> bool:
    try:
        url = base_url.removesuffix("/v1").rstrip("/") + "/api/version"
        with urllib.request.urlopen(url, timeout=1.5) as resp:
            return resp.status == 200
    except Exception:
        return False


def _ollama_installed() -> bool:
    from .runtime import _resolve_binary
    exe = _resolve_binary("ollama")
    return shutil.which(exe) is not None or Path(exe).is_file()


def _audio_devices(kind: str) -> list[str]:
    try:
        from .audio.devices import list_devices
        return list_devices(kind)
    except Exception:
        return []


def _guess_provider(base_url: str) -> str:
    for name, (url, _model, _effort) in CLOUD_PROVIDERS.items():
        if url and base_url.rstrip("/") == url:
            return name
    return "Custom" if base_url else "Google Gemini (free tier)"


class SettingsWindow:
    def __init__(self, root: tk.Tk, env_path: Path) -> None:
        self.root = root
        self.env_path = env_path
        self.saved = False
        env = read_env(env_path)
        self.env = env
        get = lambda k, d="": env.get(k, d)  # noqa: E731

        root.title("VCB settings")
        root.resizable(False, False)
        frame = ttk.Frame(root, padding=14)
        frame.grid(sticky="nsew")
        frame.columnconfigure(1, weight=1)
        self.frame = frame
        self.row = 0

        # -- brain ------------------------------------------------------------
        self._heading("Brain (language model)")
        self.backend = tk.StringVar(value=get("VCB_BACKEND", "ollama"))
        modes = ttk.Frame(frame)
        modes.grid(row=self.row, column=0, columnspan=3, sticky="w")
        for value, text in (("ollama", "Local (Ollama)"), ("cloud", "Cloud API"),
                            ("llama_cpp", "Advanced: llama.cpp GGUF")):
            ttk.Radiobutton(modes, text=text, value=value, variable=self.backend,
                            command=self._refresh).pack(side="left", padx=(0, 12))
        self.row += 1

        self.ollama_box = self._group()
        self.ollama_model = tk.StringVar(
            value=get("VCB_OLLAMA_MODEL", OLLAMA_MODELS[0]))
        self._combo(self.ollama_box, 0, "Model", self.ollama_model, OLLAMA_MODELS,
                    hint="downloaded automatically on first start")
        self.ollama_status = ttk.Label(self.ollama_box, foreground="gray")
        self.ollama_status.grid(row=1, column=1, sticky="w")
        self.ollama_link = ttk.Button(
            self.ollama_box, text="Get Ollama",
            command=lambda: webbrowser.open("https://ollama.com/download"))
        self.ollama_link.grid(row=1, column=2, sticky="e")
        self.ollama_base = get("VCB_OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")

        self.cloud_box = self._group()
        default_url, default_model, default_effort = CLOUD_PROVIDERS["Google Gemini (free tier)"]
        base = get("VCB_CLOUD_BASE_URL", default_url)
        self.provider = tk.StringVar(value=_guess_provider(base))
        cb = self._combo(self.cloud_box, 0, "Provider", self.provider,
                         tuple(CLOUD_PROVIDERS), readonly=True)
        cb.bind("<<ComboboxSelected>>", self._provider_changed)
        self.key_link = ttk.Button(self.cloud_box, text="Get a key",
                                   command=self._open_key_page)
        self.key_link.grid(row=0, column=2, sticky="w", padx=(6, 0))
        self.cloud_url = tk.StringVar(value=base)
        self._entry(self.cloud_box, 1, "API base URL", self.cloud_url)
        self.cloud_key = tk.StringVar(value=get("VCB_CLOUD_API_KEY"))
        self._entry(self.cloud_box, 2, "API key", self.cloud_key, show="*")
        self.cloud_model = tk.StringVar(value=get("VCB_CLOUD_MODEL", default_model))
        self._entry(self.cloud_box, 3, "Model", self.cloud_model)
        self.cloud_effort = tk.StringVar(
            value=get("VCB_CLOUD_REASONING_EFFORT",
                      default_effort if "VCB_CLOUD_BASE_URL" not in env else ""))
        self._combo(self.cloud_box, 4, "Thinking", self.cloud_effort, THINKING_LEVELS,
                    hint="lower = faster replies; empty = provider default")

        self.gguf_box = self._group()
        self.gguf = tk.StringVar(value=get("VCB_ACTIVE_MODEL"))
        self._file(self.gguf_box, 0, "Model", self.gguf, self._pick_gguf,
                   hint="a .gguf file or a preset name; needs llama-server on PATH")

        # -- voice ------------------------------------------------------------
        self._heading("Voice")
        engine = get("VCB_TTS_ENGINE", "chatterbox-turbo")
        self.engine = tk.StringVar(value=engine)
        modes = ttk.Frame(frame)
        modes.grid(row=self.row, column=0, columnspan=3, sticky="w")
        for value, text in (("chatterbox-turbo", "Clone a voice (Chatterbox)"),
                            ("kokoro", "Preset voice (Kokoro)")):
            ttk.Radiobutton(modes, text=text, value=value, variable=self.engine,
                            command=self._refresh).pack(side="left", padx=(0, 12))
        if engine not in ("chatterbox-turbo", "kokoro"):
            ttk.Radiobutton(modes, text=engine, value=engine, variable=self.engine,
                            command=self._refresh).pack(side="left")
        self.row += 1

        self.clone_box = self._group()
        self.voice_wav = tk.StringVar(value=get("VCB_VOICE_WAV"))
        self._file(self.clone_box, 0, "Voice sample", self.voice_wav, self._pick_wav,
                   hint="5-15 s of clean speech (.wav); empty = built-in voice")

        self.kokoro_box = self._group()
        self.kokoro_voice = tk.StringVar(value=get("VCB_KOKORO_VOICE", "af_heart"))
        self._combo(self.kokoro_box, 0, "Voice", self.kokoro_voice, KOKORO_VOICES,
                    readonly=True)

        # -- personality ------------------------------------------------------
        self._heading("Personality")
        prompts = available_prompts()
        self.persona = tk.StringVar(value=get("VCB_ACTIVE_PROMPT", "assistant"))
        self._combo(frame, self.row, "Persona", self.persona, prompts, readonly=True,
                    hint="add your own .txt in user\\prompts")
        self.row += 1
        self.user_name = tk.StringVar(value=get("VCB_USER_NAME"))
        self._entry(frame, self.row, "Your name", self.user_name,
                    hint="optional; how the assistant refers to you")
        self.row += 1

        # -- audio ------------------------------------------------------------
        self._heading("Audio devices")
        mics = [DEFAULT_LABEL, *_audio_devices("input")]
        speakers = [DEFAULT_LABEL, *_audio_devices("output")]
        self.mic = tk.StringVar(value=get("VCB_INPUT_DEVICE") or DEFAULT_LABEL)
        self._combo(frame, self.row, "Microphone", self.mic, mics, readonly=True)
        self.row += 1
        self.speaker = tk.StringVar(value=get("VCB_OUTPUT_DEVICE") or DEFAULT_LABEL)
        self._combo(frame, self.row, "Speaker", self.speaker, speakers, readonly=True)
        self.row += 1

        # -- options ----------------------------------------------------------
        self._heading("Options")
        opts = ttk.Frame(frame)
        opts.grid(row=self.row, column=0, columnspan=3, sticky="w")
        self.row += 1
        truthy = lambda k, d: get(k, d).lower() in ("1", "true", "yes", "on")  # noqa: E731
        self.memory = tk.BooleanVar(value=truthy("VCB_MEMORY_ENABLED", "true"))
        self.ai_led = tk.BooleanVar(value=truthy("VCB_AI_LED", "false"))
        self.avatar = tk.BooleanVar(value=truthy("VCB_AVATAR_ENABLED", "false"))
        for var, text in ((self.memory, "Long-term memory"),
                          (self.ai_led, "Assistant starts conversations"),
                          (self.avatar, "Unreal avatar")):
            ttk.Checkbutton(opts, text=text, variable=var).pack(side="left", padx=(0, 12))

        buttons = ttk.Frame(frame)
        buttons.grid(row=self.row, column=0, columnspan=3, sticky="e", pady=(14, 0))
        ttk.Button(buttons, text="Cancel", command=root.destroy).pack(side="right")
        ttk.Button(buttons, text="Save", command=self._save).pack(side="right", padx=6)
        root.bind("<Escape>", lambda _e: root.destroy())
        self._refresh()

    # -- layout helpers ---------------------------------------------------

    def _heading(self, text: str) -> None:
        pady = (0, 4) if self.row == 0 else (12, 4)
        ttk.Label(self.frame, text=text, font=("Segoe UI", 10, "bold")).grid(
            row=self.row, column=0, columnspan=3, sticky="w", pady=pady)
        self.row += 1

    def _group(self) -> ttk.Frame:
        box = ttk.Frame(self.frame)
        box.grid(row=self.row, column=0, columnspan=3, sticky="ew")
        box.columnconfigure(1, weight=1)
        self.row += 1
        return box

    def _hint(self, parent, row: int, hint: str | None) -> None:
        if hint:
            ttk.Label(parent, text=hint, foreground="gray").grid(
                row=row, column=2, sticky="w", padx=(6, 0))

    def _entry(self, parent, row, label, var, show=None, hint=None):
        ttk.Label(parent, text=label, width=13).grid(row=row, column=0, sticky="w", pady=2)
        e = ttk.Entry(parent, textvariable=var, width=44, show=show or "")
        e.grid(row=row, column=1, sticky="ew", pady=2)
        self._hint(parent, row, hint)
        return e

    def _combo(self, parent, row, label, var, values, readonly=False, hint=None):
        ttk.Label(parent, text=label, width=13).grid(row=row, column=0, sticky="w", pady=2)
        cb = ttk.Combobox(parent, textvariable=var, values=list(values), width=42,
                          state="readonly" if readonly else "normal")
        cb.grid(row=row, column=1, sticky="ew", pady=2)
        self._hint(parent, row, hint)
        return cb

    def _file(self, parent, row, label, var, command, hint=None):
        ttk.Label(parent, text=label, width=13).grid(row=row, column=0, sticky="w", pady=2)
        inner = ttk.Frame(parent)
        inner.grid(row=row, column=1, sticky="ew", pady=2)
        inner.columnconfigure(0, weight=1)
        ttk.Entry(inner, textvariable=var, width=36).grid(row=0, column=0, sticky="ew")
        ttk.Button(inner, text="Browse...", command=command).grid(row=0, column=1, padx=(4, 0))
        self._hint(parent, row, hint)

    # -- behaviour --------------------------------------------------------

    def _refresh(self) -> None:
        backend = self.backend.get()
        for box, name in ((self.ollama_box, "ollama"), (self.cloud_box, "cloud"),
                          (self.gguf_box, "llama_cpp")):
            box.grid() if backend == name else box.grid_remove()
        engine = self.engine.get()
        if self.provider.get() in PROVIDER_KEY_PAGES:
            self.key_link.grid()
        else:
            self.key_link.grid_remove()
        self.clone_box.grid() if engine != "kokoro" else self.clone_box.grid_remove()
        self.kokoro_box.grid() if engine == "kokoro" else self.kokoro_box.grid_remove()
        if backend == "ollama":
            if _ollama_running(self.ollama_base):
                self.ollama_status.config(text="Ollama is running", foreground="green")
                self.ollama_link.grid_remove()
            elif _ollama_installed():
                self.ollama_status.config(text="Ollama installed (VCB will start it)",
                                          foreground="green")
                self.ollama_link.grid_remove()
            else:
                self.ollama_status.config(
                    text="Ollama not found - install it first", foreground="red")
                self.ollama_link.grid()

    def _provider_changed(self, _event=None) -> None:
        url, model, effort = CLOUD_PROVIDERS[self.provider.get()]
        self.cloud_url.set(url)
        if model:
            self.cloud_model.set(model)
        self.cloud_effort.set(effort)
        self._refresh()

    def _open_key_page(self) -> None:
        page = PROVIDER_KEY_PAGES.get(self.provider.get())
        if page:
            webbrowser.open(page)

    def _pick_wav(self) -> None:
        path = filedialog.askopenfilename(
            title="Voice sample to clone", filetypes=[("WAV audio", "*.wav")])
        if path:
            self.voice_wav.set(path)

    def _pick_gguf(self) -> None:
        path = filedialog.askopenfilename(
            title="GGUF model", filetypes=[("GGUF model", "*.gguf")])
        if path:
            self.gguf.set(path)

    def _store_voice(self, path: str) -> str:
        """Copy a picked voice sample into user/voices so it keeps working if
        the original is moved; returns the path to save."""
        if not path:
            return ""
        src = Path(resolve_path(path))
        voices = USER_DIR / "voices"
        if not src.is_file() or voices in src.parents:
            return path
        voices.mkdir(parents=True, exist_ok=True)
        dest = voices / src.name
        shutil.copyfile(src, dest)
        return str(dest.relative_to(USER_DIR.parent))

    def _save(self) -> None:
        backend = self.backend.get()
        if backend == "cloud" and not self.cloud_key.get().strip():
            messagebox.showerror("VCB settings", "Enter your API key for the cloud provider.")
            return
        if backend == "cloud" and not self.cloud_url.get().strip():
            messagebox.showerror("VCB settings", "Enter the provider's API base URL.")
            return
        if backend == "llama_cpp" and not self.gguf.get().strip():
            messagebox.showerror("VCB settings", "Choose a .gguf model file.")
            return
        voice = self.voice_wav.get().strip()
        if voice and not Path(resolve_path(voice)).is_file():
            messagebox.showerror("VCB settings", f"Voice sample not found:\n{voice}")
            return
        device = lambda v: None if v.get() == DEFAULT_LABEL else v.get()  # noqa: E731
        bool_str = lambda v: "true" if v.get() else "false"  # noqa: E731
        updates: dict[str, str | None] = {
            "VCB_BACKEND": backend,
            "VCB_OLLAMA_MODEL": self.ollama_model.get().strip(),
            "VCB_CLOUD_BASE_URL": self.cloud_url.get().strip(),
            "VCB_CLOUD_API_KEY": self.cloud_key.get().strip() or None,
            "VCB_CLOUD_MODEL": self.cloud_model.get().strip(),
            "VCB_CLOUD_REASONING_EFFORT": self.cloud_effort.get().strip() or None,
            "VCB_TTS_ENGINE": self.engine.get(),
            "VCB_VOICE_WAV": self._store_voice(voice) or None,
            "VCB_KOKORO_VOICE": self.kokoro_voice.get(),
            "VCB_ACTIVE_PROMPT": self.persona.get(),
            "VCB_USER_NAME": self.user_name.get().strip() or None,
            "VCB_INPUT_DEVICE": device(self.mic),
            "VCB_OUTPUT_DEVICE": device(self.speaker),
            "VCB_MEMORY_ENABLED": bool_str(self.memory),
            "VCB_AI_LED": bool_str(self.ai_led),
            "VCB_AVATAR_ENABLED": bool_str(self.avatar),
        }
        if backend == "llama_cpp":
            updates["VCB_ACTIVE_MODEL"] = self.gguf.get().strip()
        update_env(self.env_path, updates)
        self.saved = True
        self.root.destroy()


def main() -> int:
    root = tk.Tk()
    win = SettingsWindow(root, ENV_FILE)
    root.mainloop()
    return 0 if win.saved else 1


if __name__ == "__main__":
    sys.exit(main())
