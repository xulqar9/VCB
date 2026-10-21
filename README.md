# VCB — Voice Chat Bot

A local, real-time, streaming voice AI. Talk to it like a phone call: it listens
continuously, replies as it thinks (not after it's done thinking), and you can
interrupt it mid-sentence and it'll actually stop, not just mute the speaker.

Runs fully on your own machine by default — speech recognition, the language
model (via Ollama), and the voice are all local — or point it at a cloud API if
you'd rather not run a model yourself. Windows users get a one-click launcher:
no Python setup, just pick a model and (optionally) a voice sample to clone.

## Why this exists

Most "voice assistant" demos wait for the whole LLM reply, then synthesize the
whole thing, then play it. That's not a conversation, it's a slow request/response
loop with a voice on top. VCB streams every stage: the moment the LLM writes a
sentence, it's chunked, sent to TTS, and playing — while the LLM is still writing
the next one. Interrupting cancels the actual in-flight generation, not just the
audio buffer.

## Features

- **Real-time streaming pipeline** — audio starts on the first sentence chunk;
  never waits for a full reply before speaking.
- **True barge-in** — talking over the assistant cancels the LLM stream, flushes
  queued/playing audio, and (if it had already said part of a reply) picks the
  cut-off thought back up naturally a few turns later if you say "keep going".
- **Any OpenAI-compatible LLM** — **Ollama** (default; model downloaded for you),
  a **cloud API** (OpenAI, OpenRouter, Groq, ...), or a local **llama.cpp** server
  (auto-launched for you) — same code path for all three.
- **Local STT/TTS** — faster-whisper for transcription; Chatterbox-Turbo (clones
  any voice from a short `.wav`, with natural `<laugh>`/`<sigh>` tags) or Kokoro
  preset voices, plus optional XTTS v2, Orpheus, and Sesame CSM engines.
- **Long-term memory** — recalls relevant facts and past conversation by semantic
  similarity, keeps a rolling summary, and consolidates old turns in the background.
- **Tool calling / skills** — the model can call built-in tools (calculator, timers,
  remember/recall, notes, date/time) via a lightweight JSON protocol, no native
  function-calling API required.
- **AI-led mode** — optionally the assistant can drive the conversation itself:
  greet you first, and speak up if you've gone quiet for a while.
- **Unreal MetaHuman avatar bridge** *(optional)* — stream audio, emotion tags, and
  body-action cues to an Unreal Engine avatar over a local socket; see
  [unreal/VcbAvatar](unreal/VcbAvatar).
- **Configured for real use, not just a demo** — device auto-reconnect, GPU pinning,
  Windows Job Object child-process cleanup, and a real automated test suite
  (`pytest`, 100+ tests, no GPU required to run them).

## Pipeline

```
mic ─▶ VAD (Silero) ─▶ STT (faster-whisper) ─▶ conversation + memory
                                                      │
speaker ◀─ playback ◀─ TTS ◀─ sentence chunker ◀─ LLM stream (OpenAI-compatible)
```

- Every stage is async; nothing blocks the event loop. GPU/CPU-heavy work
  (STT, TTS, embeddings) runs in worker threads.
- A playback "generation" counter marks queued/playing audio so a barge-in can
  drop anything stale instead of just muting it.
- STT/LLM/TTS are all behind small abstract interfaces (`STTEngine`, `LLMEngine`,
  `TTSEngine`), so swapping an engine is a one-line change in `main.py`.

## Quick start (Windows)

**You need:** Windows 10/11 and a microphone. An NVIDIA GPU (8 GB+ VRAM) is
strongly recommended; without one VCB still runs, using lighter models on the CPU.
You *don't* need Python or anything else installed.

1. Download VCB (green **Code** button → **Download ZIP**, or `git clone`) and
   extract it to a folder with ~15 GB free.
2. Double-click **`VCB.bat`**. The first run sets up its own private Python and
   downloads everything it needs into `.runtime\` (about 5 GB on an NVIDIA PC;
   allow 5–15 minutes). Later starts take seconds.
3. A settings window opens. Pick:
   - **Brain** — *Local (Ollama)*: free and private; install
     [Ollama](https://ollama.com/download) first, VCB downloads the model for you.
     Or *Cloud API*: paste an API key, no GPU needed for the model. **Google
     Gemini** has a free tier (key at [aistudio.google.com/api-keys](https://aistudio.google.com/api-keys));
     OpenAI, OpenRouter, Groq, or any OpenAI-compatible provider work too.
   - **Voice** — *Clone a voice*: choose a 5–15 second `.wav` of clean speech
     (leave empty for the built-in voice). Or a *Preset voice*.
   - **Persona**, your name, mic and speaker.
4. Click **Save**, and start talking. The first start also downloads the speech
   models (a few GB) once.

Change settings any time with **`VCB Settings.bat`**. To uninstall, delete the
folder (plus Hugging Face's model cache in `%USERPROFILE%\.cache\huggingface`).

| Launcher option | What it does |
|---|---|
| `VCB.bat --settings` | open the settings window only |
| `VCB.bat --extras` | also install the optional XTTS / Orpheus / CSM voice engines |
| `VCB.bat --reinstall` | rebuild the private Python environment from scratch |
| `VCB.bat --ai-led` (any `main.py` flag) | passed straight through to VCB |

### LLM backends

| Backend | What it needs | Good for |
|---|---|---|
| `ollama` *(default)* | [Ollama](https://ollama.com) installed; the model is pulled automatically | Easiest local option — free, private, auto GPU selection |
| `cloud` | An API key for an OpenAI-compatible provider | No GPU at all; adds network latency + per-token cost, sends the transcript to the provider |
| `llama_cpp` *(advanced)* | A GGUF file + `llama-server` on PATH | Full control over quant/context/GPU split; VCB launches it for you |

## Configuration

Settings are layered: CLI flag > `.env` / environment variable > code default.

1. **`.env`** — written by the settings window; see [.env.example](.env.example)
   for every option (GPU pinning, audio devices, feature toggles, ...). It's
   git-ignored, so it's safe for API keys.
2. **`user/`** *(git-ignored)* — your personal additions:
   - `user/prompts/*.txt` — your own personas (a file here overrides a
     same-named one in [prompts/](prompts)),
   - `user/models.json` — extra llama.cpp model presets, e.g.
     `{"my-model": {"label": "my-model", "model_path": "models\\my.gguf", "extra_args": ["--no-jinja"]}}`,
   - `user/voices/` — voice samples picked in the settings window are copied here.
3. **[vcb/config.py](vcb/config.py)** — every other tunable (VAD sensitivity,
   chunker sizes, memory thresholds, barge-in/resume behavior, ...) as typed
   dataclasses.

### CLI flags

```powershell
python main.py --backend ollama --llm-model llama3.1:8b-instruct-q4_K_M
python main.py --prompt coach --ai-led
python main.py --ephemeral                      # recall works, nothing saved to disk
python main.py --avatar --avatar-local-audio
```

Run `python main.py --help` for the full list.

### Personas

System prompts are plain text files in [prompts/](prompts) (public examples) and
`user/prompts/` (your own). Pick one in the settings window or with `--prompt`.

## Development setup

The launcher's environment lives in `.runtime\venv`; you can use it directly
(`.runtime\venv\Scripts\python.exe main.py`). To manage your own environment
instead, with [uv](https://docs.astral.sh/uv/):

```powershell
uv venv --python 3.12; .venv\Scripts\activate
uv pip install torch==2.7.1 torchaudio==2.7.1 -r requirements.txt `
    -c requirements-lock.txt --override requirements-overrides.txt `
    --extra-index-url https://download.pytorch.org/whl/cu128 --index-strategy unsafe-best-match
```

- [requirements.txt](requirements.txt) — core dependencies (Chatterbox + Kokoro voices).
- [requirements-extras.txt](requirements-extras.txt) — optional XTTS / Orpheus / CSM engines.
- [requirements-lock.txt](requirements-lock.txt) — exact tested versions (constraints).
- [requirements-overrides.txt](requirements-overrides.txt) — resolves chatterbox-tts's
  over-strict torch/transformers pins.

## Testing

```powershell
python -m pytest
```

The suite (100+ tests) covers the pieces that are easy to get subtly wrong in a
streaming/async system: barge-in and stale-audio rejection, LLM retry-duplication
safety, resume-window persistence, the sentence chunker, sanitizer, memory
recall, skills, and backend/config resolution — all without needing a GPU, audio
device, or running LLM server.

## Project layout

```
VCB.bat                 double-click launcher (first run installs everything)
VCB Settings.bat        opens the settings window
scripts/launch.ps1      the launcher logic (uv + private Python + dependencies)
main.py                 entry point + CLI
vcb/
  orchestrator.py        ties everything together: mic -> VAD -> turn -> TTS
  state.py               explicit pipeline phase (idle/listening/thinking/speaking)
  config.py              every tunable, as typed dataclasses
  settings.py            tkinter settings window (writes .env)
  hardware.py            CPU fallback when there's no CUDA GPU
  runtime.py             process lifecycle (Ollama / llama-server, transcript window)
  vad.py, chunker.py     speech segmentation, sentence-boundary chunking
  sanitize.py            strips tags/artifacts before TTS
  conversation.py        message history + trimming
  audio/                 mic capture, speaker playback, device lookup
  llm/                   OpenAI-compatible streaming client
  stt/                   faster-whisper
  tts/                   chatterbox, kokoro (+ optional xtts, orpheus, sesame csm)
  memory/                embeddings, recall, consolidation
  skills/                JSON tool-calling + built-in tools
  avatar/                Unreal MetaHuman TCP bridge
prompts/                example personas
user/                   your personas, voices, model presets (git-ignored)
tests/                  pytest suite (no GPU/device required)
unreal/VcbAvatar/       optional companion Unreal Engine plugin
```

## Status / known limitations

This is a personal project, still evolving. A few things worth knowing before
you rely on it:

- Windows only for now (the launcher, transcript window, and process cleanup
  are Windows-specific).
- STT and TTS synthesis run in worker threads that Python can't forcibly cancel;
  a barge-in discards their output immediately, but the in-flight computation
  itself finishes in the background.
- Without an NVIDIA GPU, VCB falls back to a smaller Whisper model and the
  Kokoro voice on the CPU — usable, but noticeably slower to respond.
- Voice cloning is for voices you have the right to use — your own, or with the
  speaker's consent.

## License

VCB is free for personal and other **non-commercial** use under the
[PolyForm Noncommercial License 1.0.0](LICENSE): use it, change it, share it.
Selling it or using it in a commercial product or business requires the
author's permission -- open an issue or contact
[@xulqar9](https://github.com/xulqar9) to ask.

This repository contains only VCB's own code. Libraries and models are not
included: the launcher downloads them from their official sources (PyPI,
Hugging Face) and each keeps its own license -- e.g. Chatterbox, Kokoro and
Whisper are MIT/Apache-2.0, Kokoro's English phonemizer is GPL-3.0, the optional
XTTS v2 engine is Coqui's non-commercial CPML, and LLMs follow their provider's
terms.
