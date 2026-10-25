"""Central configuration. Every tunable lives here.

Common user-facing settings (API keys, active model/prompt/TTS engine, memory,
avatar, AI-led mode, ...) can be overridden without touching this file by
setting environment variables or putting them in a `.env` file at the repo
root (see .env.example). Real environment variables always win over `.env`;
`.env` only fills in what isn't already set; code defaults apply if neither is.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from .envfile import read_env

_ROOT_DIR = Path(__file__).resolve().parent.parent
# Personal, git-ignored overrides: user/prompts/*.txt (personas that add to or
# replace prompts/), user/models.json (extra llama.cpp model presets), voices.
USER_DIR = _ROOT_DIR / "user"


def resolve_path(path: str) -> str:
    """Relative paths in config/.env are relative to the repo root, not the
    current working directory, so VCB behaves the same however it's launched."""
    if not path:
        return path
    p = Path(path)
    return str(p if p.is_absolute() else _ROOT_DIR / p)


def _load_dotenv(path: Path) -> None:
    """Minimal `.env` loader (KEY=VALUE lines, '#' comments) -- no dependency
    needed for a handful of simple settings. Real env vars always win; any
    that shadow a different .env value are recorded in ENV_OVERRIDES."""
    for key, value in read_env(path).items():
        current = os.environ.get(key)
        if current is None:
            os.environ[key] = value
        elif current != value:
            ENV_OVERRIDES[key] = (current, value)


# key -> (value in effect from the real environment, ignored .env value). Often
# a stale terminal: VS Code's Python extension copies .env into new terminals
# once, so later edits to .env are shadowed in that terminal.
ENV_OVERRIDES: dict[str, tuple[str, str]] = {}


ENV_FILE = Path(os.environ.get("VCB_ENV_FILE") or _ROOT_DIR / ".env")
_load_dotenv(ENV_FILE)


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


def _env_opt_str(name: str) -> str | None:
    """Env var as a string, or None when unset/blank."""
    val = os.environ.get(name, "").strip()
    return val or None


def _env_device(name: str) -> int | str | None:
    """Audio device from env: a PortAudio index or a (partial) device name;
    unset/blank = system default."""
    val = _env_opt_str(name)
    if val is None:
        return None
    return int(val) if val.lstrip("-").isdigit() else val


# How the assistant refers to the person it talks to inside VCB's own nudges
# (AI-led cues, resume hints). Personas in prompts/ set their own wording.
USER_NAME: str = _env_str("VCB_USER_NAME", "").strip() or "the user"

# System prompts live as plain .txt files in prompts/ (project root). Personal
# ones go in user/prompts/ (git-ignored); a file there wins over a same-named
# one in prompts/. Point VCB_ACTIVE_PROMPT at a name (without .txt) to switch.
PROMPTS_DIR = _ROOT_DIR / "prompts"
USER_PROMPTS_DIR = USER_DIR / "prompts"
ACTIVE_PROMPT: str = _env_str("VCB_ACTIVE_PROMPT", "assistant")


def available_prompts() -> list[str]:
    names = {p.stem for d in (PROMPTS_DIR, USER_PROMPTS_DIR) for p in d.glob("*.txt")}
    return sorted(names, key=str.lower)


def load_system_prompt(name: str) -> str:
    for d in (USER_PROMPTS_DIR, PROMPTS_DIR):
        path = d / f"{name}.txt"
        if path.is_file():
            return path.read_text(encoding="utf-8").strip()
    available = ", ".join(available_prompts()) or "none"
    raise FileNotFoundError(
        f"System prompt {name!r} not found in {PROMPTS_DIR} or {USER_PROMPTS_DIR} "
        f"(available: {available})"
    )


@dataclass
class AudioConfig:
    input_sample_rate: int = 16_000      # what VAD + whisper expect
    output_sample_rate: int = 24_000     # what Kokoro produces
    frame_ms: int = 32                   # 512 samples @ 16 kHz, Silero's native frame
    input_device: int | str | None = field(
        default_factory=lambda: _env_device("VCB_INPUT_DEVICE")
    )                                    # None = system default
    output_device: int | str | None = field(
        default_factory=lambda: _env_device("VCB_OUTPUT_DEVICE")
    )
    mic_queue_max_frames: int = 256      # ~8 s; overflow drops oldest
    playback_prime_ms: int = 250         # jitter buffer: build this lead before playback
                                         # starts, so streamed TTS won't underrun mid-laugh

    @property
    def frame_samples(self) -> int:
        return self.input_sample_rate * self.frame_ms // 1000


@dataclass
class VADConfig:
    threshold: float = 0.5               # speech probability trigger
    min_speech_ms: int = 250             # shorter bursts are ignored (coughs, clicks)
    end_silence_ms: int = 600            # silence that ends an utterance
    pre_roll_ms: int = 320               # audio kept from before speech onset
    max_utterance_s: float = 30.0        # hard cap, force-flush


@dataclass
class TurnTakingConfig:
    """When the assistant stops talking and when it starts replying."""
    # While the assistant is speaking, the user has to keep talking this long
    # (actual voiced time) before it stops. Shorter sounds are transcribed
    # first: a backchannel ("yeah", "mhm", "right") is ignored and the
    # assistant just keeps going; anything else still interrupts it.
    barge_in_min_speech_s: float = 0.8
    ignore_backchannels: bool = True
    # If an utterance ends mid-sentence ("...and", "I hit the..."), wait this
    # long for the rest before replying (0 = reply immediately).
    incomplete_wait_s: float = 1.2


@dataclass
class STTConfig:
    # "faster-whisper" (CTranslate2: NVIDIA GPU or CPU) or "whisper-torch"
    # (plain PyTorch: anywhere torch has a GPU, i.e. also AMD ROCm). main.py
    # picks whisper-torch on AMD automatically.
    engine: str = field(default_factory=lambda: _env_str("VCB_STT_ENGINE", "faster-whisper"))
    model: str = "large-v3-turbo"        # whisper model name or path
    device: str = "cuda"                 # main.py drops to cpu if no CUDA GPU
    device_index: int = 0                # which GPU; chosen automatically with several
    compute_type: str = "float16"
    cpu_model: str = "small"             # used instead when running on CPU
    cpu_compute_type: str = "int8"
    language: str = "en"
    beam_size: int = 1                   # greedy = lowest latency
    # Give Whisper the last bit of conversation plus remembered facts as a
    # prompt, so names and jargon ("Plat 3", "boost", "whiffed") are heard
    # right. Kept short: long prompts make Whisper hallucinate.
    use_context: bool = True
    context_chars: int = 200


@dataclass
class LLMConfig:
    # Any OpenAI-compatible server: llama.cpp server, LM Studio, Ollama, vLLM.
    base_url: str = "http://127.0.0.1:8080/v1"
    api_key: str = "not-needed"
    model: str = ""                   # set from MODEL_PRESETS[ACTIVE_MODEL] in __post_init__
    max_tokens: int = 200
    temperature: float = 0.3          # LOWER - makes model more obedient to prompt
    top_p: float = 0.95                # Slightly higher
    frequency_penalty: float = 0.0    # REMOVE - can interfere with compliance patterns
    presence_penalty: float = 0.0     # REMOVE
    # Add if your framework supports it:
    repetition_penalty: float = 1.0     # Neutral (1.0 = no penalty)
    # How long "thinking" models reason before answering (none/minimal/low/
    # medium/high, model-dependent). None = don't send it (provider default).
    reasoning_effort: str | None = None
    request_timeout_s: float = 60.0
    max_retries: int = 2
    retry_backoff_s: float = 1.0
    # Loaded from prompts/{ACTIVE_PROMPT}.txt in AppConfig.__post_init__.
    # Set this to a non-empty string to override the file.
    system_prompt: str = ""
    history_max_turns: int = 20          # user+assistant pairs kept in context


@dataclass
class TTSConfig:
    # kokoro
    voice: str = field(default_factory=lambda: _env_str("VCB_KOKORO_VOICE", "af_heart"))
    lang_code: str = "a"                 # 'a' = American English
    speed: float = 1.0
    # voice cloning (chatterbox / xtts). Missing file: chatterbox falls back to
    # its built-in voice. Relative paths are resolved against the repo root.
    voice_wav: str = field(
        default_factory=lambda: _env_str("VCB_VOICE_WAV", r"user\voice.wav")
    )
    # TTS device ("cuda" = picked automatically when there are several GPUs,
    # "cuda:N" = that GPU, "cpu"); main.py drops to cpu if there's no GPU.
    device: str = "cuda"
    language: str = "en"                 # xtts language
    exaggeration: float = 0.5            # chatterbox emotion intensity (0-1)
    # orpheus (preset voices, emits <laugh>/<sigh>/etc.)
    orpheus_voice: str = field(                   # tara leah jess leo dan mia zac zoe
        default_factory=lambda: _env_str("VCB_ORPHEUS_VOICE", "tara"))
    orpheus_n_gpu_layers: int = -1       # -1 = offload all layers to GPU
    orpheus_n_ctx: int = 8192            # cap context; 0 would grab a 14GB KV cache
    orpheus_pre_buffer_s: float = 0.5    # lower = faster first audio
    orpheus_temperature: float = 0.8     # Orpheus default; raise for livelier, lower for stable
    orpheus_top_p: float = 0.95
    orpheus_top_k: int = 40
    orpheus_min_p: float = 0.05
    # sesame csm (transformers CsmForConditionalGeneration, Mimi codec @ 24 kHz)
    csm_model_id: str = "unsloth/csm-1b"  # ungated mirror; "sesame/csm-1b" needs granted access
    csm_device: str = "cuda:0"           # first visible GPU (see VCB_CUDA_DEVICES)
    csm_dtype: str = "bfloat16"          # bfloat16 | float16 | float32
    csm_speaker_id: int = 0              # CSM prepends [speaker] to each utterance
    csm_max_new_tokens: int = 1024       # audio frames cap (~0.08s each); guards runaway
    # Voice seeding: without a reference CSM invents a new voice each call. Point
    # VCB_CSM_REF_AUDIO at a short clean wav and give its exact transcript to lock
    # one consistent voice (both must be set to take effect).
    csm_ref_audio: str = field(
        default_factory=lambda: _env_str("VCB_CSM_REF_AUDIO", "")
    )                                           # reference wav that fixes the voice
    csm_ref_text: str = (                       # EXACT transcript of csm_ref_audio
        "Hello, this is a universal test sentence. Can the advanced xylophonic "
        "system clearly articulate this and express a hint of excitement? "
        "The quick brown fox certainly hopes so."
    )
    csm_compile: bool = False            # torch.compile: needs triton, but REGRESSES
                                         # CSM (constant recompiles) -> leave off
    # indextts (isolated runtime installed by the launcher; clones voice_wav)
    indextts_dir: str = r".runtime\indextts"
    # GPUs to run IndexTTS on (nvidia-smi numbers): "auto" = the voice's GPU plus
    # one more with room (renders ahead, no pauses mid-reply), or e.g. "1" / "0,1".
    indextts_gpus: str = field(
        default_factory=lambda: _env_str("VCB_INDEXTTS_GPUS", "auto"))
    indextts_steps: int = 25             # mel diffusion steps (IndexTTS default);
                                         # fewer is faster but audibly rougher
    indextts_bf16: bool = True           # False = fp32: ~3x slower
    indextts_emotion_strength: float = 1.0  # scales the mood vectors (0 = neutral)
    chunk_queue_max: int = 8             # pending text chunks awaiting synthesis


@dataclass
class ChunkerConfig:
    min_chars: int = 24                  # don't synthesize tiny fragments
    max_chars: int = 240                 # force a split even mid-sentence
    first_chunk_min_chars: int = 8       # ship the first chunk ASAP for low latency
    first_chunk_clause_split: bool = True  # first chunk may end at a comma


@dataclass
class ProsodyConfig:
    """Human cadence + consistent loudness applied to TTS audio in the tts loop."""
    enabled: bool = True
    # Silence inserted after a chunk, chosen by its trailing punctuation (seconds).
    pause_sentence_s: float = 0.35       # . !
    pause_question_s: float = 0.40       # ?
    pause_ellipsis_s: float = 0.55       # ... (trailing off)
    pause_clause_s: float = 0.15         # , ; :
    pause_default_s: float = 0.0         # mid-sentence hard split -> no pause
    # Loudness: makeup gain then a transparent peak limiter (no pumping/coloring).
    gain: float = 1.0                    # 1.0 = pass through untouched; raise to boost
    ceiling: float = 0.98                # soft-limit only peaks above this (0 = off)


# ---------------------------------------------------------------------------
# llama.cpp model registry (advanced; most people should use the ollama or
# cloud backend instead). Built-in presets are examples. Add your own in
# user/models.json (git-ignored), e.g.
#   {"my-model": {"label": "my-model", "model_path": "models\\my.gguf",
#                 "ctx_size": 16384, "extra_args": ["--no-jinja"]}}
# then set VCB_ACTIVE_MODEL=my-model. VCB_ACTIVE_MODEL may also be a direct
# path to any .gguf file.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ModelPreset:
    label: str                           # name advertised to the OpenAI API
    model_path: str                      # path to the .gguf on disk
    ctx_size: int = 16384
    n_gpu_layers: int = 99               # 99 = offload everything to the GPU
    # Extra llama-server CLI args (e.g. Mistral GGUFs ship a Jinja template
    # llama.cpp can't parse: ("--no-jinja", "--chat-template", "mistral-v3-tekken")).
    extra_args: tuple[str, ...] = ()


MODEL_PRESETS: dict[str, ModelPreset] = {
    "qwen3-4b": ModelPreset(
        label="qwen3-4b-instruct-2507",
        model_path=r"models\Qwen3-4B-Instruct-2507\Qwen3-4B-Instruct-2507-Q4_K_M.gguf",
    ),
}

USER_MODELS_FILE = USER_DIR / "models.json"


def load_user_presets(path: Path = USER_MODELS_FILE) -> dict[str, ModelPreset]:
    if not path.is_file():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8-sig"))
    presets = {}
    for name, spec in raw.items():
        spec = dict(spec)
        spec["extra_args"] = tuple(spec.get("extra_args", ()))
        presets[name] = ModelPreset(**spec)
    return presets


MODEL_PRESETS.update(load_user_presets())

ACTIVE_MODEL: str = _env_str("VCB_ACTIVE_MODEL", "qwen3-4b")


def resolve_model_preset(name: str) -> ModelPreset:
    if name in MODEL_PRESETS:
        return MODEL_PRESETS[name]
    if name.lower().endswith(".gguf"):
        return ModelPreset(label=Path(name).stem, model_path=name)
    raise ValueError(
        f"VCB_ACTIVE_MODEL {name!r} is neither a preset ({', '.join(MODEL_PRESETS)}) "
        "nor a path to a .gguf file"
    )


@dataclass
class ServerConfig:
    # Auto-launch a local llama.cpp server so you don't manage it by hand.
    auto_launch: bool = True
    binary: str = "llama-server"         # resolved via PATH / winget install dir
    model_path: str = ""                 # set from the active preset in configure_backend()
    host: str = "127.0.0.1"
    port: int = 8080
    n_gpu_layers: int = 99
    ctx_size: int = 16384
    # Pin llama-server to specific GPU(s) by nvidia-smi index, e.g. "0" or "0,1".
    # Unset = all GPUs.
    cuda_visible_devices: str | None = field(
        default_factory=lambda: _env_opt_str("VCB_LLM_CUDA_DEVICES")
    )
    extra_args: tuple[str, ...] = ()     # set from the active MODEL_PRESET
    startup_timeout_s: float = 180.0     # allow time for first model load
    transcript_window: bool = True       # open a 2nd terminal for the chat log
    transcript_path: str = r"logs\chat.log"


@dataclass
class OllamaConfig:
    """Backend option for users who'd rather not manage llama.cpp GGUFs/flags by
    hand. Ollama auto-detects and spreads across whatever GPU(s) are available,
    which is friendlier for low-experience users, at the cost of a bit of control
    and (typically) a small throughput/latency hit versus a hand-tuned llama.cpp
    server on the same quant. Uses Ollama's OpenAI-compatible endpoint, so it goes
    through the same OpenAICompatLLM client as llama.cpp."""
    base_url: str = field(
        default_factory=lambda: _env_str("VCB_OLLAMA_BASE_URL", "http://127.0.0.1:11434/v1")
    )
    api_key: str = "ollama"              # ignored by Ollama, but the client needs a value
    model: str = field(
        default_factory=lambda: _env_str("VCB_OLLAMA_MODEL", "llama3.1:8b")
    )   # pulled automatically on first use if missing
    # Context window VCB runs the model with. Ollama's own default can be 32k,
    # which multiplies GPU memory use; a voice conversation + memory fits in
    # far less. VCB applies this via a lightweight "<model>-vcb" copy.
    num_ctx: int = field(default_factory=lambda: int(_env_str("VCB_OLLAMA_NUM_CTX", "8192")))
    # Ollama is normally started once as a background service (`ollama serve` or
    # the desktop app); leave this off and just make sure that's running. Turn it
    # on if you'd rather VCB launch/stop it for you each session.
    auto_launch: bool = field(
        default_factory=lambda: _env_bool("VCB_OLLAMA_AUTO_LAUNCH", False)
    )
    binary: str = "ollama"               # resolved via PATH
    startup_timeout_s: float = 30.0


@dataclass
class CloudConfig:
    """Backend option for a hosted OpenAI-compatible API (OpenAI, OpenRouter,
    Groq, Together, ...) instead of any local model. No GPU/process needed, but
    expect network latency and per-token cost, and audio never leaves the
    machine only up to the point the transcript is sent to the provider.
    NEVER put a real key in this file -- set the environment variable named by
    api_key_env before launching VCB."""
    base_url: str = field(
        default_factory=lambda: _env_str("VCB_CLOUD_BASE_URL", "https://api.openai.com/v1")
    )
    api_key_env: str = "VCB_CLOUD_API_KEY"   # env var read at startup
    # Thinking models (Gemini 3, o-series, ...) can spend many seconds reasoning
    # before the first word -- set "minimal"/"low" for snappy conversation.
    reasoning_effort: str | None = field(
        default_factory=lambda: _env_opt_str("VCB_CLOUD_REASONING_EFFORT")
    )
    model: str = field(default_factory=lambda: _env_str("VCB_CLOUD_MODEL", "gpt-4o-mini"))


@dataclass
class ResumeConfig:
    """When your speech cuts the assistant off mid-sentence, the model itself
    decides on the next turn whether you merely backchannelled (so it continues
    its thought) or started something new (so it responds to that)."""
    enabled: bool = True
    # A cut-off thought stays resumable for a few turns, so a brief backchannel
    # ("wait", "sorry, go on") doesn't burn it before you actually say "continue".
    window_turns: int = 3                # turns the resume offer survives
    resumed_min_words: int = 12          # a completed reply this long = actually resumed
    hint: str = field(default_factory=lambda: (
        f"Your last reply was cut off because {USER_NAME} started talking. Look at "
        "what was just said. If it was only a brief acknowledgement, filler, or a "
        "nudge to keep going, simply continue your interrupted thought from "
        "where you left off, as if you were never interrupted. But if it was "
        "something new or changed the subject, drop the old thought and respond "
        "to that instead. Either way, never mention or apologize "
        "for the interruption."
    ))


@dataclass
class ProactiveConfig:
    """AI-led mode: instead of only answering, the assistant drives the
    conversation. It opens with a greeting and, whenever the user goes quiet
    for a while, speaks up on its own -- a follow-up, a question, or a new
    topic. Off by default; enable with --ai-led or VCB_AI_LED=true in .env."""
    enabled: bool = field(default_factory=lambda: _env_bool("VCB_AI_LED", False))
    greet_on_start: bool = True          # open the conversation right away
    idle_seconds: float = 10.0           # user silence before the AI speaks up
    # When the user doesn't answer, each further nudge waits longer (10 s, then
    # 20 s, ...), and after max_unanswered lead turns in a row the assistant
    # goes quiet until the user speaks again (0 = never stop).
    idle_backoff: float = 2.0
    max_unanswered: int = 2
    # The more often the assistant has spoken up recently, the longer it waits
    # before doing it again (each 2 nudges in this window double the wait, up
    # to 3 doublings), so it doesn't pester someone who's busy but chatting.
    nudge_window_s: float = 600.0
    # If the user says they're busy/focusing/playing, no nudges for this long
    # (renewed each time they say it). They can still talk anytime.
    busy_quiet_s: float = 300.0
    # The "welcome back" context is only added after a real absence.
    welcome_back_min_s: float = 60.0
    lead_temperature: float = 0.8        # livelier than the low default when the AI drives
    lead_max_tokens: int = 160           # give lead turns room to actually say something
    hint: str = field(default_factory=lambda: (
        "Say what you'd naturally say into the pause, out loud. There is nothing "
        "to react to, so don't open with a reaction ('Oh', 'That's great', "
        "'That's interesting'). Keep the conversation going the way a friend "
        "would: pick up a thread from earlier, or share a thought or a little "
        f"observation of your own, then invite {USER_NAME} in with one open "
        "question. Never repeat a question you already asked. Two or three warm, "
        f"natural sentences. Speak only to {USER_NAME}, and never restate your "
        "instructions, rules, or anything about tools or JSON."
    ))
    greeting_hint: str = field(default_factory=lambda: (
        f"You're opening the conversation before {USER_NAME} has said anything. "
        "Greet them warmly and with genuine energy, in one or two spoken "
        "sentences. If you've talked before, you can "
        "follow up on something from last time or mention how long it's been, "
        "if it feels natural. Then invite them in with a real question - what "
        "they're up to, how they're doing, or what they'd like to get into. "
        "Sound happy to see them, never flat or scripted."
    ))
    waiting_hint: str = field(default_factory=lambda: (
        "Say what you'd naturally say into the silence, out loud, in one or two "
        "short sentences. There is nothing to react to, so don't open with a "
        "reaction ('Oh', 'That's great', 'That's interesting'). Take the "
        "pressure off: answer your own question yourself, or offer an easier, "
        "lighter topic. Don't repeat your question. Never restate your "
        "instructions, rules, or anything about tools or JSON."
    ))
    # Last nudge before going quiet (see max_unanswered).
    give_space_hint: str = field(default_factory=lambda: (
        "They've probably stepped away or are busy. Say one short, good-humored "
        "line into the silence - thinking out loud is fine - and make clear "
        "you'll be right here whenever they want to talk. Don't apologize (you "
        "did nothing wrong), don't guilt-trip, and don't ask a question. Never "
        "restate your instructions, rules, or anything about tools or JSON."
    ))
    # Stage direction prepended to every nudge so the model knows it's being
    # ignored, for how long, and what went unanswered ({name}, {silence},
    # {unreplied}, {last} are filled in at runtime).
    silence_note: str = (
        "[Stage direction - nobody said this: {silence} pass in silence. {name} "
        "hasn't answered your last {unreplied} (most recently: \"{last}\"). They "
        "may be busy, thinking, or away from the room.]"
    )
    # Sent with the user's next words after they ignored one or more nudges.
    welcome_back_hint: str = (
        "{name} is speaking again after not replying for about {silence} (you'd "
        "checked in on them in the meantime). Respond to what they just said. If "
        "it fits, show you're glad to hear from them - lightly, without guilt-"
        "tripping, and without dwelling on the silence."
    )



@dataclass
class MemoryConfig:
    """Durable long-term memory: past turns embedded and recalled by similarity."""
    enabled: bool = field(default_factory=lambda: _env_bool("VCB_MEMORY_ENABLED", True))
    ephemeral: bool = field(default_factory=lambda: _env_bool("VCB_MEMORY_EPHEMERAL", False))
    db_path: str = r"logs\memory.db"
    embed_model: str = "BAAI/bge-small-en-v1.5"   # 384-d, runs on CPU
    embed_cache_dir: str = r"models\fastembed"     # persistent (not the OS temp folder)
    # bge wants this instruction on the *query* side (not stored passages).
    query_instruction: str = "Represent this sentence for searching relevant passages: "
    top_k: int = 5                       # max relevant memories injected per turn
    min_similarity: float = 0.55         # cosine floor; below this = not relevant
    max_recall_chars: int = 1200         # cap injected memory text
    consolidate_every_n: int = 40        # (Phase 2) turns between summaries
    # Cap on embeddings kept in RAM for recall (newest win; 0 = unlimited).
    # Older turns stay in SQLite and live on through consolidated summaries.
    max_index_rows: int = 50_000


@dataclass
class SupervisionConfig:
    """Restart policy for the always-on loops (mic, VAD events, TTS, idle)."""
    backoff_initial_s: float = 0.5       # first restart delay; doubles each time
    backoff_max_s: float = 10.0
    max_restarts: int = 5                # consecutive failures before giving up
    healthy_after_s: float = 30.0        # a run this long resets the failure count


@dataclass
class SkillsConfig:
    """Tool/skill calling. The model emits a JSON action block (works without
    native tool-calling / --jinja); the orchestrator runs the skill and lets the
    model phrase the spoken reply."""
    enabled: bool = True
    max_tool_iters: int = 3              # cap tool calls per turn (loop guard)
    # How tools are offered to the model: "native" = the API's function calling
    # (reliable on cloud models, which tend to ignore a text protocol), "json" =
    # VCB's own JSON-in-text protocol (works with any local model), "auto" =
    # native for the cloud backend, json otherwise.
    tool_mode: str = field(default_factory=lambda: _env_str("VCB_TOOL_MODE", "auto"))
    enabled_skills: tuple[str, ...] = (
        "datetime", "calculator", "set_timer",
        "set_reminder", "list_reminders", "cancel_reminder",
        "remember", "recall", "add_note", "list_notes",
    )


@dataclass
class AvatarConfig:
    """Stream TTS audio + conversation state/emotion to an external 3D avatar app
    (Unreal MetaHuman) over a local TCP socket. Off by default. When enabled with
    local_playback=False, VCB stops playing audio itself so the avatar app is the
    single audio source and lip sync stays perfectly in sync."""
    enabled: bool = field(default_factory=lambda: _env_bool("VCB_AVATAR_ENABLED", False))
    host: str = "127.0.0.1"
    port: int = 8765
    local_playback: bool = field(
        default_factory=lambda: _env_bool("VCB_AVATAR_LOCAL_AUDIO", False)
    )         # also play audio through this PC's speakers
    queue_max: int = 2048                # outbound frame backlog; overflow drops oldest
    speech_end_margin_ms: int = 300      # hold speech_end this long past the estimated
                                         # playhead so UE's buffer drains before the A2F
                                         # stream closes (prevents clipped tails)
    # Body actions the LLM may trigger by writing an inline <action:name> tag mid-
    # speech. name -> when to use it (fed to the model). Keep in sync with the
    # ActionMontages map in ABP_VcbBody. teach_actions appends the vocabulary to the
    # system prompt (only when the avatar is enabled).
    teach_actions: bool = True
    actions: dict[str, str] = field(default_factory=lambda: {
        "wave": "greeting or saying hello/goodbye",
        "dance": "celebrating, excited, or the mood turns joyful/musical",
        "hipdance": "being playful, silly, or showing off a groovy move",
    })


def build_action_instructions(actions: dict[str, str]) -> str:
    """Voice-native system-prompt block teaching the model the <action:name> tags."""
    if not actions:
        return ""
    lines = [f"- <action:{name}> - {when}" for name, when in actions.items()]
    listing = "\n".join(lines)
    example = next(iter(actions))
    return (
        "\n\nYOUR BODY\n"
        "You control a real animated character with a body, so you can move and "
        "express yourself physically. Trigger a movement by writing an inline tag "
        "<action:NAME> at the exact moment it should happen. The tag is silent - it "
        "is never spoken aloud, it only makes your body move. Use them naturally and "
        "sparingly, only when they genuinely fit; never force one.\n"
        "Actions you can perform:\n"
        f"{listing}\n"
        f"Example: \"Oh, hey! <action:{example}> Great to see you.\"\n"
        "Only use names from this list; put the tag right where the motion happens; "
        "at most one or two per reply; never say the word action or read a tag aloud."
    )



@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    vad: VADConfig = field(default_factory=VADConfig)
    stt: STTConfig = field(default_factory=STTConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    tts: TTSConfig = field(default_factory=TTSConfig)
    chunker: ChunkerConfig = field(default_factory=ChunkerConfig)
    prosody: ProsodyConfig = field(default_factory=ProsodyConfig)
    server: ServerConfig = field(default_factory=ServerConfig)
    barge_in: bool = True                # user speech interrupts playback
    # The user's time zone as an IANA name ("Europe/Berlin", "America/New_York").
    # Empty = whatever this PC uses (incl. its daylight-saving rules).
    timezone: str = field(default_factory=lambda: _env_str("VCB_TIMEZONE", "").strip())
    turn_taking: TurnTakingConfig = field(default_factory=TurnTakingConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    skills: SkillsConfig = field(default_factory=SkillsConfig)
    proactive: ProactiveConfig = field(default_factory=ProactiveConfig)
    avatar: AvatarConfig = field(default_factory=AvatarConfig)
    supervision: SupervisionConfig = field(default_factory=SupervisionConfig)
    # Which LLM backend to talk to: "ollama" (default; local, auto GPU selection,
    # model pulled automatically), "cloud" (hosted OpenAI-compatible API, no GPU
    # needed), or "llama_cpp" (advanced: auto-launches a local llama-server on a
    # GGUF from MODEL_PRESETS / user/models.json). All three speak through the
    # same OpenAICompatLLM client. Override with VCB_BACKEND in .env.
    backend: str = field(default_factory=lambda: _env_str("VCB_BACKEND", "ollama"))
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    cloud: CloudConfig = field(default_factory=CloudConfig)
    log_level: str = field(default_factory=lambda: _env_str("VCB_LOG_LEVEL", "INFO"))

    def __post_init__(self) -> None:
        self.configure_backend()
        # Load the selected system prompt file unless one was set explicitly.
        if not self.llm.system_prompt:
            self.llm.system_prompt = load_system_prompt(ACTIVE_PROMPT)

    def configure_backend(self) -> None:
        """Point llm.base_url/api_key/model (and server auto-launch) at the
        selected backend. Called once at construction; call again after
        changing .backend or its sub-config (e.g. from CLI args)."""
        self.llm.reasoning_effort = None
        if self.backend == "llama_cpp":
            preset = resolve_model_preset(ACTIVE_MODEL)
            self.server.model_path = resolve_path(preset.model_path)
            self.server.ctx_size = preset.ctx_size
            self.server.n_gpu_layers = preset.n_gpu_layers
            self.server.extra_args = preset.extra_args
            self.llm.base_url = f"http://{self.server.host}:{self.server.port}/v1"
            self.llm.api_key = "not-needed"
            self.llm.model = preset.label
        elif self.backend == "ollama":
            self.llm.base_url = self.ollama.base_url
            self.llm.api_key = self.ollama.api_key
            self.llm.model = self.ollama.model
        elif self.backend == "cloud":
            self.llm.base_url = self.cloud.base_url
            self.llm.api_key = os.environ.get(self.cloud.api_key_env, "")
            self.llm.model = self.cloud.model
            self.llm.reasoning_effort = self.cloud.reasoning_effort
        else:
            raise ValueError(
                f"unknown backend {self.backend!r} (use llama_cpp, ollama, or cloud)"
            )

