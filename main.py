"""VCB entry point: python main.py (end users run VCB.bat, which calls this).

Launches (and later tears down) the LLM backend if needed and a second
console window that shows the live conversation transcript. This terminal
shows system/STT/debug logs; the popup shows only the chat.
"""

import os
import warnings
from pathlib import Path

# Upstream deprecation noise (chatterbox's watermarker, diffusers) that end
# users can't act on.
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated.*")
warnings.filterwarnings("ignore", category=FutureWarning, module="diffusers")
# Windows without Developer Mode can't symlink; HF caching still works.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

# Relative paths (logs/, user/, models/) are relative to the repo root.
os.chdir(Path(__file__).resolve().parent)

# Importing config loads .env, so GPU pinning below can come from it. config
# itself doesn't import torch, so CUDA isn't initialized yet.
from vcb import config as _config  # noqa: E402

# Optionally restrict ALL of VCB's GPU work (STT, TTS, llama-server) to some
# GPUs, e.g. VCB_CUDA_DEVICES=1 to leave GPU 0 free for a game or Unreal.
# PCI order makes the indices match nvidia-smi. Unset = use every GPU.
if _config._env_opt_str("VCB_CUDA_DEVICES"):
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ["VCB_CUDA_DEVICES"])

import argparse  # noqa: E402
import asyncio  # noqa: E402
import contextlib  # noqa: E402
import logging  # noqa: E402

from vcb.config import AppConfig, available_prompts, build_action_instructions, load_system_prompt  # noqa: E402
from vcb.hardware import adapt_to_hardware, cuda_available  # noqa: E402
from vcb.llm import OpenAICompatLLM  # noqa: E402
from vcb.orchestrator import Orchestrator  # noqa: E402
from vcb.runtime import ManagedProcesses, setup_logging  # noqa: E402
from vcb.stt import FasterWhisperSTT  # noqa: E402
from vcb.tts import ChatterboxTurboTTS, KokoroTTS, OrpheusTTS, SesameCSMTTS, XTTSv2TTS  # noqa: E402

# TTS engine: "chatterbox-turbo" (default; clones your voice .wav, voices
# <laugh>/<sigh> tags), "kokoro" (preset voices, light, runs on CPU too), and
# optional extras "xtts", "orpheus", "csm" (see requirements-extras.txt).
# Set VCB_TTS_ENGINE in .env (or the settings window).
TTS_ENGINE = os.environ.get("VCB_TTS_ENGINE", "chatterbox-turbo")

_TTS_ENGINES = {
    "kokoro": KokoroTTS,
    "xtts": XTTSv2TTS,
    "chatterbox-turbo": ChatterboxTurboTTS,
    "orpheus": OrpheusTTS,
    "csm": SesameCSMTTS,
}


async def amain(cfg: AppConfig, tts_engine: str, transcript_path) -> None:
    orchestrator = Orchestrator(
        cfg,
        stt=FasterWhisperSTT(cfg.stt),
        llm=OpenAICompatLLM(cfg.llm),
        tts=_TTS_ENGINES[tts_engine](cfg.tts),
        transcript_path=transcript_path,
    )
    await orchestrator.run()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="VCB voice assistant")
    p.add_argument("--no-memory", action="store_true",
                   help="disable long-term memory for this run")
    p.add_argument("--ephemeral", "--temp", action="store_true", dest="ephemeral",
                   help="temporary session: recall works but nothing is saved to disk")
    p.add_argument("--clear-memory", action="store_true",
                   help="wipe all saved memory, then start normally")
    p.add_argument("--ai-led", action="store_true", dest="ai_led",
                   help="AI drives the conversation: greets first and speaks up "
                        "whenever you go quiet, instead of only replying")
    p.add_argument("--avatar", action="store_true",
                   help="stream audio + state to an external 3D avatar app "
                        "(Unreal MetaHuman) over a local socket")
    p.add_argument("--avatar-local-audio", action="store_true", dest="avatar_local_audio",
                   help="with --avatar, ALSO play audio on this PC (default: the "
                        "avatar app is the only audio source)")
    prompts = available_prompts()
    p.add_argument("--prompt", dest="prompt", default=None, metavar="NAME",
                   help="persona to use from prompts/ or user/prompts/ "
                        f"(available: {', '.join(prompts) or 'none'})")
    p.add_argument("--backend", choices=["llama_cpp", "ollama", "cloud"], default=None,
                   help="LLM backend: ollama (default; local, model downloaded "
                        "automatically), cloud (hosted OpenAI-compatible API, no "
                        "GPU needed), or llama_cpp (advanced: auto-launches a "
                        "local llama-server on a GGUF file)")
    p.add_argument("--llm-model", dest="llm_model", default=None, metavar="NAME",
                   help="override the model for the selected backend (an Ollama "
                        "tag, or a cloud model id)")
    p.add_argument("--base-url", dest="base_url", default=None, metavar="URL",
                   help="override the backend's API base URL (ollama/cloud only)")
    p.add_argument("--ollama-auto-launch", action="store_true", dest="ollama_auto_launch",
                   help="with --backend ollama, always launch/stop `ollama serve` "
                        "for this session (by default an already-running Ollama "
                        "is reused, and it's only started if it isn't running)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = AppConfig()
    if args.no_memory:
        cfg.memory.enabled = False
    if args.ephemeral:
        cfg.memory.ephemeral = True
    if args.ai_led:
        cfg.proactive.enabled = True
    if args.avatar:
        cfg.avatar.enabled = True
    if args.avatar_local_audio:
        cfg.avatar.local_playback = True
    if args.prompt:
        cfg.llm.system_prompt = load_system_prompt(args.prompt)
    if cfg.avatar.enabled and cfg.avatar.teach_actions:
        cfg.llm.system_prompt += build_action_instructions(cfg.avatar.actions)
    if args.backend:
        cfg.backend = args.backend
    if args.ollama_auto_launch:
        cfg.ollama.auto_launch = True
    if args.llm_model:
        if cfg.backend == "ollama":
            cfg.ollama.model = args.llm_model
        elif cfg.backend == "cloud":
            cfg.cloud.model = args.llm_model
        else:
            cfg.llm.model = args.llm_model
    if args.base_url:
        if cfg.backend == "ollama":
            cfg.ollama.base_url = args.base_url
        elif cfg.backend == "cloud":
            cfg.cloud.base_url = args.base_url
    if args.backend or args.llm_model or args.base_url or args.ollama_auto_launch:
        cfg.configure_backend()   # re-derive llm.base_url/api_key/model
    if args.clear_memory:
        from vcb.memory import MemoryStore
        store = MemoryStore(cfg.memory.db_path, ephemeral=False)
        store.clear()
        store.close()
        print("memory cleared")
    transcript = setup_logging(cfg.log_level, cfg.server.transcript_path)
    log = logging.getLogger("vcb")
    for key, (env_value, file_value) in _config.ENV_OVERRIDES.items():
        if any(s in key for s in ("KEY", "TOKEN", "SECRET")):
            env_value = file_value = "<hidden>"
        log.warning("%s=%r from this terminal's environment overrides .env (%r). "
                    "If you changed settings, open a new terminal or run VCB.bat.",
                    key, env_value, file_value)
    tts_engine = adapt_to_hardware(cfg, TTS_ENGINE, cuda_available())
    procs = ManagedProcesses(cfg)
    try:
        procs.start_server()                 # blocks until the backend is ready
        procs.start_transcript_window(transcript)
        asyncio.run(amain(cfg, tts_engine, transcript))
    except KeyboardInterrupt:
        print("\nbye")
    except Exception as exc:
        # Our own setup errors (RuntimeError) carry a user-facing fix; anything
        # else is a bug, so keep the traceback.
        logging.getLogger("vcb").error(
            "VCB stopped: %s", exc, exc_info=not isinstance(exc, RuntimeError)
        )
        raise SystemExit(1) from None
    finally:
        procs.shutdown()                     # kill server + transcript window


if __name__ == "__main__":
    with contextlib.suppress(KeyboardInterrupt):
        main()
