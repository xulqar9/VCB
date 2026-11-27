"""Benchmark worker: runs ONE engine in its own process.

Why a separate process: the engine is pinned to one GPU with CUDA_VISIBLE_DEVICES,
a hung synthesis can be killed without losing the benchmark, and when the process
exits the driver frees every byte of its VRAM (the cleanest "unload" there is).
The parent (tts_bench/runner.py) talks to it with JSON lines:

    stdin  <- {"op": "init", ...}                          once, first
              {"op": "synth", "id": 1, "text": "...", "seed": 1235,
               "save_to": "C:/.../n01_1.wav" | null, "timeout_s": 30}
              {"op": "quit"}
    stdout -> {"event": "ready", "load_s": 12.3, ...}      after load()
              {"event": "load_error", ...}                 load() failed, then exit
              {"event": "result", "id": 1, "ok": true, ...}
              {"event": "bye", "blocked_network": [...]}

Everything else the engine prints goes to stderr, so stdout stays a clean channel
(the same trick as vcb/tts/indextts_worker.py).
"""

from __future__ import annotations

import asyncio
import dataclasses
import gc
import importlib
import json
import os
import random
import re
import sys
import tempfile
import threading
import time
import traceback
import wave

ANGLE_TAG = re.compile(r"<\s*([a-zA-Z_]+)\s*>")
WATCHDOG_GRACE_S = 5.0      # extra seconds before the worker kills itself


def _claim_stdout():
    """Keep the real stdout for the protocol; point fd 1 (Python prints and
    native libraries alike) at stderr."""
    proto = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return proto


def _send(proto, obj: dict) -> None:
    proto.write(json.dumps(obj).encode("utf-8") + b"\n")


def _short(exc: BaseException, limit: int = 500) -> str:
    text = f"{exc}".strip().replace("\r", " ").replace("\n", " ")
    return text[:limit]


def _torch():
    """torch if the engine already imported it (never import it here: some
    engines run their model in a child process and have no torch at all)."""
    return sys.modules.get("torch")


def _cuda_live() -> bool:
    torch = _torch()
    try:
        return bool(torch is not None and torch.cuda.is_initialized())
    except Exception:
        return False


def _seed_everything(seed: int | None) -> None:
    """The same seed goes to every engine for the same run number."""
    if seed is None:
        return
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed % 2**32)
    except Exception:
        pass
    torch = _torch()
    if torch is not None:
        try:
            torch.manual_seed(seed)
        except Exception:
            pass


def _build_options(cls, options: dict, reference_text: str):
    """Adapters that need settings declare a dataclass `Options` (the VCB
    adapters take only a TTSConfig, so they have none)."""
    opt_cls = getattr(cls, "Options", None)
    if opt_cls is None:
        if options:
            raise TypeError(f"{cls.__name__} takes no options, but the config sets: "
                            f"{', '.join(sorted(options))}")
        return None
    names = {f.name for f in dataclasses.fields(opt_cls)}
    unknown = sorted(set(options) - names)
    if unknown:
        raise TypeError(f"unknown option(s) for {cls.__name__}: {', '.join(unknown)} "
                        f"(valid: {', '.join(sorted(names))})")
    values = dict(options)
    if "reference_text" in names and reference_text and "reference_text" not in values:
        values["reference_text"] = reference_text
    return opt_cls(**values)


def _write_wav(path: str, audio, sample_rate: int) -> None:
    """16-bit mono PCM through the stdlib: the worker needs nothing beyond numpy."""
    import numpy as np
    pcm = (np.clip(audio, -1.0, 1.0) * 32767.0).round().astype("<i2")
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + ".part"
    with wave.open(tmp, "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(int(sample_rate))
        fh.writeframes(pcm.tobytes())
    os.replace(tmp, path)


def _resolve_converter(module, ref: str | None):
    """The tag converter the registry names: a function of the adapter module, or
    "module:function" for one that lives elsewhere (VCB's own engines turn tags into
    words with vcb.sanitize)."""
    if not ref:
        return None
    if ":" in ref:
        mod_name, _, func = ref.partition(":")
        return getattr(importlib.import_module(mod_name), func)
    return getattr(module, ref)


def _vcb_text_step(engine):
    """What VCB does to a line before an engine hears it (orchestrator._push_spoken and the
    TTS loop): sanitize_for_speech keeps the tags this engine performs and voices or drops
    the others, then soften_shouting. VCB's own engines get the same text here."""
    from vcb.sanitize import sanitize_for_speech, soften_shouting
    keep_tags = bool(getattr(engine, "supports_tags", False))
    mood_tags = frozenset(getattr(engine, "mood_tags", ())) | frozenset(getattr(engine, "extra_tags", ()))
    return lambda text: soften_shouting(sanitize_for_speech(text, keep_tags, mood_tags))


def _tag_counts(converter, text: str) -> tuple[int, int]:
    """(tags in the sentence, tags the engine cannot perform and therefore drops)."""
    tags = ANGLE_TAG.findall(text)
    if converter is None:
        return len(tags), 0
    return len(tags), sum(1 for t in tags if not converter(f"<{t}>").strip())


def _memory_report() -> dict:
    torch = _torch()
    if not _cuda_live():
        return {}
    try:
        return {
            "torch_peak_alloc_mib": round(torch.cuda.max_memory_allocated() / 2**20, 1),
            "torch_peak_reserved_mib": round(torch.cuda.max_memory_reserved() / 2**20, 1),
        }
    except Exception:
        return {}


async def _synthesize(engine, converter, req: dict, prepare=None) -> dict:
    import numpy as np

    text = req["text"]
    _seed_everything(req.get("seed"))
    if _cuda_live():
        try:
            _torch().cuda.reset_peak_memory_stats()
        except Exception:
            pass
    # `spoken` is what the engine is called with: the sentence itself, or for VCB's own
    # engines what VCB's text step makes of it. Tag counts always come from the sentence as
    # written, and `text_sent` is what the model finally sees after its tag conversion.
    spoken = prepare(text) if prepare else text
    tags_in, tags_dropped = _tag_counts(converter, text)
    text_sent = converter(spoken) if converter else spoken
    info = {"text_sent": text_sent, "tags_in": tags_in, "tags_dropped": tags_dropped}

    # Timed section: from the call until the last audio chunk arrives. Engines
    # without a streaming path yield one chunk, so their first-audio time equals
    # their total time.
    chunks: list = []
    ttfa = None
    t0 = time.perf_counter()
    try:
        async for chunk in engine.synthesize_stream(spoken):
            arr = np.asarray(chunk, dtype=np.float32).reshape(-1)
            if arr.size and ttfa is None:
                ttfa = time.perf_counter() - t0
            chunks.append(arr)
    except Exception as exc:
        info.update(ok=False, error_type=type(exc).__name__, error=_short(exc),
                    total_s=time.perf_counter() - t0, **_memory_report())
        return info
    total = time.perf_counter() - t0

    audio = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    sample_rate = int(engine.sample_rate)
    info.update(total_s=total, ttfa_s=ttfa, n_chunks=len(chunks), sample_rate=sample_rate,
                **_memory_report())
    # A run only counts if it produced real audio.
    if audio.size == 0:
        return {**info, "ok": False, "error_type": "EmptyAudio", "error": "engine returned no audio"}
    if not np.isfinite(audio).all():
        return {**info, "ok": False, "error_type": "InvalidAudio", "error": "NaN or Inf samples"}
    if float(np.abs(audio).max()) < 1e-4:
        return {**info, "ok": False, "error_type": "SilentAudio", "error": "audio is silent"}
    info.update(ok=True, audio_s=audio.size / sample_rate,
                peak=float(np.abs(audio).max()))
    if req.get("save_to"):
        _write_wav(req["save_to"], audio, sample_rate)
        info["wav"] = req["save_to"]
    return info


def _versions() -> dict:
    out = {"python": sys.version.split()[0],
           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    torch = _torch()
    if torch is not None:
        out["torch"] = getattr(torch, "__version__", None)
        out["cuda_runtime"] = getattr(getattr(torch, "version", None), "cuda", None)
        if _cuda_live():
            try:       # proves which physical GPU the engine really ran on
                out["gpu_name"] = torch.cuda.get_device_name(0)
            except Exception:
                pass
    return out


def _network_guard(init: dict):
    """Cut the process off from the network unless downloads were allowed; returns a function
    that lists what was refused. It runs before any engine code is imported."""
    if not init.get("offline"):
        return lambda: []
    from tts_bench import netguard
    netguard.block_outbound()
    return netguard.blocked_attempts


async def _serve(proto, init: dict) -> int:
    for path in (init["tools_dir"], init["repo_root"]):
        if path not in sys.path:
            sys.path.insert(0, path)
    blocked = _network_guard(init)

    try:
        module = importlib.import_module(init["module"])
        cls = getattr(module, init["cls"])
        converter = _resolve_converter(module, init.get("tag_converter"))
        from vcb.config import TTSConfig

        # The two settings every engine gets from the benchmark, plus the ones the registry says
        # this engine needs (CSM's reference clip); everything else keeps VCB's code defaults
        # (the parent hides .env and VCB_* variables from this process).
        cfg = TTSConfig(voice_wav=init["voice_wav"], device=init["device"], **(init.get("tts_config") or {}))
        options = _build_options(cls, init.get("options") or {}, init.get("reference_text", ""))
        engine = cls(cfg, options) if options is not None else cls(cfg)
        prepare = _vcb_text_step(engine) if init.get("vcb_native") else None
        t0 = time.perf_counter()
        await engine.load()
        load_s = time.perf_counter() - t0
    except BaseException as exc:  # includes SystemExit raised by a dependency
        traceback.print_exc()         # lands in the engine's worker.log
        _send(proto, {"event": "load_error", "error_type": type(exc).__name__,
                      "error": _short(exc), "blocked_network": blocked()})
        return 3

    _send(proto, {"event": "ready", "load_s": load_s, "sample_rate": int(engine.sample_rate),
                  "supports_tags": bool(getattr(engine, "supports_tags", False)),
                  "pid": os.getpid(), "blocked_network": blocked(), **_versions(), **_memory_report()})

    while True:
        line = await asyncio.to_thread(sys.stdin.buffer.readline)
        if not line:                      # the parent went away
            break
        req = json.loads(line.decode("utf-8"))
        if req.get("op") == "quit":
            break
        if req.get("op") != "synth":
            continue
        # Safety net: if the parent cannot kill us (it lives on the other side of
        # a WSL boundary, say), a hung synthesis ends the process after the limit.
        watchdog = threading.Timer(float(req.get("timeout_s", 30)) + WATCHDOG_GRACE_S,
                                   lambda: os._exit(70))
        watchdog.daemon = True
        watchdog.start()
        try:
            result = await _synthesize(engine, converter, req, prepare)
        except BaseException as exc:
            result = {"ok": False, "error_type": type(exc).__name__, "error": _short(exc)}
        finally:
            watchdog.cancel()
        _send(proto, {"event": "result", "id": req.get("id"), **result})

    # Unload: let the engine release what it holds, then drop the references and
    # return cached CUDA blocks. The process exit that follows frees the rest.
    try:
        await engine.close()
    except Exception as exc:
        print(f"engine.close() failed: {exc}", file=sys.stderr)
    del engine
    gc.collect()
    if _cuda_live():
        try:
            _torch().cuda.synchronize()
            _torch().cuda.empty_cache()
        except Exception:
            pass
    _send(proto, {"event": "bye", "blocked_network": blocked()})
    return 0


def main() -> int:
    # Never read the user's .env (API keys and personal settings) in a benchmark
    # process, and keep VCB's own configuration out of the measurement.
    os.environ.setdefault("VCB_ENV_FILE", os.path.join(tempfile.gettempdir(), "vcb-bench-no-env"))
    proto = _claim_stdout()
    first = sys.stdin.buffer.readline()
    if not first:
        return 2
    init = json.loads(first.decode("utf-8"))
    if init.get("op") != "init":
        print("first message must be init", file=sys.stderr)
        return 2
    return asyncio.run(_serve(proto, init))


if __name__ == "__main__":
    sys.exit(main())
