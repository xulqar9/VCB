"""Step-Audio-EditX (StepFun) through the official repository code.

Step-Audio-EditX has no pip package: `repo_dir` must point at a checkout of
https://github.com/stepfun-ai/Step-Audio-EditX (set up with `uv sync`), whose
top-level modules (tokenizer, tts, ...) are put on sys.path when the engine
loads. It runs on vLLM, which exists for Linux only (use WSL2 on Windows).
Weights: Step-Audio-EditX (8.3 GB) and Step-Audio-Tokenizer (1.4 GB), both read
from the local Hugging Face cache; vllm, torch and the repo code are imported
inside load(), so importing this module costs nothing.

LICENSE: the code is Apache-2.0, but no license is stated for the weights on the
Hub: check with StepFun before any commercial use. The README forbids voice
cloning without consent, so use only reference clips you have the rights to.

It is a cloning and editing model with no built-in voice, so it runs only with a
reference clip plus the clip's exact transcript. It does not voice inline sound
tags: it adds them by EDITING speech it already synthesised. A sentence with
tags therefore takes two passes (clone the plain text, then a "paralinguistic"
edit of that audio with the tags inserted) and the measured time covers both;
a sentence without tags takes one. Output is not streamed.

Optional adapter: it is not wired into VCB's engine factory.
"""

from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import TTSConfig, resolve_path
from .base import TTSEngine

log = logging.getLogger("vcb.tts.step_audio_editx")

# The current README table (Jan 2026 weights): [laugh] [sigh] [breath] and [uhm]
# ("Hesitation sound"). The first release spelled them [Laughter] [Sigh]
# [Breathing] [Uhm]; which spelling the Hub weights prefer is not confirmed here.
# Unmapped tags are dropped, or the edit step would read the word out loud.
_STEP_TAGS = {
    "laugh": "[laugh]",
    "sigh": "[sigh]",
    "breath": "[breath]",
    "hesitation": "[uhm]",
}
_ANGLE_TAG = re.compile(r"<\s*([a-zA-Z_]+)\s*>")


def _convert(text: str, table: dict[str, str]) -> str:
    out = _ANGLE_TAG.sub(lambda m: table.get(m.group(1).lower()) or "\x00", text)
    out = re.sub(r" *\x00 *(?=[,.;:!?])", "", out)
    return re.sub(r" *\x00 *", " ", out).strip()


def to_step_audio_tags(text: str) -> str:
    return _convert(text, _STEP_TAGS)


def _plain_text(text: str) -> str:
    """The sentence with every tag removed: what the first (cloning) pass speaks."""
    return _convert(text, {})


def _snapshot(repo: str) -> str:
    """Local folder of a model. An existing path is used as is; a Hub id is
    resolved from the Hugging Face cache (downloaded only when the environment
    is not offline, i.e. the benchmark was started with --allow-download)."""
    local = resolve_path(repo)
    if local and os.path.isdir(local):
        return local
    from huggingface_hub import snapshot_download

    offline = os.environ.get("HF_HUB_OFFLINE", "").strip().lower() in ("1", "true", "yes", "on")
    try:
        return snapshot_download(repo, local_files_only=offline)
    except Exception as exc:
        raise FileNotFoundError(
            f"could not get {repo} from the Hugging Face cache ({type(exc).__name__}); "
            "download it first: python tools/tts_benchmark.py download --engines step-audio-editx"
        ) from exc


def _repo_dir(value: str) -> str:
    if not value.strip():
        raise ValueError(
            "Step-Audio-EditX needs repo_dir: a checkout of https://github.com/stepfun-ai/Step-Audio-EditX "
            "([engines.step-audio-editx.options] repo_dir in tools/tts_bench.toml)"
        )
    path = resolve_path(value)
    missing = [p for p in ("tts.py", "tokenizer.py", "model_loader.py") if not (Path(path) / p).is_file()]
    if missing:
        raise FileNotFoundError(
            f"repo_dir {path} is not a Step-Audio-EditX checkout (missing {', '.join(missing)})")
    return path


def _mono_float32(audio) -> np.ndarray:
    if hasattr(audio, "detach"):
        audio = audio.detach().float().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


class StepAudioEditXTTS(TTSEngine):
    sample_rate = 24_000
    supports_tags = True
    extra_tags = frozenset(_STEP_TAGS)

    @dataclass
    class Options:
        repo_dir: str = ""                                   # checkout of the official repo (required)
        model: str = "stepfun-ai/Step-Audio-EditX"           # Hub id (cached) or a local folder
        tokenizer: str = "stepfun-ai/Step-Audio-Tokenizer"
        reference_text: str = ""                             # exact transcript of the clip (required)
        # The official CLI's defaults (tts_infer.py), which also fit a 16-24 GB card.
        gpu_memory_utilization: float = 0.5
        max_model_len: int = 3072
        max_num_seqs: int = 1
        dtype: str = "bfloat16"
        cosyvoice_dtype: str = "bfloat16"
        cosyvoice_cuda_graph: bool = True
        enforce_eager: bool = False
        chdir_repo: bool = False                             # run from the checkout if its code needs relative paths

    def __init__(self, cfg: TTSConfig, options: StepAudioEditXTTS.Options | None = None) -> None:
        self.cfg = cfg
        self.opt = options or self.Options()
        self._ref = ""
        self._tts = None
        # One dedicated thread keeps load and every generation on the same thread
        # (vLLM's client and the vocoder's CUDA graph are not made for thread hopping).
        self._pool: ThreadPoolExecutor | None = None

    async def load(self) -> None:
        o = self.opt
        repo = _repo_dir(o.repo_dir)
        if not self.cfg.voice_wav:
            raise ValueError("Step-Audio-EditX has no built-in voice: run it with a reference clip (clone mode)")
        self._ref = resolve_path(self.cfg.voice_wav)
        if not Path(self._ref).is_file():
            raise FileNotFoundError(f"reference clip not found: {self._ref}")
        if not o.reference_text.strip():
            raise ValueError("Step-Audio-EditX needs the exact transcript of the reference clip (reference_text)")
        if sys.platform == "win32":
            raise RuntimeError("Step-Audio-EditX needs vLLM, which has no Windows build: run the benchmark inside WSL2")
        if self.cfg.device == "cpu":
            raise RuntimeError("Step-Audio-EditX needs a CUDA GPU")

        def _load():
            os.environ.setdefault("VLLM_ATTENTION_BACKEND", "TRITON_ATTN")   # as the official CLI sets it
            model_dir = _snapshot(o.model)
            tokenizer_dir = _snapshot(o.tokenizer)
            if repo not in sys.path:
                sys.path.insert(0, repo)
            if o.chdir_repo:
                os.chdir(repo)
            from tokenizer import StepAudioTokenizer
            from tts import StepAudioTTS

            audio_tokenizer = StepAudioTokenizer(tokenizer_dir, model_source="local")
            return StepAudioTTS(
                model_dir, audio_tokenizer, model_source="local",
                gpu_memory_utilization=o.gpu_memory_utilization, max_model_len=o.max_model_len,
                max_num_seqs=o.max_num_seqs, dtype=o.dtype, enforce_eager=o.enforce_eager,
                cosyvoice_dtype=o.cosyvoice_dtype, cosyvoice_cuda_graph=o.cosyvoice_cuda_graph,
            )

        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="step-audio")
        self._tts = await asyncio.get_running_loop().run_in_executor(self._pool, _load)
        log.info("step-audio-editx loaded (%s), voice: %s", o.model, self._ref)

    def _generate(self, text: str) -> tuple[np.ndarray, int]:
        o = self.opt
        tagged = to_step_audio_tags(text)
        plain = _plain_text(text)
        if not plain:
            raise ValueError("nothing left to say once the tags are removed")
        audio, rate = self._tts.clone(
            prompt_wav_path=self._ref, prompt_text=o.reference_text.strip(), target_text=plain)
        if tagged != plain:
            # Sound tags are applied by editing the speech just produced.
            import soundfile

            workdir = tempfile.mkdtemp(prefix="stepaudio_")
            try:
                first_pass = os.path.join(workdir, "first_pass.wav")
                soundfile.write(first_pass, _mono_float32(audio), int(rate))
                audio, rate = self._tts.edit(
                    prompt_wav_path=first_pass, prompt_text=plain,
                    edit_type="paralinguistic", target_text=tagged)
            finally:
                shutil.rmtree(workdir, ignore_errors=True)
        return _mono_float32(audio), int(rate)

    async def synthesize(self, text: str) -> np.ndarray:
        assert self._tts is not None and self._pool is not None, "call load() first"
        audio, rate = await asyncio.get_running_loop().run_in_executor(self._pool, self._generate, text)
        self.sample_rate = rate
        return audio

    async def close(self) -> None:
        self._tts = None
        if self._pool is not None:
            self._pool.shutdown(wait=False)
            self._pool = None
