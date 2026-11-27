"""What the benchmark knows about each engine, without importing any of them.

Everything here is plain data (module/class names as strings, package pins,
weight sizes, licenses), so listing engines or checking an installation never
loads a model or pulls in torch. An engine's adapter module is imported only by
the worker process that actually runs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

MODES = ("default", "clone")


@dataclass(frozen=True)
class WeightRepo:
    repo: str                              # Hugging Face repo id
    size_gb: float                         # as listed on the Hub (decimal GB); for VCB's own engines, what VCB loads
    modes: tuple[str, ...] = MODES         # voice modes that need this repo


@dataclass(frozen=True)
class EngineSpec:
    key: str                               # name used on the command line
    title: str
    module: str                            # adapter module, imported by the worker only
    cls: str
    modes: tuple[str, ...]                 # "default" = built-in voice, "clone" = reference clip
    needs_reference_text: bool             # cloning needs the clip's exact transcript
    tag_converter: str | None              # canonical tags -> what the engine receives: a function of the
                                           # adapter module, or "module:function" for one from another module
    tag_coverage: dict                     # canonical tag -> does the engine perform it natively
    weights: tuple[WeightRepo, ...]
    license_code: str
    license_weights: str
    weights_commercial: bool | None        # False = non-commercial, None = not stated
    license_url: str
    platform: str                          # where it can run
    packages: tuple[str, ...]              # exact pip requirements for the engine's own venv
    setup_notes: tuple[str, ...]           # non-pip install steps
    requires_modules: tuple[str, ...]      # modules the engine's interpreter must provide
    hidden: bool = False                   # test-only engines stay out of `list` / `all`
    required_options: tuple[str, ...] = () # options that must name an existing folder (a git clone)
    linux_only: bool = False               # its runtime has no Windows build (use WSL2)
    # One of VCB's own engines. It is handed text the way VCB hands it (the sanitize step of the
    # orchestrator), VCB installs it (this tool never downloads it), and it is configured through
    # VCB's TTSConfig.
    vcb_native: bool = False
    # One of VCB's own engines that VCB no longer offers or installs (RETIRED_TTS_ENGINES in
    # vcb/hardware.py). It still runs wherever an earlier VCB installed it.
    retired: bool = False
    tts_config: dict = field(default_factory=dict)    # extra TTSConfig fields; "$voice_wav" and
                                                      # "$reference_text" stand for the benchmark's own values
    requires_files: tuple[str, ...] = ()   # files the engine needs that no Hugging Face cache holds
                                           # (relative to the repo, "~" and %VARIABLES% expanded)

    def weight_repos(self, modes: tuple[str, ...] | None = None) -> list[WeightRepo]:
        modes = modes or self.modes
        return [w for w in self.weights if any(m in modes for m in w.modes)]

    def download_gb(self, modes: tuple[str, ...] | None = None) -> float:
        return round(sum(w.size_gb for w in self.weight_repos(modes)), 2)


_TORCH_CU128 = ("torch==2.9.1", "torchaudio==2.9.1")   # from https://download.pytorch.org/whl/cu128


def _tags(laugh=False, sigh=False, hesitation=False, breath=False) -> dict:
    return {"laugh": laugh, "sigh": sigh, "hesitation": hesitation, "breath": breath}


# Engines without sound tags. VCB's own text step has already turned <laugh> and <sigh> into the
# words it speaks instead ("Haha!", "Haaah.") and dropped the other tags; this converter only
# removes whatever tag is left, so what the engine receives is what the CSV reports.
_PLAIN_TEXT = "vcb.sanitize:strip_speech_tags"
_VCB_PLATFORM = "Windows (VCB's own venv; VCB is a Windows application)"
_VCB_INSTALLS = ("Installed and updated by VCB itself (`VCB.bat --extras` installs every voice); "
                 "this tool never downloads it.")
_VCB_RETIRED = ("Retired from VCB: its settings no longer offer it and VCB no longer installs it, so it "
                "runs only where an earlier VCB installed it; this tool never downloads it.")
# Said wherever one of VCB's own engines is missing.
VCB_INSTALL_HINT = "pick the voice in `VCB Settings.bat` and start `VCB.bat` once, or run `VCB.bat --extras`"
RETIRED_INSTALL_HINT = ("VCB no longer installs this retired voice; to bring it back, see "
                        "RETIRED_TTS_ENGINES in vcb/hardware.py")


def install_hint(spec: EngineSpec) -> str:
    """How to install one of VCB's own engines that is missing."""
    return RETIRED_INSTALL_HINT if spec.retired else VCB_INSTALL_HINT

ENGINES: dict[str, EngineSpec] = {s.key: s for s in (
    EngineSpec(
        key="chatterbox",
        title="Chatterbox Turbo (VCB baseline)",
        module="vcb.tts.chatterbox_turbo_tts", cls="ChatterboxTurboTTS",
        modes=("default", "clone"), needs_reference_text=False,
        tag_converter="_to_turbo_tags",
        tag_coverage=_tags(laugh=True, sigh=True),
        weights=(WeightRepo("ResembleAI/chatterbox-turbo", 4.04),),
        license_code="MIT", license_weights="MIT", weights_commercial=True,
        license_url="https://huggingface.co/ResembleAI/chatterbox-turbo",
        platform="Windows or Linux (VCB's own venv)",
        packages=(), setup_notes=("Already installed in VCB's venv (chatterbox-tts 0.1.7).", _VCB_INSTALLS),
        requires_modules=("torch", "chatterbox"),
        vcb_native=True,
    ),
    EngineSpec(
        key="kokoro",
        title="Kokoro 82M (VCB)",
        module="vcb.tts.kokoro_tts", cls="KokoroTTS",
        modes=("default",), needs_reference_text=False,
        tag_converter=_PLAIN_TEXT, tag_coverage=_tags(),
        weights=(WeightRepo("hexgrad/Kokoro-82M", 0.33, modes=("default",)),),
        license_code="Apache-2.0", license_weights="Apache-2.0", weights_commercial=True,
        license_url="https://huggingface.co/hexgrad/Kokoro-82M",
        platform=_VCB_PLATFORM, packages=(),
        setup_notes=(_VCB_INSTALLS, "Preset voice only (VCB's `voice`, af_heart); it cannot clone."),
        requires_modules=("torch", "kokoro"),
        vcb_native=True,
    ),
    EngineSpec(
        key="xtts",
        title="XTTS v2 (VCB, Coqui)",
        module="vcb.tts.xtts_tts", cls="XTTSv2TTS",
        modes=("clone",), needs_reference_text=False,
        tag_converter=_PLAIN_TEXT, tag_coverage=_tags(),
        weights=(),
        license_code="MPL-2.0 (coqui-tts)",
        license_weights="Coqui Public Model License (CPML): non-commercial",
        weights_commercial=False,
        license_url="https://coqui.ai/cpml",
        platform=_VCB_PLATFORM, packages=(),
        setup_notes=(_VCB_RETIRED,
                     "Clones the reference clip; it has no built-in voice.",
                     "The model (1.8 GB) is in Coqui's own folder, %LOCALAPPDATA%\\tts, not in the "
                     "Hugging Face cache (the check looks there; TTS_HOME or XDG_DATA_HOME would move it).",
                     "Coqui's model manager asks huggingface.co for a changed config.json every time "
                     "XTTS loads and downloads the whole model again when it differs. The benchmark "
                     "blocks that request (see `--allow-download`)."),
        requires_modules=("torch", "TTS"),
        requires_files=("%LOCALAPPDATA%/tts/tts_models--multilingual--multi-dataset--xtts_v2/model.pth",),
        vcb_native=True, retired=True,
    ),
    EngineSpec(
        key="orpheus",
        title="Orpheus 3B (VCB, llama.cpp)",
        module="vcb.tts.orpheus_tts", cls="OrpheusTTS",
        modes=("default",), needs_reference_text=False,
        tag_converter="to_orpheus_tags",
        tag_coverage=_tags(laugh=True, sigh=True),
        weights=(WeightRepo("isaiahbjork/orpheus-3b-0.1-ft-Q4_K_M-GGUF", 2.36, modes=("default",)),
                 WeightRepo("onnx-community/snac_24khz-ONNX", 0.05, modes=("default",))),
        license_code="MIT (orpheus-cpp)",
        license_weights="Apache-2.0 on the Hub card (the base model is Llama-3.2-3B: Meta's "
                        "Llama 3.2 licence may also apply); the SNAC codec is MIT",
        weights_commercial=True,
        license_url="https://huggingface.co/canopylabs/orpheus-3b-0.1-ft",
        platform=_VCB_PLATFORM + ", llama.cpp CUDA build", packages=(),
        setup_notes=(_VCB_INSTALLS, "Preset voice only (VCB's `orpheus_voice`, tara); it cannot clone.",
                     "Only the SNAC decoder file is loaded from its repo (0.05 of 0.48 GB)."),
        requires_modules=("orpheus_cpp", "llama_cpp"),
        vcb_native=True,
    ),
    EngineSpec(
        key="csm",
        title="Sesame CSM 1B (VCB)",
        module="vcb.tts.sesame_csm_tts", cls="SesameCSMTTS",
        modes=("default", "clone"), needs_reference_text=True,
        tag_converter=_PLAIN_TEXT, tag_coverage=_tags(),
        weights=(WeightRepo("unsloth/csm-1b", 4.17),),
        license_code="Apache-2.0", license_weights="Apache-2.0", weights_commercial=True,
        license_url="https://huggingface.co/unsloth/csm-1b",
        platform=_VCB_PLATFORM, packages=(),
        setup_notes=(_VCB_RETIRED,
                     "Without a reference clip CSM invents a new voice for every line (that is how VCB "
                     "runs it), so its built-in voice changes from sentence to sentence."),
        requires_modules=("torch", "transformers"),
        tts_config={"csm_ref_audio": "$voice_wav", "csm_ref_text": "$reference_text"},
        vcb_native=True, retired=True,
    ),
    EngineSpec(
        key="indextts",
        title="IndexTTS 2.5 (VCB, bilibili)",
        module="vcb.tts.indextts_tts", cls="IndexTTSEngine",
        modes=("clone",), needs_reference_text=False,
        tag_converter=_PLAIN_TEXT, tag_coverage=_tags(),
        weights=(),
        license_code="bilibili Model Use License Agreement",
        license_weights="bilibili Model Use License Agreement: royalty-free, commercial use allowed "
                        "below 100 M monthly users and RMB 1 bn revenue (a separate licence above); "
                        "the model may not be used to improve other AI models except non-commercial ones",
        weights_commercial=True,
        license_url="https://huggingface.co/IndexTeam/IndexTTS-2.5/raw/main/LICENSE",
        platform=_VCB_PLATFORM + ", runs in its own venv under .runtime", packages=(),
        setup_notes=(_VCB_RETIRED,
                     "Clones the reference clip; it has no built-in voice.",
                     "The test sentences carry no mood tags (<excited>, <sad>, ...), which IndexTTS uses "
                     "for its emotion, so that emotion control is not exercised here.",
                     "Every line is rendered from scratch here: VCB's look-ahead (`prepare`) is not "
                     "used, so its first-audio time is longer than in the live app."),
        requires_modules=("numpy",),
        requires_files=(".runtime/indextts/installed.txt", ".runtime/indextts/checkpoints/gpt.pth"),
        vcb_native=True, retired=True,
    ),
    EngineSpec(
        key="breeze",
        title="Breeze TTS 2 (BreezeBlue)",
        module="vcb.tts.breeze_tts", cls="BreezeTTS",
        modes=("default", "clone"), needs_reference_text=True,
        tag_converter="to_breeze_tags",
        tag_coverage={"laugh": True, "sigh": True, "hesitation": True, "breath": True},
        weights=(WeightRepo("BreezeBlue/Breeze-TTS-2", 7.68),),
        license_code="Apache-2.0 (code only)",
        license_weights="BreezeBlue Research and Non-Commercial License v1.1 "
                        "(weights, derivatives and self-hosted outputs: non-commercial)",
        weights_commercial=False,
        license_url="https://huggingface.co/BreezeBlue/Breeze-TTS-2/raw/main/LICENSE",
        platform="README says Linux; native Windows is unverified",
        packages=_TORCH_CU128 + ("qwen-tts==0.1.1", "transformers==4.57.3",
                                 "huggingface-hub>=0.34", "numpy>=2.0", "soundfile>=0.13"),
        setup_notes=("git clone https://github.com/breezeblue-ai/breeze-tts (no PyPI package); "
                     "point [engines.breeze.options] repo_dir at it.",),
        requires_modules=("torch", "qwen_tts", "huggingface_hub", "soundfile"),
        required_options=("repo_dir",),
    ),
    EngineSpec(
        key="voxtral",
        title="Voxtral TTS (Mistral)",
        module="vcb.tts.voxtral_tts", cls="VoxtralTTS",
        modes=("default",), needs_reference_text=False,
        tag_converter="to_voxtral_text",
        tag_coverage={"laugh": False, "sigh": False, "hesitation": False, "breath": False},
        weights=(WeightRepo("mistralai/Voxtral-4B-TTS-2603", 8.04, modes=("default",)),),
        license_code="CC BY-NC 4.0", license_weights="CC BY-NC 4.0 (non-commercial)",
        weights_commercial=False,
        license_url="https://huggingface.co/mistralai/Voxtral-4B-TTS-2603",
        platform="Linux only (vLLM-Omni): use WSL2; the open weights have no voice cloning",
        packages=("vllm==0.20.0", "vllm-omni==0.20.0", "mistral_common>=1.10.0",
                  "torch==2.11.0", "torchaudio==2.11.0", "torchvision==0.26.0",
                  "flashinfer-python==0.6.8.post1", "flashinfer-cubin==0.6.8.post1"),
        setup_notes=("Run the benchmark inside WSL2 for this engine (see the comments in tools/tts_bench.toml).",
                     "Install `vllm==0.20.0` with `uv pip install --torch-backend auto`, then "
                     "`vllm-omni==0.20.0` and `'mistral_common>=1.10.0'`; the torch, torchaudio, "
                     "torchvision and flashinfer pins above are vllm 0.20.0's own."),
        requires_modules=("torch", "vllm", "vllm_omni", "mistral_common"),
        linux_only=True,
    ),
    EngineSpec(
        key="step-audio-editx",
        title="Step-Audio-EditX (StepFun)",
        module="vcb.tts.step_audio_editx_tts", cls="StepAudioEditXTTS",
        modes=("clone",), needs_reference_text=True,
        tag_converter="to_step_audio_tags",
        tag_coverage={"laugh": True, "sigh": True, "hesitation": True, "breath": True},
        weights=(WeightRepo("stepfun-ai/Step-Audio-EditX", 8.31, modes=("clone",)),
                 WeightRepo("stepfun-ai/Step-Audio-Tokenizer", 1.41, modes=("clone",))),
        license_code="Apache-2.0",
        license_weights="not stated: the Hub card has no license field and the repo has no "
                        "weights license file (tokenizer repo: Apache-2.0)",
        weights_commercial=None,
        license_url="https://github.com/stepfun-ai/Step-Audio-EditX",
        platform="Linux only (vLLM wheel, Python 3.12+): use WSL2; has no default voice",
        packages=("vllm 0.14.0rc2.dev125+gc826c72a9 (wheel from wheels.vllm.ai, pinned by the repo)",
                  "torch>=2.9.1", "torchaudio>=2.9.1", "transformers>=4.57.3", "numpy>=2.2.6",
                  "onnxruntime-gpu>=1.23.2", "funasr>=1.3.0", "librosa>=0.11.0",
                  "and the rest of the repo's pyproject.toml (40 direct dependencies, locked by its uv.lock)"),
        setup_notes=("git clone https://github.com/stepfun-ai/Step-Audio-EditX, then `uv sync` "
                     "inside it (Python 3.12 or 3.13); sox and ffmpeg system packages are also needed. "
                     "funasr_detach and stepvocoder are folders of the clone, not packages.",),
        requires_modules=("torch", "vllm"),
        required_options=("repo_dir",),
        linux_only=True,
    ),
    EngineSpec(
        key="qwen3-tts",
        title="Qwen3-TTS 1.7B (Alibaba)",
        module="vcb.tts.qwen3_tts", cls="Qwen3TTS",
        modes=("default", "clone"), needs_reference_text=True,
        tag_converter="to_qwen3_text",
        tag_coverage={"laugh": False, "sigh": False, "hesitation": False, "breath": False},
        weights=(WeightRepo("Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice", 4.52, modes=("default",)),
                 WeightRepo("Qwen/Qwen3-TTS-12Hz-1.7B-Base", 4.54, modes=("clone",))),
        license_code="Apache-2.0", license_weights="Apache-2.0", weights_commercial=True,
        license_url="https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        platform="Windows or Linux (pure Python; community-verified on Windows)",
        packages=_TORCH_CU128 + ("qwen-tts==0.1.1", "accelerate==1.12.0",
                                 "transformers==4.57.3", "soundfile"),
        setup_notes=("qwen-tts pins transformers==4.57.3, which conflicts with VCB's "
                     "transformers 4.57.6: install it in its own venv.",),
        requires_modules=("torch", "qwen_tts"),
    ),
    EngineSpec(
        key="fake",
        title="Fake engine (harness self-test)",
        module="tts_bench.fake_engine", cls="FakeTTS",
        modes=("default", "clone"), needs_reference_text=False,
        tag_converter="to_fake_tags",
        tag_coverage={"laugh": True, "sigh": True, "hesitation": False, "breath": False},
        weights=(), license_code="n/a", license_weights="n/a", weights_commercial=True,
        license_url="", platform="anywhere", packages=(), setup_notes=(),
        requires_modules=(), hidden=True,
    ),
    EngineSpec(
        key="fake-native",
        title="Fake engine standing in for one of VCB's own (harness self-test)",
        module="tts_bench.fake_engine", cls="FakeNativeTTS",
        modes=("default", "clone"), needs_reference_text=True,
        tag_converter=_PLAIN_TEXT, tag_coverage=_tags(),
        weights=(), license_code="n/a", license_weights="n/a", weights_commercial=True,
        license_url="", platform="anywhere", packages=(), setup_notes=(),
        requires_modules=(), hidden=True,
        tts_config={"csm_ref_audio": "$voice_wav", "csm_ref_text": "$reference_text"},
        vcb_native=True,
    ),
)}

# The voices VCB offers, then its retired ones, then the four optional engines: the report lists
# engines in this order.
VCB_ENGINES = ("chatterbox", "kokoro", "orpheus")
RETIRED_VCB_ENGINES = ("xtts", "csm", "indextts")
OPTIONAL_ENGINES = ("breeze", "voxtral", "step-audio-editx", "qwen3-tts")
BENCH_ENGINES = VCB_ENGINES + RETIRED_VCB_ENGINES + OPTIONAL_ENGINES
GROUPS = {"all": BENCH_ENGINES, "vcb": VCB_ENGINES}      # names that stand for several engines
DEFAULT_ENGINES = ("chatterbox",)      # nothing but the baseline runs unless you pick more


def get_engine(key: str) -> EngineSpec:
    try:
        return ENGINES[key]
    except KeyError:
        known = ", ".join(k for k, s in ENGINES.items() if not s.hidden)
        raise ValueError(f"unknown engine {key!r} (available: {known}, or 'vcb' / 'all' for several)") from None


def select_engines(arg: str | None) -> list[EngineSpec]:
    """Engines named on the command line: "a,b", "vcb" for VCB's own engines, or
    "all" for every benchmark engine."""
    if not arg:
        return [ENGINES[k] for k in DEFAULT_ENGINES]
    keys: list[str] = []
    for part in arg.split(","):
        part = part.strip()
        if not part:
            continue
        for key in GROUPS.get(part, (part,)):
            if key not in keys:
                keys.append(key)
    return [get_engine(k) for k in keys]


def variant_name(engine: str, mode: str) -> str:
    """Label of one (engine, voice mode) run, also its output folder:
    "breeze" for the built-in voice, "breeze_clone" for the reference clip."""
    return engine if mode == "default" else f"{engine}_{mode}"
