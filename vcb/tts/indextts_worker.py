"""IndexTTS-2.5 synthesis worker. NOT imported by VCB: it runs inside IndexTTS's
own isolated Python (.runtime/indextts/venv: Python 3.11, torch 2.8, pinned
transformers), which can't share VCB's environment. VCB's IndexTTSEngine starts
it as a subprocess and talks to it over pipes:

    stdin  <- one JSON request per line: {"text": "...", "emo": [8 floats] | null}
    stdout -> one JSON header line per reply, then (for audio) n float32 samples:
              {"ready": true, "sr": 22050}   once, after the model loaded
              {"n": 12345}                   + 4*n bytes of mono float32 audio
              {"error": "..."}               a request failed (worker keeps running)

Everything IndexTTS prints goes to stderr so stdout stays a clean channel.

    python indextts_worker.py --model-dir DIR --voice WAV [--device cuda:0] [--fp32]
    python indextts_worker.py --model-dir DIR --prepare   # download helper models
"""

import argparse
import json
import os
import sys
import warnings


def _claim_stdout():
    """Keep the real stdout for the protocol and point everything else (Python
    prints and native libraries writing to fd 1) at stderr."""
    proto = os.fdopen(os.dup(1), "wb", buffering=0)
    os.dup2(2, 1)
    sys.stdout = sys.stderr
    return proto


def _send(proto, header, payload=b""):
    proto.write(json.dumps(header).encode() + b"\n" + payload)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--voice", default="")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--fp32", action="store_true", help="full precision: ~3x slower")
    ap.add_argument("--steps", type=int, default=25,
                    help="mel diffusion steps (IndexTTS uses 25; fewer = faster)")
    ap.add_argument("--prepare", action="store_true",
                    help="only download the helper models IndexTTS fetches on first load")
    args = ap.parse_args()
    warnings.filterwarnings("ignore")

    model_dir = os.path.abspath(args.model_dir)
    if args.prepare:
        from indextts.utils.model_download import ensure_models_available
        ensure_models_available(model_dir)
        return 0

    proto = _claim_stdout()
    import numpy as np
    from indextts.infer_v2_5 import IndexTTS2

    tts = IndexTTS2(
        cfg_path=os.path.join(model_dir, "config.yaml"), model_dir=model_dir,
        use_bf16=not args.fp32, device=args.device, use_cuda_kernel=False,
    )
    voice = os.path.abspath(args.voice)

    # IndexTTS hardcodes 25 diffusion steps for the mel decoder (the slowest
    # stage on short sentences); override the step argument of its call:
    # cfm.inference(cond, lengths, ref_mel, style, f0, steps, ...).
    cfm = tts.s2mel.models["cfm"]
    if args.steps != 25:
        orig_inference = cfm.inference

        def fewer_steps(*a, **kw):
            if len(a) > 5:
                a = (*a[:5], args.steps, *a[6:])
            return orig_inference(*a, **kw)

        cfm.inference = fewer_steps

    def synth(text, emo):
        sr, wav = tts.infer(spk_audio_prompt=voice, text=text, output_path=None,
                            lang="EN", emo_vector=emo)
        return sr, (np.asarray(wav, dtype=np.float32).reshape(-1) / 32767.0)

    sr, _ = synth("Hello there, nice to meet you.", None)   # pay first-call costs now
    import torch
    vram = torch.cuda.max_memory_reserved() / 2**30 if torch.cuda.is_available() else 0.0
    _send(proto, {"ready": True, "sr": sr, "vram_gb": round(vram, 2)})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
            if req.get("quit"):
                break
            _, audio = synth(req["text"], req.get("emo"))
            data = audio.astype("<f4").tobytes()
            _send(proto, {"n": len(audio)}, data)
        except Exception as exc:  # report and keep serving
            _send(proto, {"error": f"{type(exc).__name__}: {exc}"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
