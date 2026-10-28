"""Make the CUDA 12 DLLs from the nvidia-* wheels findable on Windows.

CTranslate2 (faster-whisper) and onnxruntime-gpu's CUDA provider link
cuBLAS/cuDNN by name; the wheels drop them in site-packages\\nvidia\\*\\bin,
which isn't on the DLL search path. Harmless on other platforms / without
the wheels installed."""

import os
import sys

_PACKAGES = ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime", "cufft")


def register_cuda_dlls() -> None:
    if sys.platform != "win32":
        return
    import site

    for base in site.getsitepackages():
        for pkg in _PACKAGES:
            path = os.path.join(base, "nvidia", pkg, "bin")
            if os.path.isdir(path):
                os.add_dll_directory(path)
                os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
