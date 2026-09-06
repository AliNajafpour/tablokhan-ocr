import os
import sys
from pathlib import Path


ROOT = Path(__file__).parent
_dll_handles = []
_configured = False


def gpu_device():
    global _configured
    if not _configured:
        # PaddleOCR imports ModelScope, which imports Torch. Load Torch before
        # Paddle's CUDA DLL folders are added so Windows resolves the right DLLs.
        import torch

        local_paddle = ROOT / "tmp" / "paddle-gpu"
        if local_paddle.is_dir():
            sys.path.insert(0, str(local_paddle))
        cuda_bins = list((ROOT / "tmp" / "paddle-cuda" / "nvidia").glob("*/bin"))
        if cuda_bins:
            os.environ["PATH"] = os.pathsep.join(map(str, cuda_bins)) + os.pathsep + os.environ["PATH"]
            if hasattr(os, "add_dll_directory"):
                _dll_handles.extend(os.add_dll_directory(str(path)) for path in cuda_bins)
        _configured = True

    import paddle
    if not paddle.is_compiled_with_cuda() or paddle.device.cuda.device_count() == 0:
        raise RuntimeError("Paddle GPU is unavailable. Install paddlepaddle-gpu and restart the app.")
    return "gpu:0"
